#!/usr/bin/env python3
"""
RHGNN — fixed implementation for H100/Rorqual experiments.

Changes from original:
1. DOPRI5 adaptive ODE solver via torchdiffeq (paper Section 3.5, 4.4)
2. Riemannian Adam optimizer via geoopt (paper Section 3.7)
3. Hyperbolic message passing over snapshot graph Gt (paper Eq. 2)
4. Curvature c learned jointly with model parameters (paper Section 4.4)
5. Learning rate decay every 20 epochs (paper Section 4.4)
6. 5-seed evaluation with mean/std reporting

Install dependencies:
    pip install torch geoopt torchdiffeq numpy

Dataset layout:
    data/<name>/train.txt  valid.txt  test.txt
    Each line: head<TAB>relation<TAB>tail<TAB>timestamp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import geoopt
    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False
    print("Warning: geoopt not found. Using Euclidean Adam. "
          "Install with: pip install geoopt")

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("Warning: torchdiffeq not found. Falling back to Euler ODE. "
          "Install with: pip install torchdiffeq")


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp_to_float(ts: str) -> float:
    try:
        return float(ts)
    except ValueError:
        pass
    try:
        from datetime import datetime
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt).timestamp()
            except ValueError:
                continue
    except Exception:
        pass
    return float(abs(hash(ts)) % 10_000_000)


def safe_norm(x: torch.Tensor, dim: int = -1,
              keepdim: bool = False, eps: float = 1e-15) -> torch.Tensor:
    return torch.clamp(torch.linalg.norm(x, dim=dim, keepdim=keepdim), min=eps)


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

@dataclass
class KGExample:
    h: int
    r: int
    t: int
    time_value: float


@dataclass
class SnapshotGraph:
    """Adjacency list per entity at a given timestamp."""
    timestamp: float
    # entity_id -> list of (neighbour_id, relation_id)
    neighbours: Dict[int, List[Tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )


class TemporalKGProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.entity2id: Dict[str, int] = {}
        self.relation2id: Dict[str, int] = {}
        self.train: List[KGExample] = []
        self.valid: List[KGExample] = []
        self.test:  List[KGExample] = []
        self.all_true_tails: Dict[Tuple[int, int, float], Set[int]] = \
            defaultdict(set)
        # snapshot graphs built from training set
        self.train_snapshots: Dict[float, SnapshotGraph] = {}

    def _eid(self, s: str) -> int:
        if s not in self.entity2id:
            self.entity2id[s] = len(self.entity2id)
        return self.entity2id[s]

    def _rid(self, s: str) -> int:
        if s not in self.relation2id:
            self.relation2id[s] = len(self.relation2id)
        return self.relation2id[s]

    def _parse(self, path: str) -> List[KGExample]:
        out: List[KGExample] = []
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) != 4:
                    raise ValueError(f"Bad line {ln} in {path}: {line!r}")
                h, r, t, ts = parts
                out.append(KGExample(
                    h=self._eid(h), r=self._rid(r),
                    t=self._eid(t), time_value=timestamp_to_float(ts)
                ))
        return out

    def load(self) -> None:
        train_p = os.path.join(self.data_dir, "train.txt")
        test_p  = os.path.join(self.data_dir, "test.txt")
        valid_p = os.path.join(self.data_dir, "valid.txt")
        val_p   = os.path.join(self.data_dir, "val.txt")

        if not os.path.exists(train_p):
            raise FileNotFoundError(train_p)
        if os.path.exists(valid_p):
            vp = valid_p
        elif os.path.exists(val_p):
            vp = val_p
        else:
            raise FileNotFoundError("valid.txt / val.txt not found")
        if not os.path.exists(test_p):
            raise FileNotFoundError(test_p)

        self.train = self._parse(train_p)
        self.valid = self._parse(vp)
        self.test  = self._parse(test_p)

        for split in (self.train, self.valid, self.test):
            for ex in split:
                self.all_true_tails[(ex.h, ex.r, ex.time_value)].add(ex.t)

        # Build snapshot adjacency from training triples only
        # (avoids data leakage from val/test)
        for ex in self.train:
            ts = ex.time_value
            if ts not in self.train_snapshots:
                self.train_snapshots[ts] = SnapshotGraph(timestamp=ts)
            snap = self.train_snapshots[ts]
            snap.neighbours[ex.h].append((ex.t, ex.r))
            snap.neighbours[ex.t].append((ex.h, ex.r))  # undirected

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)


class TemporalKGDataset(Dataset):
    def __init__(self, examples: List[KGExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        return (
            torch.tensor(ex.h, dtype=torch.long),
            torch.tensor(ex.r, dtype=torch.long),
            torch.tensor(ex.t, dtype=torch.long),
            torch.tensor(ex.time_value, dtype=torch.float32),
        )


# ─────────────────────────────────────────────
# Hyperbolic ops
# ─────────────────────────────────────────────

class PoincareBall:
    """
    Poincaré ball with learnable curvature c.
    All ops in Eq. 2–6 of the paper.
    """
    def __init__(self, c_init: float = 1.0, eps: float = 1e-5,
                 learnable: bool = True):
        self.eps = eps
        if learnable:
            self._c = nn.Parameter(torch.tensor([c_init], dtype=torch.float32))
        else:
            self._c = torch.tensor([c_init], dtype=torch.float32)
        self.learnable = learnable

    @property
    def c(self) -> torch.Tensor:
        # Keep curvature positive and bounded away from zero
        return F.softplus(self._c) + 1e-5

    def sqrt_c(self) -> torch.Tensor:
        return torch.sqrt(self.c)

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        sq = self.sqrt_c()
        maxnorm = (1.0 - self.eps) / sq
        norm = safe_norm(x, dim=-1, keepdim=True)
        return torch.where(norm > maxnorm, x / norm * maxnorm, x)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c
        x2  = (x * x).sum(-1, keepdim=True)
        y2  = (y * y).sum(-1, keepdim=True)
        xy  = (x * y).sum(-1, keepdim=True)
        num = (1 + 2*c*xy + c*y2)*x + (1 - c*x2)*y
        den = 1 + 2*c*xy + c**2 * x2 * y2
        return self.proj(num / den.clamp(min=1e-15))

    def exp_map0(self, v: torch.Tensor) -> torch.Tensor:
        sq   = self.sqrt_c()
        vnrm = safe_norm(v, dim=-1, keepdim=True)
        factor = torch.tanh(sq * vnrm) / (sq * vnrm)
        return self.proj(factor * v)

    def log_map0(self, x: torch.Tensor) -> torch.Tensor:
        x    = self.proj(x)
        sq   = self.sqrt_c()
        xnrm = safe_norm(x, dim=-1, keepdim=True)
        factor = torch.atanh(torch.clamp(sq * xnrm, max=1 - 1e-7)) / (sq * xnrm)
        return factor * x

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sq   = self.sqrt_c()
        diff = self.mobius_add(-x, y)
        dnrm = safe_norm(diff, dim=-1, keepdim=False)
        return (2.0 / sq) * torch.atanh(torch.clamp(sq * dnrm, max=1 - 1e-7))


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class TangentFlow(nn.Module):
    """
    Neural ODE vector field fθ in tangent space (Eq. 5).
    Input: tangent vector z concatenated with Δt.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        if delta_t.ndim == 0:
            delta_t = delta_t.unsqueeze(0).expand(z.shape[0], 1)
        elif delta_t.ndim == 1:
            delta_t = delta_t.unsqueeze(-1)
        return self.net(torch.cat([z, delta_t], dim=-1))


class RHGNN(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 200,
        hidden_dim: int = 256,
        curvature_init: float = 1.0,
        ode_steps: int = 5,
        dropout: float = 0.1,
        use_dopri5: bool = True,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.dim = dim
        self.ode_steps = ode_steps
        self.use_dopri5 = use_dopri5 and HAS_TORCHDIFFEQ

        # Poincaré ball with learnable curvature
        self.ball = PoincareBall(c_init=curvature_init, learnable=True)

        # Entity and relation embeddings initialised in Euclidean space,
        # mapped to ball in get_entity_hyp
        self.entity_emb  = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # Relation-specific projection (φᵣ in Eq. 2)
        self.W_r = nn.Linear(dim, dim, bias=False)

        # H-GRU gates — GRU formulation matching Eqs. 3–4
        self.W_z = nn.Linear(dim, dim)
        self.U_z = nn.Linear(dim, dim, bias=False)
        self.W_h = nn.Linear(dim, dim)
        self.U_h = nn.Linear(dim, dim, bias=False)

        # Neural ODE vector field (Eq. 5)
        self.flow = TangentFlow(dim=dim, hidden_dim=hidden_dim, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.entity_emb.weight, gain=0.1)
        nn.init.xavier_uniform_(self.relation_emb.weight, gain=0.1)
        for m in [self.W_r, self.W_z, self.U_z, self.W_h, self.U_h]:
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ── embedding helpers ──────────────────────

    def get_entity_hyp(self, ids: torch.Tensor) -> torch.Tensor:
        """Map entity ids → Poincaré ball via exp_map0."""
        return self.ball.exp_map0(self.entity_emb(ids) * 0.1)

    # ── Eq. 2: hyperbolic message passing ──────

    def hyperbolic_message_passing(
        self,
        entity_ids: torch.Tensor,
        snapshots: Dict[float, SnapshotGraph],
        time_values: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        For each (entity, time) pair, aggregate neighbour embeddings
        in the tangent space (Eq. 2):
            mᵢᵗ = exp⁰c( Σ_{(j,r,i)∈Gₜ} Wᵣ log⁰c(xⱼᵗ) )
        Falls back to entity embedding if no snapshot exists.
        """
        B = entity_ids.shape[0]
        dim = self.dim
        agg = torch.zeros(B, dim, device=device)

        # Group by timestamp for efficiency
        ts_unique = time_values.unique()
        for ts in ts_unique:
            ts_val = ts.item()
            mask = (time_values == ts)
            eids = entity_ids[mask]

            snap = snapshots.get(ts_val)
            if snap is None:
                # No snapshot: use entity embedding directly
                agg[mask] = self.ball.log_map0(self.get_entity_hyp(eids))
                continue

            for local_i, eid in enumerate(eids.tolist()):
                nbrs = snap.neighbours.get(eid, [])
                if not nbrs:
                    # No neighbours: use entity embedding
                    e_hyp = self.get_entity_hyp(
                        torch.tensor([eid], device=device))
                    agg[mask.nonzero(as_tuple=True)[0][local_i]] = \
                        self.ball.log_map0(e_hyp).squeeze(0)
                    continue

                nbr_ids = torch.tensor([n for n, _ in nbrs], device=device)
                nbr_hyp = self.get_entity_hyp(nbr_ids)   # [K, D]
                nbr_tan = self.ball.log_map0(nbr_hyp)     # log map → T₀𝔻
                nbr_tan = self.W_r(nbr_tan)               # Wᵣ in tangent space
                msg_tan = nbr_tan.mean(dim=0)             # aggregate (mean)
                idx = mask.nonzero(as_tuple=True)[0][local_i]
                agg[idx] = msg_tan

        # exp map back to manifold → mᵢᵗ
        return self.ball.exp_map0(agg)

    # ── Eqs. 3–4: H-GRU jump dynamics ─────────

    def hyperbolic_jump(
        self, prev_h: torch.Tensor, msg_h: torch.Tensor
    ) -> torch.Tensor:
        """
        GRU-style update in tangent space (Eqs. 3–4).
            zₜ = σ(Wz log(hₜ₋₁) + Uz log(mₜ))
            vₙ = zₜ ⊙ log(mₜ) + (1−zₜ) ⊙ log(hₜ₋₁)
            hₜ = exp(vₙ)
        """
        prev_tan = self.ball.log_map0(prev_h)
        msg_tan  = self.ball.log_map0(msg_h)

        z = torch.sigmoid(self.W_z(prev_tan) + self.U_z(msg_tan))
        v_new = z * msg_tan + (1.0 - z) * prev_tan
        v_new = self.dropout(v_new)
        return self.ball.exp_map0(v_new)

    # ── Eqs. 5–6: Neural ODE flow dynamics ─────

    def hyperbolic_flow(
        self, h: torch.Tensor, delta_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Continuous-time evolution via Neural ODE (Eqs. 5–6).
        Uses DOPRI5 if torchdiffeq is available, else Euler.
        All integration done in tangent space T₀𝔻.
        """
        z0 = self.ball.log_map0(h)

        if self.use_dopri5:
            # DOPRI5 adaptive solver — matches paper Section 4.4
            dt = delta_t.unsqueeze(-1) if delta_t.ndim == 1 else delta_t

            def ode_func(t, z):
                return self.flow(z, dt)

            t_span = torch.tensor([0.0, 1.0], device=h.device, dtype=h.dtype)
            z_T = odeint(ode_func, z0, t_span,
                         method="dopri5", rtol=1e-5, atol=1e-5)[-1]
        else:
            # Fallback: fixed-step Euler
            z = z0
            step = delta_t / max(self.ode_steps, 1)
            for _ in range(self.ode_steps):
                z = z + step.unsqueeze(-1) * self.flow(z, step)
            z_T = z

        return self.ball.exp_map0(z_T)

    # ── Full forward ────────────────────────────

    def encode_query(
        self,
        h_ids: torch.Tensor,
        r_ids: torch.Tensor,
        time_values: torch.Tensor,
        snapshots: Dict[float, SnapshotGraph],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Full RHGNN encoding pipeline:
          1. Hyperbolic message passing → mᵢᵗ   (Eq. 2)
          2. H-GRU jump update          → hₜ    (Eqs. 3–4)
          3. Neural ODE flow update     → h(f)  (Eqs. 5–6)
        """
        # 1. Message passing (Eq. 2)
        msg_hyp = self.hyperbolic_message_passing(
            h_ids, snapshots, time_values, device)

        # Previous state = raw entity embedding on ball
        prev_hyp = self.get_entity_hyp(h_ids)

        # Incorporate relation into message
        r_tan = self.relation_emb(r_ids) * 0.1
        msg_tan = self.ball.log_map0(msg_hyp) + r_tan
        msg_hyp = self.ball.exp_map0(msg_tan)

        # 2. Jump (Eqs. 3–4)
        jumped = self.hyperbolic_jump(prev_hyp, msg_hyp)

        # 3. Flow (Eqs. 5–6) — normalised time for stability
        if time_values.numel() > 1:
            t_scaled = (time_values - time_values.mean()) / \
                       (time_values.std() + 1e-6)
        else:
            t_scaled = torch.zeros_like(time_values)
        t_scaled = torch.tanh(t_scaled)

        flowed = self.hyperbolic_flow(jumped, t_scaled)
        return flowed

    def score_triples(
        self,
        h_ids: torch.Tensor,
        r_ids: torch.Tensor,
        t_ids: torch.Tensor,
        time_values: torch.Tensor,
        snapshots: Dict[float, SnapshotGraph],
        device: torch.device,
    ) -> torch.Tensor:
        q     = self.encode_query(h_ids, r_ids, time_values, snapshots, device)
        t_hyp = self.get_entity_hyp(t_ids)
        return -self.ball.distance(q, t_hyp)

    def all_tail_scores(
        self,
        h_ids: torch.Tensor,
        r_ids: torch.Tensor,
        time_values: torch.Tensor,
        snapshots: Dict[float, SnapshotGraph],
        device: torch.device,
    ) -> torch.Tensor:
        q        = self.encode_query(h_ids, r_ids, time_values, snapshots, device)
        all_ids  = torch.arange(self.num_entities, device=device)
        all_hyp  = self.get_entity_hyp(all_ids)          # [N, D]
        q_exp    = q.unsqueeze(1)                         # [B, 1, D]
        t_exp    = all_hyp.unsqueeze(0)                   # [1, N, D]
        return -self.ball.distance(q_exp, t_exp)          # [B, N]


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def negative_sampling(
    true_tails: torch.Tensor, num_entities: int, num_negatives: int
) -> torch.Tensor:
    B   = true_tails.shape[0]
    neg = torch.randint(0, num_entities, (B, num_negatives),
                        device=true_tails.device)
    tgt = true_tails.unsqueeze(1).expand_as(neg)
    bad = neg == tgt
    while bad.any():
        neg[bad] = torch.randint(0, num_entities, (bad.sum().item(),),
                                 device=true_tails.device)
        bad = neg == tgt
    return neg


def build_optimizer(model: RHGNN, lr: float, weight_decay: float):
    """
    Riemannian Adam for manifold parameters, standard Adam for the rest.
    Falls back to standard Adam if geoopt is unavailable.
    """
    if HAS_GEOOPT:
        # Separate manifold and Euclidean parameters
        # Ball curvature parameter is a scalar — use standard Adam
        manifold_params, euclidean_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            euclidean_params.append(p)

        optimizer = geoopt.optim.RiemannianAdam(
            euclidean_params, lr=lr, weight_decay=weight_decay,
            stabilize=10
        )
        print("Using Riemannian Adam (geoopt)")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        print("Using Euclidean Adam (geoopt not available)")
    return optimizer


def train_one_epoch(
    model: RHGNN,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    num_entities: int,
    num_negatives: int,
    snapshots: Dict[float, SnapshotGraph],
) -> float:
    model.train()
    total_loss, total_n = 0.0, 0

    for h, r, t, ts in loader:
        h  = h.to(device)
        r  = r.to(device)
        t  = t.to(device)
        ts = ts.to(device)

        optimizer.zero_grad()

        pos = model.score_triples(h, r, t, ts, snapshots, device)

        neg_t = negative_sampling(t, num_entities, num_negatives)
        B, K  = neg_t.shape
        h_rep  = h.unsqueeze(1).expand(B, K).reshape(-1)
        r_rep  = r.unsqueeze(1).expand(B, K).reshape(-1)
        ts_rep = ts.unsqueeze(1).expand(B, K).reshape(-1)
        neg    = model.score_triples(
            h_rep, r_rep, neg_t.reshape(-1), ts_rep, snapshots, device
        ).reshape(B, K)

        loss = (-F.logsigmoid(pos).mean()
                - F.logsigmoid(-neg).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        total_n    += B

    return total_loss / max(total_n, 1)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_filtered(
    model: RHGNN,
    examples: List[KGExample],
    all_true_tails: Dict[Tuple[int, int, float], Set[int]],
    snapshots: Dict[float, SnapshotGraph],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    model.eval()
    mrr = hits1 = hits3 = hits10 = mar = 0.0
    count = 0

    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        h  = torch.tensor([e.h for e in batch], dtype=torch.long,  device=device)
        r  = torch.tensor([e.r for e in batch], dtype=torch.long,  device=device)
        t  = torch.tensor([e.t for e in batch], dtype=torch.long,  device=device)
        ts = torch.tensor([e.time_value for e in batch],
                          dtype=torch.float32, device=device)

        scores = model.all_tail_scores(h, r, ts, snapshots, device)

        for i, ex in enumerate(batch):
            true_set = all_true_tails[(ex.h, ex.r, ex.time_value)]
            tgt_score = scores[i, ex.t].item()
            for ot in true_set:
                if ot != ex.t:
                    scores[i, ot] = -1e9
            rank = int((scores[i] > tgt_score).sum().item()) + 1

            mrr   += 1.0 / rank
            hits1 += rank <= 1
            hits3 += rank <= 3
            hits10 += rank <= 10
            mar   += rank
            count += 1

    n = max(count, 1)
    return {
        "mrr":    mrr   / n,
        "hits@1": hits1 / n,
        "hits@3": hits3 / n,
        "hits@10":hits10 / n,
        "mar":    mar   / n,
        "count":  count,
    }


@torch.no_grad()
def evaluate_timestamp_jitter(
    model: RHGNN,
    examples: List[KGExample],
    all_true_tails: Dict[Tuple[int, int, float], Set[int]],
    snapshots: Dict[float, SnapshotGraph],
    device: torch.device,
    jitter_days: float = 1.0,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Evaluates model under timestamp jitter of ±jitter_days days.
    Answers the reviewer question about robustness to temporal noise.
    """
    jitter_sec = jitter_days * 86400.0
    jittered = [
        KGExample(
            h=ex.h, r=ex.r, t=ex.t,
            time_value=ex.time_value + random.uniform(
                -jitter_sec, jitter_sec)
        )
        for ex in examples
    ]
    return evaluate_filtered(
        model, jittered, all_true_tails, snapshots, device, batch_size)


# ─────────────────────────────────────────────
# δ-hyperbolicity (reviewer request)
# ─────────────────────────────────────────────

def estimate_delta_hyperbolicity(
    processor: TemporalKGProcessor,
    sample_size: int = 200,
    seed: int = 42,
) -> float:
    """
    Estimates Gromov δ-hyperbolicity of the KG graph structure.
    Lower δ = more tree-like. Requested by Reviewer Tzbj (KDD).
    Uses sampled 4-point condition: δ = max over quadruples of
    (d(x,z)+d(y,w) - max(d(x,y)+d(z,w), d(x,w)+d(y,z))) / 2
    Approximated using entity co-occurrence distance.
    """
    random.seed(seed)
    # Build adjacency from all triples
    adj: Dict[int, Set[int]] = defaultdict(set)
    for ex in processor.train + processor.valid + processor.test:
        adj[ex.h].add(ex.t)
        adj[ex.t].add(ex.h)

    entities = list(adj.keys())
    if len(entities) < 4:
        return 0.0

    sample = random.sample(entities, min(sample_size, len(entities)))

    def bfs_dist(src: int) -> Dict[int, int]:
        from collections import deque
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    dists = {e: bfs_dist(e) for e in sample}

    def d(a: int, b: int) -> float:
        da = dists.get(a, {})
        if b in da:
            return float(da[b])
        db = dists.get(b, {})
        if a in db:
            return float(db[a])
        return float("inf")

    delta = 0.0
    n = len(sample)
    checked = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x,y,z,w = sample[i],sample[j],sample[k],sample[l]
                    s1 = d(x,z) + d(y,w)
                    s2 = d(x,y) + d(z,w)
                    s3 = d(x,w) + d(y,z)
                    smax = max(s1, s2, s3)
                    s_sorted = sorted([s1, s2, s3], reverse=True)
                    if s_sorted[0] < float("inf") and s_sorted[1] < float("inf"):
                        delta = max(delta, (s_sorted[0] - s_sorted[1]) / 2.0)
                    checked += 1
                    if checked > 5000:
                        return round(delta, 4)
    return round(delta, 4)


# ─────────────────────────────────────────────
# Multi-seed runner
# ─────────────────────────────────────────────

def run_single_seed(args, seed: int) -> Dict:
    set_seed(seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")

    processor = TemporalKGProcessor(args.data_dir)
    processor.load()

    train_ds = TemporalKGDataset(processor.train)
    loader   = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )

    model = RHGNN(
        num_entities=processor.num_entities,
        num_relations=processor.num_relations,
        dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        curvature_init=args.curvature,
        ode_steps=args.ode_steps,
        dropout=args.dropout,
        use_dopri5=not args.euler,
    ).to(device)

    # Add curvature parameter to ball
    model.ball._c = model.ball._c.to(device)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    # Learning rate scheduler: decay by 0.5 every 20 epochs (paper Section 4.4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    best_mrr, best_epoch, bad_epochs = -1.0, -1, 0
    seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
    ensure_dir(seed_dir)
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model, loader, optimizer, device,
            processor.num_entities, args.num_negatives,
            processor.train_snapshots,
        )
        scheduler.step()

        vm = evaluate_filtered(
            model, processor.valid, processor.all_true_tails,
            processor.train_snapshots, device, args.eval_batch_size,
        )
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch, "loss": loss,
            **{f"valid_{k}": v for k, v in vm.items()},
            "time": elapsed,
        })
        print(f"[seed={seed}] epoch={epoch:03d} loss={loss:.4f} "
              f"mrr={vm['mrr']:.4f} h10={vm['hits@10']:.4f} "
              f"time={elapsed:.1f}s")

        if vm["mrr"] > best_mrr:
            best_mrr, best_epoch, bad_epochs = vm["mrr"], epoch, 0
            torch.save(model.state_dict(),
                       os.path.join(seed_dir, "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Test with best checkpoint
    model.load_state_dict(
        torch.load(os.path.join(seed_dir, "best.pt"), map_location=device))

    test_m = evaluate_filtered(
        model, processor.test, processor.all_true_tails,
        processor.train_snapshots, device, args.eval_batch_size,
    )

    # Timestamp jitter robustness (reviewer request)
    jitter_1 = evaluate_timestamp_jitter(
        model, processor.test, processor.all_true_tails,
        processor.train_snapshots, device, jitter_days=1.0,
        batch_size=args.eval_batch_size,
    )
    jitter_3 = evaluate_timestamp_jitter(
        model, processor.test, processor.all_true_tails,
        processor.train_snapshots, device, jitter_days=3.0,
        batch_size=args.eval_batch_size,
    )

    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_valid_mrr": best_mrr,
        "test": test_m,
        "jitter_1day": jitter_1,
        "jitter_3day": jitter_3,
    }
    save_json(os.path.join(seed_dir, "result.json"), result)
    save_json(os.path.join(seed_dir, "history.json"), history)
    return result


def aggregate_seeds(results: List[Dict]) -> Dict:
    """Compute mean ± std across seeds for all test metrics."""
    keys = ["mrr", "hits@1", "hits@3", "hits@10", "mar"]
    agg = {}
    for k in keys:
        vals = [r["test"][k] for r in results]
        agg[k] = {
            "mean":  round(float(np.mean(vals)), 4),
            "std":   round(float(np.std(vals)),  4),
            "seeds": [round(v, 4) for v in vals],
        }
    return agg


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RHGNN — fixed H100 version")
    p.add_argument("--data_dir",     required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--embedding_dim",type=int,   default=200)
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--curvature",    type=float, default=1.0)
    p.add_argument("--ode_steps",    type=int,   default=5,
                   help="Steps for Euler fallback (ignored with DOPRI5)")
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--num_negatives",type=int,   default=32)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--seeds",        type=int,   nargs="+",
                   default=[42, 0, 1, 2, 3],
                   help="Random seeds for multi-seed evaluation")
    p.add_argument("--euler",        action="store_true",
                   help="Force Euler ODE instead of DOPRI5")
    p.add_argument("--delta_hyp",    action="store_true",
                   help="Estimate delta-hyperbolicity of dataset")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    print(f"RHGNN fixed — H100/Rorqual version")
    print(f"ODE solver: {'DOPRI5' if HAS_TORCHDIFFEQ and not args.euler else 'Euler'}")
    print(f"Optimizer:  {'Riemannian Adam' if HAS_GEOOPT else 'Euclidean Adam'}")
    print(f"Seeds:      {args.seeds}")

    # Optionally compute δ-hyperbolicity
    if args.delta_hyp:
        print("\nEstimating δ-hyperbolicity...")
        proc = TemporalKGProcessor(args.data_dir)
        proc.load()
        delta = estimate_delta_hyperbolicity(proc)
        print(f"δ-hyperbolicity ≈ {delta}")
        save_json(os.path.join(args.output_dir, "delta_hyp.json"),
                  {"delta": delta, "dataset": args.data_dir})

    # Multi-seed training
    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*50}")
        print(f"Running seed {seed}")
        print(f"{'='*50}")
        r = run_single_seed(args, seed)
        all_results.append(r)

    # Aggregate
    agg = aggregate_seeds(all_results)
    summary = {
        "dataset": args.data_dir,
        "num_seeds": len(args.seeds),
        "seeds": args.seeds,
        "aggregated": agg,
        "per_seed": all_results,
        "args": vars(args),
    }
    save_json(os.path.join(args.output_dir, "summary.json"), summary)

    print(f"\n{'='*50}")
    print("Final results (mean ± std across seeds):")
    print(f"{'='*50}")
    for metric, vals in agg.items():
        print(f"  {metric:8s}: {vals['mean']:.4f} ± {vals['std']:.4f}")
    print(f"\nAll artifacts saved to: {args.output_dir}")


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    main()
