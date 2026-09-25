#!/usr/bin/env python3
"""
CompGCN + DistMult KGE experiment for Rorqual HPC.

Architecture:
  - Encoder: Compositional Graph Convolution (CompGCN)
    Jointly embeds entities AND relations using composition operators
    (subtract, multiply, circular correlation). More parameter-efficient
    than R-GCN since it shares weights across relations via composition.
  - Decoder: DistMult scoring  score(h,r,t) = <e_h, w_r, e_t>

Evaluation: filtered MRR, MR, Hits@1/3/10 (link prediction, both directions).

Output: one CSV row per run — same schema as train_rgcn_kge.py but with
        additional hits_at_1 and hits_at_3 columns for GNN models.

Usage (via Slurm):
    python train_compgcn_kge.py \
        --dataset FB15k-237 \
        --data-dir ~/kg_experiments/data \
        --results-dir ~/kg_experiments/results/compgcn \
        --seed 0 --epochs 150
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────
# 1. Reproducibility
# ──────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ──────────────────────────────────────────────
# 2. Dataset loading (same as train_rgcn_kge.py)
# ──────────────────────────────────────────────

def load_triples(path: Path):
    triples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            triples.append(tuple(parts))
    return triples

def build_vocab(all_triples):
    entities, relations = set(), set()
    for h, r, t in all_triples:
        entities.add(h); entities.add(t); relations.add(r)
    entity2id   = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    return entity2id, relation2id

def triples_to_ids(triples, entity2id, relation2id):
    return [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]

def build_filter_dict(all_id_triples):
    tail_filter = defaultdict(set)
    head_filter = defaultdict(set)
    for h, r, t in all_id_triples:
        tail_filter[(h, r)].add(t)
        head_filter[(t, r)].add(h)
    return tail_filter, head_filter

# ──────────────────────────────────────────────
# 3. CompGCN Layer
# ──────────────────────────────────────────────

class CompGCNLayer(nn.Module):
    """
    Single CompGCN layer.

    Key difference from R-GCN:
    - R-GCN: learns W_r per relation (num_relations weight matrices)
    - CompGCN: shares ONE weight matrix W, but composes entity+relation
      embeddings before aggregation. Relation embeddings are updated too.

    Composition operators:
      sub:  phi(e, r) = e - r
      mult: phi(e, r) = e * r
      corr: phi(e, r) = circular correlation (FFT-based)
    """

    def __init__(self, in_dim: int, out_dim: int, comp_op: str = "sub",
                 dropout: float = 0.1):
        super().__init__()
        self.comp_op = comp_op

        # Three weight matrices: original, inverse, self-loop
        self.W_orig = nn.Linear(in_dim, out_dim, bias=False)
        self.W_inv  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)

        # Relation update matrix
        self.W_rel  = nn.Linear(in_dim, out_dim, bias=False)

        self.bn      = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_orig.weight)
        nn.init.xavier_uniform_(self.W_inv.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.W_rel.weight)

    def compose(self, e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.comp_op == "sub":
            return e - r
        elif self.comp_op == "mult":
            return e * r
        elif self.comp_op == "corr":
            # Circular correlation via FFT
            e_fft = torch.fft.rfft(e, dim=-1)
            r_fft = torch.fft.rfft(r, dim=-1)
            return torch.fft.irfft(e_fft.conj() * r_fft, n=e.size(-1), dim=-1)
        else:
            raise ValueError(f"Unknown comp_op: {self.comp_op}")

    def forward(self, x: torch.Tensor, rel_emb: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor,
                num_base_relations: int) -> tuple:
        """
        x          : [num_entities, in_dim]
        rel_emb    : [num_relations_with_inv, in_dim]
        edge_index : [2, num_edges]
        edge_type  : [num_edges]  (includes inverse relation ids)
        Returns updated (entity_emb, relation_emb)
        """
        num_entities = x.size(0)
        src, tgt = edge_index[0], edge_index[1]

        # Compose source entity with relation
        rel_per_edge = rel_emb[edge_type]            # [E, in_dim]
        composed     = self.compose(x[src], rel_per_edge)  # [E, in_dim]

        # Separate original vs inverse edges
        is_inv = edge_type >= num_base_relations

        # Aggregate original edges
        msg_orig = torch.zeros(num_entities, composed.size(-1), device=x.device)
        mask_orig = ~is_inv
        if mask_orig.any():
            msg_orig.scatter_add_(0,
                tgt[mask_orig].unsqueeze(1).expand(-1, composed.size(-1)),
                self.dropout(composed[mask_orig]))

        # Aggregate inverse edges
        msg_inv = torch.zeros(num_entities, composed.size(-1), device=x.device)
        if is_inv.any():
            msg_inv.scatter_add_(0,
                tgt[is_inv].unsqueeze(1).expand(-1, composed.size(-1)),
                self.dropout(composed[is_inv]))

        # Degree normalisation
        deg = torch.zeros(num_entities, device=x.device)
        deg.scatter_add_(0, tgt, torch.ones(tgt.size(0), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(1)

        # Combine: original + inverse + self-loop
        out = (self.W_orig(msg_orig / deg) +
               self.W_inv(msg_inv / deg) +
               self.W_self(self.dropout(x)))

        out = self.bn(out)
        out = F.relu(out)

        # Update relation embeddings
        new_rel = self.W_rel(rel_emb)

        return out, new_rel

# ──────────────────────────────────────────────
# 4. CompGCN Encoder
# ──────────────────────────────────────────────

class CompGCNEncoder(nn.Module):
    def __init__(self, num_entities: int, num_relations: int,
                 hidden_dim: int = 200, num_layers: int = 2,
                 comp_op: str = "sub", dropout: float = 0.1):
        super().__init__()
        self.num_base_relations = num_relations

        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        # Store both original and inverse relation embeddings
        self.rel_emb    = nn.Embedding(num_relations * 2, hidden_dim)

        self.layers = nn.ModuleList([
            CompGCNLayer(hidden_dim, hidden_dim, comp_op, dropout)
            for _ in range(num_layers)
        ])

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> tuple:
        x   = self.entity_emb.weight
        rel = self.rel_emb.weight
        for layer in self.layers:
            x, rel = layer(x, rel, edge_index, edge_type,
                           self.num_base_relations)
        return x, rel

# ──────────────────────────────────────────────
# 5. DistMult Decoder (same as RGCN)
# ──────────────────────────────────────────────

class DistMultDecoder(nn.Module):
    def score(self, h, r_emb, t):
        return (h * r_emb * t).sum(dim=-1)

    def score_tails(self, h, r_emb, all_e):
        hr = (h * r_emb).unsqueeze(2)       # [B, dim, 1]
        e  = all_e.t().unsqueeze(0)         # [1, dim, N]
        return (hr * e).sum(dim=1)          # [B, N]

    def score_heads(self, t, r_emb, all_e):
        tr = (t * r_emb).unsqueeze(2)
        e  = all_e.t().unsqueeze(0)
        return (tr * e).sum(dim=1)

# ──────────────────────────────────────────────
# 6. Full Model
# ──────────────────────────────────────────────

class CompGCNDistMult(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim=200,
                 num_layers=2, comp_op="sub", dropout=0.1):
        super().__init__()
        self.encoder = CompGCNEncoder(num_entities, num_relations,
                                      hidden_dim, num_layers, comp_op, dropout)
        self.decoder = DistMultDecoder()
        self.num_entities = num_entities

    def encode(self, edge_index, edge_type):
        return self.encoder(edge_index, edge_type)

    def get_entity_repr(self, edge_index, edge_type):
        x, rel = self.encode(edge_index, edge_type)
        return x, rel

# ──────────────────────────────────────────────
# 7. Graph construction (bidirectional)
# ──────────────────────────────────────────────

def build_graph_tensors(train_triples, device):
    heads = [h for h, r, t in train_triples]
    rels  = [r for h, r, t in train_triples]
    tails = [t for h, r, t in train_triples]
    max_r = max(rels)
    # Inverse edges get relation id = original_id + num_relations
    src    = heads + tails
    dst    = tails + heads
    rtype  = rels  + [r + max_r + 1 for r in rels]
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_type  = torch.tensor(rtype,      dtype=torch.long, device=device)
    return edge_index, edge_type

# ──────────────────────────────────────────────
# 8. Negative sampling
# ──────────────────────────────────────────────

def negative_sample(batch_h, batch_r, batch_t, num_entities):
    bsz = batch_h.size(0)
    corrupt_tail = torch.rand(bsz, device=batch_h.device) > 0.5
    neg_entities = torch.randint(0, num_entities, (bsz,), device=batch_h.device)
    neg_h = batch_h.clone()
    neg_t = batch_t.clone()
    neg_t = torch.where(corrupt_tail,  neg_entities, neg_t)
    neg_h = torch.where(~corrupt_tail, neg_entities, neg_h)
    return neg_h, batch_r, neg_t

# ──────────────────────────────────────────────
# 9. Training
# ──────────────────────────────────────────────

def train_epoch(model, optimizer, train_ids, edge_index, edge_type,
                batch_size, device):
    model.train()
    random.shuffle(train_ids)
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(train_ids), batch_size):
        batch = train_ids[i: i + batch_size]
        if not batch:
            continue

        # Compute inside loop so gradients flow
        ent_repr, rel_repr = model.get_entity_repr(edge_index, edge_type)

        bh = torch.tensor([h for h, r, t in batch], device=device)
        br = torch.tensor([r for h, r, t in batch], device=device)
        bt = torch.tensor([t for h, r, t in batch], device=device)

        r_emb_pos = rel_repr[br]
        pos_scores = model.decoder.score(ent_repr[bh], r_emb_pos, ent_repr[bt])

        neg_h, neg_r, neg_t = negative_sample(bh, br, bt, model.num_entities)
        r_emb_neg  = rel_repr[neg_r]
        neg_scores = model.decoder.score(ent_repr[neg_h], r_emb_neg, ent_repr[neg_t])

        loss = F.margin_ranking_loss(
            pos_scores, neg_scores,
            torch.ones_like(neg_scores),
            margin=1.0,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)

# ──────────────────────────────────────────────
# 10. Filtered evaluation (MRR, MR, Hits@1/3/10)
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, test_ids, edge_index, edge_type,
             tail_filter, head_filter, device, batch_size=256):
    model.eval()
    ent_repr, rel_repr = model.get_entity_repr(edge_index, edge_type)
    ranks = []

    for i in range(0, len(test_ids), batch_size):
        batch = test_ids[i: i + batch_size]
        bh = torch.tensor([h for h, r, t in batch], device=device)
        br = torch.tensor([r for h, r, t in batch], device=device)
        bt = torch.tensor([t for h, r, t in batch], device=device)

        r_emb = rel_repr[br]

        # Tail prediction
        tail_scores = model.decoder.score_tails(ent_repr[bh], r_emb, ent_repr)
        for j, (h, r, t) in enumerate(batch):
            true_tails = tail_filter[(h, r)] - {t}
            scores = tail_scores[j].clone()
            scores[list(true_tails)] = -1e9
            rank = int((scores >= scores[t]).sum().item())
            ranks.append(rank)

        # Head prediction
        head_scores = model.decoder.score_heads(ent_repr[bt], r_emb, ent_repr)
        for j, (h, r, t) in enumerate(batch):
            true_heads = head_filter[(t, r)] - {h}
            scores = head_scores[j].clone()
            scores[list(true_heads)] = -1e9
            rank = int((scores >= scores[h]).sum().item())
            ranks.append(rank)

    ranks_t = torch.tensor(ranks, dtype=torch.float)
    return {
        "mrr":       (1.0 / ranks_t).mean().item(),
        "mr":        ranks_t.mean().item(),
        "hits_at_1": (ranks_t <= 1).float().mean().item(),
        "hits_at_3": (ranks_t <= 3).float().mean().item(),
        "hits_at_10":(ranks_t <= 10).float().mean().item(),
    }

# ──────────────────────────────────────────────
# 11. Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CompGCN + DistMult KGE")
    parser.add_argument("--dataset",     type=str, default="FB15k-237",
                        choices=["FB15k-237", "WN18RR", "YAGO3-10"])
    parser.add_argument("--data-dir",    type=str, default="~/kg_experiments/data")
    parser.add_argument("--results-dir", type=str, default="~/kg_experiments/results/compgcn")
    parser.add_argument("--seed",        type=int, required=True)
    parser.add_argument("--epochs",      type=int, default=150)
    parser.add_argument("--hidden-dim",  type=int, default=200)
    parser.add_argument("--num-layers",  type=int, default=2)
    parser.add_argument("--comp-op",     type=str, default="sub",
                        choices=["sub", "mult", "corr"])
    parser.add_argument("--batch-size",  type=int, default=1024)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--eval-every",  type=int, default=5)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

    # ── Data ──
    data_root = Path(args.data_dir).expanduser() / args.dataset
    train_raw = load_triples(data_root / "train.txt")
    valid_raw = load_triples(data_root / "valid.txt")
    test_raw  = load_triples(data_root / "test.txt")

    all_raw = train_raw + valid_raw + test_raw
    entity2id, relation2id = build_vocab(all_raw)
    num_entities  = len(entity2id)
    num_relations = len(relation2id)
    print(f"Dataset : {args.dataset} | entities: {num_entities} | relations: {num_relations}")
    print(f"Train: {len(train_raw)}  Valid: {len(valid_raw)}  Test: {len(test_raw)}")

    train_ids = triples_to_ids(train_raw, entity2id, relation2id)
    valid_ids = triples_to_ids(valid_raw, entity2id, relation2id)
    test_ids  = triples_to_ids(test_raw,  entity2id, relation2id)

    all_ids = train_ids + valid_ids + test_ids
    tail_filter, head_filter = build_filter_dict(all_ids)

    edge_index, edge_type = build_graph_tensors(train_ids, device)
    num_relations_with_inv = num_relations * 2

    # ── Model ──
    model = CompGCNDistMult(
        num_entities=num_entities,
        num_relations=num_relations,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        comp_op=args.comp_op,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params  : {total_params:,}")
    print(f"Seed    : {args.seed} | Epochs: {args.epochs} | comp_op: {args.comp_op}")
    print("=" * 60)

    best_mrr    = 0.0
    best_metrics = {}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, optimizer, train_ids, edge_index, edge_type,
                           args.batch_size, device)
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, valid_ids, edge_index, edge_type,
                               tail_filter, head_filter, device)
            print(f"Epoch {epoch:4d} | loss={loss:.4f} | "
                  f"MRR={metrics['mrr']:.4f} | "
                  f"H@1={metrics['hits_at_1']:.4f} | "
                  f"H@10={metrics['hits_at_10']:.4f}")
            if metrics["mrr"] > best_mrr:
                best_mrr     = metrics["mrr"]
                best_metrics = metrics
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | loss={loss:.4f}")

    # ── Final test ──
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_ids, edge_index, edge_type,
                            tail_filter, head_filter, device)
    elapsed = time.time() - start

    print(f"Test MRR={test_metrics['mrr']:.4f} | "
          f"H@1={test_metrics['hits_at_1']:.4f} | "
          f"H@3={test_metrics['hits_at_3']:.4f} | "
          f"H@10={test_metrics['hits_at_10']:.4f} | "
          f"time={elapsed:.1f}s")

    # ── Save ──
    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "seed":             args.seed,
        "epochs":           args.epochs,
        "dataset":          args.dataset,
        "model":            "CompGCN",
        "mrr":              test_metrics["mrr"],
        "mr":               test_metrics["mr"],
        "hits_at_1":        test_metrics["hits_at_1"],
        "hits_at_3":        test_metrics["hits_at_3"],
        "hits_at_10":       test_metrics["hits_at_10"],
        "best_valid_mrr":   best_mrr,
        "runtime_seconds":  elapsed,
        "device":           str(device),
        "hidden_dim":       args.hidden_dim,
        "num_layers":       args.num_layers,
        "comp_op":          args.comp_op,
    }

    # Naming convention matches your existing files
    outfile = results_dir / f"{args.dataset}_CompGCN_ep{args.epochs}_seed{args.seed}.csv"
    pd.DataFrame([row]).to_csv(outfile, index=False)
    print(f"Saved → {outfile}")


if __name__ == "__main__":
    main()
