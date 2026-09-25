#!/usr/bin/env python3
"""
R-GCN + DistMult KGE experiment for Rorqual HPC.

Architecture:
  - Encoder: Relational Graph Convolutional Network (R-GCN)
    Builds entity representations by aggregating neighbour messages
    per relation type over the training graph.
  - Decoder: DistMult scoring  score(h,r,t) = <e_h, w_r, e_t>

Evaluation: filtered MRR, MR, Hits@1/3/10 (link prediction, both directions).

Output: one CSV row per run, same format as run_one_experiment.py so
        your existing aggregate_results.py works unchanged.

Usage (via Slurm — never run training directly on login node):
    python train_rgcn_kge.py \\
        --dataset FB15k-237 \\
        --data-dir ~/kg_experiments/data \\
        --results-dir ~/kg_experiments/results \\
        --seed 0 --epochs 200
"""

import argparse
import json
import math
import os
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
# 2. Dataset loading
# ──────────────────────────────────────────────

def load_triples(path: Path):
    """Read a tab-separated triple file → list of (head, relation, tail) strings."""
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
    """Build entity and relation string→int maps."""
    entities, relations = set(), set()
    for h, r, t in all_triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    return entity2id, relation2id


def triples_to_ids(triples, entity2id, relation2id):
    return [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]


def build_filter_dict(all_id_triples):
    """
    For filtered evaluation: map (h,r) → set of true tail ids
    and (t,r) → set of true head ids.
    """
    tail_filter = defaultdict(set)
    head_filter = defaultdict(set)
    for h, r, t in all_id_triples:
        tail_filter[(h, r)].add(t)
        head_filter[(t, r)].add(h)
    return tail_filter, head_filter


# ──────────────────────────────────────────────
# 3. R-GCN encoder (pure PyTorch, no PyG required)
# ──────────────────────────────────────────────

class RGCNLayer(nn.Module):
    """
    Single R-GCN layer with basis decomposition.

    For each relation r, the weight matrix W_r is expressed as a linear
    combination of B shared basis matrices:
        W_r = sum_b a_{r,b} * V_b
    This reduces parameters from num_relations * d_in * d_out
    to num_bases * d_in * d_out + num_relations * num_bases.
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 num_bases: int = 30, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        # Basis matrices  [num_bases, in_dim, out_dim]
        self.basis = nn.Parameter(torch.empty(self.num_bases, in_dim, out_dim))
        # Combination coefficients  [num_relations, num_bases]
        self.coeff = nn.Parameter(torch.empty(num_relations, self.num_bases))
        # Self-loop weight
        self.self_loop_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.coeff)
        nn.init.xavier_uniform_(self.self_loop_weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """
        x           : [num_entities, in_dim]
        edge_index  : [2, num_edges]   (source, target)
        edge_type   : [num_edges]      relation id per edge
        Returns     : [num_entities, out_dim]
        """
        num_entities = x.size(0)

        # Compute relation-specific weight matrices via basis decomposition
        # W_r = coeff[r] @ basis  →  [num_relations, in_dim, out_dim]
        W = torch.einsum("rb,bio->rio", self.coeff, self.basis)

       # Message passing in chunks to avoid OOM
        src, tgt = edge_index[0], edge_index[1]
        x_src = self.dropout(x[src])                         # [E, in_dim]
        chunk_size = 4096
        messages = torch.zeros(x_src.size(0), self.out_dim, device=x.device)
        for start in range(0, x_src.size(0), chunk_size):
            end = min(start + chunk_size, x_src.size(0))
            W_chunk = W[edge_type[start:end]]                # [chunk, in_dim, out_dim]
            messages[start:end] = torch.bmm(
                x_src[start:end].unsqueeze(1), W_chunk
            ).squeeze(1)


        # Aggregate messages (mean aggregation, normalised by degree)
        agg = torch.zeros(num_entities, self.out_dim, device=x.device)
        agg.scatter_add_(0, tgt.unsqueeze(1).expand_as(messages), messages)

        # Degree normalisation
        deg = torch.zeros(num_entities, device=x.device)
        deg.scatter_add_(0, tgt, torch.ones(tgt.size(0), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(1)
        agg = agg / deg

        # Self-loop
        agg = agg + self.dropout(x) @ self.self_loop_weight + self.bias
        return F.relu(agg)


class RGCNEncoder(nn.Module):
    """Two-layer R-GCN."""

    def __init__(self, num_entities: int, num_relations: int,
                 hidden_dim: int = 200, num_bases: int = 30, dropout: float = 0.2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        self.layer1 = RGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases, dropout)
        self.layer2 = RGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases, dropout)
        nn.init.xavier_uniform_(self.entity_emb.weight)

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.entity_emb.weight
        x = self.layer1(x, edge_index, edge_type)
        x = self.layer2(x, edge_index, edge_type)
        return x


# ──────────────────────────────────────────────
# 4. DistMult decoder
# ──────────────────────────────────────────────

class DistMultDecoder(nn.Module):
    def __init__(self, num_relations: int, hidden_dim: int):
        super().__init__()
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        h, t : entity embedding vectors  [batch, dim]
        r    : relation indices           [batch]
        Returns scalar scores            [batch]
        """
        r_emb = self.relation_emb(r)
        return (h * r_emb * t).sum(dim=-1)

    def score_tails(self, h: torch.Tensor, r: torch.Tensor,
                    all_entities: torch.Tensor) -> torch.Tensor:
        """Score against all entities for tail prediction. [batch, num_entities]"""
        r_emb = self.relation_emb(r)               # [batch, dim]
        hr = (h * r_emb).unsqueeze(2)              # [batch, dim, 1]
        e = all_entities.t().unsqueeze(0)          # [1, dim, num_entities]
        return (hr * e).sum(dim=1)                 # [batch, num_entities]

    def score_heads(self, t: torch.Tensor, r: torch.Tensor,
                    all_entities: torch.Tensor) -> torch.Tensor:
        """Score against all entities for head prediction. [batch, num_entities]"""
        r_emb = self.relation_emb(r)
        tr = (t * r_emb).unsqueeze(2)
        e = all_entities.t().unsqueeze(0)
        return (tr * e).sum(dim=1)


# ──────────────────────────────────────────────
# 5. Full model
# ──────────────────────────────────────────────

class RGCNDistMult(nn.Module):
    def __init__(self, num_entities: int, num_relations: int,
                 hidden_dim: int = 200, num_bases: int = 30, dropout: float = 0.2):
        super().__init__()
        self.encoder = RGCNEncoder(num_entities, num_relations,
                                   hidden_dim, num_bases, dropout)
        self.decoder = DistMultDecoder(num_relations, hidden_dim)
        self.num_entities = num_entities

    def forward(self, edge_index, edge_type, h_idx, r_idx, t_idx):
        entity_repr = self.encoder(edge_index, edge_type)
        h = entity_repr[h_idx]
        t = entity_repr[t_idx]
        return self.decoder.score(h, r_idx, t)

    def get_entity_repr(self, edge_index, edge_type):
        return self.encoder(edge_index, edge_type)


# ──────────────────────────────────────────────
# 6. Negative sampling (uniform corruption)
# ──────────────────────────────────────────────

def negative_sample(batch_h, batch_r, batch_t, num_entities: int,
                    neg_per_pos: int = 1):
    """Corrupt head or tail uniformly at random."""
    bsz = batch_h.size(0)
    # Randomly decide whether to corrupt head or tail
    corrupt_tail = torch.rand(bsz, device=batch_h.device) > 0.5

    neg_entities = torch.randint(0, num_entities, (bsz * neg_per_pos,),
                                 device=batch_h.device)

    neg_h = batch_h.repeat(neg_per_pos)
    neg_r = batch_r.repeat(neg_per_pos)
    neg_t = batch_t.repeat(neg_per_pos)
    mask = corrupt_tail.repeat(neg_per_pos)
    neg_t = torch.where(mask, neg_entities, neg_t)
    neg_h = torch.where(~mask, neg_entities, neg_h)
    return neg_h, neg_r, neg_t


# ──────────────────────────────────────────────
# 7. Training
# ──────────────────────────────────────────────

def build_graph_tensors(train_triples, num_entities, device):
    """Build edge_index and edge_type tensors (bidirectional)."""
    heads = [h for h, r, t in train_triples]
    rels  = [r for h, r, t in train_triples]
    tails = [t for h, r, t in train_triples]
    # Add inverse edges  (tail → head with relation + num_relations)
    src  = heads + tails
    dst  = tails + heads
    rtype = rels + [r + max(rels) + 1 for r in rels]
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_type  = torch.tensor(rtype, dtype=torch.long, device=device)
    return edge_index, edge_type


def train_epoch(model, optimizer, train_ids, edge_index, edge_type,
                batch_size: int, neg_per_pos: int, device):
    model.train()
    random.shuffle(train_ids)
    total_loss = 0.0
    n_batches = 0

    # Pre-compute entity representations once per epoch (faster)
    with torch.no_grad():
        entity_repr_frozen = model.get_entity_repr(edge_index, edge_type)

    for i in range(0, len(train_ids), batch_size):
        batch = train_ids[i: i + batch_size]
        if not batch:
            continue
        bh = torch.tensor([h for h, r, t in batch], device=device)
        br = torch.tensor([r for h, r, t in batch], device=device)
        bt = torch.tensor([t for h, r, t in batch], device=device)

        # Positive scores
        h_emb = entity_repr_frozen[bh]
        t_emb = entity_repr_frozen[bt]
        pos_scores = model.decoder.score(h_emb, br, t_emb)

        # Negative samples
        neg_h, neg_r, neg_t = negative_sample(bh, br, bt, model.num_entities,
                                               neg_per_pos)
        nh_emb = entity_repr_frozen[neg_h]
        nt_emb = entity_repr_frozen[neg_t]
        neg_scores = model.decoder.score(nh_emb, neg_r, nt_emb)

        # Self-adversarial (margin-based) loss
        margin = 1.0
        loss = F.margin_ranking_loss(
            pos_scores.repeat(neg_per_pos),
            neg_scores,
            torch.ones_like(neg_scores),
            margin=margin,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────
# 8. Filtered evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, test_ids, edge_index, edge_type, tail_filter, head_filter,
             device, batch_size: int = 256):
    model.eval()
    entity_repr = model.get_entity_repr(edge_index, edge_type)
    all_e = entity_repr  # [num_entities, dim]

    ranks = []

    for i in range(0, len(test_ids), batch_size):
        batch = test_ids[i: i + batch_size]
        bh = torch.tensor([h for h, r, t in batch], device=device)
        br = torch.tensor([r for h, r, t in batch], device=device)
        bt = torch.tensor([t for h, r, t in batch], device=device)

        h_emb = entity_repr[bh]
        t_emb = entity_repr[bt]

        # ---- Tail prediction ----
        tail_scores = model.decoder.score_tails(h_emb, br, all_e)  # [B, N]
        for j, (h, r, t) in enumerate(batch):
            true_tails = tail_filter[(h, r)] - {t}
            scores = tail_scores[j].clone()
            scores[list(true_tails)] = -1e9  # filter
            rank = (scores >= scores[t]).sum().item()
            ranks.append(rank)

        # ---- Head prediction ----
        head_scores = model.decoder.score_heads(t_emb, br, all_e)  # [B, N]
        for j, (h, r, t) in enumerate(batch):
            true_heads = head_filter[(t, r)] - {h}
            scores = head_scores[j].clone()
            scores[list(true_heads)] = -1e9
            rank = (scores >= scores[h]).sum().item()
            ranks.append(rank)

    ranks_t = torch.tensor(ranks, dtype=torch.float)
    mrr = (1.0 / ranks_t).mean().item()
    mr  = ranks_t.mean().item()
    h1  = (ranks_t <= 1).float().mean().item()
    h3  = (ranks_t <= 3).float().mean().item()
    h10 = (ranks_t <= 10).float().mean().item()
    return {"mrr": mrr, "mr": mr, "hits_at_1": h1, "hits_at_3": h3, "hits_at_10": h10}


# ──────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="R-GCN + DistMult KGE on Rorqual")
    parser.add_argument("--dataset",     type=str, default="FB15k-237",
                        choices=["FB15k-237", "WN18RR", "YAGO3-10"])
    parser.add_argument("--data-dir",    type=str, default="~/kg_experiments/data")
    parser.add_argument("--results-dir", type=str, default="~/kg_experiments/results")
    parser.add_argument("--seed",        type=int, required=True)
    parser.add_argument("--epochs",      type=int, default=200)
    parser.add_argument("--hidden-dim",  type=int, default=200)
    parser.add_argument("--num-bases",   type=int, default=30)
    parser.add_argument("--batch-size",  type=int, default=1024)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--neg-per-pos", type=int, default=1)
    parser.add_argument("--eval-every",  type=int, default=25)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──
    data_root = Path(args.data_dir).expanduser() / args.dataset
    train_raw = load_triples(data_root / "train.txt")
    valid_raw = load_triples(data_root / "valid.txt")
    test_raw  = load_triples(data_root / "test.txt")

    all_raw = train_raw + valid_raw + test_raw
    entity2id, relation2id = build_vocab(all_raw)
    num_entities  = len(entity2id)
    num_relations = len(relation2id)
    print(f"Dataset: {args.dataset}  |  entities: {num_entities}  |  relations: {num_relations}")
    print(f"Train: {len(train_raw)}  Valid: {len(valid_raw)}  Test: {len(test_raw)}")

    train_ids = triples_to_ids(train_raw, entity2id, relation2id)
    valid_ids = triples_to_ids(valid_raw, entity2id, relation2id)
    test_ids  = triples_to_ids(test_raw,  entity2id, relation2id)

    all_ids = train_ids + valid_ids + test_ids
    tail_filter, head_filter = build_filter_dict(all_ids)

    # R-GCN uses only train graph for message passing
    # Inverse edges use relation ids [num_relations .. 2*num_relations-1]
    edge_index, edge_type = build_graph_tensors(train_ids, num_entities, device)
    # Encoder actually needs 2 * num_relations (original + inverse)
    num_relations_with_inv = 2 * num_relations

    # ── Model ──
    model = RGCNDistMult(
        num_entities=num_entities,
        num_relations=num_relations_with_inv,
        hidden_dim=args.hidden_dim,
        num_bases=args.num_bases,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    print(f"\nSeed: {args.seed}  |  Epochs: {args.epochs}  |  Hidden: {args.hidden_dim}")
    print("=" * 60)

    best_mrr = 0.0
    best_metrics = {}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, optimizer, train_ids, edge_index, edge_type,
                           args.batch_size, args.neg_per_pos, device)
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, valid_ids, edge_index, edge_type,
                               tail_filter, head_filter, device)
            print(f"Epoch {epoch:4d} | loss={loss:.4f} | "
                  f"MRR={metrics['mrr']:.4f} | H@10={metrics['hits_at_10']:.4f}")
            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                best_metrics = metrics
        else:
            print(f"Epoch {epoch:4d} | loss={loss:.4f}")

    # Final test evaluation
    print("\nEvaluating on test set …")
    test_metrics = evaluate(model, test_ids, edge_index, edge_type,
                            tail_filter, head_filter, device)
    elapsed = time.time() - start
    print(f"Test MRR={test_metrics['mrr']:.4f}  H@1={test_metrics['hits_at_1']:.4f}  "
          f"H@10={test_metrics['hits_at_10']:.4f}  time={elapsed:.1f}s")

    # ── Save ──
    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "seed":           args.seed,
        "epochs":         args.epochs,
        "dataset":        args.dataset,
        "model":          "RGCN-DistMult",
        "mrr":            test_metrics["mrr"],
        "mr":             test_metrics["mr"],
        "hits_at_1":      test_metrics["hits_at_1"],
        "hits_at_3":      test_metrics["hits_at_3"],
        "hits_at_10":     test_metrics["hits_at_10"],
        "best_valid_mrr": best_mrr,
        "runtime_seconds": elapsed,
        "device":         str(device),
        "hidden_dim":     args.hidden_dim,
        "num_bases":      args.num_bases,
    }

    outfile = results_dir / f"{args.dataset}_RGCN-DistMult_seed{args.seed}.csv"
    pd.DataFrame([row]).to_csv(outfile, index=False)

    print(json.dumps(row, indent=2))
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
