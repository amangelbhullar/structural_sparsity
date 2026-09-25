#!/usr/bin/env python3
import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Data loading
# ============================================================

@dataclass
class KGData:
    train: List[Tuple[str, str, str]]
    valid: List[Tuple[str, str, str]]
    test: List[Tuple[str, str, str]]
    entity_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]


def read_triples(path: Path) -> List[Tuple[str, str, str]]:
    triples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def load_fb15k237_local(data_dir: Path) -> KGData:
    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    test_path = data_dir / "test.txt"

    for p in [train_path, valid_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    train = read_triples(train_path)
    valid = read_triples(valid_path)
    test = read_triples(test_path)

    entities = set()
    relations = set()

    for triples in [train, valid, test]:
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)

    entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
    relation_to_id = {r: i for i, r in enumerate(sorted(relations))}

    return KGData(
        train=train,
        valid=valid,
        test=test,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )


def triples_to_ids(
    triples: List[Tuple[str, str, str]],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
) -> torch.Tensor:
    rows = []
    for h, r, t in triples:
        rows.append([entity_to_id[h], relation_to_id[r], entity_to_id[t]])
    return torch.tensor(rows, dtype=torch.long)


# ============================================================
# Graph building
# ============================================================

def build_train_graph(train_triples: torch.Tensor, num_relations: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        edge_index: [2, 2*num_edges]
        edge_type:  [2*num_edges]
    Includes inverse edges.
    """
    heads = train_triples[:, 0]
    rels = train_triples[:, 1]
    tails = train_triples[:, 2]

    src = torch.cat([heads, tails], dim=0)
    dst = torch.cat([tails, heads], dim=0)
    edge_index = torch.stack([src, dst], dim=0)

    edge_type = torch.cat([rels, rels + num_relations], dim=0)
    return edge_index, edge_type


# ============================================================
# CompGCN Layer
# ============================================================

class CompGCNLayer(nn.Module):
    def __init__(self, dim: int, num_rel_total: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_rel_total = num_rel_total

        self.w_in = nn.Linear(dim, dim, bias=False)
        self.w_out = nn.Linear(dim, dim, bias=False)
        self.w_loop = nn.Linear(dim, dim, bias=False)
        self.rel_transform = nn.Linear(dim, dim, bias=False)

        self.loop_rel = nn.Parameter(torch.empty(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.xavier_uniform_(self.w_out.weight)
        nn.init.xavier_uniform_(self.w_loop.weight)
        nn.init.xavier_uniform_(self.rel_transform.weight)
        nn.init.xavier_uniform_(self.loop_rel)

    @staticmethod
    def compose(ent: torch.Tensor, rel: torch.Tensor, op: str = "sub") -> torch.Tensor:
        if op == "sub":
            return ent - rel
        if op == "mult":
            return ent * rel
        if op == "corr":
            # simple circular correlation approximation via FFT
            ent_f = torch.fft.rfft(ent, dim=-1)
            rel_f = torch.fft.rfft(rel, dim=-1)
            out = torch.fft.irfft(torch.conj(ent_f) * rel_f, n=ent.size(-1), dim=-1)
            return out
        raise ValueError(f"Unsupported composition op: {op}")

    def forward(
        self,
        ent_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_base_relations: int,
        comp_op: str = "sub",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_entities = ent_emb.size(0)
        src, dst = edge_index

        edge_rel = rel_emb[edge_type]
        msg_in = self.compose(ent_emb[src], edge_rel, op=comp_op)

        is_inverse = edge_type >= num_base_relations
        transformed = torch.where(
            is_inverse.unsqueeze(-1),
            self.w_out(msg_in),
            self.w_in(msg_in),
        )

        agg = torch.zeros_like(ent_emb)
        agg.index_add_(0, dst, transformed)

        deg = torch.zeros(num_entities, device=ent_emb.device, dtype=ent_emb.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=ent_emb.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        agg = agg / deg

        loop_msg = self.compose(ent_emb, self.loop_rel.expand_as(ent_emb), op=comp_op)
        loop_msg = self.w_loop(loop_msg)

        out_ent = agg + loop_msg + self.bias
        out_ent = F.relu(out_ent)
        out_ent = self.dropout(out_ent)

        out_rel = self.rel_transform(rel_emb)
        return out_ent, out_rel


# ============================================================
# CompGCN model + DistMult decoder
# ============================================================

class CompGCNModel(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 200,
        num_layers: int = 2,
        dropout: float = 0.1,
        comp_op: str = "sub",
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_rel_total = num_relations * 2
        self.dim = dim
        self.comp_op = comp_op

        self.entity_emb = nn.Parameter(torch.empty(num_entities, dim))
        self.relation_emb = nn.Parameter(torch.empty(self.num_rel_total, dim))

        self.layers = nn.ModuleList([
            CompGCNLayer(dim=dim, num_rel_total=self.num_rel_total, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_emb)
        nn.init.xavier_uniform_(self.relation_emb)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.entity_emb
        r = self.relation_emb
        for layer in self.layers:
            x, r = layer(
                ent_emb=x,
                rel_emb=r,
                edge_index=edge_index,
                edge_type=edge_type,
                num_base_relations=self.num_relations,
                comp_op=self.comp_op,
            )
        return x, r

    def score_distmult(
        self,
        entity_repr: torch.Tensor,
        relation_repr: torch.Tensor,
        triples: torch.Tensor,
    ) -> torch.Tensor:
        h = entity_repr[triples[:, 0]]
        r = relation_repr[triples[:, 1]]
        t = entity_repr[triples[:, 2]]
        return torch.sum(h * r * t, dim=-1)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        triples: torch.Tensor,
    ) -> torch.Tensor:
        ent_repr, rel_repr = self.encode(edge_index, edge_type)
        return self.score_distmult(ent_repr, rel_repr, triples)


# ============================================================
# Negative sampling
# ============================================================

def sample_negative_triples(
    pos_triples: torch.Tensor,
    num_entities: int,
) -> torch.Tensor:
    neg = pos_triples.clone()
    mask = torch.rand(pos_triples.size(0), device=pos_triples.device) < 0.5
    random_entities = torch.randint(0, num_entities, (pos_triples.size(0),), device=pos_triples.device)

    neg[mask, 0] = random_entities[mask]
    neg[~mask, 2] = random_entities[~mask]
    return neg


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_auc_ap(
    model: CompGCNModel,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    triples: torch.Tensor,
    num_entities: int,
    batch_size: int = 4096,
) -> Tuple[float, float]:
    model.eval()

    all_scores = []
    all_labels = []

    for start in range(0, triples.size(0), batch_size):
        batch = triples[start:start + batch_size]
        neg = sample_negative_triples(batch, num_entities)

        pos_scores = model(edge_index, edge_type, batch)
        neg_scores = model(edge_index, edge_type, neg)

        scores = torch.cat([pos_scores, neg_scores], dim=0).sigmoid().detach().cpu().numpy()
        labels = np.concatenate([
            np.ones(len(pos_scores), dtype=np.int64),
            np.zeros(len(neg_scores), dtype=np.int64),
        ])

        all_scores.append(scores)
        all_labels.append(labels)

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return float(auc), float(ap)


# ============================================================
# Training
# ============================================================

def train(
    model: CompGCNModel,
    train_triples: torch.Tensor,
    valid_triples: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_entities: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_valid_ap = -math.inf
    best_state = None

    train_triples = train_triples.to(device)
    valid_triples = valid_triples.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(train_triples.size(0), device=device)
        total_loss = 0.0

        for start in range(0, train_triples.size(0), batch_size):
            idx = perm[start:start + batch_size]
            pos_batch = train_triples[idx]
            neg_batch = sample_negative_triples(pos_batch, num_entities)

            pos_scores = model(edge_index, edge_type, pos_batch)
            neg_scores = model(edge_index, edge_type, neg_batch)

            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores),
            ], dim=0)

            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pos_batch.size(0)

        valid_auc, valid_ap = evaluate_auc_ap(
            model=model,
            edge_index=edge_index,
            edge_type=edge_type,
            triples=valid_triples,
            num_entities=num_entities,
        )

        avg_loss = total_loss / train_triples.size(0)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_loss:.6f} | "
            f"valid_auc={valid_auc:.4f} | "
            f"valid_ap={valid_ap:.4f}",
            flush=True,
        )

        if valid_ap > best_valid_ap:
            best_valid_ap = valid_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_valid_ap": float(best_valid_ap)}


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CompGCN on FB15k-237 local files")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing train.txt, valid.txt, test.txt")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--comp-op", type=str, default="sub", choices=["sub", "mult", "corr"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    data_dir = Path(args.data_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    kg = load_fb15k237_local(data_dir)
    train_ids = triples_to_ids(kg.train, kg.entity_to_id, kg.relation_to_id)
    valid_ids = triples_to_ids(kg.valid, kg.entity_to_id, kg.relation_to_id)
    test_ids = triples_to_ids(kg.test, kg.entity_to_id, kg.relation_to_id)

    num_entities = len(kg.entity_to_id)
    num_relations = len(kg.relation_to_id)

    edge_index, edge_type = build_train_graph(train_ids, num_relations)

    model = CompGCNModel(
        num_entities=num_entities,
        num_relations=num_relations,
        dim=args.dim,
        num_layers=args.layers,
        dropout=args.dropout,
        comp_op=args.comp_op,
    ).to(device)

    start_time = time.time()

    train_summary = train(
        model=model,
        train_triples=train_ids,
        valid_triples=valid_ids,
        edge_index=edge_index,
        edge_type=edge_type,
        num_entities=num_entities,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    test_auc, test_ap = evaluate_auc_ap(
        model=model,
        edge_index=edge_index.to(device),
        edge_type=edge_type.to(device),
        triples=test_ids.to(device),
        num_entities=num_entities,
    )

    runtime_seconds = time.time() - start_time

    row = {
        "model": "CompGCN",
        "dataset": "FB15k-237",
        "seed": args.seed,
        "epochs": args.epochs,
        "dim": args.dim,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "comp_op": args.comp_op,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "best_valid_ap": train_summary["best_valid_ap"],
        "test_auc": test_auc,
        "test_ap": test_ap,
        "runtime_seconds": runtime_seconds,
        "device": str(device),
    }

    out_csv = results_dir / f"FB15k-237_CompGCN_ep{args.epochs}_seed{args.seed}.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)

    out_json = results_dir / f"FB15k-237_CompGCN_ep{args.epochs}_seed{args.seed}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print(json.dumps(row, indent=2), flush=True)
    print(f"Saved CSV to: {out_csv}", flush=True)
    print(f"Saved JSON to: {out_json}", flush=True)


if __name__ == "__main__":
    main()
