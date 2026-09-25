#!/usr/bin/env python3
import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def build_mappings(all_triples: List[Tuple[str, str, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    entities = sorted({x for h, r, t in all_triples for x in (h, t)})
    relations = sorted({r for _, r, _ in all_triples})
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}
    return ent2id, rel2id


def encode_triples(
    triples: List[Tuple[str, str, str]],
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
) -> torch.Tensor:
    return torch.tensor(
        [[ent2id[h], rel2id[r], ent2id[t]] for h, r, t in triples],
        dtype=torch.long
    )


def build_filters(all_encoded: torch.Tensor) -> Tuple[Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int], Set[int]]]:
    """
    For filtered ranking:
      tail_filter[(h, r)] = all true tails
      head_filter[(r, t)] = all true heads
    """
    tail_filter: Dict[Tuple[int, int], Set[int]] = {}
    head_filter: Dict[Tuple[int, int], Set[int]] = {}

    for h, r, t in all_encoded.tolist():
        tail_filter.setdefault((h, r), set()).add(t)
        head_filter.setdefault((r, t), set()).add(h)

    return tail_filter, head_filter


def build_train_graph(train_triples: torch.Tensor, num_relations: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adds inverse edges.
    edge_index: [2, 2E]
    edge_type:  [2E], where inverse relations are offset by +num_relations
    """
    h = train_triples[:, 0]
    r = train_triples[:, 1]
    t = train_triples[:, 2]

    src = torch.cat([h, t], dim=0)
    dst = torch.cat([t, h], dim=0)
    edge_index = torch.stack([src, dst], dim=0)

    edge_type = torch.cat([r, r + num_relations], dim=0)
    return edge_index, edge_type


def compute_relation_norm(edge_index: torch.Tensor, num_entities: int) -> torch.Tensor:
    """
    Simple node in-degree normalization for destination nodes.
    """
    dst = edge_index[1]
    deg = torch.zeros(num_entities, dtype=torch.float)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    deg = deg.clamp(min=1.0)
    return 1.0 / deg[dst]


# ============================================================
# Negative sampling
# ============================================================

def sample_negative_triples(
    pos_triples: torch.Tensor,
    num_entities: int,
    known_triples: Set[Tuple[int, int, int]],
    max_tries: int = 10,
) -> torch.Tensor:
    """
    Corrupt head or tail and try to avoid sampled false negatives.
    """
    neg = pos_triples.clone()

    for i in range(neg.size(0)):
        h, r, t = neg[i].tolist()
        corrupt_head = bool(random.getrandbits(1))

        for _ in range(max_tries):
            e = random.randrange(num_entities)
            cand = (e, r, t) if corrupt_head else (h, r, e)
            if cand not in known_triples:
                if corrupt_head:
                    neg[i, 0] = e
                else:
                    neg[i, 2] = e
                break
        else:
            # fallback if every try collides
            e = random.randrange(num_entities)
            if corrupt_head:
                neg[i, 0] = e
            else:
                neg[i, 2] = e

    return neg


# ============================================================
# DistMult decoder
# ============================================================

def distmult_score(h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.sum(h * r * t, dim=-1)


# ============================================================
# R-GCN
# ============================================================

class RGCNLayer(nn.Module):
    def __init__(self, dim: int, num_rel_total: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_rel_total = num_rel_total

        self.rel_weight = nn.Parameter(torch.empty(num_rel_total, dim, dim))
        self.self_loop = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.rel_weight)
        nn.init.xavier_uniform_(self.self_loop.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_norm: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        msg = torch.bmm(
            x[src].unsqueeze(1),
            self.rel_weight[edge_type]
        ).squeeze(1)

        msg = msg * edge_norm.unsqueeze(-1)

        out = torch.zeros_like(x)
        out.index_add_(0, dst, msg)

        out = out + self.self_loop(x) + self.bias
        out = F.relu(out)
        out = self.dropout(out)
        return out


class RGCNModel(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 200,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_rel_total = num_relations * 2

        self.entity_emb = nn.Parameter(torch.empty(num_entities, dim))
        self.relation_emb = nn.Parameter(torch.empty(num_relations, dim))

        self.layers = nn.ModuleList([
            RGCNLayer(dim=dim, num_rel_total=self.num_rel_total, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_emb)
        nn.init.xavier_uniform_(self.relation_emb)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.entity_emb
        for layer in self.layers:
            x = layer(x, edge_index, edge_type, edge_norm)
        return x, self.relation_emb

    def score_triples(
        self,
        triples: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_norm: torch.Tensor,
    ) -> torch.Tensor:
        ent_repr, rel_repr = self.encode(edge_index, edge_type, edge_norm)
        h = ent_repr[triples[:, 0]]
        r = rel_repr[triples[:, 1]]
        t = ent_repr[triples[:, 2]]
        return distmult_score(h, r, t)


# ============================================================
# CompGCN
# ============================================================

class CompGCNLayer(nn.Module):
    def __init__(self, dim: int, num_rel_total: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_rel_total = num_rel_total

        self.w_in = nn.Linear(dim, dim, bias=False)
        self.w_out = nn.Linear(dim, dim, bias=False)
        self.w_loop = nn.Linear(dim, dim, bias=False)
        self.w_rel = nn.Linear(dim, dim, bias=False)

        self.loop_rel = nn.Parameter(torch.empty(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.xavier_uniform_(self.w_out.weight)
        nn.init.xavier_uniform_(self.w_loop.weight)
        nn.init.xavier_uniform_(self.w_rel.weight)
        nn.init.xavier_uniform_(self.loop_rel)
        nn.init.zeros_(self.bias)

    @staticmethod
    def compose(ent: torch.Tensor, rel: torch.Tensor, op: str) -> torch.Tensor:
        if op == "sub":
            return ent - rel
        if op == "mult":
            return ent * rel
        if op == "corr":
            ef = torch.fft.rfft(ent, dim=-1)
            rf = torch.fft.rfft(rel, dim=-1)
            return torch.fft.irfft(torch.conj(ef) * rf, n=ent.size(-1), dim=-1)
        raise ValueError(f"Unknown composition op: {op}")

    def forward(
        self,
        ent_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_base_relations: int,
        comp_op: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        edge_rel = rel_emb[edge_type]
        msg = self.compose(ent_emb[src], edge_rel, comp_op)

        is_inverse = edge_type >= num_base_relations
        transformed = torch.where(
            is_inverse.unsqueeze(-1),
            self.w_out(msg),
            self.w_in(msg),
        )

        out = torch.zeros_like(ent_emb)
        out.index_add_(0, dst, transformed)

        deg = torch.zeros(ent_emb.size(0), device=ent_emb.device, dtype=ent_emb.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=ent_emb.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        out = out / deg

        loop_msg = self.compose(ent_emb, self.loop_rel.expand_as(ent_emb), comp_op)
        loop_msg = self.w_loop(loop_msg)

        out = out + loop_msg + self.bias
        out = F.relu(out)
        out = self.dropout(out)

        rel_out = self.w_rel(rel_emb)
        return out, rel_out


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
        # only base relations for DistMult decoder
        return x, r[:self.num_relations]

    def score_triples(self, triples: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        ent_repr, rel_repr = self.encode(edge_index, edge_type)
        h = ent_repr[triples[:, 0]]
        r = rel_repr[triples[:, 1]]
        t = ent_repr[triples[:, 2]]
        return distmult_score(h, r, t)


# ============================================================
# Training and evaluation
# ============================================================

def batch_bce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([
        torch.ones_like(pos_scores),
        torch.zeros_like(neg_scores),
    ], dim=0)
    return F.binary_cross_entropy_with_logits(scores, labels)


@torch.no_grad()
def evaluate_filtered_ranking(
    model_name: str,
    model: nn.Module,
    triples: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_norm: torch.Tensor,
    num_entities: int,
    head_filter: Dict[Tuple[int, int], Set[int]],
    tail_filter: Dict[Tuple[int, int], Set[int]],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    ranks = []

    if model_name == "rgcn":
        ent_repr, rel_repr = model.encode(edge_index, edge_type, edge_norm)
    else:
        ent_repr, rel_repr = model.encode(edge_index, edge_type)

    ent_repr = ent_repr.to(device)
    rel_repr = rel_repr.to(device)

    all_entity_idx = torch.arange(num_entities, device=device)

    for h, r, t in triples.tolist():
        h = int(h)
        r = int(r)
        t = int(t)

        # tail prediction
        hr = ent_repr[h] * rel_repr[r]
        tail_scores = torch.matmul(hr, ent_repr.T)

        filt_tails = tail_filter[(h, r)]
        if len(filt_tails) > 0:
            filt_idx = [x for x in filt_tails if x != t]
            if filt_idx:
                tail_scores[torch.tensor(filt_idx, device=device)] = -1e9

        true_tail_score = tail_scores[t].item()
        tail_rank = 1 + int((tail_scores > true_tail_score).sum().item())
        ranks.append(tail_rank)

        # head prediction
        rt = rel_repr[r] * ent_repr[t]
        head_scores = torch.matmul(ent_repr, rt)

        filt_heads = head_filter[(r, t)]
        if len(filt_heads) > 0:
            filt_idx = [x for x in filt_heads if x != h]
            if filt_idx:
                head_scores[torch.tensor(filt_idx, device=device)] = -1e9

        true_head_score = head_scores[h].item()
        head_rank = 1 + int((head_scores > true_head_score).sum().item())
        ranks.append(head_rank)

    ranks = np.array(ranks, dtype=np.float64)

    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@10": float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
    }


def train_one_run(args: argparse.Namespace) -> Dict[str, float]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    train_raw = read_triples(data_dir / "train.txt")
    valid_raw = read_triples(data_dir / "valid.txt")
    test_raw = read_triples(data_dir / "test.txt")

    all_raw = train_raw + valid_raw + test_raw
    ent2id, rel2id = build_mappings(all_raw)

    train_triples = encode_triples(train_raw, ent2id, rel2id)
    valid_triples = encode_triples(valid_raw, ent2id, rel2id)
    test_triples = encode_triples(test_raw, ent2id, rel2id)
    all_triples = torch.cat([train_triples, valid_triples, test_triples], dim=0)

    num_entities = len(ent2id)
    num_relations = len(rel2id)

    known_triples = {tuple(x) for x in all_triples.tolist()}
    head_filter, tail_filter = build_filters(all_triples)

    edge_index, edge_type = build_train_graph(train_triples, num_relations)
    edge_norm = compute_relation_norm(edge_index, num_entities)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    edge_norm = edge_norm.to(device)
    train_triples = train_triples.to(device)
    valid_triples = valid_triples.to(device)
    test_triples = test_triples.to(device)

    if args.model == "rgcn":
        model = RGCNModel(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=args.dim,
            num_layers=args.layers,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "compgcn":
        model = CompGCNModel(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=args.dim,
            num_layers=args.layers,
            dropout=args.dropout,
            comp_op=args.comp_op,
        ).to(device)
    else:
        raise ValueError("model must be 'rgcn' or 'compgcn'")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_valid_mrr = -1.0
    best_state = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(train_triples.size(0), device=device)

        total_loss = 0.0
        total_examples = 0

        for start in range(0, train_triples.size(0), args.batch_size):
            idx = perm[start:start + args.batch_size]
            pos_batch = train_triples[idx]

            neg_batch_cpu = sample_negative_triples(
                pos_batch.detach().cpu(),
                num_entities=num_entities,
                known_triples=known_triples,
            )
            neg_batch = neg_batch_cpu.to(device)

            if args.model == "rgcn":
                pos_scores = model.score_triples(pos_batch, edge_index, edge_type, edge_norm)
                neg_scores = model.score_triples(neg_batch, edge_index, edge_type, edge_norm)
            else:
                pos_scores = model.score_triples(pos_batch, edge_index, edge_type)
                neg_scores = model.score_triples(neg_batch, edge_index, edge_type)

            loss = batch_bce_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pos_batch.size(0)
            total_examples += pos_batch.size(0)

        train_loss = total_loss / max(total_examples, 1)

        valid_metrics = evaluate_filtered_ranking(
            model_name=args.model,
            model=model,
            triples=valid_triples,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_norm=edge_norm,
            num_entities=num_entities,
            head_filter=head_filter,
            tail_filter=tail_filter,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"valid_mrr={valid_metrics['mrr']:.6f} | "
            f"valid_h@10={valid_metrics['hits@10']:.6f}",
            flush=True,
        )

        if valid_metrics["mrr"] > best_valid_mrr:
            best_valid_mrr = valid_metrics["mrr"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_filtered_ranking(
        model_name=args.model,
        model=model,
        triples=test_triples,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_norm=edge_norm,
        num_entities=num_entities,
        head_filter=head_filter,
        tail_filter=tail_filter,
        device=device,
    )

    runtime_seconds = time.time() - start_time

    result = {
        "model": args.model,
        "dataset": args.dataset_name,
        "seed": args.seed,
        "dim": args.dim,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "comp_op": args.comp_op if args.model == "compgcn" else None,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "best_valid_mrr": best_valid_mrr,
        "test_mrr": test_metrics["mrr"],
        "test_hits@1": test_metrics["hits@1"],
        "test_hits@3": test_metrics["hits@3"],
        "test_hits@10": test_metrics["hits@10"],
        "test_mean_rank": test_metrics["mean_rank"],
        "runtime_seconds": runtime_seconds,
        "device": str(device),
    }

    out_base = f"{args.model}_{args.dataset_name}_seed{args.seed}"
    pd.DataFrame([result]).to_csv(results_dir / f"{out_base}.csv", index=False)
    with (results_dir / f"{out_base}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed RGCN / CompGCN KGE runner")
    parser.add_argument("--model", type=str, required=True, choices=["rgcn", "compgcn"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="fb15k237")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--comp-op", type=str, default="sub", choices=["sub", "mult", "corr"])

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train_one_run(args)
