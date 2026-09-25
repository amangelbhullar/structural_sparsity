#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@dataclass
class StaticKGData:
    entity2id: Dict[str, int]
    relation2id: Dict[str, int]
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    all_true_tails: Dict[Tuple[int, int], Set[int]]
    num_entities: int
    num_relations: int


def read_static_triples(file_path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def build_static_mappings(all_triples: List[Tuple[str, str, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    entities = sorted({h for h, _, _ in all_triples} | {t for _, _, t in all_triples})
    relations = sorted({r for _, r, _ in all_triples})
    return {e: i for i, e in enumerate(entities)}, {r: i for i, r in enumerate(relations)}


def encode_static_triples(
    triples: List[Tuple[str, str, str]],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> torch.Tensor:
    rows = [[entity2id[h], relation2id[r], entity2id[t]] for h, r, t in triples]
    return torch.tensor(rows, dtype=torch.long)


def build_all_true_tails(
    all_triples: List[Tuple[str, str, str]],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> Dict[Tuple[int, int], Set[int]]:
    out: Dict[Tuple[int, int], Set[int]] = {}
    for h, r, t in all_triples:
        key = (entity2id[h], relation2id[r])
        out.setdefault(key, set()).add(entity2id[t])
    return out


def load_static_dataset(data_root: str, dataset: str) -> StaticKGData:
    base = os.path.join(data_root, dataset)
    train_path = os.path.join(base, "train.txt")
    valid_path = os.path.join(base, "valid.txt")
    test_path = os.path.join(base, "test.txt")

    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Missing train/valid/test files in {base}")

    train_triples = read_static_triples(train_path)
    valid_triples = read_static_triples(valid_path)
    test_triples = read_static_triples(test_path)

    all_triples = train_triples + valid_triples + test_triples
    entity2id, relation2id = build_static_mappings(all_triples)

    train = encode_static_triples(train_triples, entity2id, relation2id)
    valid = encode_static_triples(valid_triples, entity2id, relation2id)
    test = encode_static_triples(test_triples, entity2id, relation2id)
    all_true_tails = build_all_true_tails(all_triples, entity2id, relation2id)

    return StaticKGData(
        entity2id=entity2id,
        relation2id=relation2id,
        train=train,
        valid=valid,
        test=test,
        all_true_tails=all_true_tails,
        num_entities=len(entity2id),
        num_relations=len(relation2id),
    )


def negative_sample(batch: torch.Tensor, num_entities: int, device: torch.device) -> torch.Tensor:
    neg = batch.clone()
    mask = torch.rand(batch.size(0), device=device) < 0.5
    rand_entities = torch.randint(0, num_entities, (batch.size(0),), device=device)
    neg[mask, 0] = rand_entities[mask]
    neg[~mask, 2] = rand_entities[~mask]
    return neg


class SDGNN(nn.Module):
    """
    SD-GNN (HS-GNN in the paper): uniform, per-relation structural gate.
    beta_{jri} = beta_r for all (j, i): the gate depends only on relation
    type r, not on the specific head/tail entities or any query.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        emb_dim: int,
        hidden_dim: int,
        sparsity_lambda: float = 1e-4,
    ):
        super().__init__()
        self.entity = nn.Embedding(num_entities, emb_dim)
        self.relation = nn.Embedding(num_relations, emb_dim)

        # Per-relation scalar gate: one learned logit per relation type,
        # squashed to (0, 1). No dependence on head/tail embeddings.
        self.relation_gate_logit = nn.Embedding(num_relations, 1)
        nn.init.zeros_(self.relation_gate_logit.weight)

        self.msg_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

        self.sparsity_lambda = sparsity_lambda

        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def forward_components(self, triples: torch.Tensor):
        h = self.entity(triples[:, 0])
        r = self.relation(triples[:, 1])
        t = self.entity(triples[:, 2])

        alpha = torch.sigmoid(self.relation_gate_logit(triples[:, 1]))  # beta_r only, no h/t

        msg = self.msg_proj(torch.cat([h, r], dim=-1))
        sparse_msg = alpha * msg

        score = -torch.norm(h + sparse_msg - t, p=2, dim=-1)
        sparse_penalty = alpha.mean()
        rho = alpha.mean().detach()

        return score, sparse_penalty, rho

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        score, _, _ = self.forward_components(triples)
        return score


@torch.no_grad()
def evaluate_filtered(
    model: SDGNN,
    triples: torch.Tensor,
    data: StaticKGData,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    triples = triples.to(device)
    all_entities = torch.arange(data.num_entities, device=device)
    ranks = []

    for triple in triples:
        h, r, t = triple.tolist()
        heads = torch.full((data.num_entities,), h, dtype=torch.long, device=device)
        rels = torch.full((data.num_entities,), r, dtype=torch.long, device=device)
        candidates = torch.stack([heads, rels, all_entities], dim=1)

        scores = model.score_triples(candidates)
        true_score = scores[t].item()

        filt = data.all_true_tails[(h, r)]
        if len(filt) > 1:
            filt_idx = torch.tensor([x for x in filt if x != t], dtype=torch.long, device=device)
            if filt_idx.numel() > 0:
                scores[filt_idx] = -1e9

        rank = int((scores > true_score).sum().item()) + 1
        ranks.append(rank)

    ranks_t = torch.tensor(ranks, dtype=torch.float)
    return {
        "MRR": float((1.0 / ranks_t).mean().item()),
        "Hits@1": float((ranks_t <= 1).float().mean().item()),
        "Hits@3": float((ranks_t <= 3).float().mean().item()),
        "Hits@10": float((ranks_t <= 10).float().mean().item()),
        "MeanRank": float(ranks_t.mean().item()),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_static_dataset(args.data_root, args.dataset)

    save_json(
        os.path.join(args.output_dir, "config.json"),
        {
            **vars(args),
            "num_entities": data.num_entities,
            "num_relations": data.num_relations,
            "device": str(device),
        },
    )
    save_json(os.path.join(args.output_dir, "status.json"), {"status": "started", "model": "SD-GNN", "dataset": args.dataset})

    model = SDGNN(
        num_entities=data.num_entities,
        num_relations=data.num_relations,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        sparsity_lambda=args.sparsity_lambda,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_triples = data.train.to(device)

    print(f"Model: SD-GNN", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Entities: {data.num_entities}", flush=True)
    print(f"Relations: {data.num_relations}", flush=True)
    print(f"Train triples: {train_triples.size(0)}", flush=True)
    print(f"Device: {device}", flush=True)

    best_valid_mrr = -1.0
    best_epoch = -1
    best_test_metrics = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(train_triples.size(0), device=device)
        shuffled = train_triples[perm]

        total_loss = 0.0
        total_task = 0.0
        total_sparse = 0.0
        total_rho = 0.0
        num_batches = 0

        for i in range(0, shuffled.size(0), args.batch_size):
            batch = shuffled[i:i + args.batch_size]
            neg = negative_sample(batch, data.num_entities, device)

            pos_score, sparse_penalty, rho = model.forward_components(batch)
            neg_score, _, _ = model.forward_components(neg)

            task_loss = F.softplus(neg_score - pos_score).mean()
            sparse_loss = args.sparsity_lambda * sparse_penalty
            loss = task_loss + sparse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_task += task_loss.item()
            total_sparse += sparse_loss.item()
            total_rho += rho.item()
            num_batches += 1

        epoch_info = {
            "epoch": epoch,
            "loss": total_loss / max(1, num_batches),
            "task_loss": total_task / max(1, num_batches),
            "sparse_loss": total_sparse / max(1, num_batches),
            "rho": total_rho / max(1, num_batches),
            "elapsed_sec": time.time() - start_time,
        }
        save_json(os.path.join(args.output_dir, f"train_epoch{epoch}.json"), epoch_info)

        print(
            f"Epoch {epoch:03d} | loss={epoch_info['loss']:.6f} | task={epoch_info['task_loss']:.6f} "
            f"| sparse={epoch_info['sparse_loss']:.6f} | rho={epoch_info['rho']:.4f}",
            flush=True,
        )

        if epoch % args.valid_every == 0 or epoch == args.epochs:
            valid_metrics = evaluate_filtered(model, data.valid, data, device)
            save_json(os.path.join(args.output_dir, f"valid_epoch{epoch}.json"), valid_metrics)

            print(
                f"[VALID] epoch={epoch} MRR={valid_metrics['MRR']:.6f} "
                f"Hits@1={valid_metrics['Hits@1']:.6f} Hits@3={valid_metrics['Hits@3']:.6f} "
                f"Hits@10={valid_metrics['Hits@10']:.6f}",
                flush=True,
            )

            if valid_metrics["MRR"] > best_valid_mrr:
                best_valid_mrr = valid_metrics["MRR"]
                best_epoch = epoch
                best_test_metrics = evaluate_filtered(model, data.test, data, device)
                save_json(os.path.join(args.output_dir, "best_test_metrics.json"), best_test_metrics)

    save_json(
        os.path.join(args.output_dir, "status.json"),
        {
            "status": "completed",
            "model": "SD-GNN",
            "dataset": args.dataset,
            "best_epoch": best_epoch,
            "best_valid_mrr": best_valid_mrr,
            "training_time_sec": time.time() - start_time,
        },
    )

    if best_test_metrics is not None:
        print("[BEST TEST]", flush=True)
        for k, v in best_test_metrics.items():
            print(f"{k}: {v:.6f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SD-GNN on static knowledge graphs.")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--data_root", default="data", type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--emb_dim", default=64, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--valid_every", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--sparsity_lambda", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
