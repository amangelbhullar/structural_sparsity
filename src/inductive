#!/usr/bin/env python3
"""
Inductive AS-GNN / ASR-GNN for GraIL benchmark splits.

Key difference from transductive:
- Train graph entities have learned embeddings
- Test graph entities are UNSEEN — initialised from neighbour mean
- AS-GNN  : gate = sigmoid(MLP([h_emb, r_emb, t_emb]))
- ASR-GNN : query q = MLP([h_emb, r_emb]), gate = sigmoid(MLP([h,r,t,q]))
- Both use mean-neighbour initialisation for unseen entities at test time
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── data loading ──────────────────────────────────────────────────────────────
def load_triples(path: str) -> List[Tuple[str,str,str]]:
    triples = []
    if not os.path.exists(path):
        return triples
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples

class InductiveKGData:
    """
    Holds train graph (seen entities) + test graph (unseen entities).
    Entities in test_graph/train.txt are the support edges for unseen entities.
    Entities in test_graph/test.txt are the query triples to rank.
    """
    def __init__(self, train_dir: str, test_dir: str):
        # ── training graph ────────────────────────────────────────────────
        train_triples = load_triples(os.path.join(train_dir, 'train.txt'))
        valid_triples = load_triples(os.path.join(train_dir, 'valid.txt'))

        # build entity/relation vocab from training graph
        self.entity2id: Dict[str,int] = {}
        self.relation2id: Dict[str,int] = {}

        all_train = train_triples + valid_triples
        for h, r, t in all_train:
            if h not in self.entity2id:
                self.entity2id[h] = len(self.entity2id)
            if t not in self.entity2id:
                self.entity2id[t] = len(self.entity2id)
            if r not in self.relation2id:
                self.relation2id[r] = len(self.relation2id)

        self.num_train_entities = len(self.entity2id)
        self.num_relations      = len(self.relation2id)

        # training triples as tensor
        self.train_triples = self._to_tensor(train_triples)
        self.valid_triples = self._to_tensor(valid_triples)

        # filter dict for training graph
        self.train_true_tails: Dict[Tuple[int,int], set] = defaultdict(set)
        for h, r, t in self.train_triples.tolist():
            self.train_true_tails[(h, r)].add(t)

        # ── test / inference graph ────────────────────────────────────────
        support_triples = load_triples(os.path.join(test_dir, 'train.txt'))
        test_triples    = load_triples(os.path.join(test_dir, 'test.txt'))
        valid_test      = load_triples(os.path.join(test_dir, 'valid.txt'))

        # add unseen entities to vocab (relations must already exist)
        for h, r, t in support_triples + test_triples + valid_test:
            if h not in self.entity2id:
                self.entity2id[h] = len(self.entity2id)
            if t not in self.entity2id:
                self.entity2id[t] = len(self.entity2id)
            # relations: use UNK id if not seen in training
            if r not in self.relation2id:
                self.relation2id[r] = len(self.relation2id)

        self.num_entities = len(self.entity2id)

        # support and test tensors
        self.support_triples = self._to_tensor(support_triples)
        self.test_triples    = self._to_tensor(test_triples)
        self.valid_test      = self._to_tensor(valid_test)

        # filter dict for test graph (support + test)
        self.test_true_tails: Dict[Tuple[int,int], set] = defaultdict(set)
        all_test = support_triples + test_triples + valid_test
        for h, r, t in self._to_tensor(all_test).tolist():
            self.test_true_tails[(h, r)].add(t)

        # unseen entity ids
        self.unseen_entity_ids = set(range(self.num_train_entities, self.num_entities))

        # adjacency for mean-neighbour init: unseen entity -> list of (rel, neighbour)
        self.unseen_adj: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
        for h, r, t in self.support_triples.tolist():
            if h in self.unseen_entity_ids:
                self.unseen_adj[h].append((r, t))
            if t in self.unseen_entity_ids:
                self.unseen_adj[t].append((r, h))

        print(f"Train entities : {self.num_train_entities}")
        print(f"Total entities : {self.num_entities}")
        print(f"Unseen entities: {len(self.unseen_entity_ids)}")
        print(f"Relations      : {self.num_relations}")
        print(f"Train triples  : {len(self.train_triples)}")
        print(f"Support triples: {len(self.support_triples)}")
        print(f"Test triples   : {len(self.test_triples)}")

    def _to_tensor(self, triples: List[Tuple[str,str,str]]) -> torch.Tensor:
        rows = []
        for h, r, t in triples:
            hid = self.entity2id.get(h, 0)
            rid = self.relation2id.get(r, 0)
            tid = self.entity2id.get(t, 0)
            rows.append([hid, rid, tid])
        if not rows:
            return torch.zeros(0, 3, dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long)

# ── models ────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = None):
        super().__init__()
        h = hidden or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, out_dim)
        )
    def forward(self, x): return self.net(x)

class ASGNNInductive(nn.Module):
    """
    AS-GNN (Adaptive Sparse GNN) — inductive version.
    Gate conditioned on structure [h, r, t] — no query projection.
    Unseen entities initialised via mean-neighbour pooling.
    """
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int,
                 sparsity_lambda: float = 1e-4):
        super().__init__()
        self.emb_dim         = emb_dim
        self.sparsity_lambda = sparsity_lambda

        self.entity_emb   = nn.Embedding(num_entities,  emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)

        self.gate_mlp = MLP(3 * emb_dim, 1)
        self.msg_mlp  = MLP(2 * emb_dim, emb_dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def init_unseen(self, data: InductiveKGData, device: torch.device):
        """Mean-neighbour initialisation for unseen entities."""
        with torch.no_grad():
            for eid, neighbours in data.unseen_adj.items():
                if not neighbours:
                    continue
                vecs = []
                for rid, nid in neighbours:
                    # only use seen neighbours
                    if nid < data.num_train_entities:
                        v = self.entity_emb.weight[nid] + \
                            self.relation_emb.weight[rid]
                        vecs.append(v)
                if vecs:
                    mean_vec = torch.stack(vecs).mean(0)
                    self.entity_emb.weight[eid] = mean_vec

    def forward_components(self, triples: torch.Tensor):
        h_emb = self.entity_emb(triples[:, 0])
        r_emb = self.relation_emb(triples[:, 1])
        t_emb = self.entity_emb(triples[:, 2])

        gate_in = torch.cat([h_emb, r_emb, t_emb], dim=-1)
        alpha   = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(-1)

        msg_in = torch.cat([h_emb, r_emb], dim=-1)
        msg    = alpha.unsqueeze(-1) * self.msg_mlp(msg_in)

        score        = -torch.norm(h_emb + msg - t_emb, p=2, dim=-1)
        sparse_loss  = alpha.mean()
        return score, sparse_loss, alpha.mean().item()

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        score, _, _ = self.forward_components(triples)
        return score


class ASRGNNInductive(nn.Module):
    """
    ASR-GNN (Adaptive Sparse Relational GNN) — inductive version.
    Query projection q = MLP([h, r]) conditions the gate.
    This is the KEY novelty: gate sees q so active subgraph changes per query.
    In inductive setting: q is computed from relation embedding + neighbour-
    initialised entity embedding — still meaningful even for unseen entities
    because it captures the RELATIONAL context of the query.
    """
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int,
                 sparsity_lambda: float = 1e-4):
        super().__init__()
        self.emb_dim         = emb_dim
        self.sparsity_lambda = sparsity_lambda

        self.entity_emb   = nn.Embedding(num_entities,  emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)

        self.query_mlp = MLP(2 * emb_dim, emb_dim)          # q = MLP([h,r])
        self.gate_mlp  = MLP(4 * emb_dim, 1)                # gate sees [h,r,t,q]
        self.msg_mlp   = MLP(2 * emb_dim, emb_dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def init_unseen(self, data: InductiveKGData, device: torch.device):
        """Mean-neighbour initialisation for unseen entities."""
        with torch.no_grad():
            for eid, neighbours in data.unseen_adj.items():
                if not neighbours:
                    continue
                vecs = []
                for rid, nid in neighbours:
                    if nid < data.num_train_entities:
                        v = self.entity_emb.weight[nid] + \
                            self.relation_emb.weight[rid]
                        vecs.append(v)
                if vecs:
                    mean_vec = torch.stack(vecs).mean(0)
                    self.entity_emb.weight[eid] = mean_vec

    def forward_components(self, triples: torch.Tensor):
        h_emb = self.entity_emb(triples[:, 0])
        r_emb = self.relation_emb(triples[:, 1])
        t_emb = self.entity_emb(triples[:, 2])

        # query projection — encodes (h, r) context
        q = self.query_mlp(torch.cat([h_emb, r_emb], dim=-1))

        # query-adaptive gate — sees structure AND query
        gate_in = torch.cat([h_emb, r_emb, t_emb, q], dim=-1)
        alpha   = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(-1)

        msg_in = torch.cat([h_emb, r_emb], dim=-1)
        msg    = alpha.unsqueeze(-1) * self.msg_mlp(msg_in)

        # score anchored on q (not raw h_emb) — key novelty
        score       = -torch.norm(q + msg - t_emb, p=2, dim=-1)
        sparse_loss = alpha.mean()
        return score, sparse_loss, alpha.mean().item()

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        score, _, _ = self.forward_components(triples)
        return score

# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate_inductive(model, test_triples: torch.Tensor,
                       true_tails: Dict, num_entities: int,
                       device: torch.device, batch_size: int = 256) -> Dict:
    model.eval()
    ranks = []
    test_entities = sorted(set(
        test_triples[:, 0].tolist() + test_triples[:, 2].tolist()
    ))
    all_test_ents = torch.tensor(test_entities, dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i+batch_size].to(device)
            for quad in batch:
                h, r, t = quad.tolist()

                heads = torch.full((len(test_entities),), h,
                                   dtype=torch.long, device=device)
                rels  = torch.full((len(test_entities),), r,
                                   dtype=torch.long, device=device)
                cands = torch.stack([heads, rels, all_test_ents], dim=1)

                scores = model.score_triples(cands)
                scores = torch.nan_to_num(scores, nan=-1e9)

                true_score = scores[test_entities.index(t)].item()

                # filter
                filt = true_tails.get((h, r), set())
                if len(filt) > 1:
                    for filt_t in filt:
                        if filt_t != t and filt_t in test_entities:
                            scores[test_entities.index(filt_t)] = -1e9

                rank = int((scores > true_score).sum().item()) + 1
                ranks.append(rank)

    ranks_t = torch.tensor(ranks, dtype=torch.float)
    return {
        'MRR':    float((1.0 / ranks_t).mean()),
        'Hits@1': float((ranks_t <= 1).float().mean()),
        'Hits@3': float((ranks_t <= 3).float().mean()),
        'Hits@10':float((ranks_t <= 10).float().mean()),
        'MeanRank': float(ranks_t.mean()),
    }

# ── training ──────────────────────────────────────────────────────────────────
def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    train_dir = os.path.join(args.data_root, args.dataset, 'train_graph')
    test_dir  = os.path.join(args.data_root, args.dataset, 'test_graph')
    data = InductiveKGData(train_dir, test_dir)

    # output dir
    out_dir = Path(args.output_dir) / f"{args.dataset}_{args.model}_ind_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config
    cfg = vars(args)
    cfg.update({'num_entities': data.num_entities,
                'num_train_entities': data.num_train_entities,
                'num_relations': data.num_relations})
    json.dump(cfg, open(out_dir / 'config.json', 'w'), indent=2)

    # build model
    ModelClass = ASRGNNInductive if args.model == 'ASR-GNN' else ASGNNInductive
    model = ModelClass(
        num_entities   = data.num_entities,
        num_relations  = data.num_relations,
        emb_dim        = args.emb_dim,
        sparsity_lambda= args.sparsity_lambda,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_mrr = -1.0
    best_metrics   = {}
    t0 = time.time()

    train_triples = data.train_triples.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()

        # freeze unseen entity embeddings during training
        # (they don't appear in training graph)
        idx = torch.randperm(len(train_triples))
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(train_triples), args.batch_size):
            batch = train_triples[idx[i:i+args.batch_size]]
            if len(batch) == 0:
                continue

            # negative sampling — corrupt tail
            neg_tail = torch.randint(0, data.num_train_entities,
                                     (len(batch),), device=device)
            neg = batch.clone()
            neg[:, 2] = neg_tail

            pos_score, sparse_loss, rho = model.forward_components(batch)
            neg_score, _,           _   = model.forward_components(neg)

            task_loss = F.softplus(neg_score - pos_score).mean()
            loss      = task_loss + args.sparsity_lambda * sparse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        # save epoch log
        json.dump({'epoch': epoch, 'loss': avg_loss, 'rho': rho,
                   'elapsed_sec': time.time() - t0},
                  open(out_dir / f'train_epoch{epoch}.json', 'w'), indent=2)

        # validate every valid_every epochs
        if epoch % args.valid_every == 0:
            # initialise unseen entity embeddings from support edges
            model.init_unseen(data, device)

            valid_metrics = evaluate_inductive(
                model, data.valid_test.to(device),
                data.test_true_tails,
                data.num_entities, device
            )
            json.dump(valid_metrics,
                      open(out_dir / f'valid_epoch{epoch}.json', 'w'), indent=2)

            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | rho={rho:.3f} | "
                  f"valid MRR={valid_metrics['MRR']:.4f} | "
                  f"H@10={valid_metrics['Hits@10']:.4f}")

            if valid_metrics['MRR'] > best_valid_mrr:
                best_valid_mrr = valid_metrics['MRR']
                torch.save(model.state_dict(), out_dir / 'best_model.pt')

                # test with best model
                test_metrics = evaluate_inductive(
                    model, data.test_triples.to(device),
                    data.test_true_tails,
                    data.num_entities, device
                )
                best_metrics = test_metrics
                json.dump(test_metrics,
                          open(out_dir / 'best_test_metrics.json', 'w'), indent=2)

    # final status
    status = {
        'status': 'completed', 'model': args.model,
        'dataset': args.dataset, 'seed': args.seed,
        'best_valid_mrr': best_valid_mrr,
        'training_time_sec': time.time() - t0,
    }
    status.update(best_metrics)
    json.dump(status, open(out_dir / 'status.json', 'w'), indent=2)
    print(f"\nDone. Best valid MRR={best_valid_mrr:.4f}")
    print(f"Test: MRR={best_metrics.get('MRR',0):.4f} "
          f"H@1={best_metrics.get('Hits@1',0):.4f} "
          f"H@10={best_metrics.get('Hits@10',0):.4f}")

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',           default='ASR-GNN',
                        choices=['AS-GNN', 'ASR-GNN'])
    parser.add_argument('--dataset',         default='fb237_ind_v1')
    parser.add_argument('--data_root',       default=os.path.expanduser(
                            '~/kg_experiments/data'))
    parser.add_argument('--output_dir',      default=os.path.expanduser(
                            '~/kg_experiments/results/inductive'))
    parser.add_argument('--emb_dim',         type=int,   default=200)
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--batch_size',      type=int,   default=1024)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--sparsity_lambda', type=float, default=1e-4)
    parser.add_argument('--valid_every',     type=int,   default=5)
    parser.add_argument('--seed',            type=int,   default=0)
    args = parser.parse_args()
    train(args)
