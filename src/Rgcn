"""
R-GCN Link Prediction on FB15k-237
Matches existing results naming convention:
    results/rgcn/FB15k-237_RGCN_ep{epochs}_seed{seed}.csv

Usage (from ~/kg_experiments/):
    python src/rgcn_fb15k237.py
    python src/rgcn_fb15k237.py --epochs 150 --seed 0

Run all 10 seeds to match your other models:
    for seed in {0..9}; do
        python src/rgcn_fb15k237.py --epochs 150 --seed $seed
    done
"""

import argparse
import csv
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import RGCNConv

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",      type=int,   default=150)
parser.add_argument("--seed",        type=int,   default=0)
parser.add_argument("--hidden",      type=int,   default=128)
parser.add_argument("--lr",          type=float, default=0.01)
parser.add_argument("--num_bases",   type=int,   default=30)
parser.add_argument("--dropout",     type=float, default=0.2)
parser.add_argument("--dataset",     type=str,   default="FB15k-237")
parser.add_argument("--data_dir",    type=str,   default="./data")
parser.add_argument("--results_dir", type=str,   default="./results/rgcn")
args = parser.parse_args()

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

os.makedirs(args.results_dir, exist_ok=True)

# Matches your convention: {Dataset}_{Model}_ep{epochs}_seed{seed}.csv
model_name  = "RGCN"
base_name   = f"{args.dataset}_{model_name}_ep{args.epochs}_seed{args.seed}"
METRICS_CSV = os.path.join(args.results_dir, f"{base_name}.csv")
CHECKPOINT  = os.path.join(args.results_dir, f"{base_name}.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device      : {device}")
print(f"Seed        : {args.seed}")
print(f"Output CSV  : {METRICS_CSV}")

# ── Dataset ───────────────────────────────────────────────────────────────────
print(f"\nLoading {args.dataset}...")
train_data = FB15k_237(args.data_dir, split="train")[0].to(device)
val_data   = FB15k_237(args.data_dir, split="val")[0].to(device)
test_data  = FB15k_237(args.data_dir, split="test")[0].to(device)

num_nodes     = train_data.num_nodes
num_relations = int(train_data.edge_type.max()) + 1
print(f"Nodes       : {num_nodes:,}")
print(f"Relations   : {num_relations}")
print(f"Train edges : {train_data.edge_index.size(1):,}")

# ── Model ─────────────────────────────────────────────────────────────────────
class RGCNLinkPredictor(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_channels, num_bases, dropout):
        super().__init__()
        self.entity_emb = torch.nn.Embedding(num_nodes, hidden_channels)
        self.conv1 = RGCNConv(hidden_channels, hidden_channels,
                               num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels,
                               num_relations=num_relations, num_bases=num_bases)
        self.rel_emb = torch.nn.Embedding(num_relations, hidden_channels)
        self.dropout = dropout
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

    def encode(self, edge_index, edge_type):
        x = self.entity_emb.weight
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

    def decode(self, z, edge_index, edge_type):
        head = z[edge_index[0]]
        tail = z[edge_index[1]]
        rel  = self.rel_emb(edge_type)
        return (head * rel * tail).sum(dim=-1)


model = RGCNLinkPredictor(num_nodes, num_relations,
                           args.hidden, args.num_bases, args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print(f"Parameters  : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

# ── Negative sampling ─────────────────────────────────────────────────────────
def sample_negatives(edge_index, edge_type, num_nodes):
    neg_tail = torch.randint(0, num_nodes, (edge_index.size(1),), device=device)
    return torch.stack([edge_index[0], neg_tail]), edge_type

# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(eval_data, max_eval=500):
    model.eval()
    z = model.encode(train_data.edge_index, train_data.edge_type)
    edge_index, edge_type = eval_data.edge_index, eval_data.edge_type
    hits1 = hits3 = hits10 = mrr_total = 0.0
    n = min(edge_index.size(1), max_eval)
    for i in range(n):
        head, tail, rel = edge_index[0, i], edge_index[1, i], edge_type[i]
        tails  = torch.arange(num_nodes, device=device)
        scores = model.decode(z,
                              torch.stack([head.repeat(num_nodes), tails]),
                              rel.repeat(num_nodes))
        rank = int((scores > scores[tail]).sum().item()) + 1
        mrr_total += 1.0 / rank
        if rank <= 1:  hits1  += 1
        if rank <= 3:  hits3  += 1
        if rank <= 10: hits10 += 1
    return {
        "MRR":     round(mrr_total / n, 6),
        "Hits@1":  round(hits1  / n, 6),
        "Hits@3":  round(hits3  / n, 6),
        "Hits@10": round(hits10 / n, 6),
    }

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    model.train()
    neg_ei, neg_et = sample_negatives(
        train_data.edge_index, train_data.edge_type, num_nodes)
    z = model.encode(train_data.edge_index, train_data.edge_type)
    pos_s = model.decode(z, train_data.edge_index, train_data.edge_type)
    neg_s = model.decode(z, neg_ei, neg_et)
    loss = (F.binary_cross_entropy_with_logits(pos_s, torch.ones_like(pos_s)) +
            F.binary_cross_entropy_with_logits(neg_s, torch.zeros_like(neg_s)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss)

# ── CSV — matches your column style ───────────────────────────────────────────
csv_file   = open(METRICS_CSV, "w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "epoch", "loss",
    "val_MRR", "val_Hits@1", "val_Hits@3", "val_Hits@10",
    "test_MRR", "test_Hits@1", "test_Hits@3", "test_Hits@10",
    "time_s", "seed", "model", "dataset", "epochs"
])
csv_writer.writeheader()

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"{'Epoch':>6} {'Loss':>10} {'val_MRR':>9} {'val_H@1':>9} {'val_H@10':>9} {'Time':>7}")
print("-" * 60)

best_mrr   = 0.0
start_time = time.time()

for epoch in range(1, args.epochs + 1):
    t0      = time.time()
    loss    = train()
    elapsed = time.time() - t0

    if epoch % 10 == 0 or epoch == 1:
        val_m  = evaluate(val_data,  max_eval=500)
        test_m = evaluate(test_data, max_eval=500)

        print(f"{epoch:>6} {loss:>10.4f} "
              f"{val_m['MRR']:>9.4f} "
              f"{val_m['Hits@1']:>9.4f} "
              f"{val_m['Hits@10']:>9.4f} "
              f"{elapsed:>6.1f}s")

        csv_writer.writerow({
            "epoch":       epoch,       "loss":        round(loss, 6),
            "val_MRR":     val_m["MRR"],  "val_Hits@1":  val_m["Hits@1"],
            "val_Hits@3":  val_m["Hits@3"], "val_Hits@10": val_m["Hits@10"],
            "test_MRR":    test_m["MRR"],  "test_Hits@1": test_m["Hits@1"],
            "test_Hits@3": test_m["Hits@3"], "test_Hits@10":test_m["Hits@10"],
            "time_s":      round(elapsed, 2),
            "seed":        args.seed,   "model":   model_name,
            "dataset":     args.dataset, "epochs": args.epochs,
        })
        csv_file.flush()

        if val_m["MRR"] > best_mrr:
            best_mrr = val_m["MRR"]
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"         ↑ best val MRR {best_mrr:.4f} — saved")

csv_file.close()

# ── Final test ────────────────────────────────────────────────────────────────
print(f"\nLoading best checkpoint...")
model.load_state_dict(torch.load(CHECKPOINT))
final = evaluate(test_data, max_eval=1000)
total_time = time.time() - start_time

print("\n── Final Test Results ────────────────────")
for k, v in final.items():
    print(f"  {k:<10}: {v:.4f}")
print(f"  Total time : {total_time/60:.1f} min")
print(f"  Saved to   : {METRICS_CSV}")
print("──────────────────────────────────────────")
