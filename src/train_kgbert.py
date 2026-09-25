#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_triples(path: Path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            triples.append(tuple(parts))
    return triples


def build_entity_relation_sets(all_triples):
    entities = sorted({h for h, _, _ in all_triples} | {t for _, _, t in all_triples})
    relations = sorted({r for _, r, _ in all_triples})
    return entities, relations


def build_filter_dict(all_triples):
    tails_by_hr = defaultdict(set)
    heads_by_rt = defaultdict(set)
    for h, r, t in all_triples:
        tails_by_hr[(h, r)].add(t)
        heads_by_rt[(r, t)].add(h)
    return tails_by_hr, heads_by_rt


def make_text(h, r, t):
    return f"{h} [SEP] {r} [SEP] {t}"


class TripleClassificationDataset(Dataset):
    def __init__(self, triples, entities, tokenizer, max_length, negatives_per_positive=1, seed=0):
        self.examples = []
        rng = random.Random(seed)

        entity_list = list(entities)

        for h, r, t in triples:
            self.examples.append((make_text(h, r, t), 1))

            for _ in range(negatives_per_positive):
                if rng.random() < 0.5:
                    h_neg = rng.choice(entity_list)
                    self.examples.append((make_text(h_neg, r, t), 0))
                else:
                    t_neg = rng.choice(entity_list)
                    self.examples.append((make_text(h, r, t_neg), 0))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


@torch.no_grad()
def evaluate_ranking(
    model,
    tokenizer,
    eval_triples,
    entities,
    tails_by_hr,
    heads_by_rt,
    device,
    max_length,
    batch_size,
    max_eval_triples,
):
    model.eval()

    if max_eval_triples is not None and max_eval_triples > 0:
        eval_triples = eval_triples[:max_eval_triples]

    ranks = []
    hits1 = 0
    hits3 = 0
    hits10 = 0

    entity_list = list(entities)

    def score_texts(texts):
        all_scores = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_scores.append(probs.detach().cpu())
        return torch.cat(all_scores, dim=0).numpy()

    for h, r, t in eval_triples:
        # Tail prediction
        tail_candidates = []
        tail_entities = []
        filtered_tails = tails_by_hr[(h, r)]

        for ent in entity_list:
            if ent != t and ent in filtered_tails:
                continue
            tail_candidates.append(make_text(h, r, ent))
            tail_entities.append(ent)

        tail_scores = score_texts(tail_candidates)
        target_idx = tail_entities.index(t)
        target_score = tail_scores[target_idx]
        tail_rank = int(np.sum(tail_scores > target_score)) + 1

        ranks.append(tail_rank)
        hits1 += int(tail_rank <= 1)
        hits3 += int(tail_rank <= 3)
        hits10 += int(tail_rank <= 10)

        # Head prediction
        head_candidates = []
        head_entities = []
        filtered_heads = heads_by_rt[(r, t)]

        for ent in entity_list:
            if ent != h and ent in filtered_heads:
                continue
            head_candidates.append(make_text(ent, r, t))
            head_entities.append(ent)

        head_scores = score_texts(head_candidates)
        target_idx = head_entities.index(h)
        target_score = head_scores[target_idx]
        head_rank = int(np.sum(head_scores > target_score)) + 1

        ranks.append(head_rank)
        hits1 += int(head_rank <= 1)
        hits3 += int(head_rank <= 3)
        hits10 += int(head_rank <= 10)

    ranks = np.array(ranks, dtype=np.float64)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "mr": float(np.mean(ranks)),
        "hits_at_1": float(hits1 / len(ranks)),
        "hits_at_3": float(hits3 / len(ranks)),
        "hits_at_10": float(hits10 / len(ranks)),
        "num_ranks": int(len(ranks)),
    }


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    data_dir = Path(args.data_dir)
    train_triples = read_triples(data_dir / "train.txt")
    valid_triples = read_triples(data_dir / "valid.txt")
    test_triples = read_triples(data_dir / "test.txt")

    all_triples = train_triples + valid_triples + test_triples
    entities, relations = build_entity_relation_sets(all_triples)
    tails_by_hr, heads_by_rt = build_filter_dict(all_triples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        local_files_only=True,
        num_labels=2,
    ).to(device)

    train_dataset = TripleClassificationDataset(
        train_triples,
        entities,
        tokenizer,
        args.max_length,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_valid_mrr = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            bs = batch["labels"].size(0)
            total_loss += loss.item() * bs
            total_examples += bs

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            valid_metrics = evaluate_ranking(
                model=model,
                tokenizer=tokenizer,
                eval_triples=valid_triples,
                entities=entities,
                tails_by_hr=tails_by_hr,
                heads_by_rt=heads_by_rt,
                device=device,
                max_length=args.max_length,
                batch_size=args.eval_batch_size,
                max_eval_triples=args.max_eval_triples,
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={total_loss / max(total_examples, 1):.6f} | "
                f"valid_mrr={valid_metrics['mrr']:.6f} | "
                f"hits@10={valid_metrics['hits_at_10']:.6f}",
                flush=True,
            )

            if valid_metrics["mrr"] > best_valid_mrr:
                best_valid_mrr = valid_metrics["mrr"]
                best_state = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_ranking(
        model=model,
        tokenizer=tokenizer,
        eval_triples=test_triples,
        entities=entities,
        tails_by_hr=tails_by_hr,
        heads_by_rt=heads_by_rt,
        device=device,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        max_eval_triples=args.max_eval_triples,
    )

    elapsed = time.time() - start_time

    result = {
        "model_family": "transformer",
        "model_name": "KGBERT-like",
        "base_model_path": args.model_path,
        "dataset": os.path.basename(os.path.normpath(args.data_dir)),
        "seed": args.seed,
        "num_entities": len(entities),
        "num_relations": len(relations),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "negatives_per_positive": args.negatives_per_positive,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "best_valid_mrr": best_valid_mrr,
        "test_mrr": test_metrics["mrr"],
        "test_mr": test_metrics["mr"],
        "test_hits_at_1": test_metrics["hits_at_1"],
        "test_hits_at_3": test_metrics["hits_at_3"],
        "test_hits_at_10": test_metrics["hits_at_10"],
        "training_time_seconds": elapsed,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_file = results_dir / f"kgbert_{result['dataset']}_seed{args.seed}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a KG-BERT style model.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-eval-triples", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
