#!/usr/bin/env python3
"""
Train geometric baselines (RotatE, BoxE, TransE) using PyKEEN.
Target: CoDEx-L (all 3 models), 10 seeds each.
"""
import argparse, json, os, random, time, csv
from pathlib import Path
import numpy as np
import torch
from pykeen.pipeline import pipeline
from pykeen.datasets import CoDExLarge, CoDExMedium, CoDExSmall

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

DATASETS = {'CoDEx-L': CoDExLarge, 'CoDEx-M': CoDExMedium, 'CoDEx-S': CoDExSmall}

def train(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device:{device} Model:{args.model} Dataset:{args.dataset} Seed:{args.seed}")

    out = Path(args.output_dir) / \
          f"{args.dataset}_{args.model}_ep{args.epochs}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    if (out / 'status.json').exists():
        d = json.load(open(out / 'status.json'))
        if d.get('status') == 'completed':
            print(f"Already done: MRR={d.get('mrr',0):.4f} — skipping")
            return

    t0 = time.time()
    result = pipeline(
        dataset=DATASETS[args.dataset],
        model=args.model,
        model_kwargs=dict(embedding_dim=args.emb_dim),
        optimizer='Adam',
        optimizer_kwargs=dict(lr=args.lr),
        training_kwargs=dict(num_epochs=args.epochs, batch_size=args.batch_size),
        negative_sampler='basic',
        negative_sampler_kwargs=dict(num_negs_per_pos=args.num_negs),
        evaluator='RankBasedEvaluator',
        evaluator_kwargs=dict(filtered=True),
        evaluation_kwargs=dict(batch_size=256),
        random_seed=args.seed,
        device=device,
        use_tqdm=True,
    )

    metrics = result.metric_results.to_flat_dict()
    def get(keys):
        for k in keys:
            if k in metrics: return float(metrics[k])
        return 0.0

    mrr  = get(['both.realistic.inverse_harmonic_mean_rank'])
    h1   = get(['both.realistic.hits_at_1'])
    h3   = get(['both.realistic.hits_at_3'])
    h10  = get(['both.realistic.hits_at_10'])
    mr   = get(['both.realistic.mean_rank'])
    runtime = time.time() - t0

    print(f"\nMRR={mrr:.4f} H@1={h1:.4f} H@3={h3:.4f} H@10={h10:.4f} ({runtime:.0f}s)")

    # Save CSV — same format as existing Rorqual results
    csv_path = out / f"{args.dataset}_{args.model}_ep{args.epochs}_seed{args.seed}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'dataset','model','seed','mrr','hits_at_1','hits_at_3','hits_at_10','mean_rank'])
        w.writeheader()
        w.writerow({'dataset':args.dataset,'model':args.model,'seed':args.seed,
                    'mrr':round(mrr,6),'hits_at_1':round(h1,6),
                    'hits_at_3':round(h3,6),'hits_at_10':round(h10,6),
                    'mean_rank':round(mr,2)})

    json.dump({'status':'completed','model':args.model,'dataset':args.dataset,
               'seed':args.seed,'mrr':mrr,'hits_at_1':h1,'hits_at_3':h3,
               'hits_at_10':h10,'mean_rank':mr,'runtime':runtime},
              open(out/'status.json','w'), indent=2)
    print(f"Saved to {out}")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',      default='RotatE', choices=['RotatE','BoxE','TransE'])
    p.add_argument('--dataset',    default='CoDEx-L', choices=['CoDEx-L','CoDEx-M','CoDEx-S'])
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--epochs',     type=int,   default=150)
    p.add_argument('--emb_dim',    type=int,   default=200)
    p.add_argument('--batch_size', type=int,   default=1024)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--num_negs',   type=int,   default=64)
    p.add_argument('--output_dir', default=os.path.expanduser('~/kg_experiments/results'))
    train(p.parse_args())
