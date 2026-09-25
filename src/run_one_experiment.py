import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from pykeen.models import (
    AutoSF, BoxE, ComplEx, ConvE, DistMult, MuRE,
    NodePiece, PairRE, QuatE, RESCAL, RotatE,
    SimplE, TransD, TransE, TransH, TransR, TuckER, ConvKB, CP, CrossE, DistMA, ERMLP, HolE, KG2E, NTN, TorusE
)
from pykeen.pipeline import pipeline
from pykeen.datasets import FB15k237
from pykeen.triples import TriplesFactory


def get_dataset(name: str, inverse_triples: bool = False):
    if name == "FB15k-237":
        return FB15k237(create_inverse_triples=inverse_triples)
    raise ValueError(f"Dataset {name} should be loaded from local files.")


def get_local_dataset(name: str, inverse_triples: bool = False):
    data_dir = Path.home() / "kg_experiments" / "data" / name
    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    test_path  = data_dir / "test.txt"

    if not all(p.exists() for p in [train_path, valid_path, test_path]):
        raise FileNotFoundError(f"Missing local dataset files in {data_dir}")

    training = TriplesFactory.from_path(
        train_path,
        create_inverse_triples=inverse_triples,
    )
    validation = TriplesFactory.from_path(
        valid_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    testing = TriplesFactory.from_path(
        test_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    return training, validation, testing


def get_model(name: str):
    model_map = {


        "AutoSF":    AutoSF,
        "BoxE":      BoxE,
        "ComplEx":   ComplEx,
        "ConvE":     ConvE,
        "ConvKB":    ConvKB,
        "CP":        CP,
        "CrossE":    CrossE,
        "DistMA":    DistMA,
        "DistMult":  DistMult,
        "ERMLP":     ERMLP,
        "HolE":      HolE,
        "KG2E":      KG2E,
        "MuRE":      MuRE,
        "NodePiece": NodePiece,
        "NTN":       NTN,
        "PairRE":    PairRE,
        "QuatE":     QuatE,
        "RESCAL":    RESCAL,
        "RotatE":    RotatE,
        "SimplE":    SimplE,
        "TorusE":    TorusE,
        "TransD":    TransD,
        "TransE":    TransE,
        "TransH":    TransH,
        "TransR":    TransR,
        "TuckER":    TuckER,


    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name]


def get_model_kwargs(name: str):
    if name == "NodePiece":
        return {
            "embedding_dim": 200,
            "tokenizers":    "RelationTokenizer",
            "num_tokens":    12,
        }
    if name == "NTN":
        return {"embedding_dim": 100}
    if name == "KG2E":
        return {"embedding_dim": 200}
    base = {"embedding_dim": 200}
    if name in {"TransR", "TransD"}:
        base["relation_dim"] = 200
    return base



parser = argparse.ArgumentParser()
parser.add_argument("--seed",        type=int, required=True)
parser.add_argument("--epochs",      type=int, default=150)
parser.add_argument("--dataset",     type=str, default="FB15k-237")
parser.add_argument("--model",       type=str, default="ComplEx")
parser.add_argument("--results-dir", type=str, default="../results")
args = parser.parse_args()

results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
needs_inverse = args.model == "NodePiece"

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model_cls    = get_model(args.model)
model_kwargs = get_model_kwargs(args.model)

start = time.time()

if args.dataset == "FB15k-237":
    result = pipeline(
        model=model_cls,
        dataset=get_dataset(args.dataset, inverse_triples=needs_inverse),
        device=device,
        random_seed=args.seed,
        model_kwargs=model_kwargs,
        training_kwargs=dict(
            num_epochs=args.epochs,
            batch_size=1024,
            use_tqdm_batch=False,
        ),
        optimizer_kwargs=dict(lr=1e-3),
        evaluator_kwargs=dict(filtered=True),
    )
else:
    training, validation, testing = get_local_dataset(
        args.dataset, inverse_triples=needs_inverse)
    result = pipeline(
        model=model_cls,
        training=training,
        validation=validation,
        testing=testing,
        device=device,
        random_seed=args.seed,
        model_kwargs=model_kwargs,
        training_kwargs=dict(
            num_epochs=args.epochs,
            batch_size=1024,
            use_tqdm_batch=False,
        ),
        optimizer_kwargs=dict(lr=1e-3),
        evaluator_kwargs=dict(filtered=True),
    )

elapsed = time.time() - start

flat_metrics = result.metric_results.to_flat_dict()
mrr    = flat_metrics.get("both.realistic.inverse_harmonic_mean_rank")
mr     = flat_metrics.get("both.realistic.arithmetic_mean_rank")
hits10 = flat_metrics.get("both.realistic.hits_at_10")

row = {
    "seed":            args.seed,
    "epochs":          args.epochs,
    "dataset":         args.dataset,
    "model":           args.model,
    "mrr":             mrr,
    "mr":              mr,
    "hits_at_10":      hits10,
    "runtime_seconds": elapsed,
    "device":          device,
}

outfile = results_dir / f"{args.dataset}_{args.model}_ep{args.epochs}_seed{args.seed}.csv"
pd.DataFrame([row]).to_csv(outfile, index=False)

print(json.dumps(row, indent=2))
print(f"Saved to {outfile}")
