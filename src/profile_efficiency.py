"""
Efficiency profiling for SD-GNN and ASR-GNN.
Params and FLOPs are architecture-only (weight-independent), so no
trained checkpoint is needed -- a freshly-initialized model gives
identical params/FLOPs/latency/memory numbers to a trained one.
"""
import sys, json, time, torch
sys.path.insert(0, '.')

from train_sd_gnn import SDGNN
from train_asr_gnn import ASRGNN

try:
    from fvcore.nn import FlopCountAnalysis
    HAVE_FVCORE = True
except ImportError:
    HAVE_FVCORE = False
    print("WARNING: fvcore not installed, skipping FLOPs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_ENTITIES = 14541
NUM_RELATIONS = 237
EMB_DIM = 200
HIDDEN_DIM = 200
BATCH_SIZE = 1024


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def make_dummy_triples():
    h = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,), device=device)
    r = torch.randint(0, NUM_RELATIONS, (BATCH_SIZE,), device=device)
    t = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,), device=device)
    return torch.stack([h, r, t], dim=1)


def profile_model(model, name):
    model.to(device).eval()
    n_params = count_params(model)
    triples = make_dummy_triples()

    flops = None
    if HAVE_FVCORE:
        try:
            flops = FlopCountAnalysis(model, (triples,)).total()
        except Exception as e:
            print(f"  [FLOPs failed for {name}: {e}]")

    with torch.no_grad():
        for _ in range(5):
            model.score_triples(triples)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    n_runs = 20
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model.score_triples(triples)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    latency_ms = (elapsed / n_runs) * 1000

    peak_mem_gb = None
    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    return {
        "model": name,
        "num_params": n_params,
        "flops_per_batch": flops,
        "latency_ms_per_batch": round(latency_ms, 3),
        "peak_mem_gb": round(peak_mem_gb, 4) if peak_mem_gb is not None else None,
        "batch_size": BATCH_SIZE,
        "device": str(device),
    }


if __name__ == "__main__":
    results = {}
    print("=== Profiling SD-GNN ===")
    sd_model = SDGNN(NUM_ENTITIES, NUM_RELATIONS, EMB_DIM, HIDDEN_DIM)
    results["SD-GNN"] = profile_model(sd_model, "SD-GNN")
    print(json.dumps(results["SD-GNN"], indent=2))

    print("\n=== Profiling ASR-GNN ===")
    asr_model = ASRGNN(NUM_ENTITIES, NUM_RELATIONS, EMB_DIM, HIDDEN_DIM)
    results["ASR-GNN"] = profile_model(asr_model, "ASR-GNN")
    print(json.dumps(results["ASR-GNN"], indent=2))

    with open("efficiency_profile_fb15k237.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> efficiency_profile_fb15k237.json")
