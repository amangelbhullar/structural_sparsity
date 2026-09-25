from pykeen.pipeline import pipeline
from pykeen.datasets import FB15k237
from pykeen.models import ComplEx
import torch

print("CUDA available:", torch.cuda.is_available())

# Load dataset
dataset = FB15k237()

# Run small experiment
result = pipeline(
    model=ComplEx,
    dataset=dataset,
    training_kwargs=dict(
        num_epochs=2,   # VERY SMALL for test
        batch_size=256,
    ),
    model_kwargs=dict(
        embedding_dim=100,
    ),
    optimizer_kwargs=dict(
        lr=1e-3,
    ),
    device='cpu',  # IMPORTANT: test on CPU first
)

# Print results
metrics = result.metric_results.to_dict()
print("MRR:", metrics.get("mean_reciprocal_rank"))
