import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))

x = torch.randn(2000, 2000, device="cuda" if torch.cuda.is_available() else "cpu")
y = torch.matmul(x, x)
print("Tensor device:", y.device)
print("Done")
