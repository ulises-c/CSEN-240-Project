import torch

# Check if PyTorch is installed
print(f"PyTorch Version: {torch.__version__}")

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# If CUDA is available, check the CUDA version and device count
if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
    print(f"Current CUDA Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    x = torch.rand(3, 3).cuda()  # Create a random tensor on the GPU
    print("Tensor successfully allocated on GPU:", x)
else:
    print("CUDA is not available.")
