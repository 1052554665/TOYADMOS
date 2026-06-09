import sys, torch, platform

print(f"Python      : {sys.version}")
print(f"Platform    : {platform.platform()}")
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA avail  : {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version:{torch.backends.cudnn.version()}")
print(f"GPU count   : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}    : {torch.cuda.get_device_name(i)}")
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  VRAM      : {mem:.1f} GB")