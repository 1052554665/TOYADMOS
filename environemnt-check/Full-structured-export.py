# Full structured export (saves a .json for reproducibility)

import sys, json, platform, subprocess, datetime
import torch

def get_env_report():
    report = {}

    # ── System ──────────────────────────────────────────
    report["timestamp"]  = datetime.datetime.now().isoformat()
    report["os"]         = platform.platform()
    report["python"]     = sys.version
    report["cpu_cores"]  = None

    try:
        import psutil
        report["cpu_cores"] = psutil.cpu_count(logical=False)
        ram = psutil.virtual_memory().total / 1e9
        report["ram_gb"] = round(ram, 1)
    except ImportError:
        report["cpu_cores"] = "psutil not installed"
        report["ram_gb"]    = "psutil not installed"

    # ── PyTorch ─────────────────────────────────────────
    report["torch_version"]  = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_version"]   = torch.version.cuda
    report["cudnn_version"]  = str(torch.backends.cudnn.version())
    report["gpu_count"]      = torch.cuda.device_count()

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "name":       props.name,
            "vram_gb":    round(props.total_memory / 1e9, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessors": props.multi_processor_count,
        })
    report["gpus"] = gpus

    # ── Key packages ────────────────────────────────────
    packages = {}
    for pkg in ["torchvision", "torchaudio", "numpy", "sklearn",
                "pandas", "matplotlib", "PIL", "cv2",
                "timm", "albumentations", "torchinfo"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg] = "NOT installed"
    report["packages"] = packages

    # ── pip freeze snapshot ──────────────────────────────
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            text=True
        )
        report["pip_freeze"] = freeze.strip().split("\n")
    except Exception as e:
        report["pip_freeze"] = str(e)

    return report


report = get_env_report()

# ── Pretty-print to console ──────────────────────────────
print(json.dumps(
    {k: v for k, v in report.items() if k != "pip_freeze"},  # exclude long list
    indent=2
))

# ── Save full report to disk ─────────────────────────────
with open("env_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n✓ Full report saved to env_report.json")