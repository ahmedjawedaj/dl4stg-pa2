"""Record the hardware used for training and timing (results/hardware.json)."""
import json
import platform
import subprocess

import torch

from pa2.data import default_device


def sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


info = {"platform": platform.platform(), "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "ram_gb": round(int(sysctl("hw.memsize") or 0) / 2 ** 30, 1),
        "python": platform.python_version(), "torch": torch.__version__,
        "device": str(default_device())}
json.dump(info, open("results/hardware.json", "w"), indent=1)
print(info)
