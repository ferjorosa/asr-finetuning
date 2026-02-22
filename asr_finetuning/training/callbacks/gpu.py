"""GPU stats logging callback."""

from __future__ import annotations

import shutil
import subprocess

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only


class GpuStatsMonitor(pl.Callback):
    """Log GPU memory usage and temperature from rank zero."""

    def __init__(self, log_every_n_steps: int = 1) -> None:
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive")
        self.log_every_n_steps = log_every_n_steps
        self._nvidia_smi = shutil.which("nvidia-smi")

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del outputs, batch, batch_idx
        if (trainer.global_step + 1) % self.log_every_n_steps != 0:
            return

        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            device = pl_module.device
            mem_mb = torch.cuda.memory_allocated(device) / (1024**2)
            max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            pl_module.log("gpu_mem_mb", mem_mb, on_step=True, on_epoch=False)
            pl_module.log("gpu_mem_max_mb", max_mem_mb, on_step=True, on_epoch=False)

            if self._nvidia_smi is None:
                return
            try:
                device_index = torch.cuda.current_device()
                output = subprocess.check_output(
                    [
                        self._nvidia_smi,
                        "--query-gpu=temperature.gpu",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(device_index),
                    ],
                    text=True,
                    timeout=1.0,
                ).strip()
                if output:
                    pl_module.log(
                        "gpu_temp_c", float(output), on_step=True, on_epoch=False
                    )
            except (subprocess.SubprocessError, ValueError):
                pass
