from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Optional, Any, Dict

from lightning.pytorch.callbacks import Callback


class CodeCarbonCallback(Callback):
    """
    Lightning callback wrapper for CodeCarbon's EmissionsTracker.

    - Starts/stops around training
    - Silences CodeCarbon logs if requested
    - Filters kwargs to match installed CodeCarbon version
    - Logs total kg CO2eq to the active Lightning logger
    - Writes a tiny summary file under output_dir
    - Implements state_dict/load_state_dict to satisfy Lightning's stateful-callback checks
    """

    def __init__(
        self,
        project_name: str = "training_run",
        output_dir: str = "./emissions_logs",
        measure_power_secs: int = 15,
        tracking_mode: str = "machine",   # or "process"
        gpu_ids: str | list[int] | None = "all",
        country_iso_code: Optional[str] = None,
        cloud_provider: Optional[str] = None,
        cloud_region: Optional[str] = None,
        log_level: Optional[str] = None,
        save_to_file: bool = True,
        quiet: bool = True,
    ) -> None:
        super().__init__()
        self.project_name = project_name
        self.output_dir = str(output_dir)
        self.measure_power_secs = measure_power_secs
        self.tracking_mode = tracking_mode
        self.gpu_ids = gpu_ids
        self.country_iso_code = country_iso_code
        self.cloud_provider = cloud_provider
        self.cloud_region = cloud_region
        self.log_level = log_level
        self.save_to_file = save_to_file
        self.quiet = quiet

        self.tracker = None  # set in on_fit_start
        self.total_emissions_kg: Optional[float] = None
        self._enabled = True  # flipped off if CodeCarbon import fails

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    #lightning stateful hooks
    def state_dict(self) -> Dict[str, Any]:
        # Save something small so Lightning's stateful check is happy
        return {"total_emissions_kg": self.total_emissions_kg, "_enabled": self._enabled}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.total_emissions_kg = state_dict.get("total_emissions_kg", None)
        self._enabled = state_dict.get("_enabled", True)

    #helpers
    def _build_tracker(self):
        try:
            from codecarbon import EmissionsTracker  # local import = optional dep
        except Exception as e:
            self._enabled = False
            # Don't raise — just disable quietly
            logging.getLogger(__name__).warning(f"CodeCarbon unavailable, emissions tracking disabled: {e}")
            return None

        candidate: Dict[str, Any] = {
            "project_name": self.project_name,
            "output_dir": self.output_dir,
            "measure_power_secs": self.measure_power_secs,
            "tracking_mode": self.tracking_mode,
            "gpu_ids": self.gpu_ids,
            "country_iso_code": self.country_iso_code,
            "cloud_provider": self.cloud_provider,
            "cloud_region": self.cloud_region,
            "log_level": self.log_level,
            "save_to_file": self.save_to_file,
        }

        allowed = set(inspect.signature(EmissionsTracker.__init__).parameters.keys())
        filtered = {k: v for k, v in candidate.items() if (k in allowed) and (v is not None)}
        return EmissionsTracker(**filtered)

    def _silence_codecarbon_logs(self):
        if not self.quiet:
            return
        for name in ("codecarbon", "codecarbon.emissions_tracker", "codecarbon.core"):
            logger = logging.getLogger(name)
            logger.setLevel(logging.WARNING)
            for h in list(logger.handlers):
                try:
                    h.setLevel(logging.WARNING)
                except Exception:
                    pass

    def on_fit_start(self, trainer, pl_module) -> None:
        self._silence_codecarbon_logs()
        self.tracker = self._build_tracker()
        if not self._enabled or self.tracker is None:
            return
        try:
            self.tracker.start()
        except Exception as e:
            logging.getLogger(__name__).warning(f"CodeCarbon start() failed, disabling: {e}")
            self._enabled = False
            self.tracker = None

    def on_exception(self, trainer, pl_module, exception) -> None:
        # Ensure we stop even on errors
        self._stop_and_log(trainer)

    def on_fit_end(self, trainer, pl_module) -> None:
        self._stop_and_log(trainer)

    #stop and log once
    def _stop_and_log(self, trainer) -> None:
        if not self._enabled:
            return
        if self.tracker is None:
            return

        try:
            # returns total kg CO2eq (float) in newer versions; None/0.0 otherwise
            self.total_emissions_kg = float(self.tracker.stop() or 0.0)
        except Exception as e:
            logging.getLogger(__name__).warning(f"CodeCarbon stop() failed: {e}")
            self.total_emissions_kg = None

        # Log to Lightning logger (W&B, TensorBoard, etc.)
        try:
            if trainer is not None and getattr(trainer, "logger", None) is not None:
                trainer.logger.log_metrics({"emissions/total_kg": self.total_emissions_kg},
                                           step=getattr(trainer, "global_step", 0))
        except Exception:
            pass

        try:
            out = Path(self.output_dir) / "emissions_summary.txt"
            with open(out, "w") as f:
                f.write(f"project_name: {self.project_name}\n")
                f.write(f"total_kg_co2eq: {self.total_emissions_kg}\n")
        except Exception:
            pass

        # prevent double-stop
        self.tracker = None
