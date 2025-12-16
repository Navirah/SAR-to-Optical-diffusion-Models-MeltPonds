import os
import logging

class EmissionsTrackerSession:
    """
    Lightweight CodeCarbon wrapper for training scripts.
    """

    def __init__(
        self,
        enabled: bool = True,
        project_name: str = "training_run",
        output_dir: str = "./emissions_logs",
        measure_power_secs: int = 15,
        tracking_mode: str = "machine",  # or "process"
        gpu_ids="all",
        country_iso_code=None,
        cloud_provider=None,
        cloud_region=None,
        log_level=None,
        quiet: bool = True,
    ):
        self.enabled = enabled
        self.project_name = project_name
        self.output_dir = output_dir
        self.measure_power_secs = measure_power_secs
        self.tracking_mode = tracking_mode
        self.gpu_ids = gpu_ids
        self.country_iso_code = country_iso_code
        self.cloud_provider = cloud_provider
        self.cloud_region = cloud_region
        self.log_level = log_level
        self.quiet = quiet

        self._tracker = None
        self.total_emissions_kg = None
        self._available = False

        os.makedirs(self.output_dir, exist_ok=True)

        if self.enabled:
            try:
                from codecarbon import EmissionsTracker  # noqa
                self._available = True
            except Exception:
                self._available = False

    def _silence_logs(self):
        if not self.quiet:
            return
        for name in ("codecarbon", "codecarbon.emissions_tracker", "codecarbon.core"):
            logging.getLogger(name).setLevel(logging.WARNING)

    def start(self):
        if not (self.enabled and self._available):
            return

        self._silence_logs()

        from codecarbon import EmissionsTracker
        import inspect

        kwargs = {
            "project_name": self.project_name,
            "output_dir": self.output_dir,
            "measure_power_secs": self.measure_power_secs,
            "tracking_mode": self.tracking_mode,
            "gpu_ids": self.gpu_ids,
            "country_iso_code": self.country_iso_code,
            "cloud_provider": self.cloud_provider,
            "cloud_region": self.cloud_region,
            "log_level": self.log_level,
            "save_to_file": True,
        }

        allowed = set(inspect.signature(EmissionsTracker.__init__).parameters)
        kwargs = {k: v for k, v in kwargs.items() if k in allowed and v is not None}

        self._tracker = EmissionsTracker(**kwargs)
        self._tracker.start()

    def stop(self, wandb_run=None):
        if not self._tracker:
            return None

        try:
            self.total_emissions_kg = float(self._tracker.stop() or 0.0)
        except Exception:
            self.total_emissions_kg = None

        # Log to W&B if present
        if wandb_run is not None and self.total_emissions_kg is not None:
            try:
                import wandb
                wandb.log({"emissions/total_kg": self.total_emissions_kg})
            except Exception:
                pass

        # Write summary
        try:
            with open(os.path.join(self.output_dir, "emissions_summary.txt"), "w") as f:
                f.write(f"project_name: {self.project_name}\n")
                f.write(f"total_kg_co2eq: {self.total_emissions_kg}\n")
        except Exception:
            pass

        return self.total_emissions_kg
