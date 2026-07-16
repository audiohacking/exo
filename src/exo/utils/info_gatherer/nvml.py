import contextlib
from typing import Self

from exo.shared.types.profiling import SystemPerformanceProfile
from exo.utils.pydantic_ext import TaggedModel


class NvmlMetrics(TaggedModel):
    """GPU telemetry gathered via NVIDIA NVML (temperature, power, utilization)."""

    system_profile: SystemPerformanceProfile

    @classmethod
    def gather(cls) -> Self | None:
        """Poll NVML for GPU stats. Blocking — run via to_thread."""
        try:
            import pynvml as nvml  # pyright: ignore[reportMissingModuleSource]
        except ImportError:
            return None

        try:
            nvml.nvmlInit()
        except nvml.NVMLError:
            return None

        try:
            device_count = nvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None

            gpu_usages: list[float] = []
            temps: list[float] = []
            total_power_watts = 0.0

            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                with contextlib.suppress(nvml.NVMLError):
                    gpu_usages.append(
                        float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    )
                with contextlib.suppress(nvml.NVMLError):
                    temps.append(
                        float(
                            nvml.nvmlDeviceGetTemperature(
                                handle, nvml.NVML_TEMPERATURE_GPU
                            )
                        )
                    )
                with contextlib.suppress(nvml.NVMLError):
                    total_power_watts += nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

            return cls(
                system_profile=SystemPerformanceProfile(
                    gpu_usage=sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0.0,
                    temp=sum(temps) / len(temps) if temps else 0.0,
                    sys_power=total_power_watts,
                )
            )
        finally:
            with contextlib.suppress(nvml.NVMLError):
                nvml.nvmlShutdown()
