from typing import Callable

from . import _C


class Detector:
    def __init__(self, toggle: Callable, flag_name: str) -> None:
        self._toggle = toggle
        self._get_flag = getattr(_C, f"get_{flag_name}")
        self._clear_flag = getattr(_C, f"clear_{flag_name}")

    def __enter__(self):
        self._clear_flag()
        self._toggle(True)
        return self

    def __exit__(self, *args):
        self._toggle(False)


class SynchronizationDetector(Detector):
    def __init__(self):
        super().__init__(_C.detect_synchronization, "synchronization_detected")

    @property
    def any_synchronization_point(self):
        return self._get_flag()


class KernelLaunchDetector(Detector):
    def __init__(self):
        super().__init__(_C.detect_kernel_launch, "kernel_launch_detected")

    @property
    def any_kernel_launch(self):
        return self._get_flag()


class OperationDetector:
    def __init__(self):
        self._sync_detector = SynchronizationDetector()
        self._kernel_detector = KernelLaunchDetector()

    def __enter__(self):
        self._sync_detector.__enter__()
        self._kernel_detector.__enter__()
        return self

    def __exit__(self, *args):
        self._kernel_detector.__exit__()
        self._sync_detector.__exit__()

    @property
    def uses_device(self):
        return self._kernel_detector.any_kernel_launch

    @property
    def has_sync(self):
        return self._sync_detector.any_synchronization_point
