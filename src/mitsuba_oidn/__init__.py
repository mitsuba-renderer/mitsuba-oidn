"""Python bindings for Intel Open Image Denoise."""

import importlib.metadata
import importlib.util
import os
import sys

from ._mitsuba_oidn_ext import (
    Buffer,
    Device,
    DeviceType,
    Error,
    ErrorCode,
    Filter,
    Format,
    PhysicalDevice,
    Quality,
    Storage,
    image_info,
    oidn_version,
    physical_devices,
    _set_thread_pool,
)

__version__ = importlib.metadata.version("mitsuba-oidn")

_MIN_DRJIT_VERSION = (1, 5, 0)


def _install_thread_pool():
    """Share the nanothread pool of the installed Dr.Jit package"""
    spec = importlib.util.find_spec("drjit")
    if spec is None or spec.origin is None:
        raise ImportError("mitsuba_oidn requires the 'drjit' package")

    version = importlib.metadata.version("drjit")
    parts = tuple(int(p) for p in version.split(".")[:3] if p.isdigit())
    if parts < _MIN_DRJIT_VERSION:
        raise ImportError(
            f"mitsuba_oidn requires drjit>={'.'.join(map(str, _MIN_DRJIT_VERSION))}, "
            f"found {version}"
        )

    if sys.platform == "darwin":
        name = "libnanothread.dylib"
    elif sys.platform == "win32":
        name = "nanothread.dll"
    else:
        name = "libnanothread.so"

    path = os.path.join(os.path.dirname(spec.origin), name)
    if not os.path.exists(path):
        raise ImportError(f"mitsuba_oidn: thread pool library not found at {path}")
    _set_thread_pool(path)


_install_thread_pool()

from ._denoise import denoise

__all__ = [
    "Buffer",
    "Device",
    "DeviceType",
    "Error",
    "ErrorCode",
    "Filter",
    "Format",
    "PhysicalDevice",
    "Quality",
    "Storage",
    "denoise",
    "image_info",
    "oidn_version",
    "physical_devices",
]
