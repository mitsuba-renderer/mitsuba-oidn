"""High-level denoising entry point"""

import atexit
import collections
import threading

from ._mitsuba_oidn_ext import (
    Device,
    DeviceType,
    Format,
    Quality,
    Storage,
    image_info,
)

_DL_CPU = 1
_DL_CUDA = 2
_DL_CUDA_HOST = 3
_DL_ROCM_HOST = 11
_DL_CUDA_MANAGED = 13

_lock = threading.Lock()
_devices = {}
_filters = collections.OrderedDict()
_FILTER_CACHE_SIZE = 4


def _release_caches():
    """Release cached OIDN objects before the interpreter tears down modules"""
    with _lock:
        _filters.clear()
        _devices.clear()


atexit.register(_release_caches)


def _device_for(device_type, device_id):
    """Return a cached device matching the DLPack device of an input array"""
    key = (device_type, device_id) if device_type in (_DL_CUDA, _DL_CUDA_MANAGED) else None
    dev = _devices.get(key)
    if dev is None:
        if key is None:
            dev = Device(DeviceType.Default)
        else:
            dev = Device.cuda(device_id)
        _devices[key] = dev
    return dev


def _cached_filter(key, device, filter_type):
    flt = _filters.get(key)
    if flt is not None:
        _filters.move_to_end(key)
        return flt
    flt = device.new_filter(filter_type)
    _filters[key] = flt
    while len(_filters) > _FILTER_CACHE_SIZE:
        _filters.popitem(last=False)
    return flt


def _framework(array):
    """Name of the array framework that produced 'array'"""
    module = type(array).__module__.split(".")[0]
    return module


def _from_dlpack(framework, view, like):
    """Convert an exported buffer view to the framework of the input"""
    if framework == "drjit":
        return type(like)(view)
    if framework == "numpy":
        import numpy

        return numpy.from_dlpack(view)
    if framework == "torch":
        import torch

        return torch.from_dlpack(view)
    if framework == "cupy":
        import cupy

        return cupy.from_dlpack(view)
    if framework in ("jax", "jaxlib"):
        import jax

        return jax.dlpack.from_dlpack(view)
    return view


def _to_contiguous_host(array, framework):
    if framework == "numpy":
        import numpy

        return numpy.ascontiguousarray(array)
    if framework == "torch":
        return array.contiguous()
    return array


def _stage(device, flt, name, array, info, framework):
    """Copy a host array into a device buffer and bind it to the filter"""
    height, width, channels, itemsize, device_type, _, _, _ = info
    if device_type not in (_DL_CPU, _DL_CUDA_HOST, _DL_ROCM_HOST):
        raise TypeError(
            f"cannot transfer the '{name}' image to the selected device; pass an "
            "array the device can access or select a matching device"
        )
    array = _to_contiguous_host(array, framework)
    buf = device.new_buffer(height * width * channels * itemsize, Storage.Device)
    buf.write(array)
    used = min(channels, 3)
    fmt = Format((Format.Half if itemsize == 2 else Format.Float).value + used - 1)
    flt.set_image(name, buf, format=fmt, width=width, height=height,
                  pixel_stride=channels * itemsize)
    return buf


def denoise(color, albedo=None, normal=None, *, hdr=False, srgb=False,
            clean_aux=False, quality=Quality.High, input_scale=None,
            filter="RT", device=None, output=None):
    """
    Denoise a rendered image.

    All images are arrays of shape (H, W) or (H, W, C) with C <= 4 and dtype
    float32 or float16. Any array supporting DLPack or the buffer protocol is
    accepted (Dr.Jit, NumPy, PyTorch, CuPy, JAX, MLX). A fourth channel is
    ignored.

    The result has the same framework and lives on the same device as 'color'.
    Its shape is (H, W, min(C, 3)), or (H, W) for 2D input, unless 'output' supplies a preallocated array,
    in which case that array is filled in place and returned.

    Parameters:
        color: noisy beauty image (LDR values in [0, 1] or HDR values >= 0)
        albedo: optional albedo image with values in [0, 1]
        normal: optional shading normal image with values in [-1, 1]
        hdr: whether 'color' has high dynamic range
        srgb: whether 'color' is sRGB-encoded (LDR only)
        clean_aux: whether the auxiliary images are noise-free
        quality: filter quality mode
        input_scale: optional scale applied to the input before filtering
        filter: OIDN filter type ("RT" or "RTLightmap")
        device: mitsuba_oidn.Device to run on; by default, CUDA arrays select a
            CUDA device with a matching ordinal and host arrays select the
            fastest physical device
        output: optional preallocated output array
    """
    framework = _framework(color)
    color_info = image_info(color)
    height, width, channels, itemsize, device_type, device_id, _, ndim = color_info

    with _lock:
        if device is None:
            device = _device_for(device_type, device_id)

        inputs = [("color", color, color_info)]
        for name, image in (("albedo", albedo), ("normal", normal)):
            if image is None:
                continue
            info = image_info(image)
            if info[:2] != (height, width):
                raise ValueError(f"the '{name}' image must match the size of 'color'")
            inputs.append((name, image, info))

        key = (id(device), filter, height, width, itemsize,
               tuple((n, i[2]) for n, _, i in inputs),
               hdr, srgb, clean_aux, quality, input_scale)
        flt = _cached_filter(key, device, filter)

        staged = []
        for name, image, info in inputs:
            if device.can_share(image):
                flt.set_image(name, image)
            else:
                staged.append(_stage(device, flt, name, image, info, framework))

        for name in ("albedo", "normal"):
            if not any(n == name for n, _, _ in inputs):
                flt.unset_image(name)

        if filter == "RT":
            flt.hdr = hdr
            flt.srgb = srgb
            flt.clean_aux = clean_aux
        flt.quality = quality
        flt.input_scale = input_scale

        if output is not None:
            flt.set_image("output", output)
            flt.execute()
            return output

        out_channels = min(channels, 3)
        nbytes = height * width * out_channels * itemsize
        dtype = "float16" if itemsize == 2 else "float32"

        if device_type in (_DL_CUDA, _DL_CUDA_MANAGED):
            storage = Storage.Device
        else:
            storage = Storage.Host
        out_buf = device.new_buffer(nbytes, storage)
        fmt = Format((Format.Half if itemsize == 2 else Format.Float).value + out_channels - 1)
        flt.set_image("output", out_buf, format=fmt, width=width, height=height)
        flt.execute()

        shape = (height, width) if ndim == 2 else (height, width, out_channels)
        view = out_buf.view(dtype, shape)

    return _from_dlpack(framework, view, color)
