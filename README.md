# mitsuba-oidn

[![PyPI](https://img.shields.io/pypi/v/mitsuba-oidn.svg?color=green)](https://pypi.org/project/mitsuba-oidn)
[![Wheels](https://github.com/mitsuba-renderer/mitsuba-oidn/actions/workflows/wheels.yml/badge.svg)](https://github.com/mitsuba-renderer/mitsuba-oidn/actions/workflows/wheels.yml)

This package provides unofficial Python bindings for [Intel Open Image
Denoise](https://www.openimagedenoise.org) (OIDN) for use with Dr.Jit and the
Mitsuba renderer.

```python
import mitsuba as mi
import mitsuba_oidn as oidn

mi.set_variant("cuda_ad_rgb")
scene = mi.load_dict(mi.cornell_box())
color = mi.render(scene, spp=16)                   # mi.TensorXf of shape (H, W, 3)
denoised = oidn.denoise(color, hdr=True)           # also an mi.TensorXf, on the GPU
```

The `color` argument can be any array type that supports DLPack or the buffer
protocol, and the result uses the same type and device.

The package runs on Intel/ARM CPUs, CUDA, and on Apple Metal. It exchanges
tensors via the buffer protocol and DLPack for compatibility with Dr.Jit,
NumPy, PyTorch, JAX, MLX, etc., and it accesses their memory directly whenever
the target device can.

## Why another binding?

Several unofficial bindings of OIDN already exist (e.g.,
[pyoidn](https://github.com/Hyiker/pyoidn), which uses cffi to expose the C API
directly). This project uses [nanobind](https://github.com/wjakob/nanobind) to
create bindings that feel more natural in Python. They automatically commit and
release resources and raise errors as Python exceptions. Pixel formats and
dimensions are inferred from nd-array signatures.

The bindings are designed to interoperate with
[Dr.Jit](https://github.com/mitsuba-renderer/drjit) and [Mitsuba
3](https://github.com/mitsuba-renderer/mitsuba3). The copy of OIDN bundled here
is modified to use Dr.Jit's
[nanothread](https://github.com/mitsuba-renderer/nanothread) thread pool
instead of spinning up another redundant thread pool via oneTBB. For this
reason, the package depends on Dr.Jit.

## Installation

```
pip install mitsuba-oidn
```

Wheels are available for Linux (x86_64, aarch64), Windows (x86_64), and
macOS (arm64), matching the platforms supported by Dr.Jit. The Linux x86_64
and Windows wheels include the CUDA device, which activates when an NVIDIA
driver is present. The macOS wheel includes the Metal device.

## The `denoise()` function

```python
oidn.denoise(color, albedo=None, normal=None, *, hdr=False, srgb=False,
             clean_aux=False, quality=oidn.Quality.High, input_scale=None,
             filter="RT", device=None, output=None)
```

Images are arrays of shape `(H, W)` or `(H, W, C)` with `C <= 4` and dtype
`float32` or `float16`. A fourth channel is ignored on input. The result has
the same framework and lives on the same device as `color`: a NumPy array
yields a NumPy array, a CUDA tensor yields a CUDA tensor. Its shape is
`(H, W, min(C, 3))`, or `(H, W)` for two-dimensional input.

- `hdr`, `srgb`, `clean_aux`, `quality`, and `input_scale` map to the
  parameters of the OIDN `RT` filter. Set `hdr=True` for linear radiance
  values without an upper bound, and `clean_aux=True` when the albedo and
  normal images are noise-free. See the [OIDN
  documentation](https://www.openimagedenoise.org/documentation.html) for
  details.
- `device` selects the device. By default, CUDA arrays use a CUDA device with
  the matching ordinal, and host arrays use the fastest physical device in the
  system, which can be overridden with the `OIDN_DEFAULT_DEVICE` environment
  variable (`cpu`, `cuda`, `metal`, or a physical device ID).
- `output` supplies a preallocated array that is filled in place and returned.
  With an RGBA output array, OIDN writes the RGB channels and leaves alpha
  untouched.
- Filters are expensive to create, so `denoise()` caches a few of them, keyed
  on image size, format, feature set, and parameters. Repeated calls at the
  same resolution pay only for the actual filtering.

When the device cannot access an input array directly, for example a NumPy
array passed to a CUDA device, `denoise()` copies it into a device buffer.
Otherwise no copies are made.

```python
import numpy as np
import torch
import mitsuba_oidn as oidn

# NumPy, CPU or Metal depending on the fastest available device
out = oidn.denoise(np.asarray(color, dtype=np.float32), hdr=True)

# PyTorch on the GPU: zero-copy in and out
color = torch.rand(1080, 1920, 3, device="cuda")
out = oidn.denoise(color, quality=oidn.Quality.Balanced)
assert out.device == color.device
```

## The object API

The `denoise()` function covers the common case. The classes below mirror the
OIDN object model for applications that need control over devices, memory,
and filter lifetime, for instance when denoising many frames or several AOVs
that share auxiliary images.

### Devices

```python
oidn.physical_devices()               # list of PhysicalDevice: id, name, type, uuid, ...

dev = oidn.Device()                   # fastest physical device
dev = oidn.Device(oidn.DeviceType.CPU)
dev = oidn.Device.from_physical(id)   # also from_uuid(), from_luid(), from_pci_address()
dev = oidn.Device.cuda(device_id=0, stream=torch.cuda.current_stream().cuda_stream)
dev = oidn.Device.metal(command_queue)  # raw id<MTLCommandQueue> pointer

dev.num_threads = 4                   # CPU only: 0 shares the Dr.Jit pool (default),
                                      # a positive value creates a private pool
dev.verbose = 1
dev.type, dev.version, dev.system_memory_supported, dev.managed_memory_supported
dev.sync()                            # wait for asynchronous work
```

A device commits itself when the first buffer or filter is created. Parameters
such as `num_threads` must be set before that point.

### Filters

```python
flt = dev.new_filter("RT")            # or "RTLightmap"

flt.set_image("color", color)         # arrays: layout inferred, zero-copy when possible
flt.set_image("albedo", albedo)
flt.set_image("normal", normal)
flt.set_image("output", output)       # must be writable
flt.hdr = True
flt.clean_aux = True
flt.quality = oidn.Quality.High
flt.input_scale = 0.5                 # None selects automatic scaling
flt.max_memory_mb = 2048
flt.set("cleanAux", True)             # generic access by OIDN parameter name
flt.set_data("weights", blob)         # user-trained weights (bytes or uint8 array)
flt.set_progress_monitor(lambda p: True)  # return False to cancel

flt.execute()                         # commits pending changes, runs, and waits
flt.execute_async(); dev.sync()
```

`set_image()` accepts arrays with the layout rules of `denoise()`. Row and
pixel strides are passed to OIDN, so slices of larger arrays work as long as
the channel dimension stays contiguous. Whether an array can be bound without
a copy depends on the device:

| Device | Bound without copy |
|---|---|
| CPU | any host array, including CUDA pinned and managed memory |
| CUDA | arrays on the same CUDA device, pinned host memory, and host memory if the GPU supports pageable memory access |
| Metal | any host array, and views of buffers created on the device |

Use `dev.can_share(array)` to test this in advance. When binding is not
possible, `set_image()` raises a `TypeError` that points at the buffer API.

Filters can also take a `Buffer` with an explicit description:

```python
flt.set_image("color", buf, format=oidn.Format.Float3, width=w, height=h,
              byte_offset=0, pixel_stride=0, row_stride=0)
```

### Buffers

Buffers are memory allocations made by a device. They are the way to work
with memory that the host cannot address, such as dedicated GPU memory, and
they provide zero-copy views on unified-memory systems.

```python
buf = dev.new_buffer(nbytes)                          # host and device accessible
buf = dev.new_buffer(nbytes, oidn.Storage.Device)     # device memory only
buf = dev.new_shared_buffer(array)                    # wrap device-accessible memory

buf.size, buf.storage, buf.device, buf.data_ptr

view = buf.view("float32", (h, w, 3))     # DLPack and buffer-protocol object
img = np.from_dlpack(view)                # or torch.from_dlpack(view), ...

buf.write(host_array); buf.read(host_array)           # copies through the host
buf.write_async(src); buf.read_async(dst); dev.sync()
```

Arrays created from `buf.view()` are recognized by `set_image()` and bound
through the underlying buffer. On Apple silicon, rendering into such a view
and denoising it involves no copies at all. On a CUDA device, a device-storage
buffer viewed through `torch.from_dlpack()` gives a tensor that OIDN wrote
directly.

### Errors

All OIDN errors raise `oidn.Error`, whose `code` attribute is an
`oidn.ErrorCode`. Cancellation through a progress monitor raises
`oidn.Error` with `ErrorCode.Cancelled`. An exception raised inside the
progress monitor cancels the filter and propagates unchanged.

## Building from source

The build needs CMake 3.21 or newer, a C++17 compiler, and a binary release
of [ISPC](https://ispc.github.io/downloads.html) unpacked into `ext/ispc`,
so that `ext/ispc/bin/ispc` exists. Metal support requires Xcode 15 or newer.
CUDA support requires CUDA 12.8 or newer and is enabled automatically when the
toolkit is found.

```
git clone --recursive https://github.com/mitsuba-renderer/mitsuba-oidn
cd mitsuba-oidn
pip install nanobind==3.0.1 scikit-build-core
pip install --no-build-isolation -ve .
pytest
```

The OIDN weights are stored with git-lfs, which must be installed before
cloning.

## License

mitsuba-oidn is licensed under the BSD 3-Clause license. It bundles Intel Open
Image Denoise, which is licensed under the Apache License 2.0.
