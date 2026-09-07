import numpy as np
import pytest

import mitsuba_oidn as oidn


def make_scene(h=96, w=128, noise=0.2, seed=0):
    """Smooth reference image, a noisy version, and clean auxiliary features"""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    albedo = np.stack([x / w, y / h, 0.5 * np.ones_like(x)], axis=-1)
    normal = np.stack([np.sin(x / 10), np.cos(y / 10), np.ones_like(x)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    clean = albedo * (0.6 + 0.4 * normal[..., 2:3])
    noisy = clean + rng.normal(0, noise, clean.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 1).astype(np.float32)
    return clean.astype(np.float32), noisy, albedo.astype(np.float32), normal.astype(np.float32)


def mse(a, b):
    return float(np.mean((np.asarray(a, dtype=np.float32) - b) ** 2))


def devices():
    result = [oidn.Device(oidn.DeviceType.CPU)]
    if any(d.type == oidn.DeviceType.Metal for d in oidn.physical_devices()):
        result.append(oidn.Device(oidn.DeviceType.Metal))
    return result


def test_version_and_physical_devices():
    assert oidn.oidn_version().startswith("2.")
    devs = oidn.physical_devices()
    assert len(devs) >= 1
    assert any(d.type == oidn.DeviceType.CPU for d in devs)
    assert all(isinstance(d.name, str) for d in devs)


@pytest.mark.parametrize("device", devices(), ids=lambda d: repr(d))
def test_denoise_numpy(device):
    clean, noisy, albedo, normal = make_scene()
    out = oidn.denoise(noisy, albedo, normal, device=device)
    assert isinstance(out, np.ndarray)
    assert out.shape == clean.shape and out.dtype == np.float32
    assert mse(out, clean) < 0.25 * mse(noisy, clean)


@pytest.mark.parametrize("device", devices(), ids=lambda d: repr(d))
def test_denoise_color_only(device):
    clean, noisy, _, _ = make_scene()
    out = oidn.denoise(noisy, device=device)
    assert mse(out, clean) < 0.5 * mse(noisy, clean)


def test_denoise_rgba_and_output():
    clean, noisy, albedo, normal = make_scene()
    rgba = np.concatenate([noisy, np.full(noisy.shape[:2] + (1,), 0.7, np.float32)], axis=-1)
    out = oidn.denoise(rgba, albedo, normal)
    assert out.shape == clean.shape

    dst = rgba.copy()
    ret = oidn.denoise(rgba, albedo, normal, output=dst)
    assert ret is dst
    assert np.all(dst[..., 3] == 0.7)
    assert mse(dst[..., :3], clean) < 0.25 * mse(noisy, clean)


def test_denoise_float16_and_strided():
    clean, noisy, albedo, normal = make_scene()
    out = oidn.denoise(noisy.astype(np.float16), device=oidn.Device(oidn.DeviceType.CPU))
    assert out.dtype == np.float16
    assert mse(out.astype(np.float32), clean) < 0.5 * mse(noisy, clean)

    # Cropped views are bound without a copy on the CPU
    crop = noisy[8:72, 16:112]
    out = oidn.denoise(crop, albedo[8:72, 16:112], normal[8:72, 16:112],
                       device=oidn.Device(oidn.DeviceType.CPU))
    assert out.shape == crop.shape
    assert mse(out, clean[8:72, 16:112]) < 0.25 * mse(crop, clean[8:72, 16:112])


def test_object_api_cpu():
    clean, noisy, albedo, normal = make_scene()
    dev = oidn.Device(oidn.DeviceType.CPU)
    assert dev.type == oidn.DeviceType.CPU
    assert dev.system_memory_supported
    assert dev.can_share(noisy)

    flt = dev.new_filter("RT")
    out = np.empty_like(noisy)
    flt.set_image("color", noisy)
    flt.set_image("albedo", albedo)
    flt.set_image("normal", normal)
    flt.set_image("output", out)
    flt.hdr = False
    flt.quality = oidn.Quality.High
    assert flt.quality == oidn.Quality.High
    assert flt.input_scale is None
    assert flt.tile_alignment > 0

    progress = []
    flt.set_progress_monitor(lambda p: progress.append(p) or True)
    flt.execute()
    assert progress and progress[-1] == pytest.approx(1.0)
    assert mse(out, clean) < 0.25 * mse(noisy, clean)


def test_buffers():
    clean, noisy, albedo, normal = make_scene()
    dev = oidn.Device(oidn.DeviceType.CPU)
    buf = dev.new_buffer(noisy.nbytes, oidn.Storage.Host)
    assert buf.size == noisy.nbytes
    assert buf.storage == oidn.Storage.Host

    view = np.from_dlpack(buf.view("float32", noisy.shape))
    view[...] = noisy
    assert dev.can_share(view)

    flt = dev.new_filter("RT")
    flt.set_image("color", view)
    flt.set_image("output", view)
    flt.execute()
    assert mse(view, clean) < 0.5 * mse(noisy, clean)

    # Explicit buffer binding and host copies
    buf2 = dev.new_buffer(noisy.nbytes)
    buf2.write(noisy)
    back = np.empty_like(noisy)
    buf2.read(back)
    assert np.array_equal(back, noisy)
    flt.set_image("color", buf2, format=oidn.Format.Float3, width=noisy.shape[1],
                  height=noisy.shape[0])
    flt.execute()


def test_errors():
    dev = oidn.Device(oidn.DeviceType.CPU)
    with pytest.raises(oidn.Error) as excinfo:
        dev.new_filter("DoesNotExist")
    assert excinfo.value.code == oidn.ErrorCode.InvalidArgument

    flt = dev.new_filter("RT")
    with pytest.raises(TypeError):
        flt.set_image("color", np.zeros((4, 4, 3), dtype=np.float64))
    with pytest.raises(TypeError):
        flt.set_image("output", np.zeros((4, 4, 3), dtype=np.float32)[..., ::-1])
    ro = np.zeros((4, 4, 3), dtype=np.float32)
    ro.flags.writeable = False
    with pytest.raises(TypeError):
        flt.set_image("output", ro)
    flt.set_image("color", ro)

    with pytest.raises(oidn.Error):
        flt.execute()  # no output image


def test_cancel():
    _, noisy, _, _ = make_scene(h=256, w=256)
    dev = oidn.Device(oidn.DeviceType.CPU)
    flt = dev.new_filter("RT")
    flt.set_image("color", noisy)
    flt.set_image("output", np.empty_like(noisy))
    flt.set_progress_monitor(lambda p: False)
    with pytest.raises(oidn.Error) as excinfo:
        flt.execute()
    assert excinfo.value.code == oidn.ErrorCode.Cancelled

    flt.set_progress_monitor(lambda p: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        flt.execute()


def test_drjit_tensors():
    from drjit.llvm import TensorXf

    clean, noisy, albedo, normal = make_scene()
    out = oidn.denoise(TensorXf(noisy), TensorXf(albedo), TensorXf(normal),
                       device=oidn.Device(oidn.DeviceType.CPU))
    assert isinstance(out, TensorXf)
    assert out.shape == clean.shape
    assert mse(out.numpy(), clean) < 0.25 * mse(noisy, clean)
