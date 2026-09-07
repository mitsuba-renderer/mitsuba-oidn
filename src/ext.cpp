// mitsuba-oidn: nanobind bindings for Intel Open Image Denoise

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <OpenImageDenoise/oidn.h>

#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <unistd.h>
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if defined(OIDN_API_NAMESPACE)
using namespace OIDN_API_NAMESPACE;
#endif

// ------------------------------------------------------------------------------------------------
// Error handling
// ------------------------------------------------------------------------------------------------

struct Error : std::runtime_error {
    OIDNError code;
    Error(OIDNError code, const char *message)
        : std::runtime_error(message ? message : "unknown error"), code(code) { }
};

static PyObject *error_type = nullptr;

// Raises the first error recorded for 'device' (or the global error slot when null)
static void check(OIDNDevice device) {
    const char *message = nullptr;
    OIDNError code = oidnGetDeviceError(device, &message);
    if (code != OIDN_ERROR_NONE)
        throw Error(code, message);
}

// ------------------------------------------------------------------------------------------------
// Thread pool (nanothread entry points resolved from a shared library)
// ------------------------------------------------------------------------------------------------

static void *load_library(const std::string &path) {
#if defined(_WIN32)
    int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], n);
    return (void *) LoadLibraryW(wpath.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

static void *load_symbol(void *lib, const char *name, bool required = true) {
#if defined(_WIN32)
    void *sym = (void *) GetProcAddress((HMODULE) lib, name);
#else
    void *sym = dlsym(lib, name);
#endif
    if (!sym && required)
        throw std::runtime_error(std::string("mitsuba_oidn: the thread pool library lacks the "
                                             "symbol '") + name + "'");
    return sym;
}

static void set_thread_pool(const std::string &path) {
    void *lib = load_library(path);
    if (!lib)
        throw std::runtime_error("mitsuba_oidn: could not load the thread pool library '" +
                                 path + "'");

    OIDNThreadPoolAPI api{};
    api.poolCreate = (decltype(api.poolCreate)) load_symbol(lib, "pool_create");
    api.poolDestroy = (decltype(api.poolDestroy)) load_symbol(lib, "pool_destroy");
    api.poolSize = (decltype(api.poolSize)) load_symbol(lib, "pool_size");
    api.poolThreadId = (decltype(api.poolThreadId)) load_symbol(lib, "pool_thread_id");
    api.coreCount = (decltype(api.coreCount)) load_symbol(lib, "core_count");
    api.performanceCoreCount =
        (decltype(api.performanceCoreCount)) load_symbol(lib, "performance_core_count");
    api.taskSubmitDep = (decltype(api.taskSubmitDep)) load_symbol(lib, "task_submit_dep");
    // The exclusive variant keeps the waiting thread from running unrelated queued work.
    // Older nanothread versions only provide the plain one, which works as well.
    void *wait = load_symbol(lib, "task_wait_and_release_exclusive", false);
    if (!wait)
        wait = load_symbol(lib, "task_wait_and_release");
    api.taskWaitAndRelease = (decltype(api.taskWaitAndRelease)) wait;

    oidnSetThreadPoolAPI(&api);
    check(nullptr);
}

// ------------------------------------------------------------------------------------------------
// Image descriptions derived from nd-arrays
// ------------------------------------------------------------------------------------------------

using AnyArray = nb::ndarray<nb::ro>;

// DLPack device types
enum : int {
    DL_CPU = 1, DL_CUDA = 2, DL_CUDA_HOST = 3, DL_METAL = 8,
    DL_ROCM = 10, DL_ROCM_HOST = 11, DL_CUDA_MANAGED = 13
};

struct ImageInfo {
    OIDNFormat format = OIDN_FORMAT_UNDEFINED;
    size_t width = 0, height = 0, channels = 0, itemsize = 0;
    size_t pixel_stride = 0, row_stride = 0; // bytes
    void *data = nullptr;
    size_t span = 0; // bytes touched by the image, starting at 'data'
    int device_type = 0, device_id = 0;
};

static size_t page_size() {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (size_t) info.dwPageSize;
#else
    return (size_t) sysconf(_SC_PAGESIZE);
#endif
}

static ImageInfo describe_image(const AnyArray &a) {
    ImageInfo info;
    nb::dlpack::dtype dt = a.dtype();

    if (dt.code != (uint8_t) nb::dlpack::dtype_code::Float || dt.lanes != 1 ||
        (dt.bits != 32 && dt.bits != 16))
        throw nb::type_error("image arrays must have dtype float32 or float16");

    size_t ndim = a.ndim();
    if (ndim != 2 && ndim != 3)
        throw nb::type_error("image arrays must have shape (H, W) or (H, W, C)");

    info.itemsize = dt.bits / 8;
    info.height = a.shape(0);
    info.width = a.shape(1);
    info.channels = ndim == 3 ? a.shape(2) : 1;

    if (info.channels < 1 || info.channels > 4)
        throw nb::type_error("image arrays must have 1 to 4 channels");
    if (info.width == 0 || info.height == 0)
        throw nb::type_error("image arrays must not be empty");

    int64_t s0 = a.stride(0), s1 = a.stride(1), s2 = ndim == 3 ? a.stride(2) : 1;
    if (info.channels > 1 && s2 != 1)
        throw nb::type_error("the channel dimension of image arrays must be contiguous");
    if (info.width > 1 && s1 < (int64_t) info.channels)
        throw nb::type_error("image arrays must have positive, non-overlapping pixel strides");

    info.pixel_stride = (info.width > 1 ? (size_t) s1 : info.channels) * info.itemsize;
    size_t min_row_stride = info.width * info.pixel_stride;
    if (info.height > 1 && (s0 <= 0 || (size_t) s0 * info.itemsize < min_row_stride))
        throw nb::type_error("image arrays must have positive, non-overlapping row strides");
    info.row_stride = info.height > 1 ? (size_t) s0 * info.itemsize : min_row_stride;

    size_t used = info.channels < 3 ? info.channels : 3;
    int base = dt.bits == 16 ? OIDN_FORMAT_HALF : OIDN_FORMAT_FLOAT;
    info.format = (OIDNFormat) (base + (int) used - 1);

    info.data = (void *) a.data();
    info.span = (info.height - 1) * info.row_stride + (info.width - 1) * info.pixel_stride +
                used * info.itemsize;
    info.device_type = a.device_type();
    info.device_id = a.device_id();
    return info;
}

// Casts an arbitrary Python object into an nd-array, reporting whether it is writable
static AnyArray to_array(nb::handle obj, bool &writable) {
    nb::ndarray<> rw;
    if (nb::try_cast(obj, rw, false)) {
        writable = true;
        return AnyArray(rw);
    }
    AnyArray ro;
    if (nb::try_cast(obj, ro, false)) {
        writable = false;
        return ro;
    }
    throw nb::type_error("expected an array supporting DLPack or the buffer protocol");
}

static const char *device_type_name(OIDNDeviceType type) {
    switch (type) {
        case OIDN_DEVICE_TYPE_CPU: return "CPU";
        case OIDN_DEVICE_TYPE_SYCL: return "SYCL";
        case OIDN_DEVICE_TYPE_CUDA: return "CUDA";
        case OIDN_DEVICE_TYPE_HIP: return "HIP";
        case OIDN_DEVICE_TYPE_METAL: return "Metal";
        default: return "unknown";
    }
}

// ------------------------------------------------------------------------------------------------
// Device
// ------------------------------------------------------------------------------------------------

struct BufferRange {
    size_t size;
    OIDNBuffer buffer;
};

struct Device {
    OIDNDevice h = nullptr;
    bool committed = false;
    int cuda_device_id = -1;

    // Device-accessible memory ranges of live buffers, keyed by start address
    std::map<uintptr_t, BufferRange> buffers;

    explicit Device(OIDNDevice h) : h(h) {
        if (!h) {
            check(nullptr);
            throw Error(OIDN_ERROR_UNKNOWN, "device creation failed");
        }
    }

    Device(Device &&o) noexcept
        : h(o.h), committed(o.committed), cuda_device_id(o.cuda_device_id),
          buffers(std::move(o.buffers)) {
        o.h = nullptr;
    }

    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    ~Device() {
        if (h)
            oidnReleaseDevice(h);
    }

    OIDNDeviceType type() const { return (OIDNDeviceType) oidnGetDeviceInt(h, "type"); }

    void commit() {
        if (committed)
            return;
        {
            nb::gil_scoped_release guard;
            oidnCommitDevice(h);
        }
        check(h);
        committed = true;
    }

    void sync() {
        nb::gil_scoped_release guard;
        oidnSyncDevice(h);
    }

    // Finds a registered buffer containing [ptr, ptr + span)
    const BufferRange *find_buffer(const void *ptr, size_t span, size_t *offset) const {
        uintptr_t p = (uintptr_t) ptr;
        auto it = buffers.upper_bound(p);
        if (it == buffers.begin())
            return nullptr;
        --it;
        if (p + span > it->first + it->second.size)
            return nullptr;
        *offset = p - it->first;
        return &it->second;
    }

    // Returns an empty string if the device can access the image memory directly, and an
    // explanation otherwise
    std::string share_error(const ImageInfo &info) const {
        size_t offset;
        if (find_buffer(info.data, info.span, &offset))
            return "";

        OIDNDeviceType t = type();
        int dt = info.device_type;
        bool host = dt == DL_CPU || dt == DL_CUDA_HOST || dt == DL_ROCM_HOST;

        switch (t) {
            case OIDN_DEVICE_TYPE_CPU:
                if (host || dt == DL_CUDA_MANAGED)
                    return "";
                return "the CPU device cannot access arrays stored in device memory";

            case OIDN_DEVICE_TYPE_CUDA:
                if (dt == DL_CUDA || dt == DL_CUDA_MANAGED) {
                    if (cuda_device_id >= 0 && info.device_id != cuda_device_id)
                        return "the array is stored on a different CUDA device";
                    return "";
                }
                if (dt == DL_CUDA_HOST)
                    return "";
                if (host && oidnGetDeviceBool(h, "systemMemorySupported"))
                    return "";
                return "the CUDA device cannot access this array directly; pass a CUDA array "
                       "or a Buffer created on the device";

            case OIDN_DEVICE_TYPE_METAL:
                if (host)
                    return "";
                return "the Metal device can only access host arrays and Buffers created "
                       "on the device";

            default:
                return "the device cannot access this array directly; pass a Buffer "
                       "created on the device";
        }
    }
};

// Wraps the memory behind a host image in a shared OIDN buffer covering whole pages
static OIDNBuffer new_page_buffer(Device &dev, const ImageInfo &info, size_t *offset) {
    size_t ps = page_size();
    uintptr_t begin = (uintptr_t) info.data & ~(uintptr_t) (ps - 1);
    uintptr_t end = ((uintptr_t) info.data + info.span + ps - 1) & ~(uintptr_t) (ps - 1);
    OIDNBuffer buffer = oidnNewSharedBuffer(dev.h, (void *) begin, end - begin);
    check(dev.h);
    *offset = (uintptr_t) info.data - begin;
    return buffer;
}

// ------------------------------------------------------------------------------------------------
// Buffer
// ------------------------------------------------------------------------------------------------

struct Buffer {
    OIDNBuffer h = nullptr;
    nb::object device_obj;
    Device *device = nullptr;
    nb::object keep; // Python object owning shared memory, if any
    void *data = nullptr;
    size_t size = 0;

    Buffer(nb::handle device_obj, OIDNBuffer h, nb::object keep = nb::none())
        : h(h), device_obj(nb::borrow(device_obj)), device(nb::inst_ptr<Device>(device_obj)),
          keep(std::move(keep)) {
        if (!h) {
            check(device->h);
            throw Error(OIDN_ERROR_UNKNOWN, "buffer creation failed");
        }
        data = oidnGetBufferData(h);
        size = oidnGetBufferSize(h);
        if (data && size)
            device->buffers[(uintptr_t) data] = BufferRange{ size, h };
    }

    Buffer(Buffer &&o) noexcept
        : h(o.h), device_obj(std::move(o.device_obj)), device(o.device),
          keep(std::move(o.keep)), data(o.data), size(o.size) {
        o.h = nullptr;
    }

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    ~Buffer() {
        if (!h)
            return;
        if (data && size) {
            auto it = device->buffers.find((uintptr_t) data);
            if (it != device->buffers.end() && it->second.buffer == h)
                device->buffers.erase(it);
        }
        oidnReleaseBuffer(h);
    }

    OIDNStorage storage() const { return oidnGetBufferStorage(h); }

    // DLPack device of the buffer memory
    void dlpack_device(int &type, int &id) const {
        OIDNDeviceType t = device->type();
        OIDNStorage st = storage();
        id = 0;
        type = DL_CPU;
        if (t == OIDN_DEVICE_TYPE_CUDA) {
            id = device->cuda_device_id >= 0 ? device->cuda_device_id : 0;
            type = st == OIDN_STORAGE_DEVICE    ? DL_CUDA
                   : st == OIDN_STORAGE_MANAGED ? DL_CUDA_MANAGED
                                                : DL_CUDA_HOST;
        } else if (t == OIDN_DEVICE_TYPE_HIP) {
            type = st == OIDN_STORAGE_DEVICE ? DL_ROCM : DL_ROCM_HOST;
        }
    }
};

static nb::dlpack::dtype parse_dtype(const std::string &name) {
    using nb::dlpack::dtype_code;
    struct Entry { const char *name; dtype_code code; uint8_t bits; };
    static const Entry table[] = {
        { "float16", dtype_code::Float, 16 }, { "float32", dtype_code::Float, 32 },
        { "float64", dtype_code::Float, 64 }, { "int8", dtype_code::Int, 8 },
        { "int16", dtype_code::Int, 16 },     { "int32", dtype_code::Int, 32 },
        { "int64", dtype_code::Int, 64 },     { "uint8", dtype_code::UInt, 8 },
        { "uint16", dtype_code::UInt, 16 },   { "uint32", dtype_code::UInt, 32 },
        { "uint64", dtype_code::UInt, 64 },   { "bool", dtype_code::Bool, 8 }
    };
    for (const Entry &e : table) {
        if (name == e.name)
            return nb::dlpack::dtype{ (uint8_t) e.code, e.bits, 1 };
    }
    throw nb::value_error(("unsupported dtype '" + name + "'").c_str());
}

using HostArrayRW = nb::ndarray<nb::c_contig, nb::device::cpu>;
using HostArrayRO = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;
using ExportArray = nb::ndarray<nb::array_api>;

static ExportArray buffer_view(Buffer &self, const std::string &dtype_name,
                               std::vector<size_t> shape, size_t byte_offset) {
    if (!self.data)
        throw Error(OIDN_ERROR_INVALID_OPERATION,
                    "the buffer memory is not accessible through a pointer");

    nb::dlpack::dtype dtype = parse_dtype(dtype_name);
    size_t itemsize = dtype.bits / 8, count = 1;
    for (size_t s : shape)
        count *= s;
    if (byte_offset + count * itemsize > self.size)
        throw nb::value_error("the requested view exceeds the buffer size");

    int device_type, device_id;
    self.dlpack_device(device_type, device_id);

    return ExportArray((uint8_t *) self.data + byte_offset, shape.size(), shape.data(),
                       nb::handle(), nullptr, dtype, device_type, device_id);
}

// ------------------------------------------------------------------------------------------------
// Filter
// ------------------------------------------------------------------------------------------------

struct Filter {
    OIDNFilter h = nullptr;
    nb::object device_obj;
    Device *device = nullptr;
    std::string type;
    bool dirty = true;

    // Python objects backing image and data parameters
    std::map<std::string, nb::object> images, data;
    // Page-granular buffers wrapping host images (Metal)
    std::map<std::string, OIDNBuffer> shared_buffers;

    nb::object progress;
    std::exception_ptr progress_error;

    Filter(nb::handle device_obj, const std::string &type)
        : device_obj(nb::borrow(device_obj)), device(nb::inst_ptr<Device>(device_obj)),
          type(type) {
        device->commit();
        h = oidnNewFilter(device->h, type.c_str());
        if (!h) {
            check(device->h);
            throw Error(OIDN_ERROR_UNKNOWN, "filter creation failed");
        }
    }

    Filter(Filter &&o) noexcept
        : h(o.h), device_obj(std::move(o.device_obj)), device(o.device),
          type(std::move(o.type)), dirty(o.dirty), images(std::move(o.images)),
          data(std::move(o.data)), shared_buffers(std::move(o.shared_buffers)),
          progress(std::move(o.progress)), progress_error(std::move(o.progress_error)) {
        o.h = nullptr;
        o.shared_buffers.clear();
    }

    Filter(const Filter &) = delete;
    Filter &operator=(const Filter &) = delete;

    ~Filter() {
        for (auto &kv : shared_buffers)
            oidnReleaseBuffer(kv.second);
        if (h)
            oidnReleaseFilter(h);
    }

    void release_shared(const std::string &name) {
        auto it = shared_buffers.find(name);
        if (it != shared_buffers.end()) {
            oidnReleaseBuffer(it->second);
            shared_buffers.erase(it);
        }
    }

    void set_image_from_buffer(const std::string &name, nb::handle obj, Buffer &buf,
                               OIDNFormat format, size_t width, size_t height,
                               size_t byte_offset, size_t pixel_stride, size_t row_stride) {
        if (buf.device != device)
            throw nb::type_error("the buffer belongs to a different device");
        oidnSetFilterImage(h, name.c_str(), buf.h, format, width, height, byte_offset,
                           pixel_stride, row_stride);
        check(device->h);
        release_shared(name);
        images[name] = nb::borrow(obj);
        dirty = true;
    }

    void set_image_from_array(const std::string &name, nb::handle obj) {
        bool writable;
        AnyArray arr = to_array(obj, writable);
        if (name == "output" && !writable)
            throw nb::type_error("the output image must be writable");

        ImageInfo info = describe_image(arr);
        std::string err = device->share_error(info);
        if (!err.empty())
            throw nb::type_error(err.c_str());

        size_t offset = 0;
        OIDNBuffer page_buffer = nullptr;
        const BufferRange *range = device->find_buffer(info.data, info.span, &offset);

        if (range) {
            oidnSetFilterImage(h, name.c_str(), range->buffer, info.format, info.width,
                               info.height, offset, info.pixel_stride, info.row_stride);
        } else if (device->type() == OIDN_DEVICE_TYPE_METAL) {
            page_buffer = new_page_buffer(*device, info, &offset);
            oidnSetFilterImage(h, name.c_str(), page_buffer, info.format, info.width,
                               info.height, offset, info.pixel_stride, info.row_stride);
        } else {
            oidnSetSharedFilterImage(h, name.c_str(), info.data, info.format, info.width,
                                     info.height, 0, info.pixel_stride, info.row_stride);
        }

        try {
            check(device->h);
        } catch (...) {
            if (page_buffer)
                oidnReleaseBuffer(page_buffer);
            throw;
        }

        release_shared(name);
        if (page_buffer)
            shared_buffers[name] = page_buffer;
        images[name] = nb::borrow(obj);
        dirty = true;
    }

    void unset_image(const std::string &name) {
        oidnUnsetFilterImage(h, name.c_str());
        check(device->h);
        release_shared(name);
        images.erase(name);
        dirty = true;
    }

    void commit() {
        {
            nb::gil_scoped_release guard;
            oidnCommitFilter(h);
        }
        check(device->h);
        dirty = false;
    }

    // A Python exception raised by the progress monitor takes precedence over the cancellation
    // error that it triggered, which is cleared here
    void rethrow_progress_error() {
        if (progress_error) {
            const char *message = nullptr;
            oidnGetDeviceError(device->h, &message);
            std::exception_ptr e = progress_error;
            progress_error = nullptr;
            std::rethrow_exception(e);
        }
    }

    void execute(bool async) {
        if (dirty)
            commit();
        progress_error = nullptr;
        {
            nb::gil_scoped_release guard;
            if (async)
                oidnExecuteFilterAsync(h);
            else
                oidnExecuteFilter(h);
        }
        rethrow_progress_error();
        check(device->h);
    }

    static bool progress_callback(void *user, double n) {
        Filter *self = (Filter *) user;
        nb::gil_scoped_acquire guard;
        try {
            return nb::cast<bool>(self->progress(n));
        } catch (...) {
            self->progress_error = std::current_exception();
            return false;
        }
    }

    void set_progress_monitor(nb::object func) {
        if (func.is_none()) {
            oidnSetFilterProgressMonitorFunction(h, nullptr, nullptr);
            progress = nb::object();
        } else {
            progress = std::move(func);
            oidnSetFilterProgressMonitorFunction(h, progress_callback, this);
        }
        check(device->h);
    }

    bool get_bool(const std::string &name) {
        bool v = oidnGetFilterBool(h, name.c_str());
        check(device->h);
        return v;
    }
    int get_int(const std::string &name) {
        int v = oidnGetFilterInt(h, name.c_str());
        check(device->h);
        return v;
    }
    float get_float(const std::string &name) {
        float v = oidnGetFilterFloat(h, name.c_str());
        check(device->h);
        return v;
    }
    void set_bool(const std::string &name, bool v) {
        oidnSetFilterBool(h, name.c_str(), v);
        check(device->h);
        dirty = true;
    }
    void set_int(const std::string &name, int v) {
        oidnSetFilterInt(h, name.c_str(), v);
        check(device->h);
        dirty = true;
    }
    void set_float(const std::string &name, float v) {
        oidnSetFilterFloat(h, name.c_str(), v);
        check(device->h);
        dirty = true;
    }
};

// ------------------------------------------------------------------------------------------------
// Physical devices
// ------------------------------------------------------------------------------------------------

struct PhysicalDevice {
    int id;
};

static nb::object physical_data(int id, const char *name, const char *flag) {
    if (!oidnGetPhysicalDeviceBool(id, flag)) {
        check(nullptr);
        return nb::none();
    }
    size_t size = 0;
    const void *data = oidnGetPhysicalDeviceData(id, name, &size);
    check(nullptr);
    return nb::bytes((const char *) data, size);
}

// ------------------------------------------------------------------------------------------------
// Module
// ------------------------------------------------------------------------------------------------

NB_MODULE(_mitsuba_oidn_ext, m) {
    m.doc() = "Python bindings for Intel Open Image Denoise";

    // Exception type with a 'code' attribute
    error_type = PyErr_NewExceptionWithDoc(
        "mitsuba_oidn.Error", "Error raised by Open Image Denoise. The 'code' attribute "
        "holds the associated ErrorCode.", PyExc_RuntimeError, nullptr);
    if (!error_type)
        throw nb::python_error();
    m.attr("Error") = nb::borrow(error_type);

    nb::register_exception_translator(
        [](const std::exception_ptr &p, void *payload) {
            try {
                std::rethrow_exception(p);
            } catch (const Error &e) {
                nb::object type = nb::borrow((PyObject *) payload);
                nb::object exc = type(e.what());
                exc.attr("code") = nb::cast(e.code);
                PyErr_SetObject((PyObject *) payload, exc.ptr());
            }
        },
        error_type);

    nb::enum_<OIDNDeviceType>(m, "DeviceType")
        .value("Default", OIDN_DEVICE_TYPE_DEFAULT)
        .value("CPU", OIDN_DEVICE_TYPE_CPU)
        .value("SYCL", OIDN_DEVICE_TYPE_SYCL)
        .value("CUDA", OIDN_DEVICE_TYPE_CUDA)
        .value("HIP", OIDN_DEVICE_TYPE_HIP)
        .value("Metal", OIDN_DEVICE_TYPE_METAL);

    nb::enum_<OIDNError>(m, "ErrorCode")
        .value("NoError", OIDN_ERROR_NONE)
        .value("Unknown", OIDN_ERROR_UNKNOWN)
        .value("InvalidArgument", OIDN_ERROR_INVALID_ARGUMENT)
        .value("InvalidOperation", OIDN_ERROR_INVALID_OPERATION)
        .value("OutOfMemory", OIDN_ERROR_OUT_OF_MEMORY)
        .value("UnsupportedHardware", OIDN_ERROR_UNSUPPORTED_HARDWARE)
        .value("Cancelled", OIDN_ERROR_CANCELLED);

    nb::enum_<OIDNFormat>(m, "Format")
        .value("Undefined", OIDN_FORMAT_UNDEFINED)
        .value("Float", OIDN_FORMAT_FLOAT)
        .value("Float2", OIDN_FORMAT_FLOAT2)
        .value("Float3", OIDN_FORMAT_FLOAT3)
        .value("Float4", OIDN_FORMAT_FLOAT4)
        .value("Half", OIDN_FORMAT_HALF)
        .value("Half2", OIDN_FORMAT_HALF2)
        .value("Half3", OIDN_FORMAT_HALF3)
        .value("Half4", OIDN_FORMAT_HALF4);

    nb::enum_<OIDNStorage>(m, "Storage")
        .value("Undefined", OIDN_STORAGE_UNDEFINED)
        .value("Host", OIDN_STORAGE_HOST)
        .value("Device", OIDN_STORAGE_DEVICE)
        .value("Managed", OIDN_STORAGE_MANAGED);

    nb::enum_<OIDNQuality>(m, "Quality")
        .value("Default", OIDN_QUALITY_DEFAULT)
        .value("Fast", OIDN_QUALITY_FAST)
        .value("Balanced", OIDN_QUALITY_BALANCED)
        .value("High", OIDN_QUALITY_HIGH);

    m.def("_set_thread_pool", &set_thread_pool, "path"_a,
          "Load the nanothread shared library at 'path' and install it as the thread pool "
          "used by CPU devices.");

    m.def("oidn_version", []() { return std::string(OIDN_VERSION_STRING); },
          "Version of the bundled Open Image Denoise library.");

    m.def("image_info",
          [](nb::handle obj) {
              bool writable;
              AnyArray arr = to_array(obj, writable);
              ImageInfo info = describe_image(arr);
              return nb::make_tuple(info.height, info.width, info.channels,
                                    info.itemsize, info.device_type, info.device_id,
                                    writable, arr.ndim());
          },
          "array"_a,
          "Describe an image array as a tuple (height, width, channels, itemsize, "
          "dlpack_device_type, dlpack_device_id, writable, ndim).");

    // Physical devices

    nb::class_<PhysicalDevice>(m, "PhysicalDevice",
                               "A physical device (CPU or GPU) supported by the library")
        .def_ro("id", &PhysicalDevice::id)
        .def_prop_ro("name",
                     [](PhysicalDevice &d) {
                         const char *s = oidnGetPhysicalDeviceString(d.id, "name");
                         check(nullptr);
                         return std::string(s ? s : "");
                     })
        .def_prop_ro("type",
                     [](PhysicalDevice &d) {
                         int t = oidnGetPhysicalDeviceInt(d.id, "type");
                         check(nullptr);
                         return (OIDNDeviceType) t;
                     })
        .def_prop_ro("uuid", [](PhysicalDevice &d) { return physical_data(d.id, "uuid", "uuidSupported"); })
        .def_prop_ro("luid", [](PhysicalDevice &d) { return physical_data(d.id, "luid", "luidSupported"); })
        .def_prop_ro("pci_address",
                     [](PhysicalDevice &d) -> nb::object {
                         if (!oidnGetPhysicalDeviceBool(d.id, "pciAddressSupported")) {
                             check(nullptr);
                             return nb::none();
                         }
                         nb::tuple t = nb::make_tuple(oidnGetPhysicalDeviceInt(d.id, "pciDomain"),
                                                      oidnGetPhysicalDeviceInt(d.id, "pciBus"),
                                                      oidnGetPhysicalDeviceInt(d.id, "pciDevice"),
                                                      oidnGetPhysicalDeviceInt(d.id, "pciFunction"));
                         check(nullptr);
                         return t;
                     })
        .def("__repr__", [](PhysicalDevice &d) {
            const char *s = oidnGetPhysicalDeviceString(d.id, "name");
            int t = oidnGetPhysicalDeviceInt(d.id, "type");
            check(nullptr);
            return "PhysicalDevice(id=" + std::to_string(d.id) + ", type=" +
                   device_type_name((OIDNDeviceType) t) + ", name='" + (s ? s : "") + "')";
        });

    m.def("physical_devices",
          []() {
              int n = oidnGetNumPhysicalDevices();
              check(nullptr);
              std::vector<PhysicalDevice> result;
              for (int i = 0; i < n; ++i)
                  result.push_back(PhysicalDevice{ i });
              return result;
          },
          "List the supported physical devices, ordered approximately from fastest to "
          "slowest.");

    // Device

    nb::class_<Device>(m, "Device", "A logical Open Image Denoise device")
        .def("__init__",
             [](Device *self, OIDNDeviceType type) {
                 new (self) Device(oidnNewDevice(type));
             },
             "type"_a = OIDN_DEVICE_TYPE_DEFAULT,
             "Create a device of the given type. The default picks the fastest available "
             "physical device.")
        .def_static("from_physical",
                    [](int id) { return Device(oidnNewDeviceByID(id)); }, "id"_a,
                    "Create a device on the physical device with the given ID.")
        .def_static("from_uuid",
                    [](nb::bytes uuid) {
                        if (uuid.size() != OIDN_UUID_SIZE)
                            throw nb::value_error("UUIDs must have 16 bytes");
                        return Device(oidnNewDeviceByUUID(uuid.c_str()));
                    },
                    "uuid"_a, "Create a device on the physical device with the given UUID.")
        .def_static("from_luid",
                    [](nb::bytes luid) {
                        if (luid.size() != OIDN_LUID_SIZE)
                            throw nb::value_error("LUIDs must have 8 bytes");
                        return Device(oidnNewDeviceByLUID(luid.c_str()));
                    },
                    "luid"_a, "Create a device on the physical device with the given LUID.")
        .def_static("from_pci_address",
                    [](int domain, int bus, int device, int function) {
                        return Device(oidnNewDeviceByPCIAddress(domain, bus, device, function));
                    },
                    "domain"_a, "bus"_a, "device"_a, "function"_a,
                    "Create a device on the physical device with the given PCI address.")
        .def_static("cuda",
                    [](int device_id, uintptr_t stream) {
                        cudaStream_t s = (cudaStream_t) stream;
                        Device d(oidnNewCUDADevice(&device_id, &s, 1));
                        d.cuda_device_id = device_id;
                        return d;
                    },
                    "device_id"_a = 0, "stream"_a = 0,
                    "Create a CUDA device on the given CUDA device ordinal. 'stream' is a raw "
                    "cudaStream_t handle (0 selects the default stream).")
        .def_static("metal",
                    [](uintptr_t command_queue) {
                        MTLCommandQueue_id q = (MTLCommandQueue_id) command_queue;
                        return Device(oidnNewMetalDevice(&q, 1));
                    },
                    "command_queue"_a,
                    "Create a Metal device using the given raw id<MTLCommandQueue> pointer.")
        .def_prop_ro("type", &Device::type)
        .def_prop_ro("version",
                     [](Device &d) {
                         int v = oidnGetDeviceInt(d.h, "version");
                         check(d.h);
                         return std::to_string(v / 10000) + "." + std::to_string(v / 100 % 100) +
                                "." + std::to_string(v % 100);
                     })
        .def_prop_ro("system_memory_supported",
                     [](Device &d) { return oidnGetDeviceBool(d.h, "systemMemorySupported"); })
        .def_prop_ro("managed_memory_supported",
                     [](Device &d) { return oidnGetDeviceBool(d.h, "managedMemorySupported"); })
        .def_prop_ro("committed", [](Device &d) { return d.committed; })
        .def_prop_ro("cuda_device_id", [](Device &d) { return d.cuda_device_id; })
        .def_prop_rw("verbose", [](Device &d) { return oidnGetDeviceInt(d.h, "verbose"); },
                     [](Device &d, int v) { oidnSetDeviceInt(d.h, "verbose", v); check(d.h); })
        .def_prop_rw("num_threads",
                     [](Device &d) { int v = oidnGetDeviceInt(d.h, "numThreads"); check(d.h); return v; },
                     [](Device &d, int v) { oidnSetDeviceInt(d.h, "numThreads", v); check(d.h); },
                     "Thread count of CPU devices. Zero (the default) uses the shared thread "
                     "pool; a positive value creates a private pool of that size.")
        .def("get_bool", [](Device &d, const std::string &n) { bool v = oidnGetDeviceBool(d.h, n.c_str()); check(d.h); return v; }, "name"_a)
        .def("get_int", [](Device &d, const std::string &n) { int v = oidnGetDeviceInt(d.h, n.c_str()); check(d.h); return v; }, "name"_a)
        .def("set_bool", [](Device &d, const std::string &n, bool v) { oidnSetDeviceBool(d.h, n.c_str(), v); check(d.h); }, "name"_a, "value"_a)
        .def("set_int", [](Device &d, const std::string &n, int v) { oidnSetDeviceInt(d.h, n.c_str(), v); check(d.h); }, "name"_a, "value"_a)
        .def("commit", &Device::commit,
             "Commit the device parameters. Called implicitly when the first buffer or "
             "filter is created.")
        .def("sync", &Device::sync, "Wait for all asynchronous operations to complete.")
        .def("new_buffer",
             [](nb::handle_t<Device> self, size_t byte_size, OIDNStorage storage) {
                 Device &d = nb::cast<Device &>(self);
                 d.commit();
                 OIDNBuffer b = storage == OIDN_STORAGE_UNDEFINED
                                    ? oidnNewBuffer(d.h, byte_size)
                                    : oidnNewBufferWithStorage(d.h, byte_size, storage);
                 return Buffer(self, b);
             },
             "byte_size"_a, "storage"_a = OIDN_STORAGE_UNDEFINED,
             "Allocate a buffer. The default storage mode is accessible by both host and "
             "device.")
        .def("new_shared_buffer",
             [](nb::handle_t<Device> self, nb::handle array) {
                 Device &d = nb::cast<Device &>(self);
                 d.commit();
                 nb::ndarray<> arr;
                 if (!nb::try_cast(array, arr, false))
                     throw nb::type_error("expected a writable array supporting DLPack or "
                                          "the buffer protocol");
                 if (arr.ndim() > 0) {
                     size_t expected = 1;
                     for (size_t i = 0; i < arr.ndim(); ++i)
                         expected *= arr.shape(i);
                     int64_t stride = 1;
                     for (size_t i = arr.ndim(); i-- > 0;) {
                         if (arr.shape(i) > 1 && arr.stride(i) != stride)
                             throw nb::type_error("expected a C-contiguous array");
                         stride *= (int64_t) arr.shape(i);
                     }
                     (void) expected;
                 }
                 OIDNBuffer b = oidnNewSharedBuffer(d.h, arr.data(), arr.nbytes());
                 return Buffer(self, b, nb::borrow(array));
             },
             "array"_a,
             "Wrap the memory of a C-contiguous array that the device can access. The "
             "array is kept alive by the buffer.")
        .def("new_filter",
             [](nb::handle_t<Device> self, const std::string &type) {
                 return Filter(self, type);
             },
             "type"_a = "RT", "Create a filter of the given type (\"RT\" or \"RTLightmap\").")
        .def("can_share",
             [](Device &d, nb::handle obj) {
                 bool writable;
                 AnyArray arr = to_array(obj, writable);
                 return d.share_error(describe_image(arr)).empty();
             },
             "array"_a,
             "Check whether Filter.set_image() can bind the given image array without a "
             "copy.")
        .def("__repr__", [](Device &d) {
            return std::string("Device(type=") + device_type_name(d.type()) + ")";
        });

    // Buffer

    nb::class_<Buffer>(m, "Buffer", "A memory buffer allocated by or shared with a device")
        .def_prop_ro("size", [](Buffer &b) { return b.size; }, "Size in bytes")
        .def_prop_ro("storage", &Buffer::storage)
        .def_prop_ro("device", [](Buffer &b) { return b.device_obj; })
        .def_prop_ro("data_ptr", [](Buffer &b) { return (uintptr_t) b.data; },
                     "Raw address of the buffer memory (0 if not accessible via pointer)")
        .def("view", &buffer_view, "dtype"_a = "float32", "shape"_a, "byte_offset"_a = 0,
             nb::rv_policy::reference_internal,
             "Create an array view of the buffer memory. The result supports DLPack and the "
             "buffer protocol and can be consumed by numpy.from_dlpack(), "
             "torch.from_dlpack(), and similar functions.")
        .def("read",
             [](Buffer &b, HostArrayRW dst, size_t byte_offset) {
                 nb::gil_scoped_release guard;
                 oidnReadBuffer(b.h, byte_offset, dst.nbytes(), dst.data());
                 check(b.device->h);
             },
             nb::arg("dst").noconvert(), "byte_offset"_a = 0,
             "Copy buffer contents into a C-contiguous host array.")
        .def("read_async",
             [](Buffer &b, HostArrayRW dst, size_t byte_offset) {
                 oidnReadBufferAsync(b.h, byte_offset, dst.nbytes(), dst.data());
                 check(b.device->h);
             },
             nb::arg("dst").noconvert(), "byte_offset"_a = 0,
             "Asynchronous variant of read(). Call Device.sync() before using the data.")
        .def("write",
             [](Buffer &b, HostArrayRO src, size_t byte_offset) {
                 nb::gil_scoped_release guard;
                 oidnWriteBuffer(b.h, byte_offset, src.nbytes(), src.data());
                 check(b.device->h);
             },
             "src"_a, "byte_offset"_a = 0, "Copy a host array into the buffer.")
        .def("write_async",
             [](Buffer &b, HostArrayRO src, size_t byte_offset) {
                 oidnWriteBufferAsync(b.h, byte_offset, src.nbytes(), src.data());
                 check(b.device->h);
             },
             nb::arg("src").noconvert(), "byte_offset"_a = 0,
             "Asynchronous variant of write(). Call Device.sync() before reusing 'src'.")
        .def("__repr__", [](Buffer &b) {
            return "Buffer(size=" + std::to_string(b.size) + ")";
        });

    // Filter

    auto filter = nb::class_<Filter>(m, "Filter", "A denoising filter");
    filter
        .def_prop_ro("device", [](Filter &f) { return f.device_obj; })
        .def_prop_ro("type", [](Filter &f) { return f.type; })
        .def("set_image",
             [](Filter &f, const std::string &name, nb::handle obj, nb::object format,
                nb::object width, nb::object height, size_t byte_offset,
                size_t pixel_stride, size_t row_stride) {
                 if (nb::isinstance<Buffer>(obj)) {
                     if (format.is_none() || width.is_none() || height.is_none())
                         throw nb::type_error("'format', 'width', and 'height' are required "
                                              "when passing a Buffer");
                     f.set_image_from_buffer(name, obj, nb::cast<Buffer &>(obj),
                                             nb::cast<OIDNFormat>(format),
                                             nb::cast<size_t>(width), nb::cast<size_t>(height),
                                             byte_offset, pixel_stride, row_stride);
                 } else {
                     if (!format.is_none() || !width.is_none() || !height.is_none() ||
                         byte_offset || pixel_stride || row_stride)
                         throw nb::type_error("layout arguments are only accepted together "
                                              "with a Buffer; arrays describe their own "
                                              "layout");
                     f.set_image_from_array(name, obj);
                 }
             },
             "name"_a, "image"_a, "format"_a = nb::none(), "width"_a = nb::none(),
             "height"_a = nb::none(), "byte_offset"_a = 0, "pixel_stride"_a = 0,
             "row_stride"_a = 0,
             "Bind an image parameter (\"color\", \"albedo\", \"normal\", \"output\"). "
             "Arrays of shape (H, W) or (H, W, C) with C <= 4 and dtype float32 or float16 "
             "are bound without a copy when the device can access their memory. Buffers "
             "require an explicit format and size; strides of zero mean tightly packed.")
        .def("unset_image", &Filter::unset_image, "name"_a)
        .def("set_data",
             [](Filter &f, const std::string &name, nb::handle obj) {
                 HostArrayRO arr = nb::cast<HostArrayRO>(obj);
                 oidnSetSharedFilterData(f.h, name.c_str(), (void *) arr.data(), arr.nbytes());
                 check(f.device->h);
                 f.data[name] = nb::borrow(obj);
                 f.dirty = true;
             },
             "name"_a, "data"_a,
             "Bind an opaque data parameter such as \"weights\". Accepts bytes or any "
             "contiguous host array, which is kept alive by the filter.")
        .def("update_data",
             [](Filter &f, const std::string &name) {
                 oidnUpdateFilterData(f.h, name.c_str());
                 check(f.device->h);
                 f.dirty = true;
             },
             "name"_a, "Notify the filter that the contents of a data parameter changed.")
        .def("unset_data",
             [](Filter &f, const std::string &name) {
                 oidnUnsetFilterData(f.h, name.c_str());
                 check(f.device->h);
                 f.data.erase(name);
                 f.dirty = true;
             },
             "name"_a)
        .def("get_bool", &Filter::get_bool, "name"_a)
        .def("get_int", &Filter::get_int, "name"_a)
        .def("get_float", &Filter::get_float, "name"_a)
        .def("set_bool", &Filter::set_bool, "name"_a, "value"_a)
        .def("set_int", &Filter::set_int, "name"_a, "value"_a)
        .def("set_float", &Filter::set_float, "name"_a, "value"_a)
        .def("set",
             [](Filter &f, const std::string &name, nb::handle value) {
                 if (nb::isinstance<nb::bool_>(value))
                     f.set_bool(name, nb::cast<bool>(value));
                 else if (nb::isinstance<OIDNQuality>(value))
                     f.set_int(name, (int) nb::cast<OIDNQuality>(value));
                 else if (nb::isinstance<nb::int_>(value))
                     f.set_int(name, nb::cast<int>(value));
                 else if (nb::isinstance<nb::float_>(value))
                     f.set_float(name, nb::cast<float>(value));
                 else
                     throw nb::type_error("expected a bool, int, float, or Quality value");
             },
             "name"_a, "value"_a,
             "Set a scalar parameter using the OIDN parameter name, dispatching on the "
             "Python type of 'value'.")
        .def_prop_rw("hdr", [](Filter &f) { return f.get_bool("hdr"); },
                     [](Filter &f, bool v) { f.set_bool("hdr", v); },
                     "The color image has high dynamic range")
        .def_prop_rw("srgb", [](Filter &f) { return f.get_bool("srgb"); },
                     [](Filter &f, bool v) { f.set_bool("srgb", v); },
                     "The color image is sRGB-encoded (LDR only)")
        .def_prop_rw("clean_aux", [](Filter &f) { return f.get_bool("cleanAux"); },
                     [](Filter &f, bool v) { f.set_bool("cleanAux", v); },
                     "The albedo and normal images are noise-free")
        .def_prop_rw("directional", [](Filter &f) { return f.get_bool("directional"); },
                     [](Filter &f, bool v) { f.set_bool("directional", v); },
                     "RTLightmap: the input holds normalized directional coefficients")
        .def_prop_rw("input_scale",
                     [](Filter &f) -> nb::object {
                         float v = f.get_float("inputScale");
                         if (std::isnan(v))
                             return nb::none();
                         return nb::float_(v);
                     },
                     [](Filter &f, nb::object v) {
                         f.set_float("inputScale", v.is_none() ? NAN : nb::cast<float>(v));
                     },
                     nb::for_setter(nb::arg("value").none()),
                     "Scale applied to the input before filtering (None = automatic)")
        .def_prop_rw("quality", [](Filter &f) { return (OIDNQuality) f.get_int("quality"); },
                     [](Filter &f, OIDNQuality v) { f.set_int("quality", (int) v); })
        .def_prop_rw("max_memory_mb", [](Filter &f) { return f.get_int("maxMemoryMB"); },
                     [](Filter &f, int v) { f.set_int("maxMemoryMB", v); },
                     "Requested memory limit in megabytes (-1 = device-dependent default)")
        .def_prop_ro("tile_alignment", [](Filter &f) { return f.get_int("tileAlignment"); })
        .def_prop_ro("tile_overlap", [](Filter &f) { return f.get_int("tileOverlap"); })
        .def("set_progress_monitor", &Filter::set_progress_monitor, "func"_a.none(),
             "Register a callable receiving the progress in [0, 1]. Returning False "
             "requests cancellation. Pass None to remove the monitor.")
        .def("commit", &Filter::commit,
             "Commit parameter changes. Called implicitly by execute() when needed.")
        .def("execute", [](Filter &f) { f.execute(false); },
             "Run the filter and wait for completion.")
        .def("execute_async", [](Filter &f) { f.execute(true); },
             "Run the filter asynchronously. Call Device.sync() before reading the output.")
        .def("__repr__", [](Filter &f) { return "Filter(type='" + f.type + "')"; });
}
