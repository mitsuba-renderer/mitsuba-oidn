cmake_minimum_required(VERSION 3.21...3.31)

project(mitsuba_oidn LANGUAGES C CXX)

if (NOT SKBUILD)
  message(WARNING "This CMake file is meant to be executed through scikit-build-core. "
    "To build the package, run 'pip install .' or, for development, "
    "'pip install --no-build-isolation -ve .'")
endif()

find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

# ------------------------------------------------------------------------------
# Open Image Denoise (vendored, patched to use an external thread pool)
# ------------------------------------------------------------------------------

set(OIDN_APPS OFF CACHE BOOL "" FORCE)
set(OIDN_LIBRARY_NAME "mitsuba_oidn" CACHE STRING "" FORCE)
set(OIDN_API_NAMESPACE "mitsuba_oidn" CACHE STRING "" FORCE)
set(OIDN_LIBRARY_VERSIONED OFF CACHE BOOL "" FORCE)
set(OIDN_INSTALL_DEPENDENCIES OFF CACHE BOOL "" FORCE)
set(OIDN_DEVICE_CPU ON CACHE BOOL "" FORCE)

# ISPC binary unpacked into ext/ispc (see README)
if (NOT ISPC_EXECUTABLE)
  set(_ispc "${CMAKE_CURRENT_SOURCE_DIR}/ext/ispc/bin/ispc")
  if (WIN32)
    set(_ispc "${_ispc}.exe")
  endif()
  if (EXISTS "${_ispc}")
    set(ISPC_EXECUTABLE "${_ispc}" CACHE FILEPATH "Path to the ISPC executable." FORCE)
  endif()
endif()

if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  option(MITSUBA_OIDN_METAL "Build the Metal device" ON)
else()
  set(MITSUBA_OIDN_METAL OFF)
endif()

if (NOT APPLE)
  option(MITSUBA_OIDN_CUDA "Build the CUDA device (requires the CUDA toolkit)" ON)
else()
  set(MITSUBA_OIDN_CUDA OFF)
endif()

set(OIDN_DEVICE_METAL ${MITSUBA_OIDN_METAL} CACHE BOOL "" FORCE)

# All shared libraries live side by side in the package directory
if (APPLE)
  set(_rpath "@loader_path")
else()
  set(_rpath "$ORIGIN")
endif()
set(OIDN_INSTALL_RPATH "${_rpath}" CACHE STRING "" FORCE)

if (MITSUBA_OIDN_CUDA)
  find_package(CUDAToolkit 12.8 QUIET)
  if (CUDAToolkit_FOUND)
    message(STATUS "mitsuba-oidn: building the CUDA device (CUDA ${CUDAToolkit_VERSION})")
    set(OIDN_DEVICE_CUDA ON CACHE BOOL "" FORCE)
  else()
    message(STATUS "mitsuba-oidn: CUDA toolkit not found, skipping the CUDA device")
    set(OIDN_DEVICE_CUDA OFF CACHE BOOL "" FORCE)
  endif()
else()
  set(OIDN_DEVICE_CUDA OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(ext/oidn)

set(MITSUBA_OIDN_LIBS OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_device_cpu)
if (OIDN_DEVICE_METAL)
  list(APPEND MITSUBA_OIDN_LIBS OpenImageDenoise_device_metal)
endif()

# ------------------------------------------------------------------------------
# Python extension
# ------------------------------------------------------------------------------

nanobind_add_module(_mitsuba_oidn_ext
  NB_DOMAIN mitsuba_oidn
  BACKEND_MODULE nanobind_backend
  src/ext.cpp
)

target_link_libraries(_mitsuba_oidn_ext PRIVATE OpenImageDenoise)
target_compile_features(_mitsuba_oidn_ext PRIVATE cxx_std_17)

set_target_properties(_mitsuba_oidn_ext PROPERTIES INSTALL_RPATH "${_rpath}")

install(TARGETS _mitsuba_oidn_ext ${MITSUBA_OIDN_LIBS}
  COMPONENT python
  LIBRARY DESTINATION mitsuba_oidn
  RUNTIME DESTINATION mitsuba_oidn
  ARCHIVE DESTINATION mitsuba_oidn/lib
)

# Type stubs (the build-tree extension resolves the OIDN libraries via its build rpath)
if (NOT WIN32)
  nanobind_add_stub(mitsuba_oidn_stub
    MODULE _mitsuba_oidn_ext
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_mitsuba_oidn_ext.pyi"
    MARKER_FILE "${CMAKE_CURRENT_BINARY_DIR}/py.typed"
    PYTHON_PATH $<TARGET_FILE_DIR:_mitsuba_oidn_ext>
    DEPENDS _mitsuba_oidn_ext
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/_mitsuba_oidn_ext.pyi"
    "${CMAKE_CURRENT_BINARY_DIR}/py.typed"
    DESTINATION mitsuba_oidn
    COMPONENT python
  )
endif()

if (OIDN_DEVICE_CUDA)
  # OIDN builds the CUDA device as an external project that pre-installs its module
  # into a staging directory at build time
  if (WIN32)
    set(_cuda_dir "${CMAKE_INSTALL_BINDIR}")
  else()
    set(_cuda_dir "${CMAKE_INSTALL_LIBDIR}")
  endif()
  install(DIRECTORY "${OIDN_ROOT_BINARY_DIR}/devices/cuda/preinstall/${_cuda_dir}/"
    DESTINATION mitsuba_oidn
    COMPONENT python
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING PATTERN "*mitsuba_oidn_device_cuda*"
  )
endif()
