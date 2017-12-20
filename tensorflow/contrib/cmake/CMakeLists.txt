# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(tensorflow C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# Actual source is the ../../.. directory
get_filename_component(tf_contrib_source_dir ${tensorflow_SOURCE_DIR} PATH)
get_filename_component(tf_tf_source_dir ${tf_contrib_source_dir} PATH)
get_filename_component(tensorflow_source_dir ${tf_tf_source_dir} PATH)

# [CLEANUP] Not sure if this is needed (copied from Protobuf)
# CMake policies
cmake_policy(SET CMP0022 NEW)

# Options
option(tensorflow_VERBOSE "Enable for verbose output" OFF)
option(tensorflow_ENABLE_SSL_SUPPORT "Enable boringssl support" OFF)
option(tensorflow_ENABLE_GRPC_SUPPORT "Enable gRPC support" ON)
option(tensorflow_ENABLE_HDFS_SUPPORT "Enable HDFS support" OFF)
option(tensorflow_ENABLE_JEMALLOC_SUPPORT "Enable jemalloc support" OFF)
option(tensorflow_BUILD_CC_EXAMPLE "Build the C++ tutorial example" ON)
option(tensorflow_BUILD_PYTHON_BINDINGS "Build the Python bindings" ON)
option(tensorflow_BUILD_ALL_KERNELS "Build all OpKernels" ON)
option(tensorflow_BUILD_CONTRIB_KERNELS "Build OpKernels from tensorflow/contrib/..." ON)
option(tensorflow_BUILD_CC_TESTS "Build cc unit tests " OFF)
option(tensorflow_BUILD_PYTHON_TESTS "Build python unit tests " OFF)
option(tensorflow_BUILD_MORE_PYTHON_TESTS "Build more python unit tests for contrib packages" OFF)
option(tensorflow_BUILD_SHARED_LIB "Build TensorFlow as a shared library" OFF)
option(tensorflow_OPTIMIZE_FOR_NATIVE_ARCH "Enable compiler optimizations for the native processor architecture (if available)" ON)
option(tensorflow_WIN_CPU_SIMD_OPTIONS "Enables CPU SIMD instructions")
option(tensorflow_ENABLE_SNAPPY_SUPPORT "Enable SNAPPY compression support" ON)
option(tensorflow_DISABLE_EIGEN_FORCEINLINE "Disable forceinline, to speed up build on windows." OFF)

# GPU, CUDA and cuDNN options
option(tensorflow_ENABLE_GPU "Enable GPU support" OFF)
set(tensorflow_CUDA_VERSION "9.0" CACHE STRING "CUDA version to build against")
set(tensorflow_CUDNN_VERSION "7" CACHE STRING "cuDNN version to build against")

if(HAIKU)
	option(tensorflow_ENABLE_POSITION_INDEPENDENT_CODE "Enable PIE support" OFF)
else()
	option(tensorflow_ENABLE_POSITION_INDEPENDENT_CODE "Enable PIE support" ON)
endif()


if (NOT WIN32)
  # Threads: defines CMAKE_THREAD_LIBS_INIT and adds -pthread compile option
  # for targets that link ${CMAKE_THREAD_LIBS_INIT}.
  find_package (Threads)

  option(tensorflow_PATH_STATIC_LIB "Additional library search path for libcudnn_static.a, libnccl_static.a, libculibos.a" /usr/local/cuda/lib64/)
  option(tensorflow_CUDNN_INCLUDE "cudnn.h header install path" /usr/include/)
  if (NOT tensorflow_CUDNN_INCLUDE)
    # option's default value is OFF. Fill it with real default values
    set(tensorflow_CUDNN_INCLUDE /usr/include)
  endif (NOT tensorflow_CUDNN_INCLUDE)
  option(tensorflow_PATH_CUDNN_STATIC_LIB "Override PATH_STATIC_LIB for libcudnn_static.a" ${tensorflow_PATH_STATIC_LIB})
  if (NOT tensorflow_PATH_CUDNN_STATIC_LIB)
    # option's default value is OFF. Fill it with real default values
    set (tensorflow_PATH_CUDNN_STATIC_LIB ${tensorflow_PATH_STATIC_LIB})
  endif (NOT tensorflow_PATH_CUDNN_STATIC_LIB)
  option(tensorflow_PATH_NCCL_STATIC_LIB "Override PATH_STATIC_LIB for libnccl_static.a" ${tensorflow_PATH_STATIC_LIB})
  if (NOT tensorflow_PATH_NCCL_STATIC_LIB)
    # option's default value is OFF. Fill it with real default values
    set (tensorflow_PATH_NCCL_STATIC_LIB ${tensorflow_PATH_STATIC_LIB})
  endif (NOT tensorflow_PATH_NCCL_STATIC_LIB)
  option(tensorflow_CUDA_LIBRARY_PATH "Designate the default CUDA library paths" /usr/local/cuda/lib64)
  if (NOT tensorflow_CUDA_LIBRARY_PATH)
    # option's default value is OFF. Fill it with real default values
    set(tensorflow_CUDA_LIBRARY_PATH /usr/local/cuda/lib64)
  endif (NOT tensorflow_CUDA_LIBRARY_PATH)
endif()

if (WIN32)
  set(BOOL_WIN32 ON)
else (WIN32)
  set(BOOL_WIN32 OFF)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif (WIN32)

# [CLEANUP] Remove when done
# For debugging
function(SHOW_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endfunction()

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

# Location where external projects will be downloaded
set (DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
     CACHE PATH "Location where external projects will be downloaded.")
mark_as_advanced(DOWNLOAD_LOCATION)

if (tensorflow_ENABLE_POSITION_INDEPENDENT_CODE)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
	set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
endif()

if (tensorflow_DISABLE_EIGEN_FORCEINLINE)
  add_definitions(-DEIGEN_STRONG_INLINE=inline)
endif()

add_definitions(-DEIGEN_AVOID_STL_ARRAY)
if(WIN32)
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC)
  add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
  add_definitions(-DTENSORFLOW_USE_EIGEN_THREADPOOL -DEIGEN_HAS_C99_MATH)
  add_definitions(-DTF_COMPILE_LIBRARY)
  add_definitions(/bigobj /nologo /EHsc /GF /MP /Gm-)
  # Suppress warnings to reduce build log size.
  add_definitions(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_definitions(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_definitions(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  add_definitions(/wd4003 /wd4244 /wd4267 /wd4503 /wd4506 /wd4800 /wd4996)
  # Suppress linker warnings.
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Ob2")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /D_ITERATOR_DEBUG_LEVEL=0")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -std=c++11")
endif()

if (tensorflow_OPTIMIZE_FOR_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
endif()

# MSVC SIMD instructions
if (tensorflow_WIN_CPU_SIMD_OPTIONS)
  if (WIN32)
    CHECK_CXX_COMPILER_FLAG("${tensorflow_WIN_CPU_SIMD_OPTIONS}" COMPILER_OPT_WIN_CPU_SIMD_SUPPORTED)
    if(COMPILER_OPT_WIN_CPU_SIMD_SUPPORTED)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${tensorflow_WIN_CPU_SIMD_OPTIONS}")
    else()
      message(FATAL_ERROR "${tensorflow_WIN_CPU_SIMD_OPTIONS} not supported")
    endif()
  endif()
endif()

if (tensorflow_ENABLE_JEMALLOC_SUPPORT)
  add_definitions(-DTENSORFLOW_USE_JEMALLOC -DJEMALLOC_EXPORT=)
endif()

# External dependencies
include(zlib)
include(gif)
include(png)
include(jpeg)
include(lmdb)
include(eigen)
include(gemmlowp)
include(jsoncpp)
include(farmhash)
include(fft2d)
include(highwayhash)
include(nsync)
include(protobuf)
include(re2)
include(cub)
include(sqlite)
if (tensorflow_BUILD_CC_TESTS)
  include(googletest)
endif()

set(tensorflow_EXTERNAL_LIBRARIES
    ${zlib_STATIC_LIBRARIES}
    ${gif_STATIC_LIBRARIES}
    ${png_STATIC_LIBRARIES}
    ${jpeg_STATIC_LIBRARIES}
    ${lmdb_STATIC_LIBRARIES}
    ${jsoncpp_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${fft2d_STATIC_LIBRARIES}
    ${highwayhash_STATIC_LIBRARIES}
    ${nsync_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
    ${re2_STATIC_LIBRARIES}
    ${sqlite_STATIC_LIBRARIES}
)
set(tensorflow_EXTERNAL_DEPENDENCIES
    zlib_copy_headers_to_destination
    gif_copy_headers_to_destination
    png_copy_headers_to_destination
    jpeg_copy_headers_to_destination
    lmdb_copy_headers_to_destination
    jsoncpp
    farmhash_copy_headers_to_destination
    highwayhash_copy_headers_to_destination
    nsync_copy_headers_to_destination
    protobuf
    eigen
    gemmlowp
    cub
    fft2d
    re2
    sqlite_copy_headers_to_destination
)

include_directories(
    # Source and generated code.
    ${tensorflow_source_dir}
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${zlib_INCLUDE_DIR}
    ${gif_INCLUDE_DIR}
    ${png_INCLUDE_DIR}
    ${jpeg_INCLUDE_DIR}
    ${lmdb_INCLUDE_DIR}
    ${eigen_INCLUDE_DIRS}
    ${gemmlowp_INCLUDE_DIR}
    ${jsoncpp_INCLUDE_DIR}
    ${farmhash_INCLUDE_DIR}
    ${highwayhash_INCLUDE_DIR}
    ${cub_INCLUDE_DIR}
    ${nsync_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    ${re2_INCLUDE_DIR}
    ${sqlite_INCLUDE_DIR}
)

if(tensorflow_ENABLE_SSL_SUPPORT)
  include(boringssl)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${boringssl_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES boringssl)
  include_directories(${boringssl_INCLUDE_DIR})
endif()
if(tensorflow_ENABLE_GRPC_SUPPORT)
  include(grpc)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${grpc_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES grpc)
  include_directories(${GRPC_INCLUDE_DIRS})
endif()
if(tensorflow_ENABLE_JEMALLOC_SUPPORT)
  include(jemalloc)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${jemalloc_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES jemalloc)
  include_directories(${jemalloc_INCLUDE_DIRS})
endif()
if(tensorflow_ENABLE_SNAPPY_SUPPORT)
  include(snappy)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${snappy_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES snappy)
  include_directories(${snappy_INCLUDE_DIR})
endif()
if(WIN32)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES wsock32 ws2_32 shlwapi)
endif()
if(UNIX)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()
if(HAIKU)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES network)
endif()

if (tensorflow_ENABLE_GPU)
  if (NOT WIN32)
    # Default install paths for cuda libraries in Linux
    # In some Linux distros, find_package(CUDA) seems to require CMAKE_LIBRARY_PATH to include cuda-lib paths
    list(APPEND CMAKE_LIBRARY_PATH "${tensorflow_CUDA_LIBRARY_PATH}")
    list(APPEND CMAKE_LIBRARY_PATH "${tensorflow_CUDA_LIBRARY_PATH}/stubs")
  endif (NOT WIN32)

  find_package(CUDA ${tensorflow_CUDA_VERSION} REQUIRED)

  # by default we assume compute cabability 3.5 and 5.2. If you change this change it in
  # CUDA_NVCC_FLAGS and cuda_config.h below
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=\"sm_30,compute_30\";-gencode arch=compute_35,code=\"sm_35,compute_35\";-gencode arch=compute_52,code=\"sm_52,compute_52\")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--include-path ${PROJECT_BINARY_DIR}/$\{build_configuration\};--expt-relaxed-constexpr)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero
  set(CUDA_INCLUDE ${CUDA_TOOLKIT_TARGET_DIR} ${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/include)
  include_directories(${CUDA_INCLUDE})
  if (WIN32)
    add_definitions(-DGOOGLE_CUDA=1 -DTF_EXTRA_CUDA_CAPABILITIES=3.0,3.5,5.2)
  else (WIN32)
    # Without these double quotes, cmake in Linux makes it "-DTF_EXTRA_CUDA_CAPABILITIES=3.0, -D3.5, -D5.2" for cc, which incurs build breaks
    add_definitions(-DGOOGLE_CUDA=1 -D"TF_EXTRA_CUDA_CAPABILITIES=3.0,3.5,5.2")
  endif (WIN32)

  if (WIN32)
    # add cudnn
    if(NOT CUDNN_HOME)
      set(CUDNN_HOME ${CUDA_TOOLKIT_TARGET_DIR})
    endif(NOT CUDNN_HOME)
    include_directories(${CUDNN_HOME})
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_curand_LIBRARY} ${CUDA_cupti_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDNN_HOME}/lib/x64/cudnn.lib)
  else (WIN32)
    set(CUDNN_INCLUDE "${tensorflow_CUDNN_INCLUDE}")

    find_library(nccl_STATIC_LIBRARY NAMES libnccl_static.a PATHS ${tensorflow_PATH_NCCL_STATIC_LIB} ${CUDA_TOOLKIT_ROOT_DIR})
    if (NOT nccl_STATIC_LIBRARY)
      message(FATAL_ERROR "NCCL is required for GPU-build")
    else (NOT nccl_STATIC_LIBRARY)
      message("nccl-static: ${nccl_STATIC_LIBRARY}")
      # something like /usr/lib64/libnccl_static.a
    endif (NOT nccl_STATIC_LIBRARY)

    find_library(cudnn_STATIC_LIBRARY NAMES libcudnn_static.a PATHS ${tensorflow_PATH_CUDNN_STATIC_LIB} ${CUDA_TOOLKIT_ROOT_DIR})
    if (NOT cudnn_STATIC_LIBRARY)
      message(FATAL_ERROR "CUDNN is required for GPU-build")
    else (NOT cudnn_STATIC_LIBRARY)
      message("cudnn-static: ${cudnn_STATIC_LIBRARY}")
    endif (NOT cudnn_STATIC_LIBRARY)

    find_library(culibos_STATIC_LIBRARY NAMES libculibos.a PATHS ${tensorflow_PATH_STATIC_LIB} ${CUDA_TOOLKIT_ROOT_DIR})
    if (NOT culibos_STATIC_LIBRARY)
      message(FATAL_ERROR "CULIBOS is required for GPU-build")
    else (NOT culibos_STATIC_LIBRARY)
      message("culibos-static: ${culibos_STATIC_LIBRARY}")
    endif (NOT culibos_STATIC_LIBRARY)

    include_directories(${CUDNN_INCLUDE})
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_curand_LIBRARY} ${CUDA_cupti_LIBRARY} ${CUDA_cusolver_LIBRARY} ${cudnn_STATIC_LIBRARY} ${culibos_STATIC_LIBRARY} ${nccl_STATIC_LIBRARY})
  endif (WIN32)

  # Remove "." from CUDA version variable.
  string(REPLACE "." "" short_CUDA_VER ${tensorflow_CUDA_VERSION})

  # create cuda_config.h
  FILE(WRITE ${tensorflow_source_dir}/third_party/gpus/cuda/cuda_config.h
    "#ifndef CUDA_CUDA_CONFIG_H_\n"
    "#define CUDA_CUDA_CONFIG_H_\n"
    "#define TF_CUDA_CAPABILITIES CudaVersion(\"3.0\"),CudaVersion(\"3.5\"),CudaVersion(\"5.2\")\n"
    "#define TF_CUDA_VERSION \"64_${short_CUDA_VER}\"\n"
    "#define TF_CUDNN_VERSION \"64_${tensorflow_CUDNN_VERSION}\"\n"
    "#define TF_CUDA_TOOLKIT_PATH \"${CUDA_TOOLKIT_ROOT_DIR}\"\n"
    "#endif  // CUDA_CUDA_CONFIG_H_\n"
  )

  if (WIN32)
    # tf assumes in various places header files to be in cuda/include. On windows the cuda sdk
    # installs them under cuda/version/include and to avoid that we need to change tf we copy a
    # few files to cuda/include
    FILE(COPY
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda.h ${CUDA_TOOLKIT_TARGET_DIR}/include/cuComplex.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cublas_v2.h ${CUDNN_HOME}/include/cudnn.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cufft.h ${CUDA_TOOLKIT_TARGET_DIR}/include/curand.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda_runtime_api.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cusolverDn.h
      DESTINATION ${tensorflow_source_dir}/third_party/gpus/cuda/include
    )
  else(WIN32)
    # Linux has slightly differnt install paths than Windows
    FILE(COPY
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda.h ${CUDA_TOOLKIT_TARGET_DIR}/include/cuComplex.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cublas_v2.h ${CUDNN_INCLUDE}/cudnn.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cufft.h ${CUDA_TOOLKIT_TARGET_DIR}/include/curand.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda_runtime_api.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cusolverDn.h
      DESTINATION ${tensorflow_source_dir}/third_party/gpus/cuda/include
    )
  endif(WIN32)

  include_directories(${tensorflow_source_dir}/third_party/gpus)
  # add cuda libraries to tensorflow_EXTERNAL_LIBRARIES
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES})

  # NOTE(mrry): Update these flags when the version of CUDA or cuDNN used
  # in the default build is upgraded.
  if(WIN32)
    set(tensorflow_BUILD_INFO_FLAGS --build_config cuda --key_value
      msvcp_dll_name=msvcp140.dll
      cudart_dll_name=cudart64_${short_CUDA_VER}.dll
      cuda_version_number=${tensorflow_CUDA_VERSION}
      nvcuda_dll_name=nvcuda.dll
      cudnn_dll_name=cudnn64_${tensorflow_CUDNN_VERSION}.dll
      cudnn_version_number=${tensorflow_CUDNN_VERSION})
  else(WIN32)
    set(tensorflow_BUILD_INFO_FLAGS --build_config cuda --key_value
	    cuda_version_number=${tensorflow_CUDA_VERSION}
	    cudnn_version_number=${tensorflow_CUDNN_VERSION})
  endif(WIN32)
else(tensorflow_ENABLE_GPU)
  set(tensorflow_BUILD_INFO_FLAGS --build_config cpu --key_value
    msvcp_dll_name=msvcp140.dll)
endif(tensorflow_ENABLE_GPU)

# Find python executable
include(FindPythonInterp)
if(NOT ${PYTHONINTERP_FOUND})
    message(FATAL_ERROR "CMake was unable to find a python interpreter.")
endif()

# Let's get to work!
include(tf_core_framework.cmake)
include(tf_stream_executor.cmake)

include(tf_core_cpu.cmake)
include(tf_core_ops.cmake)
include(tf_core_direct_session.cmake)
include(tf_core_kernels.cmake)
if(tensorflow_ENABLE_GRPC_SUPPORT)
  include(tf_core_distributed_runtime.cmake)
endif()
# We include tf_cc_ops first, because tf_c depends on tf_cc.
include(tf_cc_ops.cmake)
include(tf_c.cmake)
include(tf_grappler.cmake)
include(tf_core_profiler.cmake)
if(tensorflow_BUILD_CC_EXAMPLE)
  include(tf_tutorials.cmake)
  include(tf_label_image_example.cmake)
endif()
include(tf_tools.cmake)
if(tensorflow_BUILD_PYTHON_BINDINGS)
  include(tf_python.cmake)
endif()
if(tensorflow_BUILD_SHARED_LIB)
  include(tf_shared_lib.cmake)
endif()
if(tensorflow_BUILD_CC_TESTS OR tensorflow_BUILD_PYTHON_TESTS)
  include(tf_tests.cmake)
endif()
