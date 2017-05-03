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
option(tensorflow_ENABLE_GPU "Enable GPU support" OFF)
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
option(tensorflow_BUILD_SHARED_LIB "Build TensorFlow as a shared library" OFF)
option(tensorflow_OPTIMIZE_FOR_NATIVE_ARCH "Enable compiler optimizations for the native processor architecture (if available)" ON)
option(tensorflow_WIN_CPU_SIMD_OPTIONS "Enables CPU SIMD instructions")
option(tensorflow_USE_SYSTEM_ZLIB "Use system library libz" OFF)
option(tensorflow_USE_SYSTEM_PNG "Use system library libpng" ON)
option(tensorflow_USE_SYSTEM_JPEG "Use system library libjpeg" ON)
option(tensorflow_USE_SYSTEM_GIF "Use system library libgif" ON)
option(tensorflow_USE_SYSTEM_JSONCPP "Use system library libjsoncpp" ON)
option(tensorflow_USE_SYSTEM_EIGEN "Use system library eigen3" OFF)
option(tensorflow_USE_SYSTEM_GEMMLOWP "Use system library gemmlowp" OFF)
option(tensorflow_USE_SYSTEM_FARMHASH "Use system library farmhash" OFF)
option(tensorflow_USE_SYSTEM_HIGHWAYHASH "Use system library highwayhash" OFF)
option(tensorflow_USE_SYSTEM_FFT2D "Use system library fft2d" OFF)
option(tensorflow_USE_SYSTEM_PROTOBUF "Use system library protobuf" OFF)



if (NOT WIN32)
  # Threads: defines CMAKE_THREAD_LIBS_INIT and adds -pthread compile option
  # for targets that link ${CMAKE_THREAD_LIBS_INIT}.
  find_package (Threads)
endif()

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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DEIGEN_AVOID_STL_ARRAY)
if(WIN32)
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC)
  add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
  add_definitions(-DTENSORFLOW_USE_EIGEN_THREADPOOL -DEIGEN_HAS_C99_MATH -D_ITERATOR_DEBUG_LEVEL=0)
  add_definitions(-DTF_COMPILE_LIBRARY)
  add_definitions(-DNDEBUG /O2)  # Equivalent of -c opt in Bazel.
  add_definitions(/bigobj /nologo /EHsc /GF /FC /MP /Gm-)
  # Suppress warnings to reduce build log size.
  add_definitions(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_definitions(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_definitions(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
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


if(tensorflow_USE_SYSTEM_ZLIB)

  message(STATUS "use system zlib")

  find_package(ZLIB REQUIRED)
  include_directories(${ZLIB_INCLUDE_DIR})
  link_libraries (${ZLIB_LIBRARY})

else(tensorflow_USE_SYSTEM_ZLIB)

  message(STATUS "do not use system zlib")

  include(zlib)
  set(tensorflow_EXTERNAL_LIBRARIES ${zlib_STATIC_LIBRARIES})
  set(tensorflow_EXTERNAL_DEPENDENCIES zlib_copy_headers_to_destination)
  include_directories(${zlib_INCLUDE_DIR})

endif(tensorflow_USE_SYSTEM_ZLIB)

if(tensorflow_USE_SYSTEM_JPEG)

  message(STATUS "use system libjpeg")

  find_package(JPEG REQUIRED)
  include_directories(${JPEG_INCLUDE_DIR})
  link_libraries (${JPEG_LIBRARY})

else(tensorflow_USE_SYSTEM_JPEG)

  message(STATUS "do not use system libjpeg")

  include(jpeg)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${jpeg_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES jpeg_copy_headers_to_destination)

  include_directories(${jpeg_INCLUDE_DIR})

endif(tensorflow_USE_SYSTEM_JPEG)

if(tensorflow_USE_SYSTEM_PNG)

  message(STATUS "use system libpng")

  find_package(PNG REQUIRED)
  include_directories(${PNG_INCLUDE_DIR})
  link_libraries (${PNG_LIBRARY})

else(tensorflow_USE_SYSTEM_PNG)

  message(STATUS "do not use system libpng")

  include(png)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${PNG_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES png_copy_headers_to_destination)

  include_directories(
    ${png_INCLUDE_DIR}
    )

endif(tensorflow_USE_SYSTEM_PNG)

if(tensorflow_USE_SYSTEM_GIF)

  message(STATUS "use system libgif")

  find_package(GIF REQUIRED)
  include_directories(${GIF_INCLUDE_DIR})
  link_libraries (${GIF_LIBRARY})

else(tensorflow_USE_SYSTEM_GIF)

  message(STATUS "do not use system libgif")

  include(gif)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${gif_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES gif_copy_headers_to_destination)

  include_directories(
    ${gif_INCLUDE_DIR}
    )

endif(tensorflow_USE_SYSTEM_GIF)


if(tensorflow_USE_SYSTEM_JSONCPP)

  message(STATUS "use system libjsoncpp")

  find_package(jsoncpp REQUIRED)
  include_directories(${JSONCPP_INCLUDE_DIR})
  link_libraries (${JSONCPP_LIBRARY})

else(tensorflow_USE_SYSTEM_JSONCPP)

  message(STATUS "do not use system libjsoncpp")

  include(jsoncpp)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${jsoncpp_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES jsoncpp)

  include_directories(
    ${jsoncpp_INCLUDE_DIR}
    )

endif(tensorflow_USE_SYSTEM_JSONCPP)

if(tensorflow_USE_SYSTEM_EIGEN)

  message(STATUS "use system eigen3")
  message(FATAL_ERROR "finding eigen3 is not yet implemented")


else(tensorflow_USE_SYSTEM_EIGEN)

  message(STATUS "do not use system eigen3")

  include(eigen)
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES eigen)

  include_directories(
    ${eigen_INCLUDE_DIRS}
    )

endif(tensorflow_USE_SYSTEM_EIGEN)

if(tensorflow_USE_SYSTEM_GEMMLOWP)

  message(STATUS "use system libgemmlowp")
  message(FATAL_ERROR "finding gemmlowp is not yet implemented")


else(tensorflow_USE_SYSTEM_GEMMLOWP)

  message(STATUS "do not use system libgemmlowp")

  include(gemmlowp)
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES gemmlowp)

  include_directories(${gemmlowp_INCLUDE_DIR})

endif(tensorflow_USE_SYSTEM_GEMMLOWP)


if(tensorflow_USE_SYSTEM_FARMHASH)

  message(STATUS "use system farmhash")
  message(FATAL_ERROR "finding farmhash is not yet implemented")

else(tensorflow_USE_SYSTEM_FARMHASH)

  message(STATUS "do not use system libfarmhash")

  include(farmhash)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${farmhash_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES farmhash_copy_headers_to_destination)

  include_directories( ${farmhash_INCLUDE_DIR}   )

endif(tensorflow_USE_SYSTEM_FARMHASH)

if(tensorflow_USE_SYSTEM_FFT2D)

  message(STATUS "use system fft2d")
  message(FATAL_ERROR "finding fft2d is not yet implemented")

else(tensorflow_USE_SYSTEM_FFT2D)

  message(STATUS "do not use system fft2d")

  include(fft2d)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${fft2d_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES fft2d)


endif(tensorflow_USE_SYSTEM_FFT2D)



if(tensorflow_USE_SYSTEM_HIGHWAYHASH)

  message(STATUS "use system highwayhash")
  message(FATAL_ERROR "finding highwayhash is not yet implemented")

else(tensorflow_USE_SYSTEM_HIGHWAYHASH)

  message(STATUS "do not use system highwayhash")

  include(highwayhash)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${highwayhash_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES highwayhash_copy_headers_to_destination)

  include_directories(${highwayhash_INCLUDE_DIR})

endif(tensorflow_USE_SYSTEM_HIGHWAYHASH)


if(tensorflow_USE_SYSTEM_PROTOBUF)

  message(STATUS "use system protobuf")

  find_package(protobuf REQUIRED)
  include_directories(${PROTOBUF_INCLUDE_DIR})
  link_libraries (${PROTOBUF_LIBRARY})

else(tensorflow_USE_SYSTEM_PROTOBUF)

  message(STATUS "do not use system protobuf")

  include(protobuf)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${protobuf_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES protobuf)

  include_directories(${PROTOBUF_INCLUDE_DIRS})

endif(tensorflow_USE_SYSTEM_PROTOBUF)



# this stuff will be needed no matter what
if (tensorflow_BUILD_CC_TESTS)
  include(googletest)
endif()

include_directories(
  #Source and generated code.
  ${tensorflow_source_dir}
  ${CMAKE_CURRENT_BINARY_DIR}
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
if(WIN32)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES wsock32 ws2_32 shlwapi)
endif()
if(UNIX)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()

if (tensorflow_ENABLE_GPU)
  if (WIN32)
    find_package(CUDA 8.0 REQUIRED)

    # by default we assume compute cabability 3.5 and 5.2. If you change this change it in
    # CUDA_NVCC_FLAGS and cuda_config.h below
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=\"sm_30,compute_30\";-gencode arch=compute_35,code=\"sm_35,compute_35\";-gencode arch=compute_52,code=\"sm_52,compute_52\")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--include-path ${PROJECT_BINARY_DIR}/$\{build_configuration\};--expt-relaxed-constexpr)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero
    set(CUDA_INCLUDE ${CUDA_TOOLKIT_TARGET_DIR} ${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/include)
    include_directories(${CUDA_INCLUDE})
    add_definitions(-DGOOGLE_CUDA=1 -DTF_EXTRA_CUDA_CAPABILITIES=3.0,3.5,5.2)

    # add cudnn
    include_directories(${CUDNN_HOME})
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_curand_LIBRARY} ${CUDA_cupti_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDNN_HOME}/lib/x64/cudnn.lib)

    # create cuda_config.h
    FILE(WRITE ${tensorflow_source_dir}/third_party/gpus/cuda/cuda_config.h
      "#ifndef CUDA_CUDA_CONFIG_H_\n"
      "#define CUDA_CUDA_CONFIG_H_\n"
      "#define TF_CUDA_CAPABILITIES CudaVersion(\"3.0\"),CudaVersion(\"3.5\"),CudaVersion(\"5.2\")\n"
      "#define TF_CUDA_VERSION \"64_80\"\n"
      "#define TF_CUDNN_VERSION \"64_5\"\n"
      "#define TF_CUDA_TOOLKIT_PATH \"${CUDA_TOOLKIT_ROOT_DIR}\"\n"
      "#endif  // CUDA_CUDA_CONFIG_H_\n"
    )

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
    include_directories(${tensorflow_source_dir}/third_party/gpus)
    # add cuda libraries to tensorflow_EXTERNAL_LIBRARIES
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES})
  endif()
endif()

# Let's get to work!
include(tf_core_framework.cmake)
# NOTE: Disabled until issue #3996 is fixed.
# include(tf_stream_executor.cmake)
if (tensorflow_ENABLE_GPU)
  if (WIN32)
    include(tf_stream_executor.cmake)
  endif()
endif()

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
if(tensorflow_BUILD_CC_EXAMPLE)
  include(tf_tutorials.cmake)
  include(tf_label_image_example.cmake)
endif()
include(tf_tools.cmake)
if(tensorflow_BUILD_PYTHON_BINDINGS)
  include(tensorboard)
  include(tf_python.cmake)
endif()
if(tensorflow_BUILD_SHARED_LIB)
  include(tf_shared_lib.cmake)
endif()
if(tensorflow_BUILD_CC_TESTS OR tensorflow_BUILD_PYTHON_TESTS)
  include(tf_tests.cmake)
endif()
