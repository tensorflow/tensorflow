#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Builds the Tensorflow Lite runtime.
#
# WARNING: This is an experimental that is subject to change.
# This has only been tested on Windows, Linux and macOS.
#
# The following are not currently supported:
# - iOS
# - Micro backend
# - Tests
# - Many features in experimental
# - Host Tools (i.e conversion / analysis tools etc.)

cmake_minimum_required(VERSION 3.16)
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release, for debug builds use"
    "'-DCMAKE_BUILD_TYPE=Debug'.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Double colon in target name means ALIAS or IMPORTED target.
cmake_policy(SET CMP0028 NEW)
# Enable MACOSX_RPATH (@rpath) for built dynamic libraries.
cmake_policy(SET CMP0042 NEW)
project(tensorflow-lite C CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../"
    ABSOLUTE
  )
endif()
set(TF_SOURCE_DIR "${TENSORFLOW_SOURCE_DIR}/tensorflow")
set(TSL_SOURCE_DIR "${TENSORFLOW_SOURCE_DIR}/third_party/xla/third_party/tsl")
set(XLA_SOURCE_DIR "${TENSORFLOW_SOURCE_DIR}/third_party/xla/")
set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(CMAKE_MODULE_PATH
  "${TFLITE_SOURCE_DIR}/tools/cmake/modules"
  ${CMAKE_MODULE_PATH}
)
set(CMAKE_PREFIX_PATH
  "${TFLITE_SOURCE_DIR}/tools/cmake/modules"
  ${CMAKE_PREFIX_PATH}
)
include(GNUInstallDirs)
include(CMakeDependentOption)
option(TFLITE_ENABLE_INSTALL "Enable install rule" OFF)
option(TFLITE_ENABLE_LABEL_IMAGE "Enable label_image example" OFF)
option(TFLITE_ENABLE_RUY "Enable experimental RUY integration" OFF)
option(TFLITE_ENABLE_RESOURCE "Enable experimental support for resources" ON)
option(TFLITE_ENABLE_NNAPI "Enable NNAPI (Android only)." ON)
cmake_dependent_option(TFLITE_ENABLE_NNAPI_VERBOSE_VALIDATION "Enable NNAPI verbose validation." OFF
                       "TFLITE_ENABLE_NNAPI" ON)
option(TFLITE_ENABLE_MMAP "Enable MMAP (unsupported on Windows)" ON)
option(TFLITE_ENABLE_GPU "Enable GPU" OFF)
option(TFLITE_ENABLE_METAL "Enable Metal delegate (iOS only)" OFF)
option(TFLITE_ENABLE_XNNPACK "Enable XNNPACK backend" ON)
option(TFLITE_ENABLE_EXTERNAL_DELEGATE "Enable External Delegate backend" ON)

option(TFLITE_KERNEL_TEST "Enable tflite kernel unit test" OFF)

if(${CMAKE_CROSSCOMPILING})
  set(TFLITE_HOST_TOOLS_DIR "" CACHE PATH "Host tools directory")
  if(TFLITE_HOST_TOOLS_DIR STREQUAL "")
    message(FATAL_ERROR "When cross-compiling, some tools need to be available to run on the host (current required tools: flatc). Please specify where those binaries can be found by using -DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path>.")
  endif()

  # Setup the host FlatBuffers compiler.
  set(FLATC_PATHS
      ${TFLITE_HOST_TOOLS_DIR}
      ${TFLITE_HOST_TOOLS_DIR}/bin
      ${TFLITE_HOST_TOOLS_DIR}/flatbuffers-flatc/bin
      )
  find_program(FLATC_BIN flatc PATHS ${FLATC_PATHS})
  if(${FLATC_BIN} STREQUAL "FLATC_BIN-NOTFOUND")
    message(FATAL_ERROR "Host 'flatc' compiler has not been found in the following locations: ${FLATC_PATHS}")
  else()
    message(STATUS "Pre-built 'flatc' compiler for cross-compilation purposes found: ${FLATC_BIN}")
    set(FLATBUFFERS_FLATC_EXECUTABLE "${FLATC_BIN}")
  endif()
else()
  set(FLATBUFFERS_FLATC_EXECUTABLE "${CMAKE_BINARY_DIR}/flatbuffers-flatc/bin/flatc")
  set(FLATC_TARGET "flatbuffers-flatc")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(_TFLITE_ENABLE_RUY "${TFLITE_ENABLE_RUY}")
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
  set(_TFLITE_ENABLE_RUY ON)
endif()
set(_TFLITE_ENABLE_NNAPI "${TFLITE_ENABLE_NNAPI}")
if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
  set(_TFLITE_ENABLE_NNAPI OFF)
endif()
set(_TFLITE_ENABLE_MMAP "${TFLITE_ENABLE_MMAP}")
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  # See https://github.com/tensorflow/tensorflow/blob/\
  # 2b96f3662bd776e277f86997659e61046b56c315/tensorflow/lite/tools/make/\
  # Makefile#L157
  set(_TFLITE_ENABLE_MMAP OFF)
endif()
# Simplifies inclusion of non-test sources and headers from a directory.
# SOURCE_DIR: Directory to search for files.
# SOURCES_VAR: Variable to append with all matching *.cc and *.h files.
# [FILTER expression0 .. expressionN]:
#   Additional regular expressions to filter the set of matching
#   files. By default, all files ending in "(_test|test_util)\\.(cc|h)" are
#   removed.
# [RECURSE]: Whether to recursively search SOURCE_DIR.
macro(populate_source_vars SOURCE_DIR SOURCES_VAR)
  cmake_parse_arguments(ARGS "RECURSE" "" "FILTER" ${ARGN})
  if(ARGS_RECURSE)
    set(GLOB_OP GLOB_RECURSE)
  else()
    set(GLOB_OP GLOB)
  endif()
  set(DEFAULT_FILE_FILTER ".*(_test|test_util)\\.(c|cc|h)$")
  file(${GLOB_OP} FOUND_SOURCES "${SOURCE_DIR}/*.*")
  list(FILTER FOUND_SOURCES INCLUDE REGEX ".*\\.(c|cc|h)$")
  list(FILTER FOUND_SOURCES EXCLUDE REGEX "${DEFAULT_FILE_FILTER}")
  foreach(FILE_FILTER ${ARGS_FILTER})
    list(FILTER FOUND_SOURCES EXCLUDE REGEX "${FILE_FILTER}")
  endforeach()
  list(APPEND ${SOURCES_VAR} ${FOUND_SOURCES})
endmacro()
# Simplifies inclusion of non-test sources and headers from a directory
# relative to TFLITE_SOURCE_DIR. See populate_source_vars() for the
# description of arguments including and following SOURCES_VAR.
macro(populate_tflite_source_vars RELATIVE_DIR SOURCES_VAR)
  populate_source_vars(
    "${TFLITE_SOURCE_DIR}/${RELATIVE_DIR}" ${SOURCES_VAR} ${ARGN}
  )
endmacro()
# Simplifies inclusion of non-test sources and headers from a directory
# relative to TF_SOURCE_DIR. See populate_source_vars() for the description of
# arguments including and following SOURCES_VAR.
macro(populate_tf_source_vars RELATIVE_DIR SOURCES_VAR)
  populate_source_vars(
    "${TF_SOURCE_DIR}/${RELATIVE_DIR}" ${SOURCES_VAR} ${ARGN}
  )
endmacro()
# Make sure all repos have licenses.
set(OVERRIDABLE_FETCH_CONTENT_LICENSE_CHECK ON)
# Additional library dependencies based upon enabled features.
set(TFLITE_TARGET_DEPENDENCIES "")
# Find TensorFlow Lite dependencies.
find_package(absl REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(farmhash REQUIRED)
find_package(fft2d REQUIRED)
find_package(FlatBuffers REQUIRED)
find_package(gemmlowp REQUIRED)
if (NOT CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86")
  find_package(NEON_2_SSE REQUIRED)
  list(APPEND TFLITE_TARGET_DEPENDENCIES NEON_2_SSE::NEON_2_SSE)
endif()
find_package(cpuinfo REQUIRED)  #CPUINFO is used by XNNPACK and RUY library
find_package(ml_dtypes REQUIRED)
find_package(ruy REQUIRED)
# Include TSL, which is in tensorflow/third_party
include_directories(
  ${TSL_SOURCE_DIR}
  ${XLA_SOURCE_DIR}
)
# Download necessary dependencies.
if(TFLITE_ENABLE_XNNPACK)
  # pthreadpool is used by XNNPACK.
  if(SYSTEM_PTHREADPOOL)
    find_library(PTHREADPOOL_LIB pthreadpool REQUIRED)
  elseif(NOT DEFINED PTHREADPOOL_SOURCE_DIR)
      message(STATUS "Downloading pthreadpool to ${CMAKE_BINARY_DIR}/pthreadpool-source (define SYSTEM_PTHREADPOOL or PTHREADPOOL_SOURCE_DIR to avoid it)")
      configure_file(cmake/DownloadPThreadPool.cmake "${CMAKE_BINARY_DIR}/pthreadpool-download/CMakeLists.txt")
      execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pthreadpool-download")
      execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pthreadpool-download")
      set(PTHREADPOOL_SOURCE_DIR "${CMAKE_BINARY_DIR}/pthreadpool-source" CACHE STRING "pthreadpool source directory")
  endif()
  # Configure pthreadpool
  if(NOT SYSTEM_PTHREADPOOL AND NOT TARGET pthreadpool)
    set(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
    set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
    set(PTHREADPOOL_ALLOW_DEPRECATED_API OFF CACHE BOOL "")
    add_subdirectory(
      "${PTHREADPOOL_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/pthreadpool")
  endif()
  list(APPEND TFLITE_TARGET_DEPENDENCIES pthreadpool)

  IF(NOT DEFINED FP16_SOURCE_DIR)
    MESSAGE(STATUS "Downloading FP16 to ${CMAKE_BINARY_DIR}/FP16-source (define FP16_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadFP16.cmake "${CMAKE_BINARY_DIR}/FP16-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/FP16-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/FP16-download")
    SET(FP16_SOURCE_DIR "${CMAKE_BINARY_DIR}/FP16-source" CACHE STRING "FP16 source directory")
  ENDIF()
endif()
set(TF_TARGET_PRIVATE_OPTIONS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang$")
  # TensorFlow uses a heap of deprecated proto fields so surpress these
  # warnings until they're fixed.
  list(APPEND TF_TARGET_PRIVATE_OPTIONS "-Wno-deprecated-declarations")
endif()
# Additional compiler flags used when compiling TF Lite.
set(TFLITE_TARGET_PUBLIC_OPTIONS "-DEIGEN_NEON_GEBP_NR=4")
set(TFLITE_TARGET_PRIVATE_OPTIONS "")
set(TFLITE_TARGET_PRIVATE_DEFINITIONS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang$")
  # TFLite uses deprecated methods in neon2sse which generates a huge number of
  # warnings so surpress these until they're fixed.
  list(APPEND TFLITE_TARGET_PRIVATE_OPTIONS "-Wno-deprecated-declarations")
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  # Use NOMINMAX to disable the min / max macros in windows.h as they break
  # use of std::min std::max.
  # Use NOGDI to ERROR macro which breaks TensorFlow logging.
  # Disable mmap, which is not available on Windows.
  list(APPEND TFLITE_TARGET_PRIVATE_OPTIONS "-DNOMINMAX" "-DNOGDI" "-DTFLITE_MMAP_DISABLED")
  # lite/kernels/conv.cc has more than 64k sections so enable /bigobj to
  # support compilation with MSVC2015.
  if(MSVC)
    list(APPEND TFLITE_TARGET_PRIVATE_OPTIONS "/bigobj")
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
      message("Disabling MSVC /O2 optimization for Win32")
      set(CompFlags
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELWITHDEBINFO
      )
      foreach (CompFlag ${CompFlags})
        string(REGEX REPLACE "(\/Ob. )" "" ${CompFlag} "${${CompFlag}}")
        string(REPLACE "/O2" "/O1" ${CompFlag} "${${CompFlag}}")
        list(REMOVE_DUPLICATES ${CompFlag})
        set(${CompFlag} "${${CompFlag}}" CACHE INTERNAL "")
      endforeach()
    endif()
    list(APPEND TFLITE_TARGET_PRIVATE_DEFINITIONS "_USE_MATH_DEFINES")
  elseif(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND TFLITE_TARGET_PRIVATE_OPTIONS "-Wa,-mbig-obj")
  endif()
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Android")
  find_library(ANDROID_LOG_LIB log)
  list(APPEND TFLITE_TARGET_DEPENDENCIES
    log
  )
endif()
# Build a list of source files to compile into the TF Lite library.
populate_tflite_source_vars("." TFLITE_SRCS)
if(CMAKE_SYSTEM_NAME MATCHES "Windows" AND BUILD_SHARED_LIBS)
  list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*simple_memory_arena_debug_dump\\.cc$")
endif()

# This particular file is excluded because the more explicit approach to enable
# XNNPACK delegate is preferred to the weak-symbol one.
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*tflite_with_xnnpack\\.cc$")

# Exclude Flex related files.
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*with_selected_ops\\.cc$")

# Exclude tensorflow_profiler_logger files.
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*tensorflow_profiler_logger\\.cc$")

# Handle TFLite logging source.
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*minimal_logging_.*\\.cc$")
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
  list(APPEND TFLITE_SRCS ${TFLITE_SOURCE_DIR}/minimal_logging_android.cc)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "iOS")
  list(APPEND TFLITE_SRCS ${TFLITE_SOURCE_DIR}/minimal_logging_ios.cc)
else()
  list(APPEND TFLITE_SRCS ${TFLITE_SOURCE_DIR}/minimal_logging_default.cc)
endif()

populate_tflite_source_vars("core" TFLITE_CORE_SRCS)
populate_tflite_source_vars(
  "core/acceleration/configuration" TFLITE_CORE_ACCELERATION_SRCS
  FILTER "xnnpack_plugin.*"
  FILTER "(_test)\\.(cc|h)$"
)
populate_tflite_source_vars("core/api" TFLITE_CORE_API_SRCS)
populate_tflite_source_vars("core/async" TFLITE_CORE_ASYNC_SRCS)
populate_tflite_source_vars("core/async/c" TFLITE_CORE_ASYNC_C_SRCS)
populate_tflite_source_vars("core/async/interop" TFLITE_CORE_ASYNC_INTEROP_SRCS)
populate_tflite_source_vars("core/async/interop/c" TFLITE_CORE_ASYNC_INTEROP_C_SRCS)
populate_tflite_source_vars("core/c" TFLITE_CORE_C_SRCS)
populate_tflite_source_vars("core/experimental/acceleration/configuration" TFLITE_CORE_EXPERIMENTAL_SRCS)
populate_tflite_source_vars("core/kernels" TFLITE_CORE_KERNELS_SRCS)
populate_tflite_source_vars("core/tools" TFLITE_CORE_TOOLS_SRCS)
populate_tflite_source_vars("c" TFLITE_C_SRCS)
populate_tflite_source_vars("delegates" TFLITE_DELEGATES_SRCS)
if(TFLITE_ENABLE_GPU)
  find_package(opencl_headers REQUIRED)
  find_package(vulkan_headers REQUIRED)
  find_package(fp16_headers REQUIRED)
  # Android NDK already has OpenGL, EGL headers.
  if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
    find_package(opengl_headers REQUIRED)
    find_package(egl_headers REQUIRED)
  endif()
  populate_tflite_source_vars(
    "delegates/gpu/cl" TFLITE_DELEGATES_GPU_CL_SRCS
    FILTER "(_test|gl_interop|gpu_api_delegate|egl_sync)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/cl/default" TFLITE_DELEGATES_GPU_CL_DEFAULT_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/cl/kernels" TFLITE_DELEGATES_GPU_CL_KERNELS_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/common/default" TFLITE_DELEGATES_GPU_COMMON_DEFAULT_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/common/memory_management"
    TFLITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/common/selectors" TFLITE_DELEGATES_GPU_COMMON_SELECTORS_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/common/selectors/default" TFLITE_DELEGATES_GPU_COMMON_SELECTORS_DEFAULT_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/common" TFLITE_DELEGATES_GPU_COMMON_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/common/task"
    TFLITE_DELEGATES_GPU_COMMON_TASK_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/common/tasks"
    TFLITE_DELEGATES_GPU_COMMON_TASKS_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/common/tasks/special"
    TFLITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "delegates/gpu/common/transformations"
    TFLITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_SRCS
    FILTER "(_test)\\.(cc|h)$"
  )
  list(APPEND TFLITE_DELEGATES_GPU_SRCS
    ${TFLITE_SOURCE_DIR}/delegates/gpu/api.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/delegate.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/delegate.h
    ${TFLITE_SOURCE_DIR}/delegates/gpu/delegate_options.cc
    ${TFLITE_SOURCE_DIR}/delegates/gpu/delegate_options.h
    ${TFLITE_SOURCE_DIR}/delegates/gpu/tflite_profile.cc
    ${TFLITE_SOURCE_DIR}/experimental/acceleration/compatibility/android_info.cc
    ${TFLITE_DELEGATES_GPU_CL_SRCS}
    ${TFLITE_DELEGATES_GPU_CL_DEFAULT_SRCS}
    ${TFLITE_DELEGATES_GPU_CL_KERNELS_SRCS}
    ${TFLITE_DELEGATES_GPU_COMMON_DEFAULT_SRCS}
    ${TFLITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_SRCS}
    ${TFLITE_DELEGATES_GPU_COMMON_SELECTORS_SRCS}
    ${TFLITE_DELEGATES_GPU_COMMON_SELECTORS_DEFAULT_SRCS}
    ${TFLITE_DELEGATES_GPU_COMMON_SRCS}
    ${TFLITE_DELEGATES_GPU_COMMON_TASK_SRCS}
    ${TFLITE_DELEGATES_GPU_COMMON_TASKS_SRCS}
    ${TFLITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_SRCS}
    ${TFLITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_SRCS}
    ${TFLITE_SOURCE_DIR}/tools/versioning/gpu_compatibility.cc
    ${TFLITE_SOURCE_DIR}/tools/versioning/op_signature.cc
  )
  include_directories(
    AFTER
    ${TFLITE_SOURCE_DIR}/delegates/gpu/common
    ${TFLITE_SOURCE_DIR}/delegates/gpu/common/task
  )
  if(TFLITE_ENABLE_METAL AND "${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    #
    # libmetal_delegate library
    #
    enable_language(OBJCXX)
    list(APPEND TFLITE_DELEGATES_METAL_SRCS
      ${TFLITE_SOURCE_DIR}/delegates/gpu/metal_delegate.mm
      ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/buffer.cc
      ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/buffer_convert.mm
      ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/common.mm
      ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/compute_task.cc
      ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/inference_context.cc
      ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/metal_arguments.cc
      ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/metal_device.cc
      ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/metal_spatial_tensor.cc
    )
    add_library(metal_delegate STATIC
      ${TFLITE_DELEGATES_METAL_SRCS}
    )
    target_include_directories(metal_delegate PUBLIC
      ${CMAKE_BINARY_DIR}/abseil-cpp
      ${CMAKE_BINARY_DIR}/flatbuffers/include
      PRIVATE ${TENSORFLOW_SOURCE_DIR}
    )
    #
    # generate flatbuffers header for inference_context
    #
    if(FLATBUFFERS_FLATC_EXECUTABLE)
      set(FLATC ${FLATBUFFERS_FLATC_EXECUTABLE})
    else()
      set(FLATC flatc)
    endif()
    add_custom_command(
        OUTPUT ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/inference_context_generated.h
        COMMAND ${FLATC} --scoped-enums
        -I ${TENSORFLOW_SOURCE_DIR}
        -o ${TFLITE_SOURCE_DIR}/delegates/gpu/metal
        -c ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/inference_context.fbs
    )
    add_custom_target(
        inference_context_cc_fbs
        DEPENDS ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/inference_context_generated.h
    )
    add_dependencies(metal_delegate inference_context_cc_fbs)
    #
    # supplementary libraries for libmetal_delegate
    #
    list(APPEND CC_SRCS
        buffer
        compute_task
        inference_context
        metal_arguments
        metal_device
        metal_spatial_tensor
    )
   SET(METAL_DELEGATE_PATH ${TFLITE_SOURCE_DIR}/delegates/gpu/metal/)

   foreach(lib_name ${CC_SRCS})
     set_source_files_properties(${METAL_DELEGATE_PATH}${lib_name}.cc  PROPERTIES LANGUAGE OBJCXX)
     add_library("${lib_name}" STATIC ${METAL_DELEGATE_PATH}${lib_name}.cc)
     target_include_directories("${lib_name}" PUBLIC
       ${CMAKE_BINARY_DIR}/abseil-cpp
       ${CMAKE_BINARY_DIR}/flatbuffers/include
     )
     set_target_properties(${lib_name} PROPERTIES LINKER_LANGUAGE OBJCXX)
     target_link_libraries(${lib_name})
   endforeach()

   list(APPEND MM_SRCS
     buffer_convert
     common
   )
   foreach(lib_name ${MM_SRCS})
     add_library("${lib_name}" STATIC ${METAL_DELEGATE_PATH}${lib_name}.mm)
     target_include_directories("${lib_name}" PUBLIC
       ${CMAKE_BINARY_DIR}/abseil-cpp
       ${CMAKE_BINARY_DIR}/flatbuffers/include
     )
     target_link_libraries(${lib_name})
   endforeach()
endif()
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DCL_DELEGATE_NO_GL" "-DEGL_NO_X11")
  list(APPEND TFLITE_TARGET_DEPENDENCIES
    absl::any
    absl::flat_hash_map
  )
endif()
if(_TFLITE_ENABLE_NNAPI)
  find_package(fp16_headers REQUIRED)
  populate_tflite_source_vars("delegates/nnapi"
    TFLITE_DELEGATES_NNAPI_SRCS
    FILTER "(_test_list|_disabled)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "nnapi" TFLITE_NNAPI_SRCS FILTER "(_disabled)\\.(cc|h)$"
  )

  list(APPEND TFLITE_NNAPI_SRCS
    "${TFLITE_SOURCE_DIR}/nnapi/sl/SupportLibrary.cc"
  )

  if(${TFLITE_ENABLE_NNAPI_VERBOSE_VALIDATION})
    list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DNNAPI_VERBOSE_VALIDATION")
  endif()
else()
  set(TFLITE_DELEGATES_NNAPI_SRCS
    "${TFLITE_SOURCE_DIR}/delegates/nnapi/nnapi_delegate_disabled.cc"
  )
  set(TFLITE_NNAPI_SRCS
    "${TFLITE_SOURCE_DIR}/nnapi/nnapi_implementation_disabled.cc"
  )
endif()
if(TFLITE_ENABLE_XNNPACK)
  find_package(fp16_headers REQUIRED)
  find_package(XNNPACK REQUIRED)
  populate_tflite_source_vars("delegates/xnnpack"
    TFLITE_DELEGATES_XNNPACK_SRCS
    FILTER ".*(_test|_tester)\\.(cc|h)"
  )
  add_custom_command(
    OUTPUT "${PROJECT_BINARY_DIR}/tensorflow/lite/delegates/xnnpack/weight_cache_schema_generated.h"
    COMMAND "${FLATBUFFERS_FLATC_EXECUTABLE}" -c
      -o "${PROJECT_BINARY_DIR}/tensorflow/lite/delegates/xnnpack/"
      --gen-mutable --gen-object-api
      "${TFLITE_SOURCE_DIR}/delegates/xnnpack/weight_cache_schema.fbs"
    DEPENDS "${FLATC_TARGET}" "${TFLITE_SOURCE_DIR}/delegates/xnnpack/weight_cache_schema.fbs"
  )

  add_library(xnnpack-delegate STATIC
    "${TFLITE_DELEGATES_XNNPACK_SRCS}"
    "${PROJECT_BINARY_DIR}/tensorflow/lite/delegates/xnnpack/weight_cache_schema_generated.h"
  )
  target_include_directories(xnnpack-delegate
    PUBLIC $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE "${PROJECT_BINARY_DIR}"
  )
  target_link_libraries(xnnpack-delegate
    Eigen3::Eigen
    flatbuffers::flatbuffers
    gemmlowp::gemmlowp
    ruy::ruy
    )

  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DTFLITE_KERNEL_USE_XNNPACK")
  target_compile_options(xnnpack-delegate
    PUBLIC ${TFLITE_TARGET_PUBLIC_OPTIONS}
    PRIVATE ${TFLITE_TARGET_PRIVATE_OPTIONS}
  )

  list(APPEND TFLITE_TARGET_DEPENDENCIES
    xnnpack-delegate
    XNNPACK
  )
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DTFLITE_BUILD_WITH_XNNPACK_DELEGATE")
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DXNNPACK_DELEGATE_ENABLE_QS8")
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DXNNPACK_DELEGATE_ENABLE_QU8")
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DXNNPACK_DELEGATE_USE_LATEST_OPS")
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DXNNPACK_DELEGATE_ENABLE_SUBGRAPH_RESHAPING")
endif()
if(TFLITE_ENABLE_EXTERNAL_DELEGATE)
  populate_tflite_source_vars("delegates/external"
    TFLITE_DELEGATES_EXTERNAL_SRCS
    FILTER ".*(_test|_tester)\\.(cc|h)"
  )
endif()
populate_tflite_source_vars("experimental/remat"
  TFLITE_EXPERIMENTAL_REMAT_SRCS
  FILTER ".*_test\\.(cc|h)$"
)
if (TFLITE_ENABLE_RESOURCE)
  populate_tflite_source_vars("experimental/resource"
    TFLITE_EXPERIMENTAL_RESOURCE_SRCS
  )
endif()
populate_tflite_source_vars("experimental/ruy"
  TFLITE_EXPERIMENTAL_RUY_SRCS
  FILTER
  ".*(test(_fast|_slow|_special_specs))\\.(cc|h)$"
  ".*(benchmark|tune_tool|example)\\.(cc|h)$"
)
populate_tflite_source_vars("experimental/ruy/profiler"
  TFLITE_EXPERIMENTAL_RUY_PROFILER_SRCS
  FILTER ".*(test|test_instrumented_library)\\.(cc|h)$"
)
if(_TFLITE_ENABLE_RUY)
  message(STATUS "RUY is enabled.")
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DTFLITE_WITH_RUY")
endif()
populate_tflite_source_vars("kernels"
  TFLITE_KERNEL_SRCS
  FILTER "(.*_test_util_internal|test_.*|.*_ops_wrapper)\\.(cc|h)"
)
populate_tflite_source_vars("kernels/internal" TFLITE_KERNEL_INTERNAL_SRCS)
populate_tflite_source_vars("kernels/internal/optimized"
  TFLITE_KERNEL_INTERNAL_OPT_SRCS
)
populate_tflite_source_vars("kernels/internal/optimized/integer_ops"
  TFLITE_KERNEL_INTERNAL_OPT_INTEGER_OPS_SRCS
)
populate_tflite_source_vars("kernels/internal/optimized/sparse_ops"
  TFLITE_KERNEL_INTERNAL_OPT_SPARSE_OPS_SRCS
)
populate_tflite_source_vars("kernels/internal/reference"
  TFLITE_KERNEL_INTERNAL_REF_SRCS
)
populate_tflite_source_vars("kernels/internal/reference/integer_ops"
  TFLITE_KERNEL_INTERNAL_REF_INTEGER_OPS_SRCS
)
populate_tflite_source_vars("kernels/internal/reference/sparse_ops"
  TFLITE_KERNEL_INTERNAL_REF_SPARSE_OPS_SRCS
)
populate_tflite_source_vars("kernels/internal/optimized/4bit"
  TFLITE_KERNEL_INTERNAL_OPT_4BIT_SRCS
  FILTER "(.*neon.*|.*sse.*)\\.(cc|h)"
)
set(TFLITE_PROFILER_SRCS
  ${TFLITE_SOURCE_DIR}/profiling/platform_profiler.cc
  ${TFLITE_SOURCE_DIR}/profiling/root_profiler.h
  ${TFLITE_SOURCE_DIR}/profiling/root_profiler.cc
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/profiler.cc
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/profiler.h
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/telemetry.cc
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/c/telemetry_setting_internal.cc
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/c/telemetry_setting_internal.h
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/c/profiler.h
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/c/telemetry_setting.h
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/telemetry_status.h
)
if(CMAKE_SYSTEM_NAME MATCHES "Android")
  list(APPEND TFLITE_PROFILER_SRCS
    ${TFLITE_SOURCE_DIR}/profiling/atrace_profiler.cc
  )
elseif(CMAKE_SYSTEM_NAME MATCHES "iOS")
  enable_language(OBJCXX)
  list(APPEND TFLITE_PROFILER_SRCS
    ${TFLITE_SOURCE_DIR}/profiling/signpost_profiler.mm
  )
  set_source_files_properties(${TFLITE_SOURCE_DIR}/profiling/signpost_profiler.mm PROPERTIES COMPILE_FLAGS -fno-objc-arc)
endif()

# TFLite library
set(_ALL_TFLITE_SRCS
  ${TFLITE_CORE_ACCELERATION_SRCS}
  ${TFLITE_CORE_API_SRCS}
  ${TFLITE_CORE_C_SRCS}
  ${TFLITE_CORE_EXPERIMENTAL_SRCS}
  ${TFLITE_CORE_KERNELS_SRCS}
  ${TFLITE_CORE_SRCS}
  ${TFLITE_CORE_ASYNC_SRCS}
  ${TFLITE_CORE_ASYNC_C_SRCS}
  ${TFLITE_CORE_ASYNC_INTEROP_SRCS}
  ${TFLITE_CORE_ASYNC_INTEROP_C_SRCS}
  ${TFLITE_CORE_TOOLS_SRCS}
  ${TFLITE_C_SRCS}
  ${TFLITE_DELEGATES_FLEX_SRCS}
  ${TFLITE_DELEGATES_GPU_SRCS}
  ${TFLITE_DELEGATES_NNAPI_SRCS}
  ${TFLITE_DELEGATES_SRCS}
  ${TFLITE_DELEGATES_EXTERNAL_SRCS}
  ${TFLITE_EXPERIMENTAL_REMAT_SRCS}
  ${TFLITE_EXPERIMENTAL_RESOURCE_SRCS}
  ${TFLITE_EXPERIMENTAL_RUY_PROFILER_SRCS}
  ${TFLITE_EXPERIMENTAL_RUY_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_INTEGER_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_SPARSE_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_4BIT_SRCS}
  ${TFLITE_KERNEL_INTERNAL_REF_INTEGER_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_REF_SPARSE_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_REF_SRCS}
  ${TFLITE_KERNEL_INTERNAL_SRCS}
  ${TFLITE_KERNEL_SRCS}
  ${TFLITE_NNAPI_SRCS}
  ${TFLITE_SRCS}
  ${TFLITE_PROFILER_SRCS}
  ${TFLITE_SOURCE_DIR}/internal/signature_def.h
  ${TFLITE_SOURCE_DIR}/kernels/internal/utils/sparsity_format_converter.cc
  ${TFLITE_SOURCE_DIR}/schema/conversion_metadata_generated.h
  ${TFLITE_SOURCE_DIR}/schema/schema_generated.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/schema/schema_utils.cc
  ${TF_SOURCE_DIR}/compiler/mlir/lite/schema/schema_generated.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/schema/conversion_metadata_generated.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/utils/control_edges.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/utils/string_utils.cc
  ${TF_SOURCE_DIR}/compiler/mlir/lite/utils/string_utils.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/experimental/remat/metadata_util.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/experimental/remat/metadata_util.cc
  ${TF_SOURCE_DIR}/compiler/mlir/lite/core/macros.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/core/model_builder_base.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/core/model_builder_base.cc
  ${TF_SOURCE_DIR}/compiler/mlir/lite/core/api/error_reporter.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/core/api/error_reporter.cc
  ${TF_SOURCE_DIR}/compiler/mlir/lite/core/api/verifier.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/allocation.h
  ${TF_SOURCE_DIR}/compiler/mlir/lite/allocation.cc
  ${TF_SOURCE_DIR}/compiler/mlir/lite/mmap_allocation.cc
  ${TF_SOURCE_DIR}/compiler/mlir/lite/mmap_allocation_disabled.cc
)

if(_TFLITE_ENABLE_MMAP)
  list(FILTER _ALL_TFLITE_SRCS EXCLUDE REGEX ".*mmap_allocation_disabled\\.cc$")
else()
  list(FILTER _ALL_TFLITE_SRCS EXCLUDE REGEX ".*mmap_allocation\\.cc$")
endif()

add_library(tensorflow-lite
  ${_ALL_TFLITE_SRCS}
)
set(_ALL_TFLITE_HDRS ${_ALL_TFLITE_SRCS})
list(FILTER _ALL_TFLITE_HDRS INCLUDE REGEX ".*\\.h$")
target_include_directories(tensorflow-lite
  PUBLIC $<BUILD_INTERFACE:${TENSORFLOW_SOURCE_DIR}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(tensorflow-lite
  PUBLIC
    Eigen3::Eigen
    absl::flags
    absl::hash
    absl::status
    absl::strings
    absl::synchronization
    absl::variant
    farmhash
    fft2d_fftsg2d
    flatbuffers::flatbuffers
    gemmlowp::gemmlowp
    ml_dtypes
    ruy::ruy
    ${CMAKE_DL_LIBS}
    ${TFLITE_TARGET_DEPENDENCIES}
)

if (NOT BUILD_SHARED_LIBS)
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DTFL_STATIC_LIBRARY_BUILD")
endif()

target_compile_options(tensorflow-lite
  PUBLIC ${TFLITE_TARGET_PUBLIC_OPTIONS}
  PRIVATE ${TFLITE_TARGET_PRIVATE_OPTIONS}
)
add_library(${PROJECT_NAME}::tensorflowlite ALIAS tensorflow-lite)

# The install targets.
if(TFLITE_ENABLE_INSTALL)
  install(
    TARGETS tensorflow-lite
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  foreach(hdr ${_ALL_TFLITE_HDRS})
    get_filename_component(dir ${hdr} DIRECTORY)
    file(RELATIVE_PATH dir ${CMAKE_CURRENT_SOURCE_DIR} ${dir})
    install(
      FILES ${hdr}
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tensorflow/lite/${dir}"
    )
  endforeach()

  install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  )

  # Generate config file that's used by find_package(tensorflow-lite CONFIG).

  include(CMakePackageConfigHelpers)

  configure_package_config_file(
    "tools/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  )

  install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  )
endif()

# The kernel tests.
if(TFLITE_KERNEL_TEST)
  enable_testing()
  add_subdirectory(${TFLITE_SOURCE_DIR}/kernels)
endif()

# Add the generated headers directory. Required for maintaining the
# tensorflow/lite directory structure for generated headers.
set(TFLITE_GENERATED_HEADERS_DIR ${CMAKE_BINARY_DIR}/tensorflow/lite)

# Add the profiling proto directory.
add_subdirectory(${TFLITE_SOURCE_DIR}/profiling/proto)

# Add the tf example directory.
add_subdirectory(${TF_SOURCE_DIR}/core/example ${CMAKE_BINARY_DIR}/example_proto_generated)

# The benchmark tool.
add_subdirectory(${TFLITE_SOURCE_DIR}/tools/benchmark)

# The label_image example.
add_subdirectory(${TFLITE_SOURCE_DIR}/examples/label_image)

# Python interpreter wrapper.
add_library(_pywrap_tensorflow_interpreter_wrapper SHARED EXCLUDE_FROM_ALL
  ${TFLITE_SOURCE_DIR}/python/interpreter_wrapper/interpreter_wrapper.cc
  ${TFLITE_SOURCE_DIR}/python/interpreter_wrapper/interpreter_wrapper_pybind11.cc
  ${TFLITE_SOURCE_DIR}/python/interpreter_wrapper/numpy.cc
  ${TFLITE_SOURCE_DIR}/python/interpreter_wrapper/python_error_reporter.cc
  ${TFLITE_SOURCE_DIR}/python/interpreter_wrapper/python_utils.cc
)

# To remove "lib" prefix.
set_target_properties(_pywrap_tensorflow_interpreter_wrapper PROPERTIES PREFIX "")

target_include_directories(_pywrap_tensorflow_interpreter_wrapper
  PUBLIC
    ${TENSORFLOW_SOURCE_DIR}
)

target_link_libraries(_pywrap_tensorflow_interpreter_wrapper
  tensorflow-lite
  ${CMAKE_DL_LIBS}
)
target_compile_options(_pywrap_tensorflow_interpreter_wrapper
  PUBLIC ${TFLITE_TARGET_PUBLIC_OPTIONS}
  PRIVATE ${TFLITE_TARGET_PRIVATE_OPTIONS}
)
