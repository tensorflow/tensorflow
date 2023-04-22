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
set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(CMAKE_MODULE_PATH
  "${TFLITE_SOURCE_DIR}/tools/cmake/modules"
  ${CMAKE_MODULE_PATH}
)
set(CMAKE_PREFIX_PATH
  "${TFLITE_SOURCE_DIR}/tools/cmake/modules"
  ${CMAKE_PREFIX_PATH}
)
include(CMakeDependentOption)
option(TFLITE_ENABLE_RUY "Enable experimental RUY integration" OFF)
option(TFLITE_ENABLE_RESOURCE "Enable experimental support for resources" ON)
option(TFLITE_ENABLE_NNAPI "Enable NNAPI (Android only)." ON)
cmake_dependent_option(TFLITE_ENABLE_NNAPI_VERBOSE_VALIDATION "Enable NNAPI verbose validation." OFF
                       "TFLITE_ENABLE_NNAPI" ON)
option(TFLITE_ENABLE_MMAP "Enable MMAP (unsupported on Windows)" ON)
option(TFLITE_ENABLE_GPU "Enable GPU" OFF)
option(TFLITE_ENABLE_XNNPACK "Enable XNNPACK backend" ON)

option(TFLITE_KERNEL_TEST "Enable tflite kernel unit test" OFF)

set(CMAKE_CXX_STANDARD 14)  # Some components require C++14.
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
# Find TensorFlow Lite dependencies.
find_package(absl REQUIRED)
find_package(eigen REQUIRED)
find_package(farmhash REQUIRED)
find_package(fft2d REQUIRED)
find_package(flatbuffers REQUIRED)
find_package(gemmlowp REQUIRED)
find_package(neon2sse REQUIRED)
find_package(ruy REQUIRED)
# Generate TensorFlow Lite FlatBuffer code.
# We used to have an actual compilation logic with flatc but decided to use
# schema_generated.h since flatc doesn't work with cross compilation.
set(TFLITE_FLATBUFFERS_SCHEMA_DIR "${TFLITE_SOURCE_DIR}/schema")
set(TF_TARGET_PRIVATE_OPTIONS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang$")
  # TensorFlow uses a heap of deprecated proto fields so surpress these
  # warnings until they're fixed.
  list(APPEND TF_TARGET_PRIVATE_OPTIONS "-Wno-deprecated-declarations")
endif()
# Additional compiler flags used when compiling TF Lite.
set(TFLITE_TARGET_PUBLIC_OPTIONS "")
set(TFLITE_TARGET_PRIVATE_OPTIONS "")
# Additional library dependencies based upon enabled features.
set(TFLITE_TARGET_DEPENDENCIES "")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang$")
  # TFLite uses deprecated methods in neon2sse which generates a huge number of
  # warnings so surpress these until they're fixed.
  list(APPEND TFLITE_TARGET_PRIVATE_OPTIONS "-Wno-deprecated-declarations")
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  # Use NOMINMAX to disable the min / max macros in windows.h as they break
  # use of std::min std::max.
  # Use NOGDI to ERROR macro which breaks TensorFlow logging.
  list(APPEND TFLITE_TARGET_PRIVATE_OPTIONS "-DNOMINMAX" "-DNOGDI")
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
  elseif(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND TFLITE_TARGET_PRIVATE_OPTIONS "-Wa,-mbig-obj")
  endif()
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Android")
  find_library(ANDROID_LOG_LIB log)
endif()
# Build a list of source files to compile into the TF Lite library.
populate_tflite_source_vars("." TFLITE_SRCS)

# This particular file is excluded because the more explicit approach to enable
# XNNPACK delegate is preferred to the weak-symbol one.
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*tflite_with_xnnpack\\.cc$")

# Exclude Flex related files.
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*with_selected_ops\\.cc$")

if(_TFLITE_ENABLE_MMAP)
  list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*mmap_allocation_disabled\\.cc$")
else()
  list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*mmap_allocation\\.cc$")
endif()
if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
  list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*minimal_logging_android\\.cc$")
endif()
if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "iOS")
  list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*minimal_logging_ios\\.cc$")
endif()
populate_tflite_source_vars("core" TFLITE_CORE_SRCS)
populate_tflite_source_vars("core/api" TFLITE_CORE_API_SRCS)
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
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DCL_DELEGATE_NO_GL" "-DEGL_NO_X11")
  list(APPEND TFLITE_TARGET_DEPENDENCIES
    absl::any
    absl::flat_hash_map
  )
endif()
if(_TFLITE_ENABLE_NNAPI)
  populate_tflite_source_vars("delegates/nnapi"
    TFLITE_DELEGATES_NNAPI_SRCS
    FILTER "(_test_list|_disabled)\\.(cc|h)$"
  )
  populate_tflite_source_vars(
    "nnapi" TFLITE_NNAPI_SRCS FILTER "(_disabled)\\.(cc|h)$"
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
  find_package(xnnpack REQUIRED)
  populate_tflite_source_vars("delegates/xnnpack"
    TFLITE_DELEGATES_XNNPACK_SRCS
    FILTER ".*(_test|_tester)\\.(cc|h)"
  )
  list(APPEND TFLITE_TARGET_DEPENDENCIES
    XNNPACK
  )
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DTFLITE_BUILD_WITH_XNNPACK_DELEGATE")
endif()
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
set(TFLITE_PROFILER_SRCS ${TFLITE_SOURCE_DIR}/profiling/platform_profiler.cc)
if(CMAKE_SYSTEM_NAME MATCHES "Android")
  list(APPEND TFLITE_PROFILER_SRCS
    ${TFLITE_SOURCE_DIR}/profiling/atrace_profiler.cc
  )
endif()

# Common include directories
set(TFLITE_INCLUDE_DIRS
  "${TENSORFLOW_SOURCE_DIR}"
  "${TFLITE_FLATBUFFERS_SCHEMA_DIR}"
)
include_directories(
  BEFORE
    ${TFLITE_INCLUDE_DIRS}
)

# TFLite library
add_library(tensorflow-lite
  ${TFLITE_CORE_API_SRCS}
  ${TFLITE_CORE_SRCS}
  ${TFLITE_C_SRCS}
  ${TFLITE_DELEGATES_FLEX_SRCS}
  ${TFLITE_DELEGATES_GPU_SRCS}
  ${TFLITE_DELEGATES_NNAPI_SRCS}
  ${TFLITE_DELEGATES_SRCS}
  ${TFLITE_DELEGATES_XNNPACK_SRCS}
  ${TFLITE_EXPERIMENTAL_RESOURCE_SRCS}
  ${TFLITE_EXPERIMENTAL_RUY_PROFILER_SRCS}
  ${TFLITE_EXPERIMENTAL_RUY_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_INTEGER_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_SPARSE_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_OPT_SRCS}
  ${TFLITE_KERNEL_INTERNAL_REF_INTEGER_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_REF_SPARSE_OPS_SRCS}
  ${TFLITE_KERNEL_INTERNAL_REF_SRCS}
  ${TFLITE_KERNEL_INTERNAL_SRCS}
  ${TFLITE_KERNEL_SRCS}
  ${TFLITE_NNAPI_SRCS}
  ${TFLITE_SRCS}
  ${TFLITE_PROFILER_SRCS}
  ${TFLITE_SOURCE_DIR}/kernels/internal/utils/sparsity_format_converter.cc
  ${TFLITE_SOURCE_DIR}/schema/schema_utils.cc
)
target_include_directories(tensorflow-lite
  PUBLIC
    ${TFLITE_INCLUDE_DIRS}
)
target_link_libraries(tensorflow-lite
  PUBLIC
    Eigen3::Eigen
    NEON_2_SSE
    absl::flags
    absl::hash
    absl::status
    absl::strings
    absl::synchronization
    absl::variant
    farmhash
    fft2d_fftsg2d
    flatbuffers
    gemmlowp
    ruy
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
add_library(tensorflow::tensorflowlite ALIAS tensorflow-lite)

# The kernel tests.
if(TFLITE_KERNEL_TEST)
  enable_testing()
  add_subdirectory(${TFLITE_SOURCE_DIR}/kernels)
endif()

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
    ${TFLITE_INCLUDE_DIRS}
)

target_link_libraries(_pywrap_tensorflow_interpreter_wrapper
  tensorflow-lite
  ${CMAKE_DL_LIBS}
)
target_compile_options(_pywrap_tensorflow_interpreter_wrapper
  PUBLIC ${TFLITE_TARGET_PUBLIC_OPTIONS}
  PRIVATE ${TFLITE_TARGET_PRIVATE_OPTIONS}
)
