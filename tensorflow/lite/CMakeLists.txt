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
# - GPU acceleration
# - Android
# - iOS
# - Micro backend
# - Tests
# - Many features in experimental
# - Host Tools (i.e conversion / analysis tools etc.)

cmake_minimum_required(VERSION 3.16)
# Double colon in target name means ALIAS or IMPORTED target.
cmake_policy(SET CMP0028 NEW)
# Enable MACOSX_RPATH (@rpath) for built dynamic libraries.
cmake_policy(SET CMP0042 NEW)
project(tensorflow-lite C CXX)
set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_SOURCE_DIR}/../../"
    ABSOLUTE
  )
endif()
set(TF_SOURCE_DIR "${TENSORFLOW_SOURCE_DIR}/tensorflow")
set(TFLITE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(CMAKE_MODULE_PATH
  "${TFLITE_SOURCE_DIR}/tools/cmake/modules"
  ${CMAKE_MODULE_PATH}
)
set(CMAKE_PREFIX_PATH
  "${TFLITE_SOURCE_DIR}/tools/cmake/modules"
  ${CMAKE_PREFIX_PATH}
)
# b/168750039: To workaround absl module not found error on Android build.
set(absl_DIR ${CMAKE_MODULE_PATH})

option(TFLITE_ENABLE_RUY "Enable experimental RUY integration" OFF)
option(TFLITE_ENABLE_RESOURCE "Enable experimental support for resources" ON)
option(TFLITE_ENABLE_NNAPI "Enable NNAPI (Android only)." ON)
option(TFLITE_ENABLE_MMAP "Enable MMAP (unsupported on Windows)" ON)
option(TFLITE_ENABLE_GPU "Enable GPU (not supported)" OFF)
# This must be enabled when converting from TF models with SELECT_TF_OPS
# enabled.
# https://www.tensorflow.org/lite/guide/ops_select#converting_the_model
# This is currently not supported.
option(TFLITE_ENABLE_FLEX "Enable SELECT_TF_OPS" OFF) # TODO: Add support
option(TFLITE_ENABLE_XNNPACK "Enable XNNPACK backend" ON)
set(CMAKE_CXX_STANDARD 14)  # Some components require C++14.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
find_package(absl REQUIRED CONFIG)
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
  elseif(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND TFLITE_TARGET_PRIVATE_OPTIONS "-Wa,-mbig-obj")
  endif()
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Android")
  find_library(ANDROID_LOG_LIB log)
endif()
# Build a list of source files to compile into the TF Lite library.
populate_tflite_source_vars("." TFLITE_SRCS)
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
if(TFLITE_ENABLE_FLEX)
  message(FATAL_ERROR "TF Lite Flex delegate is currently not supported.")
  populate_tflite_source_vars("delegates/flex" TFLITE_DELEGATES_FLEX_SRCS)
  list(APPEND TFLITE_TARGET_DEPENDENCIES
    absl::inlined_vector
    absl::optional
    absl::type_traits
  )
endif()
if(TFLITE_ENABLE_GPU)
  # Implementation is under delegates/gpu.
  message(FATAL_ERROR
    "GPU acceleration is not currently supported in CMake builds"
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
else()
  set(TFLITE_DELEGATES_NNAPI_SRCS
    "${TFLITE_SOURCE_DIR}/delegates/nnapi/nnapi_delegate_disabled.cc"
  )
  set(TFLITE_NNAPI_SRCS
    "${TFLITE_SOURCE_DIR}/nnapi/nnapi_implementation_disabled.cc"
  )
endif()
if(TFLITE_ENABLE_XNNPACK)
  find_package(xnnpack REQUIRED)
  populate_tflite_source_vars("delegates/xnnpack"
    TFLITE_DELEGATES_XNNPACK_SRCS
    FILTER ".*(_test|_tester)\\.(cc|h)"
  )
  list(APPEND TFLITE_TARGET_DEPENDENCIES
    XNNPACK
  )
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
if(TFLITE_ENABLE_RUY)
  list(APPEND TFLITE_TARGET_PUBLIC_OPTIONS "-DTFLITE_WITH_RUY")
endif()
populate_tflite_source_vars("kernels"
  TFLITE_KERNEL_SRCS
  FILTER ".*(_test_util_internal|test_main)\\.(cc|h)"
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

# Common include directories
include_directories(
  BEFORE
   "${TENSORFLOW_SOURCE_DIR}"
   "${TFLITE_FLATBUFFERS_SCHEMA_DIR}"
)

# TFLite library
add_library(tensorflowlite
  ${TFLITE_CORE_API_SRCS}
  ${TFLITE_CORE_SRCS}
  ${TFLITE_C_SRCS}
  ${TFLITE_DELEGATES_FLEX_SRCS}
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
)
target_link_libraries(tensorflowlite
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
    ${TFLITE_TARGET_DEPENDENCIES}
)
target_compile_options(tensorflowlite
  PUBLIC ${TFLITE_TARGET_PUBLIC_OPTIONS}
  PRIVATE ${TFLITE_TARGET_PRIVATE_OPTIONS}
)
add_library(tensorflow::tensorflowlite ALIAS tensorflowlite)

# Benchmark Tool
populate_source_vars("${TFLITE_SOURCE_DIR}/tools/benchmark"
  TFLITE_BENCHMARK_SRCS
  FILTER "(_test|_plus_flex_main|_performance_options_main)\\.cc$"
)
list(APPEND TFLITE_BENCHMARK_SRCS
  ${TF_SOURCE_DIR}/core/util/stats_calculator.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
  ${TFLITE_SOURCE_DIR}/profiling/platform_profiler.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summarizer.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summary_formatter.cc
  ${TFLITE_SOURCE_DIR}/profiling/time.cc
  ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/default_execution_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/utils.cc
  ${TFLITE_SOURCE_DIR}/tools/optimize/sparsity/format_converter.cc
  ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
)

list(APPEND TFLITE_BENCHMARK_LIBS
  tensorflowlite
  ${CMAKE_DL_LIBS}
)

if(TFLITE_ENABLE_XNNPACK)
  list(APPEND TFLITE_BENCHMARK_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
  )
else()
  set(TFLITE_BENCHMARK_CC_OPTIONS "-DTFLITE_WITHOUT_XNNPACK")
endif()  # TFLITE_ENABLE_XNNPACK

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  list(APPEND TFLITE_BENCHMARK_SRCS
    ${TFLITE_SOURCE_DIR}/profiling/atrace_profiler.cc
    ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
  )
  list(APPEND TFLITE_BENCHMARK_LIBS
    ${ANDROID_LOG_LIB}
    absl::strings
  )
endif()  # Android

add_executable(benchmark_model
  EXCLUDE_FROM_ALL
  ${TFLITE_BENCHMARK_SRCS}
)
target_compile_options(benchmark_model
  PRIVATE
    ${TFLITE_BENCHMARK_CC_OPTIONS}
)
target_link_libraries(benchmark_model
    ${TFLITE_BENCHMARK_LIBS}
)
