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


# Builds the Tensorflow Lite micro runtime.
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
if(CONFIG_TENSFLOW_LITE_MICRO)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_EXE_LINKER_FLAGS "--specs=nosys.specs" CACHE INTERNAL "")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(TFLITE_ENABLE_RESOURCE OFF)
set(TFLITE_ENABLE_NNAPI OFF)
set(TFLITE_ENABLE_MMAP OFF)
set(TFLITE_ENABLE_XNNPACK OFF)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release, for debug builds use"
    "'-DCMAKE_BUILD_TYPE=Debug'.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Double colon in target name means ALIAS or IMPORTED target.
cmake_policy(SET CMP0028 NEW)
# Enable MACOSX_RPATH (@rpath) for built dynamic libraries.
cmake_policy(SET CMP0042 NEW)
project(tensorflow-lite-micro C)
set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../"
    ABSOLUTE
  )
endif()
set(TF_SOURCE_DIR "${TENSORFLOW_SOURCE_DIR}/tensorflow")
set(TFLITE_SOURCE_DIR "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite")
set(TFLITE_MICRO_SOURCE_DIR "${TFLITE_SOURCE_DIR}/micro")
set(TF_THIRD_PATRY_INC_DIR "${TFLITE_SOURCE_DIR}/micro")


set(CMAKE_MODULE_PATH
  "${TFLITE_SOURCE_DIR}/tools/cmake/modules"
  ${CMAKE_MODULE_PATH}
)
set(CMAKE_PREFIX_PATH
  "${TFLITE_SOURCE_DIR}/tools/cmake/modules"
  ${CMAKE_PREFIX_PATH}
)

option(TFLITE_ENABLE_RESOURCE "Enable experimental support for resources" ON)
option(TFLITE_ENABLE_NNAPI "Enable NNAPI (Android only)." ON)

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

macro(populate_tflite_micro_source_vars RELATIVE_DIR SOURCES_VAR)
  populate_source_vars(
    "${TFLITE_MICRO_SOURCE_DIR}/${RELATIVE_DIR}" ${SOURCES_VAR} ${ARGN}
  )
endmacro()

# Find TensorFlow Lite dependencies.
find_package(fft2d REQUIRED)
find_package(flatbuffers REQUIRED)
find_package(gemmlowp REQUIRED)
find_package(ruy REQUIRED)
find_package(cmsis-nn REQUIRED)

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
populate_tflite_micro_source_vars("." TFLITE_MICRO_SRCS)
populate_tflite_source_vars("kernels"
  TFLITE_KERNEL_SRCS
  FILTER "(.*_test_util_internal|test_.*)\\.(cc|h)"
)

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
  FILTER "(.*_test_util_internal|test_.*)\\.(cc|h)"
)

populate_tflite_source_vars("micro/benchmarks" TFLITE_MICRO_BENCHMARKS_CC_SRCS)
populate_tflite_source_vars("micro/memory_planner" TFLITE_MICRO_MPLAN_CC_SRCS)

# populate_tflite_source_vars("kernels/internal" TFLITE_KERNEL_INTERNAL_SRCS)
# populate_tflite_source_vars("kernels/internal/optimized"
#  TFLITE_KERNEL_INTERNAL_OPT_SRCS
#)
#populate_tflite_source_vars("kernels/internal/optimized/integer_ops"
#  TFLITE_KERNEL_INTERNAL_OPT_INTEGER_OPS_SRCS
#)
#populate_tflite_source_vars("kernels/internal/optimized/sparse_ops"
#  TFLITE_KERNEL_INTERNAL_OPT_SPARSE_OPS_SRCS
#)
#populate_tflite_source_vars("kernels/internal/reference"
#  TFLITE_KERNEL_INTERNAL_REF_SRCS
#)
#populate_tflite_source_vars("kernels/internal/reference/integer_ops"
#  TFLITE_KERNEL_INTERNAL_REF_INTEGER_OPS_SRCS
#)
#set(TFLITE_PROFILER_SRCS ${TFLITE_SOURCE_DIR}/profiling/platform_profiler.cc)
# populate_tflite_source_vars("kernels/internal/reference/sparse_ops"
#   TFLITE_KERNEL_INTERNAL_REF_SPARSE_OPS_SRCS
# )

# Common include directories
set(TFLITE_INCLUDE_DIRS
  "${TENSORFLOW_SOURCE_DIR}"
  "${TFLITE_MICRO_SOURCE_DIR}"
  "${TFLITE_FLATBUFFERS_SCHEMA_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/../../ruy"
)
include_directories(
  BEFORE
    ${TFLITE_INCLUDE_DIRS}
)

message(STATUS " include path ${TFLITE_INCLUDE_DIRS}")

# TFLite micro library
add_library(tensorflow-lite-micro
  ${TFLITE_MICRO_SRCS}
  ${TFLITE_MICRO_BENCHMARKS_CC_SRCS}
  ${TFLITE_MICRO_MPLAN_CC_SRCS}
  ${TFLITE_SOURCE_DIR}/c/common.c
  ${TFLITE_SOURCE_DIR}/core/api/error_reporter.cc
  ${TFLITE_SOURCE_DIR}/core/api/flatbuffer_conversions.cc
  ${TFLITE_SOURCE_DIR}/core/api/op_resolver.cc
  ${TFLITE_SOURCE_DIR}/core/api/tensor_utils.cc
  ${TFLITE_SOURCE_DIR}/kernels/internal/quantization_util.cc
  ${TFLITE_SOURCE_DIR}/kernels/kernel_util.cc
  ${TFLITE_SOURCE_DIR}/schema/schema_utils.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/activations.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/add.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/arg_min_max.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/cast.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/ceil.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/circular_buffer.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/comparisons.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/concatenation.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/conv.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/conv_test_common.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/depthwise_conv.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/dequantize.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/detection_postprocess.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/elementwise.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/ethosu.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/exp.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/flexbuffers_generated_data.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/floor.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/fully_connected.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/fully_connected_common.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/hard_swish.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/kernel_runner.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/kernel_util.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/l2norm.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/logical.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/logistic.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/maximum_minimum.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/mul.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/neg.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/pack.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/pad.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/pooling.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/prelu.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/quantize.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/quantize_common.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/reduce.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/reshape.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/resize_nearest_neighbor.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/round.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/shape.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/softmax.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/softmax_common.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/split.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/split_v.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/strided_slice.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/sub.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/svdf.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/svdf_common.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/tanh.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/transpose_conv.cc
  ${TFLITE_MICRO_SOURCE_DIR}/kernels/unpack.cc
)

target_include_directories(tensorflow-lite-micro
  PUBLIC
    ${TFLITE_INCLUDE_DIRS}
)
target_link_libraries(tensorflow-lite-micro
  PUBLIC
    farmhash
    fft2d_fftsg2d
    flatbuffers
    gemmlowp
    ${TFLITE_TARGET_DEPENDENCIES}
)
target_compile_options(tensorflow-lite-micro
  PUBLIC ${TFLITE_TARGET_PUBLIC_OPTIONS}
  PRIVATE ${TFLITE_TARGET_PRIVATE_OPTIONS}
)
add_library(tensorflow::tensorflowlite-micro ALIAS tensorflow-lite-micro)

endif()
