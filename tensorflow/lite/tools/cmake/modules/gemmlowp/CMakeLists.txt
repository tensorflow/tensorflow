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

project(gemmlowp CXX)

option(GEMMLOWP_ADD_HEADERS_TO_TARGETS OFF
  "Whether to add sources to gemmlowp's interface library targets.
   This will cause all users of these libraries to also include these headers"
)

set(GEMMLOWP_SOURCE_DIR "" CACHE PATH
  "Directory that contains the gemmlowp project"
)
if(NOT GEMMLOWP_SOURCE_DIR)
  message(FATAL_ERROR "Must specify source directory")
endif()

# gemmlowp doesn't have a CMake project so this is transcribed from
# gemmlowp/BUILD.

file(GLOB GEMMLOWP_EIGHTBITINT_HEADERS
  "${GEMMLOWP_SOURCE_DIR}/eight_bit_int_gemm/*.h"
  )
file(GLOB GEMMLOWP_EIGHTBITINT_SOURCES
  "${GEMMLOWP_SOURCE_DIR}/eight_bit_int_gemm/*.cc"
)
file(GLOB GEMMLOWP_FIXEDPOINT_HEADERS "${GEMMLOWP_SOURCE_DIR}/fixedpoint/*.h")
file(GLOB GEMMLOWP_INTERNAL_HEADERS "${GEMMLOWP_SOURCE_DIR}/internal/*.h")
file(GLOB GEMMLOWP_META_HEADERS "${GEMMLOWP_SOURCE_DIR}/meta/*.h")
file(GLOB GEMMLOWP_PROFILING_HEADERS "${GEMMLOWP_SOURCE_DIR}/profiling/*.h")
file(GLOB GEMMLOWP_PUBLIC_HEADERS "${GEMMLOWP_SOURCE_DIR}/public/*.h")

set(GEMMLOWP_PRIVATE_HEADERS "")
list(APPEND GEMMLOWP_PRIVATE_HEADERS ${GEMMLOWP_FIXEDPOINT_HEADERS})
list(APPEND GEMMLOWP_PRIVATE_HEADERS ${GEMMLOWP_INTERNAL_HEADERS})

add_library(gemmlowp_private INTERFACE)
if(GEMMLOWP_ADD_HEADERS_TO_TARGETS)
  target_sources(gemmlowp_private INTERFACE ${GEMMLOWP_PRIVATE_HEADERS})
endif()
target_include_directories(gemmlowp_private INTERFACE "${GEMMLOWP_SOURCE_DIR}")

add_library(gemmlowp INTERFACE)
if(GEMMLOWP_ADD_HEADERS_TO_TARGETS)
  target_sources(gemmlowp INTERFACE ${GEMMLOWP_PUBLIC_HEADERS})
endif()
target_include_directories(gemmlowp INTERFACE "${GEMMLOWP_SOURCE_DIR}/public")
target_link_libraries(gemmlowp INTERFACE gemmlowp_private)

add_library(gemmlowp_eight_bit_int_gemm
  ${GEMMLOWP_EIGHTBITINT_SOURCES}
  ${GEMMLOWP_EIGHTBITINT_HEADERS}
)
target_include_directories(gemmlowp_eight_bit_int_gemm
  PUBLIC "${GEMMLOWP_SOURCE_DIR}/eight_bit_int_gemm"
)

add_library(gemmlowp_fixedpoint INTERFACE)
if(GEMMLOWP_ADD_HEADERS_TO_TARGETS)
  target_sources(gemmlowp_fixedpoint INTERFACE ${GEMMLOWP_FIXEDPOINT_HEADERS})
endif()
target_include_directories(gemmlowp_fixedpoint
  INTERFACE "${GEMMLOWP_SOURCE_DIR}/fixedpoint"
)
target_link_libraries(gemmlowp_fixedpoint INTERFACE gemmlowp_private)

add_library(gemmlowp_profiler INTERFACE)
if(GEMMLOWP_ADD_HEADERS_TO_TARGETS)
  target_sources(gemmlowp_profiler INTERFACE ${GEMMLOWP_PROFILING_HEADERS})
endif()
target_include_directories(gemmlowp_profiler
  INTERFACE "${GEMMLOWP_SOURCE_DIR}/profiling"
)


