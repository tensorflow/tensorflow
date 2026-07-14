#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

if(TARGET google_benchmark OR google_benchmark_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  google_benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.7.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/google_benchmark"
)
OverridableFetchContent_GetProperties(google_benchmark)
if(NOT google_benchmark_POPULATED)
  OverridableFetchContent_Populate(google_benchmark)
endif()

option(HAVE_GNU_POSIX_REGEX OFF)

add_subdirectory(
  "${google_benchmark_SOURCE_DIR}"
  "${google_benchmark_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)

include_directories(
  AFTER
  "${google_benchmark_SOURCE_DIR}/include"
)
