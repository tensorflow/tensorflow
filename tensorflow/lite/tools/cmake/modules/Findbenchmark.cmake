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

# grpc uses find_package in CONFIG mode for this package, so override the
# system installation and build from source instead.

if(TARGET benchmark OR benchmark_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark
  # Sync with tensorflow/third_party/benchmark/workspace.bzl
  GIT_TAG 0baacde3618ca617da95375e0af13ce1baadea47
  GIT_PROGRESS TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/benchmark"
)
OverridableFetchContent_GetProperties(benchmark)
if(NOT benchmark_POPULATED)
  OverridableFetchContent_Populate(benchmark)
endif()

set(BENCHMARK_SOURCE_DIR "${benchmark_SOURCE_DIR}" CACHE PATH "BENCHMARK source directory")

add_subdirectory(
  "${benchmark_SOURCE_DIR}"
  "${benchmark_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)

