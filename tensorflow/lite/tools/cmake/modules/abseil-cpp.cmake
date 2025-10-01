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

# Use absl_base as a proxy for the project being included.
if(TARGET absl_base OR abseil-cpp_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  abseil-cpp
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp
  # Sync with tensorflow/third_party/absl/workspace.bzl
  GIT_TAG 987c57f325f7fa8472fa84e1f885f7534d391b0d
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/abseil-cpp"
)
OverridableFetchContent_GetProperties(abseil-cpp)
if(NOT abseil-cpp_POPULATED)
  OverridableFetchContent_Populate(abseil-cpp)
endif()

set(ABSL_USE_GOOGLETEST_HEAD OFF CACHE BOOL "Disable googletest")
set(ABSL_RUN_TESTS OFF CACHE BOOL "Disable build of ABSL tests")
add_subdirectory(
  "${abseil-cpp_SOURCE_DIR}"
  "${abseil-cpp_BINARY_DIR}"
)

