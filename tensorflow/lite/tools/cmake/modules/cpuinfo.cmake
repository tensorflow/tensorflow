#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

if(TARGET cpuinfo OR cpuinfo_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  cpuinfo
  GIT_REPOSITORY https://github.com/pytorch/cpuinfo
  # Sync with tensorflow/workspace2.bzl
  GIT_TAG 8a9210069b5a37dd89ed118a783945502a30a4ae
  GIT_PROGRESS TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/cpuinfo"
)
OverridableFetchContent_GetProperties(cpuinfo)
if(NOT cpuinfo_POPULATED)
  OverridableFetchContent_Populate(cpuinfo)
endif()

set(CPUINFO_SOURCE_DIR "${cpuinfo_SOURCE_DIR}" CACHE PATH "CPUINFO source directory")
set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "Disable cpuinfo command-line tools")
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Disable cpuinfo unit tests")
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Disable cpuinfo cpuinfo mock tests")
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Disable cpuinfo micro-benchmarks")

add_subdirectory(
  "${cpuinfo_SOURCE_DIR}"
  "${cpuinfo_BINARY_DIR}"
)
