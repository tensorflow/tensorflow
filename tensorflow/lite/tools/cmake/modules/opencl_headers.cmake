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

if(TARGET opencl_headers OR opencl_headers_POPULATED)
  return()
endif()

include(FetchContent)

OverridableFetchContent_Declare(
  opencl_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers
  # Sync with tensorflow/third_party/opencl_headers/workspace.bzl
  GIT_TAG dcd5bede6859d26833cd85f0d6bbcee7382dc9b3
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/opencl_headers"
)

OverridableFetchContent_GetProperties(opencl_headers)
if(NOT opencl_headers)
  OverridableFetchContent_Populate(opencl_headers)
endif()

include_directories(
  AFTER
   "${opencl_headers_SOURCE_DIR}/"
)
