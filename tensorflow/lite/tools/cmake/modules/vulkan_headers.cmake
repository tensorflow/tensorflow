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

if(TARGET vulkan_headers OR vulkan_headers_POPULATED)
  return()
endif()

include(FetchContent)

OverridableFetchContent_Declare(
  vulkan_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers
  # GIT_TAG must keep in sync with tensorflow/third_party/vulkan_headers/workspace.bzl
  GIT_TAG 0e57fc1cfa56a203efe43e4dfb9b3c9e9b105593
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/vulkan_headers"
)

OverridableFetchContent_GetProperties(vulkan_headers)
if(NOT vulkan_headers)
  OverridableFetchContent_Populate(vulkan_headers)
endif()

include_directories(
  AFTER
   "${vulkan_headers_SOURCE_DIR}/include"
)
