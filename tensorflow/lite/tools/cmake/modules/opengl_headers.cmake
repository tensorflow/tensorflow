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

if(TARGET opengl_headers OR opengl_headers_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  opengl_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenGL-Registry.git
  # No reference in TensorFlow Bazel rule since it's used for GPU Delegate
  # build without using Android NDK.
  GIT_TAG 0cb0880d91581d34f96899c86fc1bf35627b4b81
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/opengl_headers"
  # Per https://www.khronos.org/legal/Khronos_Apache_2.0_CLA
  LICENSE_URL "https://www.apache.org/licenses/LICENSE-2.0.txt"
)

OverridableFetchContent_GetProperties(opengl_headers)
if(NOT opengl_headers)
  OverridableFetchContent_Populate(opengl_headers)
endif()

include_directories(
  AFTER
   "${opengl_headers_SOURCE_DIR}/api"
)
