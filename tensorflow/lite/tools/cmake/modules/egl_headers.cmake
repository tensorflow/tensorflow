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

if(TARGET egl_headers OR egl_headers_POPULATED)
  return()
endif()

include(FetchContent)

OverridableFetchContent_Declare(
  egl_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/EGL-Registry.git
  GIT_TAG 649981109e263b737e7735933c90626c29a306f2
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/egl_headers"
)

OverridableFetchContent_GetProperties(egl_headers)
if(NOT egl_headers)
  OverridableFetchContent_Populate(egl_headers)
endif()

include_directories(
  AFTER
   "${egl_headers_SOURCE_DIR}/api"
)
