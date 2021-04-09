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

if(TARGET nsync OR nsync_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  nsync
  GIT_REPOSITORY https://github.com/google/nsync.git
  GIT_TAG 1.22.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/nsync"
)
OverridableFetchContent_GetProperties(nsync)
if(NOT nsync_POPULATED)
  OverridableFetchContent_Populate(nsync)
endif()

option(NSYNC_ENABLE_TESTS OFF)

add_subdirectory(
  "${nsync_SOURCE_DIR}"
  "${nsync_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)

target_include_directories(nsync_cpp PUBLIC ${nsync_SOURCE_DIR}/public)
