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

if(TARGET ruy OR ruy_POPULATED)
  return()
endif()

include(utils)
get_dependency_tag("ruy" "${TF_SOURCE_DIR}/../third_party/ruy/workspace.bzl" RUY_TAG)

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  ruy
  GIT_REPOSITORY https://github.com/google/ruy
  GIT_TAG ${RUY_TAG}
  GIT_PROGRESS TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/ruy"
)
OverridableFetchContent_GetProperties(ruy)
if(NOT ruy_POPULATED)
  OverridableFetchContent_Populate(ruy)
endif()

set(RUY_SOURCE_DIR "${ruy_SOURCE_DIR}" CACHE PATH "RUY source directory")

add_subdirectory(
  "${ruy_SOURCE_DIR}"
  "${ruy_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)
