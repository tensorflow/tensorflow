#
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

if(TARGET ml_dtypes OR ml_dtypes_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  ml_dtypes
  GIT_REPOSITORY https://github.com/jax-ml/ml_dtypes
  # Sync with tensorflow/third_party/py/ml_dtypes/workspace.bzl
  # Github link:
  # https://github.com/jax-ml/ml_dtypes/commit/6f02f77c4fa624d8b467c36d1d959a9b49b07900
  GIT_TAG 6f02f77c4fa624d8b467c36d1d959a9b49b07900
  # It's not currently possible to shallow clone with a GIT TAG
  # as cmake attempts to git checkout the commit hash after the clone
  # which doesn't work as it's a shallow clone hence a different commit hash.
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17770
  # GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/ml_dtypes"
)
OverridableFetchContent_GetProperties(ml_dtypes)
if(NOT ml_dtypes_POPULATED)
  OverridableFetchContent_Populate(ml_dtypes)
endif()

set(ML_DTYPES_SOURCE_DIR "${ml_dtypes_SOURCE_DIR}" CACHE PATH
  "Source directory for the CMake project."
)

add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/ml_dtypes"
  "${ml_dtypes_BINARY_DIR}"
)
