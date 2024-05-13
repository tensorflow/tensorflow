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

if(TARGET gemmlowp OR gemmlowp_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  gemmlowp
  GIT_REPOSITORY https://github.com/google/gemmlowp
  # Sync with tensorflow/third_party/gemmlowp/workspace.bzl
  GIT_TAG 16e8662c34917be0065110bfcd9cc27d30f52fdf
  # It's not currently (cmake 3.17) possible to shallow clone with a GIT TAG
  # as cmake attempts to git checkout the commit hash after the clone
  # which doesn't work as it's a shallow clone hence a different commit hash.
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17770
  # GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/gemmlowp"
)

OverridableFetchContent_GetProperties(gemmlowp)
if(NOT gemmlowp_POPULATED)
  OverridableFetchContent_Populate(gemmlowp)
endif()

# gemmlowp creates a benchmark target if BUILD_TESTING is set,
# which clashes with a target created by google_benchmark.
# https://github.com/google/gemmlowp/blob/master/contrib/CMakeLists.txt#L85
set(BUILD_TESTING_TMP ${BUILD_TESTING})
set(BUILD_TESTING OFF)

set(GEMMLOWP_SOURCE_DIR "${gemmlowp_SOURCE_DIR}" CACHE PATH "Source directory")
add_subdirectory(
  "${gemmlowp_SOURCE_DIR}/contrib"
  "${gemmlowp_BINARY_DIR}"
)

set(BUILD_TESTING ${BUILD_TESTING_TMP})
