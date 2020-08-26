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

if(TARGET flatbuffers OR flatbuffers_POPULATED)
  return()
endif()

include(FetchContent)

OverridableFetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers
  GIT_TAG v1.12.0 # TODO: What version does TFLite need?
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/flatbuffers"
)
OverridableFetchContent_GetProperties(flatbuffers)
if(NOT flatbuffers_POPULATED)
  OverridableFetchContent_Populate(flatbuffers)
endif()

# Required for Windows, since it has macros called min & max which
# clashes with std::min
add_definitions(-DNOMINMAX=1)
add_subdirectory(
  "${flatbuffers_SOURCE_DIR}"
  "${flatbuffers_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)
remove_definitions(-DNOMINMAX)
