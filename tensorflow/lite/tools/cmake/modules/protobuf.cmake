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

include(OverridableFetchContent)

if(TARGET protobuf OR protobuf_POPULATED)
  return()
endif()

set(PROTOBUF_SRC_DIR "${CMAKE_BINARY_DIR}/protobuf")

OverridableFetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf
  # Sync with tensorflow/third_party/flatbuffers/workspace.bzl
  GIT_TAG v3.9.2
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SOURCE_DIR ${PROTOBUF_SRC_DIR}
)

OverridableFetchContent_GetProperties(protobuf)
if(NOT protobuf_POPULATED)
  message(STATUS "Cloning https://github.com/protocolbuffers/protobuf...")
  OverridableFetchContent_Populate(protobuf)
  add_subdirectory("${PROTOBUF_SRC_DIR}/cmake" "${CMAKE_BINARY_DIR}/protobuf-protoc")
endif()