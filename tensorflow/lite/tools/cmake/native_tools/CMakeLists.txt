#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

# Builds the native (host) tools separately (e.g. for the purposes of 
# TensorFlow Lite kernel tests/TensorFlow Lite Eval Tools cross-compilation for which 
# natively compiled 'flatc'/'protoc' binary is required.

cmake_minimum_required(VERSION 3.16)
project(native-tools C CXX)

include(GNUInstallDirs)
include(CMakeDependentOption)

# Incorporate the tflite CMake modules into the CMAKE_MODULE_PATH
set(TFLITE_NATIVE_TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
get_filename_component(TFLITE_CMAKE_DIR ${TFLITE_NATIVE_TOOLS_DIR} DIRECTORY)
set(TFLITE_CMAKE_MODULE_PATH ${TFLITE_CMAKE_DIR}/modules)

list(APPEND CMAKE_MODULE_PATH ${TFLITE_CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH ${TFLITE_CMAKE_MODULE_PATH})

option(TFLITE_NATIVE_TOOLS_BUILD_FLATC "Build libprotobuf static library and protoc compiler" ON)
option(TFLITE_NATIVE_TOOLS_BUILD_PROTOC "Build flatc compiler" ON)

if (TFLITE_NATIVE_TOOLS_BUILD_FLATC)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers)
    # flatc compiler is installed during the build phase via ExternalProject_Add (see flatbuffers.cmake)
    # TODO: Use OverridableFetchContent instead of ExternalProject_Add for flatbuffers build
    # as there are no benefits of using the latter in context of TensorFlow Lite build.
endif()

if (TFLITE_NATIVE_TOOLS_BUILD_PROTOC)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/protobuf)
    install(TARGETS protoc libprotoc libprotobuf)
endif()