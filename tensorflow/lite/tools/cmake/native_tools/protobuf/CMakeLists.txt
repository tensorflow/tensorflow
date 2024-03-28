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

# Builds the flatc compiler separately (e.g. for the purposes of TensorFlow Lite kernel tests cross-compilation
# during which a natively compiled 'flatc' binary is required.

cmake_minimum_required(VERSION 3.16)
project(protoc C CXX)

set(PROTOC_EXCLUDE_FROM_ALL FALSE)

# Install location of a (native) flatc compiler might be determined using the CMAKE_INSTALL_PREFIX variable.
# If the user provides such location, the path gets cached during the first CMake run and used later
# (for details see protobuf.cmake configuration).
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(PROTOC_INSTALL_PREFIX <INSTALL_DIR> CACHE PATH "Protoc installation directory")
else()
  set(PROTOC_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE PATH "Protoc installation directory")
endif()

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Turn off protobuf unit tests")
find_package(protobuf)
