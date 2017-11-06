# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
include (ExternalProject)

set(PROTOBUF_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/src)
set(PROTOBUF_URL https://github.com/google/protobuf.git)
set(PROTOBUF_TAG b04e5cba356212e4e8c66c61bbe0c3a20537c5b9)

if(WIN32)
  set(protobuf_STATIC_LIBRARIES 
    debug ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/$(Configuration)/libprotobufd.lib
    optimized ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/$(Configuration)/libprotobuf.lib)
  set(PROTOBUF_PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/$(Configuration)/protoc.exe)
  set(PROTOBUF_ADDITIONAL_CMAKE_OPTIONS	-Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF -A x64)
else()
  set(protobuf_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/libprotobuf.a)
  set(PROTOBUF_PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/protoc)
endif()

ExternalProject_Add(protobuf
    PREFIX protobuf
    DEPENDS zlib
    GIT_REPOSITORY ${PROTOBUF_URL}
    GIT_TAG ${PROTOBUF_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf
    CONFIGURE_COMMAND ${CMAKE_COMMAND} cmake/
        -Dprotobuf_BUILD_TESTS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DZLIB_ROOT=${ZLIB_INSTALL}
        ${PROTOBUF_ADDITIONAL_CMAKE_OPTIONS}
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
		if(tensorflow_ENABLE_POSITION_INDEPENDENT_CODE)
			-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
		else()
			-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=OFF
		endif()
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DZLIB_ROOT:STRING=${ZLIB_INSTALL}
)
