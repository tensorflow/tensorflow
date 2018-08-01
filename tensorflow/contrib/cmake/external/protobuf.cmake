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
set(PROTOBUF_TAG v3.6.0)

if(WIN32)
  if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set(protobuf_STATIC_LIBRARIES
      debug ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/$(Configuration)/libprotobufd.lib
      optimized ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/$(Configuration)/libprotobuf.lib)
    set(PROTOBUF_PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/$(Configuration)/protoc.exe)
  else()
    if(CMAKE_BUILD_TYPE EQUAL Debug)
      set(protobuf_STATIC_LIBRARIES
        ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/libprotobufd.lib)
    else()
      set(protobuf_STATIC_LIBRARIES
        ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/libprotobuf.lib)
    endif()
    set(PROTOBUF_PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/protoc.exe)
  endif()

  # This section is to make sure CONFIGURE_COMMAND use the same generator settings
  set(PROTOBUF_GENERATOR_PLATFORM)
  if (CMAKE_GENERATOR_PLATFORM)
    set(PROTOBUF_GENERATOR_PLATFORM -A ${CMAKE_GENERATOR_PLATFORM})
  endif()
  set(PROTOBUF_GENERATOR_TOOLSET)
  if (CMAKE_GENERATOR_TOOLSET)
  set(PROTOBUF_GENERATOR_TOOLSET -T ${CMAKE_GENERATOR_TOOLSET})
  endif()
  set(PROTOBUF_ADDITIONAL_CMAKE_OPTIONS	-Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
    -G${CMAKE_GENERATOR} ${PROTOBUF_GENERATOR_PLATFORM} ${PROTOBUF_GENERATOR_TOOLSET})
  # End of section
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
    BUILD_BYPRODUCTS ${PROTOBUF_PROTOC_EXECUTABLE} ${protobuf_STATIC_LIBRARIES}
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf
    # SOURCE_SUBDIR cmake/ # Requires CMake 3.7, this will allow removal of CONFIGURE_COMMAND
    # CONFIGURE_COMMAND resets some settings made in CMAKE_CACHE_ARGS and the generator used
    CONFIGURE_COMMAND ${CMAKE_COMMAND} cmake/
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${tensorflow_ENABLE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -Dprotobuf_BUILD_TESTS:BOOL=OFF
        -DZLIB_ROOT=${ZLIB_INSTALL}
        ${PROTOBUF_ADDITIONAL_CMAKE_OPTIONS}
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${tensorflow_ENABLE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -Dprotobuf_BUILD_TESTS:BOOL=OFF
        -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
        -DZLIB_ROOT:STRING=${ZLIB_INSTALL}
    GIT_SHALLOW 1
    GIT_PROGRESS 1
)
