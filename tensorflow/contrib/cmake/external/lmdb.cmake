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

set(lmdb_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/lmdb)
set(lmdb_URL https://mirror.bazel.build/github.com/LMDB/lmdb/archive/LMDB_0.9.19.tar.gz)
set(lmdb_HASH SHA256=108532fb94c6f227558d45be3f3347b52539f0f58290a7bb31ec06c462d05326)
set(lmdb_BUILD ${CMAKE_BINARY_DIR}/lmdb/src/lmdb)
set(lmdb_INSTALL ${CMAKE_BINARY_DIR}/lmdb/install)

if(WIN32)
    set(lmdb_STATIC_LIBRARIES ${lmdb_INSTALL}/lib/lmdb.lib)
else()
    set(lmdb_STATIC_LIBRARIES ${lmdb_INSTALL}/lib/liblmdb.a)
endif()

ExternalProject_Add(lmdb
    PREFIX lmdb
    URL ${lmdb_URL}
    URL_HASH ${lmdb_HASH}
    BUILD_BYPRODUCTS ${lmdb_STATIC_LIBRARIES}
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/patches/lmdb/CMakeLists.txt ${lmdb_BUILD}
    INSTALL_DIR ${lmdb_INSTALL}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${tensorflow_ENABLE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:STRING=${lmdb_INSTALL}
    GIT_SHALLOW 1
    GIT_PROGRESS 1
)

set(lmdb_HEADERS
    "${lmdb_INSTALL}/include/lmdb.h"
    "${lmdb_INSTALL}/include/midl.h"
)

## put lmdb includes in the directory where they are expected
add_custom_target(lmdb_create_destination_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${lmdb_INCLUDE_DIR}
    DEPENDS lmdb)

add_custom_target(lmdb_copy_headers_to_destination
    DEPENDS lmdb_create_destination_dir)

foreach(header_file ${lmdb_HEADERS})
  add_custom_command(TARGET lmdb_copy_headers_to_destination PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${lmdb_INCLUDE_DIR}/)
endforeach()
