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

set(farmhash_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/farmhash_archive ${CMAKE_CURRENT_BINARY_DIR}/external/farmhash_archive/util)
set(farmhash_URL https://mirror.bazel.build/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz)
set(farmhash_HASH SHA256=6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0)
set(farmhash_BUILD ${CMAKE_CURRENT_BINARY_DIR}/farmhash/src/farmhash)
set(farmhash_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/farmhash/install)
set(farmhash_INCLUDES ${farmhash_BUILD})
set(farmhash_HEADERS
    "${farmhash_BUILD}/src/farmhash.h"
)

if(WIN32)
  set(farmhash_STATIC_LIBRARIES ${farmhash_INSTALL}/lib/farmhash.lib)

  ExternalProject_Add(farmhash
      PREFIX farmhash
      URL ${farmhash_URL}
      URL_HASH ${farmhash_HASH}
      DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
      BUILD_IN_SOURCE 1
      BUILD_BYPRODUCTS ${farmhash_STATIC_LIBRARIES}
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/patches/farmhash/CMakeLists.txt ${farmhash_BUILD}
      INSTALL_DIR ${farmhash_INSTALL}
      CMAKE_CACHE_ARGS
          -DCMAKE_BUILD_TYPE:STRING=Release
          -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
          -DCMAKE_INSTALL_PREFIX:STRING=${farmhash_INSTALL})
else()
  set(farmhash_STATIC_LIBRARIES ${farmhash_INSTALL}/lib/libfarmhash.a)

  ExternalProject_Add(farmhash
      PREFIX farmhash
      URL ${farmhash_URL}
      URL_HASH ${farmhash_HASH}
      DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
      BUILD_COMMAND $(MAKE)
      INSTALL_COMMAND $(MAKE) install
      CONFIGURE_COMMAND
          ${farmhash_BUILD}/configure
          --prefix=${farmhash_INSTALL}
          --libdir=${farmhash_INSTALL}/lib
          --enable-shared=yes
          CXXFLAGS=-fPIC)

endif()

# put farmhash includes in the directory where they are expected
add_custom_target(farmhash_create_destination_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${farmhash_INCLUDE_DIR}
    DEPENDS farmhash)

add_custom_target(farmhash_copy_headers_to_destination
    DEPENDS farmhash_create_destination_dir)

foreach(header_file ${farmhash_HEADERS})
    add_custom_command(TARGET farmhash_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${farmhash_INCLUDE_DIR}/)
endforeach()
