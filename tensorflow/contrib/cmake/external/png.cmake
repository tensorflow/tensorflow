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

set(png_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/png_archive)
set(png_URL https://mirror.bazel.build/github.com/glennrp/libpng/archive/v1.6.34.tar.gz)
set(png_HASH SHA256=e45ce5f68b1d80e2cb9a2b601605b374bdf51e1798ef1c2c2bd62131dfcf9eef)
set(png_BUILD ${CMAKE_BINARY_DIR}/png/src/png)
set(png_INSTALL ${CMAKE_BINARY_DIR}/png/install)

if(WIN32)
  if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set(png_STATIC_LIBRARIES 
      debug ${CMAKE_BINARY_DIR}/png/install/lib/libpng16_staticd.lib
      optimized ${CMAKE_BINARY_DIR}/png/install/lib/libpng16_static.lib)
  else()
    if(CMAKE_BUILD_TYPE EQUAL Debug)
      set(png_STATIC_LIBRARIES 
        ${CMAKE_BINARY_DIR}/png/install/lib/libpng16_staticd.lib)
    else()
      set(png_STATIC_LIBRARIES 
        ${CMAKE_BINARY_DIR}/png/install/lib/libpng16_static.lib)
    endif()
  endif()
else()
  set(png_STATIC_LIBRARIES ${CMAKE_BINARY_DIR}/png/install/lib/libpng16.a)
endif()

set(png_HEADERS
    "${png_INSTALL}/include/libpng16/png.h"
    "${png_INSTALL}/include/libpng16/pngconf.h"
    "${png_INSTALL}/include/libpng16/pnglibconf.h"
)

ExternalProject_Add(png
    PREFIX png
    DEPENDS zlib
    URL ${png_URL}
    URL_HASH ${png_HASH}
    BUILD_BYPRODUCTS ${png_STATIC_LIBRARIES}
    INSTALL_DIR ${png_INSTALL}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${tensorflow_ENABLE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:STRING=${png_INSTALL}
	-DZLIB_ROOT:STRING=${ZLIB_INSTALL}
  -DPNG_TESTS:BOOL=OFF
)

## put png includes in the directory where they are expected
add_custom_target(png_create_destination_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${png_INCLUDE_DIR}
    DEPENDS png)

add_custom_target(png_copy_headers_to_destination
    DEPENDS png_create_destination_dir)

foreach(header_file ${png_HEADERS})
  add_custom_command(TARGET png_copy_headers_to_destination PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${png_INCLUDE_DIR}/)
endforeach()
