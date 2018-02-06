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

set(gif_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/gif_archive/giflib-5.1.4/)
set(gif_URL https://mirror.bazel.build/ufpr.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz)
set(gif_HASH SHA256=34a7377ba834397db019e8eb122e551a49c98f49df75ec3fcc92b9a794a4f6d1)
set(gif_INSTALL ${CMAKE_BINARY_DIR}/gif/install)
set(gif_BUILD ${CMAKE_BINARY_DIR}/gif/src/gif)


set(gif_HEADERS
    "${gif_INSTALL}/include/gif_lib.h"
)

if(WIN32)

  set(gif_STATIC_LIBRARIES ${gif_INSTALL}/lib/giflib.lib)

  ExternalProject_Add(gif
      PREFIX gif
      URL ${gif_URL}
      URL_HASH ${gif_HASH}
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/patches/gif/CMakeLists.txt ${gif_BUILD}
      INSTALL_DIR ${gif_INSTALL}
      DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
      CMAKE_CACHE_ARGS
          -DCMAKE_BUILD_TYPE:STRING=Release
          -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
          -DCMAKE_INSTALL_PREFIX:STRING=${gif_INSTALL}
  )

  ExternalProject_Add_Step(gif copy_unistd
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_SOURCE_DIR}/patches/gif/unistd.h ${gif_BUILD}/lib/unistd.h
      DEPENDEES patch
      DEPENDERS build
  )

else()

  set(gif_STATIC_LIBRARIES ${gif_INSTALL}/lib/libgif.a)
  set(ENV{CFLAGS} "$ENV{CFLAGS} -fPIC")

  ExternalProject_Add(gif
      PREFIX gif
      URL ${gif_URL}
      URL_HASH ${gif_HASH}
      INSTALL_DIR ${gif_INSTALL}
      DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
      BUILD_COMMAND $(MAKE)
      INSTALL_COMMAND $(MAKE) install
      CONFIGURE_COMMAND
          ${CMAKE_CURRENT_BINARY_DIR}/gif/src/gif/configure
          --with-pic
          --prefix=${gif_INSTALL}
          --libdir=${gif_INSTALL}/lib
         --enable-shared=yes
  )

endif()

# put gif includes in the directory where they are expected
add_custom_target(gif_create_destination_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${gif_INCLUDE_DIR}
    DEPENDS gif)

add_custom_target(gif_copy_headers_to_destination
    DEPENDS gif_create_destination_dir)

foreach(header_file ${gif_HEADERS})
    add_custom_command(TARGET gif_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${gif_INCLUDE_DIR}/)
endforeach()
