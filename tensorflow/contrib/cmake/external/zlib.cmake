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
if (systemlib_ZLIB)
  find_package(PkgConfig)
  pkg_search_module(ZLIB REQUIRED zlib)
  set(zlib_INCLUDE_DIR ${ZLIB_INCLUDE_DIRS})
  set(ADD_LINK_DIRECTORY ${ADD_LINK_DIRECTORY} ${ZLIB_LIBRARY_DIRS})
  set(ADD_CFLAGS ${ADD_CFLAGS} ${ZLIB_CFLAGS_OTHER})

  # To meet DEPENDS zlib from other projects.
  # If we hit this line, zlib is already built and installed to the system.
  add_custom_target(zlib)
  add_custom_target(zlib_copy_headers_to_destination)

else (systemlib_ZLIB)
  include (ExternalProject)

  set(zlib_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/zlib_archive)
  set(ZLIB_URL https://github.com/madler/zlib)
  set(ZLIB_BUILD ${CMAKE_CURRENT_BINARY_DIR}/zlib/src/zlib)
  set(ZLIB_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/zlib/install)
  set(ZLIB_TAG 50893291621658f355bc5b4d450a8d06a563053d)

  if(WIN32)
    if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
      set(zlib_STATIC_LIBRARIES
          debug ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib/zlibstaticd.lib
          optimized ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib/zlibstatic.lib)
    else()
      if(CMAKE_BUILD_TYPE EQUAL Debug)
        set(zlib_STATIC_LIBRARIES
            ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib/zlibstaticd.lib)
      else()
        set(zlib_STATIC_LIBRARIES
            ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib/zlibstatic.lib)
      endif()
    endif()
  else()
    set(zlib_STATIC_LIBRARIES
        ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib/libz.a)
  endif()

  set(ZLIB_HEADERS
      "${ZLIB_INSTALL}/include/zconf.h"
      "${ZLIB_INSTALL}/include/zlib.h"
  )

  ExternalProject_Add(zlib
      PREFIX zlib
      GIT_REPOSITORY ${ZLIB_URL}
      GIT_TAG ${ZLIB_TAG}
      INSTALL_DIR ${ZLIB_INSTALL}
      BUILD_IN_SOURCE 1
      BUILD_BYPRODUCTS ${zlib_STATIC_LIBRARIES}
      DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
      CMAKE_CACHE_ARGS
          -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${tensorflow_ENABLE_POSITION_INDEPENDENT_CODE}
          -DCMAKE_BUILD_TYPE:STRING=Release
          -DCMAKE_INSTALL_PREFIX:STRING=${ZLIB_INSTALL}
  )

  # put zlib includes in the directory where they are expected
  add_custom_target(zlib_create_destination_dir
      COMMAND ${CMAKE_COMMAND} -E make_directory ${zlib_INCLUDE_DIR}
      DEPENDS zlib)

  add_custom_target(zlib_copy_headers_to_destination
      DEPENDS zlib_create_destination_dir)

  foreach(header_file ${ZLIB_HEADERS})
      add_custom_command(TARGET zlib_copy_headers_to_destination PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${zlib_INCLUDE_DIR})
  endforeach()
endif (systemlib_ZLIB)
