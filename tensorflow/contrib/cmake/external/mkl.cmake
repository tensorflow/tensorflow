# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# NOTE: Different from mkldnn.cmake, this file is meant to download mkl libraries
set(mkl_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/include)
set(mkl_BIN_DIRS ${CMAKE_CURRENT_BINARY_DIR}/mkl/bin)
set(mkl_WIN mklml_win_2018.0.3.20180406.zip) # match for v0.14
set(mkl_MAC mklml_mac_2018.0.3.20180406.tgz)
set(mkl_LNX mklml_lnx_2018.0.3.20180406.tgz)
set(mkl_TAG v0.14)
set(mkl_URL https://github.com/intel/mkl-dnn/releases)

if (WIN32)
  set(mkl_DOWNLOAD_URL ${mkl_URL}/download/${mkl_TAG}/${mkl_WIN})
  list(APPEND mkl_STATIC_LIBRARIES
    ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/mklml.lib)
  list(APPEND mkl_STATIC_LIBRARIES
    ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libiomp5md.lib)
  list(APPEND mkl_SHARED_LIBRARIES
    ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/mklml.dll)
  list(APPEND mkl_SHARED_LIBRARIES
    ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libiomp5md.dll)
elseif (UNIX)
  set(mkl_DOWNLOAD_URL ${mkl_URL}/download/${mkl_TAG}/${mkl_LNX})
  list(APPEND mkl_SHARED_LIBRARIES
    ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libiomp5.so)
  list(APPEND mkl_SHARED_LIBRARIES
    ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libmklml_gnu.so)
  list(APPEND mkl_SHARED_LIBRARIES
    ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libmklml_intel.so)
elseif (APPLE)
  set(mkl_DOWNLOAD_URL ${mkl_URL}/download/${mkl_TAG}/${mkl_MAC})
  #TODO need more information
endif ()

ExternalProject_Add(mkl
    PREFIX mkl
    URL ${mkl_DOWNLOAD_URL}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    GIT_SHALLOW 1
    GIT_PROGRESS 1
)

# put mkl dynamic libraries in one bin directory
add_custom_target(mkl_create_destination_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${mkl_BIN_DIRS}
  DEPENDS mkl)

add_custom_target(mkl_copy_shared_to_destination DEPENDS mkl_create_destination_dir)

foreach(dll_file ${mkl_SHARED_LIBRARIES})
  add_custom_command(TARGET mkl_copy_shared_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dll_file} ${mkl_BIN_DIRS})
endforeach()
