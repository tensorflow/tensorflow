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

set(double_conversion_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/double_conversion/src/double_conversion)
set(double_conversion_URL https://github.com/google/double-conversion.git)
set(double_conversion_TAG 3992066a95b823efc8ccc1baf82a1cfc73f6e9b8)
set(double_conversion_BUILD ${double_conversion_INCLUDE_DIR})
set(double_conversion_LIBRARIES ${double_conversion_BUILD}/double-conversion/libdouble-conversion.so)
set(double_conversion_INCLUDES ${double_conversion_BUILD})

if(WIN32)
  set(double_conversion_STATIC_LIBRARIES ${double_conversion_BUILD}/$(Configuration)/double-conversion.lib)
else()
  set(double_conversion_STATIC_LIBRARIES ${double_conversion_BUILD}/libdouble-conversion.a)
endif()

set(double_conversion_HEADERS
    "${double_conversion_INCLUDE_DIR}/double-conversion/bignum-dtoa.h"
    "${double_conversion_INCLUDE_DIR}/double-conversion/cached-powers.h"
    "${double_conversion_INCLUDE_DIR}/double-conversion/double-conversion.h"
    "${double_conversion_INCLUDE_DIR}/double-conversion/fixed-dtoa.h"
    "${double_conversion_INCLUDE_DIR}/double-conversion/strtod.h"
    "${double_conversion_INCLUDE_DIR}/double-conversion/bignum.h"
    "${double_conversion_INCLUDE_DIR}/double-conversion/diy-fp.h"
    "${double_conversion_INCLUDE_DIR}/double-conversion/fast-dtoa.h"
    "${double_conversion_INCLUDE_DIR}/double-conversion/ieee.h"
    "${double_conversion_INCLUDE_DIR}/double-conversion/utils.h"
)

ExternalProject_Add(double_conversion
    PREFIX double_conversion
    GIT_REPOSITORY ${double_conversion_URL}
    GIT_TAG ${double_conversion_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    GIT_SHALLOW 1
    GIT_PROGRESS 1
)
