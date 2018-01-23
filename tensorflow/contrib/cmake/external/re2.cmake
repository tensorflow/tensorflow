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

set(re2_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/re2/install/include)
set(re2_URL https://github.com/google/re2)
set(re2_BUILD ${CMAKE_CURRENT_BINARY_DIR}/re2/src/re2)
set(re2_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/re2/install)
set(re2_TAG e7efc48)

if(WIN32)
  set(re2_STATIC_LIBRARIES ${re2_BUILD}/$(Configuration)/re2.lib)
else()
  set(re2_STATIC_LIBRARIES ${re2_BUILD}/libre2.a)
endif()

set(re2_HEADERS
    ${re2_INSTALL}/include/re2/re2.h
)

ExternalProject_Add(re2
    PREFIX re2
    GIT_REPOSITORY ${re2_URL}
    GIT_TAG ${re2_TAG}
    INSTALL_DIR ${re2_INSTALL}
    BUILD_IN_SOURCE 1
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${tensorflow_ENABLE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_INSTALL_PREFIX:STRING=${re2_INSTALL}
        -DRE2_BUILD_TESTING:BOOL=OFF
)
