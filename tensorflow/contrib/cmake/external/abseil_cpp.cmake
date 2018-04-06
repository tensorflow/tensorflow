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

set(abseil_cpp_URL https://github.com/abseil/abseil-cpp.git)
set(abseil_cpp_TAG "720c017e30339fd1786ce4aac68bc8559736e53f")
set(abseil_cpp_BUILD ${CMAKE_CURRENT_BINARY_DIR}/abseil_cpp/src/abseil_cpp)
set(abseil_cpp_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/abseil_cpp/src/abseil_cpp)

# DO NOT SUBMIT verify outputs
if(WIN32)
    if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
        set(abseil_cpp_STATIC_LIBRARIES ${abseil_cpp_BUILD}/$(Configuration)/abseil_cpp.lib)
    else()
        set(abseil_cpp_STATIC_LIBRARIES ${abseil_cpp_BUILD}/abseil_cpp.lib)
    endif()
else()
    set(abseil_cpp_STATIC_LIBRARIES ${abseil_cpp_BUILD}/libabseil_cpp.a)
endif()

set(abseil_cpp_HEADERS
    "$(abseil_cpp_INCLUDE_DIR)/*/*.h"
)

ExternalProject_Add(abseil_cpp
    PREFIX abseil_cpp
    GIT_REPOSITORY ${abseil_cpp_URL}
    GIT_TAG ${abseil_cpp_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${abseil_cpp_STATIC_LIBRARIES}
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${tensorflow_ENABLE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DSNAPPY_BUILD_TESTS:BOOL=OFF
)
