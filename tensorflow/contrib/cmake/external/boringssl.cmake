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

set(boringssl_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/boringssl/src/boringssl/include)
#set(boringssl_EXTRA_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/boringssl/src)
set(boringssl_URL https://boringssl.googlesource.com/boringssl)
set(boringssl_TAG 7f8c553d7f4db0a6ce727f2986d41bf8fe8ec4bf)
set(boringssl_BUILD ${CMAKE_BINARY_DIR}/boringssl/src/boringssl-build)
#set(boringssl_LIBRARIES ${boringssl_BUILD}/obj/so/libboringssl.so)
set(boringssl_STATIC_LIBRARIES
    ${boringssl_BUILD}/ssl/libssl.a
    ${boringssl_BUILD}/crypto/libcrypto.a
    ${boringssl_BUILD}/decrepit/libdecrepit.a
)
set(boringssl_INCLUDES ${boringssl_BUILD})

set(boringssl_HEADERS
    "${boringssl_INCLUDE_DIR}/include/*.h"
)

ExternalProject_Add(boringssl
    PREFIX boringssl
    GIT_REPOSITORY ${boringssl_URL}
    GIT_TAG ${boringssl_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    # BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${boringssl_STATIC_LIBRARIES}
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${tensorflow_ENABLE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
    GIT_SHALLOW 1
    GIT_PROGRESS 1
)
