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

set(jemalloc_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc/include)
set(jemalloc_URL https://mirror.bazel.build/github.com/jemalloc/jemalloc-cmake/archive/jemalloc-cmake.4.3.1.tar.gz)
set(jemalloc_HASH SHA256=f9be9a05fe906deb5c1c8ca818071a7d2e27d66fd87f5ba9a7bf3750bcedeaf0)
set(jemalloc_BUILD ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc)

if (WIN32)
    set(jemalloc_INCLUDE_DIRS
        ${jemalloc_INCLUDE_DIRS} 
        ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc/include/msvc_compat
    )
    if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
        set(jemalloc_STATIC_LIBRARIES ${jemalloc_BUILD}/Release/jemalloc.lib)
    else()
        set(jemalloc_STATIC_LIBRARIES ${jemalloc_BUILD}/jemalloc.lib)
    endif()
else()
    set(jemalloc_STATIC_LIBRARIES ${jemalloc_BUILD}/Release/jemalloc.a)
endif()

ExternalProject_Add(jemalloc
    PREFIX jemalloc
    URL ${jemalloc_URL}
    URL_HASH ${jemalloc_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${jemalloc_STATIC_LIBRARIES}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --target jemalloc
    INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Skipping install step."
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -Dwith-jemalloc-prefix:STRING=jemalloc_
        -Dwithout-export:BOOL=ON
)
