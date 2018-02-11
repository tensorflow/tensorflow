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

set(mkldnn_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/mkldnn/src/mkldnn/include)
set(mkldnn_URL https://github.com/01org/mkl-dnn.git)
set(mkldnn_BUILD ${CMAKE_CURRENT_BINARY_DIR}/mkldnn/src/mkldnn/src)
set(mkldnn_TAG 3063b2e4c943983f6bf5f2fb9a490d4a998cd291)

if(WIN32)
  if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set(mkldnn_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/mkldnn/src/mkldnn/src/Release/mkldnn.lib)
  else()
    set(mkldnn_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/mkldnn/src/mkldnn/src/mkldnn.lib)
  endif()
else()
    set(mkldnn_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/mkldnn/src/mkldnn/src/libmkldnn.a)
endif()

ExternalProject_Add(mkldnn
    PREFIX mkldnn
    GIT_REPOSITORY ${mkldnn_URL}
    GIT_TAG ${mkldnn_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${mkldnn_STATIC_LIBRARIES}
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DMKLINC:STRING=${MKL_INCLUDE_DIRS}
)
