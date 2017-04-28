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

set(jsoncpp_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/jsoncpp/src/jsoncpp)
#set(jsoncpp_EXTRA_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/jsoncpp/src)
set(jsoncpp_URL https://github.com/open-source-parsers/jsoncpp.git)
set(jsoncpp_TAG 4356d9b)
set(jsoncpp_BUILD ${CMAKE_CURRENT_BINARY_DIR}/jsoncpp/src/jsoncpp/src/lib_json)
set(jsoncpp_LIBRARIES ${jsoncpp_BUILD}/obj/so/libjsoncpp.so)
set(jsoncpp_INCLUDES ${jsoncpp_BUILD})

if(WIN32)
  set(jsoncpp_STATIC_LIBRARIES ${jsoncpp_BUILD}/${CMAKE_BUILD_TYPE}/jsoncpp.lib)
else()
  set(jsoncpp_STATIC_LIBRARIES ${jsoncpp_BUILD}/libjsoncpp.a)
endif()

# We only need jsoncpp.h in external/jsoncpp/jsoncpp/jsoncpp.h
# For the rest, we'll just add the build dir as an include dir.
set(jsoncpp_HEADERS
    "${jsoncpp_INCLUDE_DIR}/include/json/json.h"
)

ExternalProject_Add(jsoncpp
    PREFIX jsoncpp
    GIT_REPOSITORY ${jsoncpp_URL}
    GIT_TAG ${jsoncpp_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
	-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
)

