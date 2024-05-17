#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project(farmhash CXX)

set(FARMHASH_SOURCE_DIR "" CACHE PATH
  "Directory that contains the farmhash project"
)
if(NOT FARMHASH_SOURCE_DIR)
  message(FATAL_ERROR "Must specify source directory")
endif()

# Transcribed from farmhash/src/Makefile.am
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
  "int main(int argc, char* argv[]) { return (int)__builtin_expect(0, 0); }"
  FARMHASH_HAS_BUILTIN_EXPECT
)

add_library(farmhash
  "${FARMHASH_SOURCE_DIR}/src/farmhash.cc"
  "${FARMHASH_SOURCE_DIR}/src/farmhash.h"
)
set_target_properties(farmhash PROPERTIES PUBLIC_HEADER "${FARMHASH_SOURCE_DIR}/src/farmhash.h")
target_include_directories(farmhash PUBLIC $<BUILD_INTERFACE:${FARMHASH_SOURCE_DIR}/src> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(NOT FARMHASH_HAS_BUILTIN_EXPECT)
  target_compile_definitions(farmhash PUBLIC -DFARMHASH_NO_BUILTIN_EXPECT)
endif()

if(TFLITE_ENABLE_INSTALL)
  install(
    TARGETS farmhash
    EXPORT tensorflow-liteTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()
