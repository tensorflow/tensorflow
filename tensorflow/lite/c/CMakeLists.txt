#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.16)
project(tensorflow-lite-c C CXX)

option(TFLITE_C_BUILD_SHARED_LIBS "Build shared libraries" ON)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if (NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../"
    ABSOLUTE
  )
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 11)

if(CMAKE_SYSTEM_NAME MATCHES "Windows"
    AND (MSVC AND (CMAKE_SIZEOF_VOID_P EQUAL 4)))
  message("Disabling MSVC /O2 optimization for Win32")
  set(CompFlags
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS_RELWITHDEBINFO
  )
  foreach (CompFlag ${CompFlags})
    string(REGEX REPLACE "(\/Ob. )" "" ${CompFlag} "${${CompFlag}}")
    string(REPLACE "/O2" "/O1" ${CompFlag} "${${CompFlag}}")
    list(REMOVE_DUPLICATES ${CompFlag})
    set(${CompFlag} "${${CompFlag}}" CACHE INTERNAL "")
  endforeach()
endif()

set(TFLITE_C_LIBTYPE STATIC)
if (TFLITE_C_BUILD_SHARED_LIBS)
  set(TFLITE_C_LIBTYPE SHARED)
endif()

add_library(tensorflowlite_c ${TFLITE_C_LIBTYPE}
  builtin_op_data.h
  common.h
  common.c
  c_api_types.h
  c_api.h
  c_api.cc
  c_api_experimental.h
  c_api_experimental.cc
  c_api_internal.h
)

if (TFLITE_C_BUILD_SHARED_LIBS)
  if (WIN32)
    target_compile_definitions(tensorflowlite_c PRIVATE TFL_COMPILE_LIBRARY)
  elseif (APPLE)
    target_link_options(tensorflowlite_c PRIVATE "-Wl,-exported_symbols_list,${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/c/exported_symbols.lds")
  else ()
    target_link_options(tensorflowlite_c PRIVATE "-Wl,--version-script,${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/c/version_script.lds")
  endif()
endif()

target_link_libraries(tensorflowlite_c
  tensorflow-lite
)
