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


# Builds the minimal Tensorflow Lite example.

cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

option(LINK_TFLITE_FLEX "Enable tensorflowlite_flex library linkage" OFF)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../"
    ABSOLUTE
  )
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 17)
add_executable(minimal
  minimal.cc
)

if(LINK_TFLITE_FLEX)
  find_library(TF_LIB_FLEX tensorflowlite_flex HINTS "${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow/lite/delegates/flex/")
  if(NOT TF_LIB_FLEX)
    message(FATAL_ERROR "tensorflowlite_flex library not found")
  endif()

  target_link_libraries(minimal
    tensorflow-lite
    -Wl,--no-as-needed # Add --no-as-needed since for some toolchains (e.g. Ubuntu) --as-needed is added by default.
    ${TF_LIB_FLEX}
  )
else()
  target_link_libraries(minimal
    tensorflow-lite
  )
endif()