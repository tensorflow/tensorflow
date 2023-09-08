#
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

project(ml_dtypes CXX)

set(ML_DTYPES_SOURCE_DIR "" CACHE PATH
  "Directory that contains the ml_dtypes project"
)
if(NOT ML_DTYPES_SOURCE_DIR)
  message(FATAL_ERROR "Must specify source directory")
endif()

add_library(ml_dtypes INTERFACE)
target_include_directories(ml_dtypes INTERFACE
  "${ML_DTYPES_SOURCE_DIR}"
  "${ML_DTYPES_SOURCE_DIR}/ml_dtypes")
file(GLOB ML_DTYPES_PUBLIC_HEADERS
  ${ML_DTYPES_SOURCE_DIR}/ml_dtypes/include/*.h)
set_target_properties(ml_dtypes PROPERTIES
  PUBLIC_HEADER "${ML_DTYPES_PUBLIC_HEADERS}")

if(TFLITE_ENABLE_INSTALL)
  install(
    TARGETS ml_dtypes
    EXPORT tensorflow-liteTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()
