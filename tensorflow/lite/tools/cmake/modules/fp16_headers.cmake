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

# Skip installation when xnnpack is used since it also has fp16 headers.
if(TARGET fp16_headers OR fp16_headers_POPULATED OR TFLITE_ENABLE_XNNPACK)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  fp16_headers
  GIT_REPOSITORY https://github.com/Maratyszcza/FP16
  # Sync with https://github.com/google/XNNPACK/blob/master/cmake/DownloadFP16.cmake
  GIT_TAG 0a92994d729ff76a58f692d3028ca1b64b145d91
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/fp16_headers"
)

OverridableFetchContent_GetProperties(fp16_headers)
if(NOT fp16_headers)
  OverridableFetchContent_Populate(fp16_headers)
endif()

include_directories(
  AFTER
   "${fp16_headers_SOURCE_DIR}/include"
)
