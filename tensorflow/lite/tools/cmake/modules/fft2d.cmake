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

if(TARGET fft2d OR fft2d_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  fft2d
  URL https://storage.googleapis.com/mirror.tensorflow.org/github.com/petewarden/OouraFFT/archive/v1.0.tar.gz
  # Sync with tensorflow/workspace2.bzl
  URL_HASH SHA256=5f4dabc2ae21e1f537425d58a49cdca1c49ea11db0d6271e2a4b27e9697548eb
  SOURCE_DIR "${CMAKE_BINARY_DIR}/fft2d"
  LICENSE_FILE "readme2d.txt"
  LICENSE_URL "http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html"
)
OverridableFetchContent_GetProperties(fft2d)
if(NOT fft2d_POPULATED)
  OverridableFetchContent_Populate(fft2d)
endif()

set(FFT2D_SOURCE_DIR "${fft2d_SOURCE_DIR}" CACHE PATH "fft2d source")
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/fft2d"
  "${fft2d_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)
