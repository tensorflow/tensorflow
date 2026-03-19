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

include(ExternalProject)

if(TARGET neon2sse OR neon2sse_POPULATED)
  return()
endif()

set(NEON2SSE_URL
  https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/ARM_NEON_2_x86_SSE/archive/a15b489e1222b2087007546b4912e21293ea86ff.tar.gz
)
OverridableFetchContent_Declare(
  neon2sse
  URL "${NEON2SSE_URL}"
  # Sync with tensorflow/workspace2.bzl
  URL_HASH SHA256=019fbc7ec25860070a1d90e12686fc160cfb33e22aa063c80f52b363f1361e9d
  LICENSE_FILE LICENSE
  LICENSE_URL "${NEON2SSE_URL}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/neon2sse"
)

OverridableFetchContent_GetProperties(neon2sse)
if(NOT neon2sse_POPULATED)
  OverridableFetchContent_Populate(neon2sse)
endif()

add_subdirectory(
  "${neon2sse_SOURCE_DIR}"
  "${neon2sse_BINARY_DIR}"
)
