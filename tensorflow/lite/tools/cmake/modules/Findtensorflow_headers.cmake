#
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

# Downloads the few files from the TensorFlow repository that the TFLite runtime
# and the benchmark_model tool use (xla/tsl/..., tensorflow/core/util/... and
# third_party/fft2d/...), instead of cloning the whole TensorFlow repository.
#
# Sets TENSORFLOW_HEADERS_INCLUDE_DIR.

if(TENSORFLOW_HEADERS_INCLUDE_DIR)
  set(tensorflow_headers_FOUND TRUE)
  return()
endif()

set(_tensorflow_headers_base_url
  "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.21.0-rc0"
)
set(_tensorflow_headers_dir "${CMAKE_BINARY_DIR}/tensorflow_headers")
# <path in the TensorFlow repository>|<include path>|<sha256>
set(_tensorflow_headers_files
  "third_party/fft2d/fft.h|third_party/fft2d/fft.h|2db045d17dfd4b4fa5201e86a1653f0c0b7741e14c927f0c426007127109e825"
  "third_party/fft2d/fft2d.h|third_party/fft2d/fft2d.h|b24c63e77d5daf3affd7386085c41c046f3ad0c8dcab56a621fd8e84a5af5c1a"
  "third_party/xla/xla/tsl/framework/convolution/eigen_convolution_helpers.h|xla/tsl/framework/convolution/eigen_convolution_helpers.h|3fd52ebb0f14b9c4f3b3e225d1e7b29c712c3046d34e07936bfec0e0c1f41152"
  "third_party/xla/xla/tsl/framework/convolution/eigen_spatial_convolutions-inl.h|xla/tsl/framework/convolution/eigen_spatial_convolutions-inl.h|c0bec189723d52b4495dec62e197e2a4ffe7725bc313676984384e8b8fbb7a13"
  "third_party/xla/xla/tsl/lib/random/philox_random.h|xla/tsl/lib/random/philox_random.h|7a7659f95c59419373261af305736311bfd839b281259703b23f4df9b391a43e"
  "third_party/xla/xla/tsl/lib/random/random_distributions_utils.h|xla/tsl/lib/random/random_distributions_utils.h|1d65158a878510a1ec5835c5f26962b0ebe45bd56ecaec7e38314a8ac5e7a517"
  # Used by the benchmark_model tool.
  "tensorflow/core/util/stat_summarizer_options.h|tensorflow/core/util/stat_summarizer_options.h|e91f94b2fbb7e3a8d8b99664a1ff2443088ca1433b4372b1fcd2477b78492b3b"
  "tensorflow/core/util/stats_calculator.h|tensorflow/core/util/stats_calculator.h|8d80ca00444162a55a288b3b7369f314e5cf65a3258387aef4af1df39db08e0a"
  "third_party/xla/xla/tsl/util/stat_summarizer_options.h|xla/tsl/util/stat_summarizer_options.h|9f7d7cc5de38ae6e97a8982dc386a012ed70144c768ef2525fd25054de41c2b2"
  "third_party/xla/xla/tsl/util/stats_calculator.cc|xla/tsl/util/stats_calculator.cc|704c6d22240a521a7cbee591cd932121f0f6dcbaf179037758a7f3bd4d851e82"
  "third_party/xla/xla/tsl/util/stats_calculator.h|xla/tsl/util/stats_calculator.h|f5628ba1fbf39e7c4d2b280ff78c8daba40d9d462a97f8aa2abc0c5371b9f241"
)

foreach(_entry IN LISTS _tensorflow_headers_files)
  string(REPLACE "|" ";" _entry "${_entry}")
  list(LENGTH _entry _entry_len)
  if(NOT _entry_len EQUAL 3)
    message(FATAL_ERROR "Malformed entry in _tensorflow_headers_files: ${_entry}")
  endif()
  list(GET _entry 0 _src)
  list(GET _entry 1 _path)
  list(GET _entry 2 _sha256)
  set(_dst "${_tensorflow_headers_dir}/${_path}")
  if(EXISTS "${_dst}")
    file(SHA256 "${_dst}" _actual)
    if(_actual STREQUAL _sha256)
      continue()
    endif()
  endif()
  message(STATUS "Downloading ${_src}")
  file(DOWNLOAD "${_tensorflow_headers_base_url}/${_src}" "${_dst}"
    EXPECTED_HASH SHA256=${_sha256}
    STATUS _status
  )
  list(GET _status 0 _code)
  if(NOT _code EQUAL 0)
    message(FATAL_ERROR "Failed to download ${_src}: ${_status}")
  endif()
endforeach()

set(TENSORFLOW_HEADERS_INCLUDE_DIR "${_tensorflow_headers_dir}" CACHE PATH
  "Include directory with the TensorFlow headers used by TFLite"
)
set(tensorflow_headers_FOUND TRUE)
