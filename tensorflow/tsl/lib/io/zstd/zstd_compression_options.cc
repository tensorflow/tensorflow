/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/tsl/lib/io/zstd/zstd_compression_options.h"

namespace tsl {
namespace io {

ZstdCompressionOptions::ZstdCompressionOptions() {
  input_buffer_size = ZSTD_CStreamInSize();
  output_buffer_size = ZSTD_DStreamOutSize();

  window_log = 0;                // default
  compression_level = 3;         // ZSTD_CLEVEL_DEFAULT
  compression_strategy = 0;      // default
  nb_workers = 0;                // single-threaded by default
  flush_mode = ZSTD_e_continue;  // ZSTD_e_continue
}

}  // namespace io
}  // namespace tsl
