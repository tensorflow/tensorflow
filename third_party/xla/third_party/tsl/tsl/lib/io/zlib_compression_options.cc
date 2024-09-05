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

#include "tsl/lib/io/zlib_compression_options.h"

#include <zlib.h>

namespace tsl {
namespace io {

ZlibCompressionOptions::ZlibCompressionOptions() {
  flush_mode = Z_NO_FLUSH;
  window_bits = MAX_WBITS;
  compression_level = Z_DEFAULT_COMPRESSION;
  compression_method = Z_DEFLATED;
  compression_strategy = Z_DEFAULT_STRATEGY;
}

}  // namespace io
}  // namespace tsl
