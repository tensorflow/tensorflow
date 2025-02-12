/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_LIB_IO_SNAPPY_SNAPPY_COMPRESSION_OPTIONS_H_
#define XLA_TSL_LIB_IO_SNAPPY_SNAPPY_COMPRESSION_OPTIONS_H_

#include "xla/tsl/platform/types.h"

namespace tsl {
namespace io {

struct SnappyCompressionOptions {
  // Size of the buffer used for caching the data read from source file.
  int64_t input_buffer_size = 256 << 10;

  // Size of the sink buffer where the compressed/decompressed data produced by
  // snappy is cached.
  int64_t output_buffer_size = 256 << 10;
};

}  // namespace io
}  // namespace tsl

#endif  // XLA_TSL_LIB_IO_SNAPPY_SNAPPY_COMPRESSION_OPTIONS_H_
