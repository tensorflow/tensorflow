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

#ifndef TENSORFLOW_CORE_LIB_IO_LZ4_LZ4_COMPRESSION_OPTIONS_H_
#define TENSORFLOW_CORE_LIB_IO_LZ4_LZ4_COMPRESSION_OPTIONS_H_

#include <lz4frame.h>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace io {

class Lz4CompressionOptions {
 public:
  Lz4CompressionOptions();

  static Lz4CompressionOptions DEFAULT();

  int64 input_buffer_size;

  int64 output_buffer_size;

  int8 compression_level;

  int8 compression_strategy;

  // LZ4_EndDirective flush_mode;
};

inline Lz4CompressionOptions Lz4CompressionOptions::DEFAULT() {
  return Lz4CompressionOptions();
}

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_LZ4_LZ4_COMPRESSION_OPTIONS_H_
