/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_FULLY_CONNECTED_4BIT_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_FULLY_CONNECTED_4BIT_H_

#include <stdint.h>

#include <cstdlib>

#if defined(FC_4BIT_SSE)
#include "tensorflow/lite/kernels/internal/optimized/4bit/sse_fully_connected.h"
#elif defined(FC_4BIT_NEON)
#include "tensorflow/lite/kernels/internal/optimized/4bit/neon_fully_connected.h"
#else
#include "tensorflow/lite/kernels/internal/optimized/4bit/fully_connected_reference.h"
#endif

namespace tflite {
namespace optimized_4bit {

// Define 4-bit filter block size: 4x32 (64 bytes)
constexpr int FilterWidth = 4;
constexpr int FilterDepth = 32;

struct OpData4Bit {
  int rows_right = 1;
  int batch_size = 0;
  bool needs_prepack = true;
  uint8_t* prepacked_cache = nullptr;
  ~OpData4Bit() { free(prepacked_cache); }
};

}  // namespace optimized_4bit
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_FULLY_CONNECTED_4BIT_H_
