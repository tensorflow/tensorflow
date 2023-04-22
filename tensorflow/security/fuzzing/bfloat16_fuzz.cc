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
#include <fuzzer/FuzzedDataProvider.h>

#include <cstdint>
#include <cstdlib>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/platform/test.h"

// This is a fuzzer for tensorflow::FloatToBFloat16 and
// tensorflow::BFloat16ToFloat.

namespace {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  FuzzedDataProvider fuzzed_data(data, size);

  const int array_size = 100;

  float float_originals[array_size];
  for (int i = 0; i < array_size; ++i) {
    float_originals[i] = fuzzed_data.ConsumeFloatingPointInRange(1.0f, 1000.0f);
  }
  tensorflow::bfloat16 bfloats[array_size];
  float floats_converted[array_size];

  tensorflow::FloatToBFloat16(float_originals, bfloats, array_size);
  tensorflow::BFloat16ToFloat(bfloats, floats_converted, array_size);

  for (int i = 0; i < array_size; ++i) {
    // The relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    // Copied this logic from bfloat16_test.cc
    assert(fabs(floats_converted[i] - float_originals[i]) / float_originals[i] <
           1.0 / 128);
  }

  return 0;
}

}  // namespace
