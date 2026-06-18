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
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/platform/bfloat16.h"

// This is a fuzzer for tensorflow::FloatToBFloat16 and
// tensorflow::BFloat16ToFloat.

namespace {

void FuzzTest(const std::vector<float>& float_originals) {
  const int32_t size = float_originals.size();
  std::vector<tensorflow::bfloat16> bfloats(size);
  std::vector<float> floats_converted(size);

  tensorflow::FloatToBFloat16(float_originals.data(), bfloats.data(), size);
  tensorflow::BFloat16ToFloat(bfloats.data(), floats_converted.data(), size);

  for (int i = 0; i < float_originals.size(); ++i) {
    // The relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    // Copied this logic from bfloat16_test.cc
    assert(fabs(floats_converted[i] - float_originals[i]) / float_originals[i] <
           1.0 / 128);
  }
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithDomains(fuzztest::ContainerOf<std::vector<float>>(
        fuzztest::InRange(1.0f, 1000.0f)));

}  // namespace
