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

#include "tensorflow/compiler/mlir/lite/quantization/numerical_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace quant {

namespace {

double ComposeScale(const QuantizedMultiplier& input) {
  return input.first * std::exp2(-31 + input.second);
}

TEST(DecomposeScale, QuantizeMultiplier) {
  // Decompose multiplier larger than 1.
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0e6)), 1.0e6);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0e3)), 1.0e3);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(10.)), 10.);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(5.)), 5.);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(2.)), 2.);

  // Decompose multiplier between 1.0 and 1e-6.
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(0.0)), 0.0);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0)), 1.0);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0e-1)), 1.0e-1);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0e-2)), 1.0e-2);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0e-3)), 1.0e-3);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0e-4)), 1.0e-4);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0e-5)), 1.0e-5);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0e-6)), 1.0e-6);

  // When scale is smaller than 1.0e-6, it is decomposed to {0, 0}.
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0e-7)), 0.0);
  ASSERT_FLOAT_EQ(ComposeScale(QuantizeMultiplier(1.0e-8)), 0.0);
}

}  // namespace
}  // namespace quant
}  // namespace mlir
