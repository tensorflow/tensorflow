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

#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/math_utils.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::quant::stablehlo {
namespace {

TEST(UtilsTest, QuantizeMultiplierNormalMultipliers) {
  int32_t quantized_fraction;
  int32_t shift;

  EXPECT_TRUE(succeeded(QuantizeMultiplier(1.2, quantized_fraction, shift)));
  EXPECT_EQ(quantized_fraction, 19661);
  EXPECT_EQ(shift, 1);

  EXPECT_TRUE(succeeded(QuantizeMultiplier(15.5, quantized_fraction, shift)));
  EXPECT_EQ(quantized_fraction, 31744);
  EXPECT_EQ(shift, 4);

  EXPECT_TRUE(succeeded(QuantizeMultiplier(1, quantized_fraction, shift)));
  EXPECT_EQ(quantized_fraction, 16384);
  EXPECT_EQ(shift, 1);
}

TEST(UtilsTest, QuantizeMultiplierExtremeMultipliers) {
  int32_t quantized_fraction;
  int32_t shift;

  EXPECT_TRUE(
      succeeded(QuantizeMultiplier(0.00001f, quantized_fraction, shift)));
  EXPECT_EQ(quantized_fraction, 0);
  EXPECT_EQ(shift, 0);

  EXPECT_TRUE(succeeded(QuantizeMultiplier(40000, quantized_fraction, shift)));
  EXPECT_EQ(quantized_fraction, 32767);
  EXPECT_EQ(shift, 14);
}

TEST(UtilsTest, QuantizeMultiplierInvalidArgument) {
  int32_t quantized_fraction;
  int32_t shift;

  EXPECT_FALSE(succeeded(QuantizeMultiplier(0, quantized_fraction, shift)));
}

}  // namespace
}  // namespace mlir::quant::stablehlo
