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

#include <cmath>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/optional.h"

namespace mlir {
namespace quant {

namespace {

double ComposeScale(const QuantizedMultiplier& input) {
  return input.first * exp2(-31 + input.second);
}

TEST(NumericalUtils, QuantizeMultiplier) {
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

TEST(NumericalUtils, ActivationRange) {
  // zero point = 0
  auto a =
      CalculateQuantizedRange(1e-6, 0, absl::nullopt, absl::nullopt, -128, 127);
  ASSERT_EQ(a.first, -128);
  ASSERT_EQ(a.second, 127);

  auto b = CalculateQuantizedRange(1e-6, 0, 0.0, absl::nullopt, -128, 127);
  ASSERT_EQ(b.first, 0);
  ASSERT_EQ(b.second, 127);

  auto c = CalculateQuantizedRange(1e-6, 0, -1.0, 1.0, -128, 127);
  ASSERT_EQ(c.first, -128);
  ASSERT_EQ(c.second, 127);

  auto d = CalculateQuantizedRange(1e-6, 0, 0.0, 6.0, -128, 127);
  ASSERT_EQ(d.first, 0);
  ASSERT_EQ(d.second, 127);

  // zero point = 100
  auto e = CalculateQuantizedRange(1e-6, 100, absl::nullopt, absl::nullopt,
                                   -128, 127);
  ASSERT_EQ(e.first, -128);
  ASSERT_EQ(e.second, 127);

  auto f = CalculateQuantizedRange(1e-6, 100, 0.0, absl::nullopt, -128, 127);
  ASSERT_EQ(f.first, 100);
  ASSERT_EQ(f.second, 127);

  auto g = CalculateQuantizedRange(1e-6, 100, -1.0, 1.0, -128, 127);
  ASSERT_EQ(g.first, -128);
  ASSERT_EQ(g.second, 127);

  auto h = CalculateQuantizedRange(1e-6, 100, 0.0, 6.0, -128, 127);
  ASSERT_EQ(h.first, 100);
  ASSERT_EQ(h.second, 127);

  // zero point = -100
  auto i = CalculateQuantizedRange(1e-6, -100, absl::nullopt, absl::nullopt,
                                   -128, 127);
  ASSERT_EQ(i.first, -128);
  ASSERT_EQ(i.second, 127);

  auto j = CalculateQuantizedRange(1e-6, -100, 0.0, absl::nullopt, -128, 127);
  ASSERT_EQ(j.first, -100);
  ASSERT_EQ(j.second, 127);

  auto k = CalculateQuantizedRange(1e-6, -100, -1.0, 1.0, -128, 127);
  ASSERT_EQ(k.first, -128);
  ASSERT_EQ(k.second, 127);

  auto l = CalculateQuantizedRange(1e-6, -100, 0.0, 6.0, -128, 127);
  ASSERT_EQ(l.first, -100);
  ASSERT_EQ(l.second, 127);
}

}  // namespace
}  // namespace quant
}  // namespace mlir
