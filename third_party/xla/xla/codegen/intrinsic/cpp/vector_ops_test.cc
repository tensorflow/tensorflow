/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

#include <cstdint>

#include <gtest/gtest.h>

namespace xla::codegen {
namespace {

TEST(VectorOpsTest, SplatPreservesNegativeZero) {
  Vec4f v = Splat<Vec4f>(-0.0f);
  Vec4i bits = __builtin_bit_cast(Vec4i, v);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(bits[i], 0x80000000u);
  }
  float s = Splat<float>(-0.0f);
  EXPECT_EQ(__builtin_bit_cast(uint32_t, s), 0x80000000u);
}

TEST(VectorOpsTest, HornerPoly) {
  // P(x) = 2x^3 - 3x^2 + 5x - 7
  EXPECT_FLOAT_EQ(HornerPoly(0.0f, 2.0f, -3.0f, 5.0f, -7.0f), -7.0f);
  EXPECT_FLOAT_EQ(HornerPoly(2.0f, 2.0f, -3.0f, 5.0f, -7.0f), 7.0f);

  Vec4f x = {-1.0f, 0.0f, 1.0f, 2.0f};
  Vec4f y = HornerPoly(x, 2.0f, -3.0f, 5.0f, -7.0f);
  EXPECT_FLOAT_EQ(y[0], -17.0f);
  EXPECT_FLOAT_EQ(y[1], -7.0f);
  EXPECT_FLOAT_EQ(y[2], -3.0f);
  EXPECT_FLOAT_EQ(y[3], 7.0f);
}

}  // namespace
}  // namespace xla::codegen
