/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/bfloat16.h"

#include "absl/base/casts.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

TEST(Bfloat16Test, DefaultValueIsZero) {
  EXPECT_EQ(0.0f, static_cast<float>(bfloat16()));
}

TEST(Bfloat16Test, RepresentableFloatsRoundTripViaBfloat16) {
  const std::vector<float> values = {
      -std::numeric_limits<float>::infinity(), -1.0, -0.5, -0.0, 0.0, 0.5, 1.0,
      std::numeric_limits<float>::infinity(),
  };
  for (float v : values) {
    EXPECT_EQ(v, static_cast<float>(static_cast<bfloat16>(v)));
  }
}

TEST(Bfloat16Test, Simple) {
  bfloat16 a(12);
  // Floating point representation of 12: 0x41400000
  EXPECT_EQ(0x4140, a.value);
}

float BinaryToFloat(uint32_t sign, uint32_t exponent, uint32_t high_mantissa,
                    uint32_t low_mantissa) {
  return absl::bit_cast<float>((sign << 31) + (exponent << 23) +
                               (high_mantissa << 16) + low_mantissa);
}

struct Bfloat16TestParam {
  float input;
  float expected_truncation;
  float expected_rounding;
};

class Bfloat16Test : public ::testing::Test,
                     public ::testing::WithParamInterface<Bfloat16TestParam> {};

TEST_P(Bfloat16Test, TruncateTest) {
  bfloat16 truncated = bfloat16::truncate_to_bfloat16((GetParam().input));

  if (std::isnan(GetParam().input)) {
    EXPECT_TRUE(std::isnan(float(truncated)) || std::isinf(float(truncated)));
    return;
  }
  EXPECT_EQ(GetParam().expected_truncation, float(truncated));

  bfloat16 rounded = bfloat16::round_to_bfloat16((GetParam().input));
  if (std::isnan(GetParam().input)) {
    EXPECT_TRUE(std::isnan(float(rounded)) || std::isinf(float(rounded)));
    return;
  }
  EXPECT_EQ(GetParam().expected_rounding, float(rounded));
}

INSTANTIATE_TEST_SUITE_P(
    Bfloat16Test_Instantiation, Bfloat16Test,
    ::testing::Values(
        Bfloat16TestParam{
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b1111010111000011),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001001, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(1, 0b10000000, 0b1001000, 0b1111010111000011),
            BinaryToFloat(1, 0b10000000, 0b1001000, 0b0000000000000000),
            BinaryToFloat(1, 0b10000000, 0b1001001, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b1000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b11111111, 0b0000000, 0b0000000000000001),
            BinaryToFloat(0, 0b11111111, 0b0000000, 0b0000000000000000),
            BinaryToFloat(0, 0b11111111, 0b1000000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b11111111, 0b1111111, 0b1111111111111111),
            BinaryToFloat(0, 0b11111111, 0b1111111, 0b0000000000000000),
            BinaryToFloat(0, 0b11111111, 0b1000000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(1, 0b10000000, 0b1001000, 0b1100000000000000),
            BinaryToFloat(1, 0b10000000, 0b1001000, 0b0000000000000000),
            BinaryToFloat(1, 0b10000000, 0b1001001, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0100000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b1000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b00000000, 0b1001000, 0b1000000000000000),
            BinaryToFloat(0, 0b00000000, 0b1001000, 0b0000000000000000),
            BinaryToFloat(0, 0b00000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b00000000, 0b1111111, 0b1100000000000000),
            BinaryToFloat(0, 0b00000000, 0b1111111, 0b0000000000000000),
            BinaryToFloat(0, 0b00000001, 0b0000000, 0b0000000000000000)}));

TEST(Bfloat16Test, Conversion) {
  float a[100];
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  bfloat16 b[100];
  float c[100];
  FloatToBFloat16(a, b, 100);
  BFloat16ToFloat(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    // The relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    EXPECT_LE(fabs(c[i] - a[i]) / a[i], 1.0 / 128);
  }
}

TEST(Bfloat16Test, Epsilon) {
  EXPECT_LT(1.0f, static_cast<float>(bfloat16::epsilon() + bfloat16(1.0f)));
  EXPECT_EQ(1.0f, static_cast<float>((bfloat16::epsilon() / bfloat16(2.0f)) +
                                     bfloat16(1.0f)));
}

TEST(Bfloat16Test, Negate) {
  EXPECT_EQ(-3.0f, static_cast<float>(-bfloat16(3.0f)));
  EXPECT_EQ(4.5f, static_cast<float>(-bfloat16(-4.5f)));
}

static void BM_FloatToBFloat16(int iters) {
  testing::StopTiming();
  static const int N = 32 << 20;
  const int64 tot = static_cast<int64>(iters) * N;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * (sizeof(float) + sizeof(bfloat16)));

  float* inp = new float[N];
  bfloat16* out = new bfloat16[N];

  testing::StartTiming();
  while (iters--) {
    FloatToBFloat16(inp, out, N);
  }
  delete[] inp;
  delete[] out;
}
BENCHMARK(BM_FloatToBFloat16);

static void BM_BFloat16ToFloat(int iters) {
  testing::StopTiming();
  static const int N = 32 << 20;
  const int64 tot = static_cast<int64>(iters) * N;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * (sizeof(float) + sizeof(bfloat16)));

  bfloat16* inp = new bfloat16[N];
  float* out = new float[N];

  testing::StartTiming();
  while (iters--) {
    BFloat16ToFloat(inp, out, N);
  }
  delete[] inp;
  delete[] out;
}
BENCHMARK(BM_BFloat16ToFloat);

}  // namespace
}  // namespace tensorflow
