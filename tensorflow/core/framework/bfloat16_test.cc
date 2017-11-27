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

#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

TEST(Bfloat16Test, Simple) {
  bfloat16 a(12);
  // Floating point representation of 12: 0x41400000
  EXPECT_EQ(0x4140, a.value);
}

float BinaryToFloat(uint32_t sign, uint32_t exponent, uint32_t high_mantissa,
                    uint32_t low_mantissa) {
  return bit_cast<float>((sign << 31) + (exponent << 23) +
                         (high_mantissa << 16) + low_mantissa);
}

struct Bfloat16TestParam {
  float input;
  float expected;
};

class Bfloat16Test : public ::testing::Test,
                     public ::testing::WithParamInterface<Bfloat16TestParam> {};

TEST_P(Bfloat16Test, TruncateTest) {
  bfloat16 a(GetParam().input);
  if (std::isnan(GetParam().input)) {
    EXPECT_TRUE(std::isnan(float(a)) || std::isinf(float(a)));
    return;
  }
  EXPECT_EQ(GetParam().expected, float(a));
}

INSTANTIATE_TEST_CASE_P(
    Bfloat16Test_Instantiation, Bfloat16Test,
    ::testing::Values(
        Bfloat16TestParam{
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b1111010111000011),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(1, 0b10000000, 0b1001000, 0b1111010111000011),
            BinaryToFloat(1, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b1000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b11111111, 0b0000000, 0b0000000000000001),
            BinaryToFloat(0, 0b11111111, 0b0000000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b11111111, 0b1111111, 0b1111111111111111),
            BinaryToFloat(0, 0b11111111, 0b1111111, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(1, 0b10000000, 0b1001000, 0b1100000000000000),
            BinaryToFloat(1, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0100000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b1000000000000000),
            BinaryToFloat(0, 0b10000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b00000000, 0b1001000, 0b1000000000000000),
            BinaryToFloat(0, 0b00000000, 0b1001000, 0b0000000000000000)},
        Bfloat16TestParam{
            BinaryToFloat(0, 0b00000000, 0b1111111, 0b1100000000000000),
            BinaryToFloat(0, 0b00000000, 0b1111111, 0b0000000000000000)}));

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
