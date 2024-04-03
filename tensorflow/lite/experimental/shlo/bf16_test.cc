/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/shlo/bf16.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"

namespace shlo_ref {
namespace {

::testing::Matcher<BF16> MatchesBits(uint16_t bits) {
  return ::testing::ResultOf([](BF16 y) { return absl::bit_cast<uint16_t>(y); },
                             ::testing::Eq(bits));
}

::testing::Matcher<float> NearFloat(float x, float relative_error = 1e-3) {
  return ::testing::FloatNear(x, std::abs(x) * relative_error);
}

float BinaryToFloat(uint32_t sign, uint32_t exponent, uint32_t high_mantissa,
                    uint32_t low_mantissa) {
  float dest;
  uint32_t src =
      (sign << 31) + (exponent << 23) + (high_mantissa << 16) + low_mantissa;
  memcpy(static_cast<void*>(&dest), static_cast<const void*>(&src),
         sizeof(dest));
  return dest;
}

template <typename T>
void TestRoundtrips() {
  for (T value : {
           -std::numeric_limits<T>::infinity(),
           std::numeric_limits<T>::infinity(),
           T(-1.0),
           T(-0.5),
           T(-0.0),
           T(1.0),
           T(0.5),
           T(0.0),
       }) {
    EXPECT_EQ(value, static_cast<T>(static_cast<BF16>(value)));
  }
}

TEST(BF16Test, FloatRoundtrips) { TestRoundtrips<float>(); }

TEST(BF16Test, DoubleRoundtrips) { TestRoundtrips<double>(); }

TEST(BF16Test, Float16Roundtrips) { TestRoundtrips<BF16>(); }

TEST(BF16Test, ConversionFromFloat) {
  EXPECT_THAT(BF16(1.0f), MatchesBits(0x3f80));
  EXPECT_THAT(BF16(0.5f), MatchesBits(0x3f00));
  EXPECT_THAT(BF16(0.33333f), MatchesBits(0x3eab));
  EXPECT_THAT(BF16(3.38e38f), MatchesBits(0x7f7e));
  EXPECT_THAT(BF16(3.40e38f), MatchesBits(0x7f80));  // Becomes infinity.
}

TEST(BF16Test, RoundToNearestEven) {
  float val1 = static_cast<float>(absl::bit_cast<BF16>(uint16_t{0x3c00}));
  float val2 = static_cast<float>(absl::bit_cast<BF16>(uint16_t{0x3c01}));
  float val3 = static_cast<float>(absl::bit_cast<BF16>(uint16_t{0x3c02}));
  EXPECT_THAT(BF16(0.5f * (val1 + val2)), MatchesBits(0x3c00));
  EXPECT_THAT(BF16(0.5f * (val2 + val3)), MatchesBits(0x3c02));
}

TEST(BF16Test, ConversionFromInt) {
  EXPECT_THAT(BF16(-1), MatchesBits(0xbf80));
  EXPECT_THAT(BF16(0), MatchesBits(0x0000));
  EXPECT_THAT(BF16(1), MatchesBits(0x3f80));
  EXPECT_THAT(BF16(2), MatchesBits(0x4000));
  EXPECT_THAT(BF16(3), MatchesBits(0x4040));
  EXPECT_THAT(BF16(12), MatchesBits(0x4140));
}

TEST(BF16Test, ConversionFromBool) {
  EXPECT_THAT(BF16(false), MatchesBits(0x0000));
  EXPECT_THAT(BF16(true), MatchesBits(0x3f80));
}

TEST(BF16Test, ConversionToBool) {
  EXPECT_EQ(static_cast<bool>(BF16(3)), true);
  EXPECT_EQ(static_cast<bool>(BF16(0.33333f)), true);
  EXPECT_EQ(BF16(-0.0), false);
  EXPECT_EQ(static_cast<bool>(BF16(0.0)), false);
}

TEST(BF16Test, ExplicitConversionToFloat) {
  EXPECT_EQ(static_cast<float>(absl::bit_cast<BF16, uint16_t>(0x0000)), 0.0f);
  EXPECT_EQ(static_cast<float>(absl::bit_cast<BF16, uint16_t>(0x3f80)), 1.0f);
}

TEST(BF16Test, ImplicitConversionToFloat) {
  EXPECT_EQ((absl::bit_cast<BF16, uint16_t>(0x0000)), 0.0f);
  EXPECT_EQ((absl::bit_cast<BF16, uint16_t>(0x3f80)), 1.0f);
}

TEST(BF16Test, Zero) {
  EXPECT_EQ(BF16(0.0f), BF16(0.0f));
  EXPECT_EQ(BF16(-0.0f), BF16(0.0f));
  EXPECT_EQ(BF16(-0.0f), BF16(-0.0f));
  EXPECT_THAT(BF16(0.0f), MatchesBits(0x0000));
  EXPECT_THAT(BF16(-0.0f), MatchesBits(0x8000));
}

TEST(BF16Test, DefaultConstruct) {
  EXPECT_EQ(static_cast<float>(BF16()), 0.0f);
}

TEST(BF16Test, Conversion) {
  for (int i = 0; i < 100; ++i) {
    float a = i + 1.25;
    BF16 b = static_cast<BF16>(a);
    float c = static_cast<float>(b);
    EXPECT_LE(std::abs(c - a), a / 128);
  }
}

TEST(BF16Test, Epsilon) {
  EXPECT_LE(1.0f, static_cast<float>(std::numeric_limits<BF16>::epsilon() +
                                     BF16(1.0f)));
  EXPECT_EQ(1.0f, static_cast<float>(std::numeric_limits<BF16>::epsilon() /
                                         BF16(2.0f) +
                                     BF16(1.0f)));
}

TEST(BF16Test, Negate) {
  EXPECT_EQ(static_cast<float>(-BF16(3.0f)), -3.0f);
  EXPECT_EQ(static_cast<float>(-BF16(-4.5f)), 4.5f);
}

TEST(BF16Test, DivisionByZero) {
  EXPECT_TRUE(std::isnan(static_cast<float>(BF16(0.0 / 0.0))));
  EXPECT_TRUE(std::isinf(static_cast<float>(BF16(1.0 / 0.0))));
  EXPECT_TRUE(std::isinf(static_cast<float>(BF16(-1.0 / 0.0))));

  EXPECT_TRUE(std::isnan(BF16(0.0 / 0.0)));
  EXPECT_TRUE(std::isinf(BF16(1.0 / 0.0)));
  EXPECT_TRUE(std::isinf(BF16(-1.0 / 0.0)));
}

TEST(BF16Test, NonFinite) {
  EXPECT_FALSE(std::isinf(
      static_cast<float>(BF16(3.38e38f))));  // Largest finite number.
  EXPECT_FALSE(std::isnan(static_cast<float>(BF16(0.0f))));
  EXPECT_TRUE(
      std::isinf(static_cast<float>(absl::bit_cast<BF16, uint16_t>(0xff80))));
  EXPECT_TRUE(
      std::isnan(static_cast<float>(absl::bit_cast<BF16, uint16_t>(0xffc0))));
  EXPECT_TRUE(
      std::isinf(static_cast<float>(absl::bit_cast<BF16, uint16_t>(0x7f80))));
  EXPECT_TRUE(
      std::isnan(static_cast<float>(absl::bit_cast<BF16, uint16_t>(0x7fc0))));

  // Exactly same checks as above, just directly on the BF16 representation.
  EXPECT_FALSE(isinf(absl::bit_cast<BF16, uint16_t>(0x7bff)));
  EXPECT_FALSE(isnan(absl::bit_cast<BF16, uint16_t>(0x0000)));
  EXPECT_TRUE(isinf(absl::bit_cast<BF16, uint16_t>(0xff80)));
  EXPECT_TRUE(isnan(absl::bit_cast<BF16, uint16_t>(0xffc0)));
  EXPECT_TRUE(isinf(absl::bit_cast<BF16, uint16_t>(0x7f80)));
  EXPECT_TRUE(isnan(absl::bit_cast<BF16, uint16_t>(0x7fc0)));

  EXPECT_THAT(BF16(BinaryToFloat(0x0, 0xff, 0x40, 0x0)),  // +nan
              MatchesBits(0x7fe0));
  EXPECT_THAT(BF16(BinaryToFloat(0x1, 0xff, 0x40, 0x0)),  // -nan
              MatchesBits(0xffe0));
}

TEST(BF16Test, NumericLimits) {
  static_assert(std::numeric_limits<BF16>::is_signed);

  EXPECT_EQ(
      absl::bit_cast<uint16_t>(std::numeric_limits<BF16>::infinity()),
      absl::bit_cast<uint16_t>(BF16(std::numeric_limits<float>::infinity())));
  // There is no guarantee that casting a 32-bit NaN to bfloat16 has a precise
  // bit pattern.  We test that it is in fact a NaN, then test the signaling
  // bit (msb of significand is 1 for quiet, 0 for signaling).
  constexpr uint16_t BFLOAT16_QUIET_BIT = 0x0040;
  EXPECT_TRUE(isnan(std::numeric_limits<BF16>::quiet_NaN()));
  EXPECT_TRUE(isnan(BF16(std::numeric_limits<float>::quiet_NaN())));
  EXPECT_GT((absl::bit_cast<uint16_t>(std::numeric_limits<BF16>::quiet_NaN()) &
             BFLOAT16_QUIET_BIT),
            0);
  EXPECT_GT(
      (absl::bit_cast<uint16_t>(BF16(std::numeric_limits<float>::quiet_NaN())) &
       BFLOAT16_QUIET_BIT),
      0);

  EXPECT_TRUE(isnan(std::numeric_limits<BF16>::signaling_NaN()));
  EXPECT_TRUE(isnan(BF16(std::numeric_limits<float>::signaling_NaN())));
  EXPECT_EQ(
      0, (absl::bit_cast<uint16_t>(std::numeric_limits<BF16>::signaling_NaN()) &
          BFLOAT16_QUIET_BIT));
  EXPECT_EQ(0, (absl::bit_cast<uint16_t>(
                    BF16(std::numeric_limits<float>::signaling_NaN())) &
                BFLOAT16_QUIET_BIT));

  EXPECT_GT(std::numeric_limits<BF16>::min(), BF16(0.f));
  EXPECT_GT(std::numeric_limits<BF16>::denorm_min(), BF16(0.f));
  EXPECT_EQ(std::numeric_limits<BF16>::denorm_min() / BF16(2), BF16(0.f));
}

TEST(BF16Test, Arithmetic) {
  EXPECT_EQ(static_cast<float>(BF16(2) + BF16(2)), 4);
  EXPECT_EQ(static_cast<float>(BF16(2) + BF16(-2)), 0);
  EXPECT_THAT(static_cast<float>(BF16(0.33333f) + BF16(0.66667f)),
              NearFloat(1.0f));
  EXPECT_EQ(static_cast<float>(BF16(2.0f) * BF16(-5.5f)), -11.0f);
  EXPECT_THAT(static_cast<float>(BF16(1.0f) / BF16(3.0f)), NearFloat(0.3339f));
  EXPECT_EQ(static_cast<float>(-BF16(4096.0f)), -4096.0f);
  EXPECT_EQ(static_cast<float>(-BF16(-4096.0f)), 4096.0f);
}

TEST(BF16Test, Comparison) {
  EXPECT_TRUE(BF16(1.0f) > BF16(0.5f));
  EXPECT_TRUE(BF16(0.5f) < BF16(1.0f));
  EXPECT_FALSE((BF16(1.0f) < BF16(0.5f)));
  EXPECT_FALSE((BF16(0.5f) > BF16(1.0f)));

  EXPECT_FALSE((BF16(4.0f) > BF16(4.0f)));
  EXPECT_FALSE((BF16(4.0f) < BF16(4.0f)));

  EXPECT_FALSE((BF16(0.0f) < BF16(-0.0f)));
  EXPECT_FALSE((BF16(-0.0f) < BF16(0.0f)));
  EXPECT_FALSE((BF16(0.0f) > BF16(-0.0f)));
  EXPECT_FALSE((BF16(-0.0f) > BF16(0.0f)));

  EXPECT_TRUE(BF16(0.2f) > BF16(-1.0f));
  EXPECT_TRUE(BF16(-1.0f) < BF16(0.2f));
  EXPECT_TRUE(BF16(-16.0f) < BF16(-15.0f));

  EXPECT_TRUE(BF16(1.0f) == BF16(1.0f));
  EXPECT_TRUE(BF16(1.0f) != BF16(2.0f));

  EXPECT_FALSE((BF16(0.0 / 0.0) == BF16(0.0 / 0.0)));
  EXPECT_TRUE(BF16(0.0 / 0.0) != BF16(0.0 / 0.0));

  EXPECT_FALSE((BF16(1.0) == BF16(0.0 / 0.0)));
  EXPECT_FALSE((BF16(1.0) < BF16(0.0 / 0.0)));
  EXPECT_FALSE((BF16(1.0) > BF16(0.0 / 0.0)));
  EXPECT_TRUE(BF16(1.0) != BF16(0.0 / 0.0));

  EXPECT_TRUE(BF16(1.0) < BF16(1.0 / 0.0));
  EXPECT_TRUE(BF16(1.0) > BF16(-1.0 / 0.0));
}

constexpr float PI = 3.14159265358979323846f;

TEST(BF16Test, BasicFunctions) {
  // These calls should be found via ADL.
  EXPECT_EQ(static_cast<float>(abs(BF16(3.5f))), 3.5f);
  EXPECT_EQ(static_cast<float>(abs(BF16(3.5f))), 3.5f);
  EXPECT_EQ(static_cast<float>(abs(BF16(-3.5f))), 3.5f);
  EXPECT_EQ(static_cast<float>(abs(BF16(-3.5f))), 3.5f);

  EXPECT_EQ(static_cast<float>(floor(BF16(3.5f))), 3.0f);
  EXPECT_EQ(static_cast<float>(floor(BF16(3.5f))), 3.0f);
  EXPECT_EQ(static_cast<float>(floor(BF16(-3.5f))), -4.0f);
  EXPECT_EQ(static_cast<float>(floor(BF16(-3.5f))), -4.0f);

  EXPECT_EQ(static_cast<float>(ceil(BF16(3.5f))), 4.0f);
  EXPECT_EQ(static_cast<float>(ceil(BF16(3.5f))), 4.0f);
  EXPECT_EQ(static_cast<float>(ceil(BF16(-3.5f))), -3.0f);
  EXPECT_EQ(static_cast<float>(ceil(BF16(-3.5f))), -3.0f);

  EXPECT_FLOAT_EQ(static_cast<float>(sqrt(BF16(0.0f))), 0.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(sqrt(BF16(0.0f))), 0.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(sqrt(BF16(4.0f))), 2.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(sqrt(BF16(4.0f))), 2.0f);

  EXPECT_FLOAT_EQ(static_cast<float>(pow(BF16(0.0f), BF16(1.0f))), 0.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(pow(BF16(0.0f), BF16(1.0f))), 0.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(pow(BF16(2.0f), BF16(2.0f))), 4.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(pow(BF16(2.0f), BF16(2.0f))), 4.0f);

  EXPECT_EQ(static_cast<float>(exp(BF16(0.0f))), 1.0f);
  EXPECT_EQ(static_cast<float>(exp(BF16(0.0f))), 1.0f);
  EXPECT_THAT(static_cast<float>(exp(BF16(PI))),
              NearFloat(20.f + static_cast<float>(PI)));
  EXPECT_THAT(static_cast<float>(exp(BF16(PI))),
              NearFloat(20.f + static_cast<float>(PI)));

  EXPECT_EQ(static_cast<float>(expm1(BF16(0.0f))), 0.0f);
  EXPECT_EQ(static_cast<float>(expm1(BF16(0.0f))), 0.0f);
  EXPECT_THAT(static_cast<float>(expm1(BF16(2.0f))), NearFloat(6.375f));
  EXPECT_THAT(static_cast<float>(expm1(BF16(2.0f))), NearFloat(6.375f));

  EXPECT_EQ(static_cast<float>(log(BF16(1.0f))), 0.0f);
  EXPECT_EQ(static_cast<float>(log(BF16(1.0f))), 0.0f);
  EXPECT_THAT(static_cast<float>(log(BF16(10.0f))), NearFloat(2.296875f));
  EXPECT_THAT(static_cast<float>(log(BF16(10.0f))), NearFloat(2.296875f));

  EXPECT_EQ(static_cast<float>(log1p(BF16(0.0f))), 0.0f);
  EXPECT_EQ(static_cast<float>(log1p(BF16(0.0f))), 0.0f);
  EXPECT_THAT(static_cast<float>(log1p(BF16(10.0f))), NearFloat(2.390625f));
  EXPECT_THAT(static_cast<float>(log1p(BF16(10.0f))), NearFloat(2.390625f));
}

TEST(BF16Test, TrigonometricFunctions) {
  EXPECT_THAT(cos(BF16(0.0f)), NearFloat(BF16(std::cos(0.0f))));
  EXPECT_THAT(cos(BF16(0.0f)), NearFloat(BF16(std::cos(0.0f))));
  EXPECT_FLOAT_EQ(cos(BF16(PI)), BF16(std::cos(PI)));
  EXPECT_NEAR(cos(BF16(PI / 2)), BF16(std::cos(PI / 2)), 1e-3);
  EXPECT_NEAR(cos(BF16(3 * PI / 2)), BF16(std::cos(3 * PI / 2)), 1e-2);
  EXPECT_THAT(cos(BF16(3.5f)), NearFloat(BF16(std::cos(3.5f))));

  EXPECT_FLOAT_EQ(sin(BF16(0.0f)), BF16(std::sin(0.0f)));
  EXPECT_FLOAT_EQ(sin(BF16(0.0f)), BF16(std::sin(0.0f)));
  EXPECT_NEAR(sin(BF16(PI)), BF16(std::sin(PI)), 1e-3);
  EXPECT_THAT(sin(BF16(PI / 2)), NearFloat(BF16(std::sin(PI / 2))));
  EXPECT_THAT(sin(BF16(3 * PI / 2)), NearFloat(BF16(std::sin(3 * PI / 2))));
  EXPECT_THAT(sin(BF16(3.5f)), NearFloat(BF16(std::sin(3.5f))));

  EXPECT_FLOAT_EQ(tan(BF16(0.0f)), BF16(std::tan(0.0f)));
  EXPECT_FLOAT_EQ(tan(BF16(0.0f)), BF16(std::tan(0.0f)));
  EXPECT_NEAR(tan(BF16(PI)), BF16(std::tan(PI)), 1e-3);
  EXPECT_THAT(tan(BF16(3.5f)), NearFloat(BF16(std::tan(3.5f))));
}

}  // namespace
}  // namespace shlo_ref
