/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/float8.h"

#include <cmath>
#include <limits>
#include <string>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

template <typename Float8_>
class Float8Test : public ::testing::Test {};

// Helper utility for prettier test names.
struct Float8TestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    if constexpr (std::is_same_v<TypeParam, float8_e4m3>) {
      return "float8_e4m3";
    } else if constexpr (std::is_same_v<TypeParam, float8_e5m2>) {
      return "float8_e5m2";
    }
    return absl::StrCat(idx);
  }
};

using Float8Types = ::testing::Types<float8_e4m3, float8_e5m2>;
TYPED_TEST_SUITE(Float8Test, Float8Types, Float8TestParamNames);

TEST(Float8E4m3Test, NumericLimits) {
  EXPECT_TRUE(
      Eigen::numext::isnan(std::numeric_limits<float8_e4m3>::quiet_NaN()));
  EXPECT_TRUE(
      Eigen::numext::isnan(std::numeric_limits<float8_e4m3>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e4m3>::min()),
            std::exp2(-6));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e4m3>::max()), 448);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e4m3>::lowest()),
            -448);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e4m3>::epsilon()),
            0.125);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e4m3>::round_error()),
            0.5);
  // No infinity, represent as NaN.
  EXPECT_TRUE(
      Eigen::numext::isnan(std::numeric_limits<float8_e4m3>::infinity()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e4m3>::denorm_min()),
            std::exp2(-9));
}

TEST(Float8E5m2Test, NumericLimits) {
  EXPECT_TRUE(
      Eigen::numext::isnan(std::numeric_limits<float8_e5m2>::quiet_NaN()));
  EXPECT_TRUE(
      Eigen::numext::isnan(std::numeric_limits<float8_e5m2>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::min()),
            std::exp2(-14));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::max()), 57344);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::lowest()),
            -57344);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::epsilon()),
            0.25);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::round_error()),
            0.5);
  EXPECT_TRUE(
      Eigen::numext::isinf(std::numeric_limits<float8_e5m2>::infinity()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::denorm_min()),
            std::exp2(-16));
}

TYPED_TEST(Float8Test, FromRep) {
  using Float8 = TypeParam;
  Float8 x = Float8::FromRep(0x4F);
  EXPECT_EQ(x.rep(), 0x4F);
}

TYPED_TEST(Float8Test, Negate) {
  using Float8 = TypeParam;
  Float8 x = -Float8::FromRep(0x4F);
  EXPECT_EQ(x.rep(), 0x80 | 0x4F);

  Float8 nan = -std::numeric_limits<Float8>::quiet_NaN();
  EXPECT_TRUE(Eigen::numext::isnan(nan));
}

TYPED_TEST(Float8Test, BitCasts) {
  using Float8 = TypeParam;
  Float8 x = Float8::FromRep(0x47);
  EXPECT_EQ(Eigen::numext::bit_cast<uint8_t>(x), 0x47);
  EXPECT_EQ(Eigen::numext::bit_cast<Float8>(x.rep()).rep(), 0x47);
}

TYPED_TEST(Float8Test, UpCasts) {
  using Float8 = TypeParam;

  // Loop through each float8 value.
  for (int i = 0x00; i <= 0xFF; ++i) {
    // Cast up to each other floating-point type, and verify they are the same.
    Float8 f8 = Float8::FromRep(i);
    double f64 = static_cast<double>(f8);
    float f32 = static_cast<float>(f8);
    Eigen::bfloat16 bf16 = static_cast<Eigen::bfloat16>(f8);
    Eigen::half f16 = static_cast<Eigen::half>(f8);

    if (Eigen::numext::isnan(f8)) {
      EXPECT_TRUE(Eigen::numext::isnan(f64));
      EXPECT_TRUE(Eigen::numext::isnan(f32));
      EXPECT_TRUE(Eigen::numext::isnan(bf16));
      EXPECT_TRUE(Eigen::numext::isnan(f16));
    } else {
      EXPECT_EQ(f64, f32);
      EXPECT_EQ(f32, bf16);
      EXPECT_EQ(bf16, f16);
    }
  }
}

TYPED_TEST(Float8Test, DownCasts) {
  using Float8 = TypeParam;
  for (int i = 0x00; i <= 0xFF; ++i) {
    float x = static_cast<float>(Float8::FromRep(i));

    Float8 f64 = static_cast<Float8>(static_cast<double>(x));
    Float8 f32 = static_cast<Float8>(static_cast<float>(x));
    Float8 bf16 = static_cast<Float8>(static_cast<Eigen::bfloat16>(x));
    Float8 f16 = static_cast<Float8>(static_cast<Eigen::half>(x));

    if (Eigen::numext::isnan(x)) {
      EXPECT_TRUE(Eigen::numext::isnan(f64));
      EXPECT_TRUE(Eigen::numext::isnan(f32));
      EXPECT_TRUE(Eigen::numext::isnan(bf16));
      EXPECT_TRUE(Eigen::numext::isnan(f16));
    } else {
      EXPECT_EQ(f64.rep(), i) << i;
      EXPECT_EQ(f32.rep(), i) << i;
      EXPECT_EQ(bf16.rep(), i) << i;
      EXPECT_EQ(f16.rep(), i) << i;
    }
  }
}

TYPED_TEST(Float8Test, ConvertFromWithSaturation) {
  using Float8 = TypeParam;

  // Saturation above max value.
  Float8 upper =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          static_cast<float>(std::numeric_limits<Float8>::max()) * 2);
  EXPECT_EQ(upper, std::numeric_limits<Float8>::max());

  Float8 lower =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          static_cast<float>(std::numeric_limits<Float8>::lowest()) * 2);
  EXPECT_EQ(lower, std::numeric_limits<Float8>::lowest());

  // Special values remain with saturation.
  Float8 nan =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/true>(
          std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(Eigen::numext::isnan(nan));
  Float8 inf =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/true>(
          std::numeric_limits<float>::infinity());
  // E4M3 doesn't have inf, so check inf -> NaN conversion.
  EXPECT_TRUE(std::numeric_limits<Float8>::has_infinity
                  ? Eigen::numext::isinf(inf)
                  : Eigen::numext::isnan(inf));
  Float8 ninf =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/true>(
          -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::numeric_limits<Float8>::has_infinity
                  ? Eigen::numext::isinf(ninf)
                  : Eigen::numext::isnan(ninf));
}

TYPED_TEST(Float8Test, ConvertFromWithTruncation) {
  using Float8 = TypeParam;

  // Truncation and rounding of a number ever-so-slightly less than 2.
  float less_than_two = Eigen::numext::bit_cast<float>(0x3FFFFFFF);
  Float8 truncated =
      Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
          less_than_two);
  EXPECT_LT(static_cast<float>(truncated), 2);

  Float8 rounded =
      Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
          less_than_two);
  EXPECT_EQ(static_cast<float>(rounded), 2);

  // Truncation and rounding of a subnormal.
  for (int i = 0x01; i < 0x04; ++i) {
    float less_than_subnorm =
        std::nexttoward(static_cast<float>(Float8::FromRep(i)), 0);

    Float8 truncated_subnorm =
        Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
            less_than_subnorm);
    EXPECT_EQ(truncated_subnorm.rep(), i - 1);

    Float8 rounded_subnorm =
        Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
            less_than_subnorm);
    EXPECT_EQ(rounded_subnorm.rep(), i);
  }
}

TYPED_TEST(Float8Test, ConvertTo) {
  using Float8 = TypeParam;

  // Converting to higher precision types doesn't result in either
  // truncation or saturation, so let's just ensure they all provide the
  // same results.
  for (int i = 0x00; i <= 0xFF; ++i) {
    // Cast up to each other floating-point type, and verify they are the same.
    Float8 f8 = Float8::FromRep(i);
    float f32 = static_cast<float>(f8);
    if (Eigen::numext::isnan(f8)) {
      EXPECT_TRUE(
          std::isnan(Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                /*kTruncate=*/false>(f8)));
      EXPECT_TRUE(
          std::isnan(Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                /*kTruncate=*/true>(f8)));
      EXPECT_TRUE(
          std::isnan(Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                /*kTruncate=*/false>(f8)));
      EXPECT_TRUE(
          std::isnan(Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                /*kTruncate=*/true>(f8)));
    } else {
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                 /*kTruncate=*/false>(f8)));
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                 /*kTruncate=*/true>(f8)));
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                 /*kTruncate=*/false>(f8)));
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                 /*kTruncate=*/true>(f8)));
    }
  }
}

TEST(Float8Test, Float8E5m2_To_Float8E4m3) {
  for (int i = 0x00; i <= 0xFF; ++i) {
    float8_e5m2 e5m2 = float8_e5m2::FromRep(i);
    float8_e4m3 e4m3 = static_cast<float8_e4m3>(e5m2);
    float8_e4m3 expected = static_cast<float8_e4m3>(static_cast<float>(e5m2));
    EXPECT_EQ(e4m3.rep(), expected.rep()) << i;
  }

  // Saturation.
  float8_e5m2 max = std::numeric_limits<float8_e5m2>::max();
  float8_e4m3 saturated = float8_e4m3::ConvertFrom</*kSaturate=*/true>(max);
  EXPECT_EQ(saturated, std::numeric_limits<float8_e4m3>::max());
  saturated = float8_e5m2::ConvertTo<float8_e4m3, /*kSaturate=*/true>(max);
  EXPECT_EQ(saturated, std::numeric_limits<float8_e4m3>::max());

  // Truncation - only occurs for e4m3 subnormals.
  float8_e5m2 less_than_subnorm = float8_e5m2::FromRep(0x1F);  // 2^-7 - 2^-10.
  float8_e4m3 rounded_subnorm =
      float8_e4m3::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
          less_than_subnorm);
  EXPECT_EQ(rounded_subnorm.rep(), 0x04);
  float8_e4m3 truncated_subnorm =
      float8_e4m3::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
          less_than_subnorm);
  EXPECT_EQ(truncated_subnorm.rep(), 0x03);
}

TEST(Float8Test, Float8E4m3_To_Float8E5m2) {
  for (int i = 0x00; i <= 0xFF; ++i) {
    float8_e4m3 e4m3 = float8_e4m3::FromRep(i);
    float8_e5m2 e5m2 = static_cast<float8_e5m2>(e4m3);
    float8_e5m2 expected = static_cast<float8_e5m2>(static_cast<float>(e4m3));
    EXPECT_EQ(e5m2.rep(), expected.rep()) << i;
  }

  // Truncation and rounding of a number ever-so-slightly less than 2.
  float8_e4m3 less_than_two = float8_e4m3::FromRep(0x3F);
  float8_e5m2 truncated =
      float8_e5m2::template ConvertFrom</*kSaturate=*/false,
                                        /*kTruncate=*/true>(less_than_two);
  EXPECT_LT(static_cast<float>(truncated), 2);

  float8_e5m2 rounded =
      float8_e5m2::template ConvertFrom</*kSaturate=*/false,
                                        /*kTruncate=*/false>(less_than_two);
  EXPECT_EQ(static_cast<float>(rounded), 2);
}

TEST(Float8Test, Half_To_Float8E5m2) {
  // Special values, NaN.
  Eigen::half inf =
      Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(0x7C00));
  EXPECT_EQ(static_cast<float8_e5m2>(inf).rep(), 0x7C);
  Eigen::half ninf =
      Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(0xFC00));
  EXPECT_EQ(static_cast<float8_e5m2>(ninf).rep(), 0xFC);

  Eigen::half nan =
      Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(0x7C01));
  EXPECT_EQ(static_cast<float8_e5m2>(nan).rep(), 0x7D);
  Eigen::half nnan =
      Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(0xFC01));
  EXPECT_EQ(static_cast<float8_e5m2>(nnan).rep(), 0xFD);

  // Rounding vs truncation.
  Eigen::half less_than_two =
      Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ((float8_e5m2::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
                 less_than_two)
                 .rep()),
            0x40);
  EXPECT_EQ((float8_e5m2::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
                 less_than_two)
                 .rep()),
            0x3F);
  EXPECT_EQ((float8_e5m2::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
                 -less_than_two)
                 .rep()),
            0xC0);
  EXPECT_EQ((float8_e5m2::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
                 -less_than_two)
                 .rep()),
            0xBF);
}

}  // namespace
}  // namespace tensorflow
