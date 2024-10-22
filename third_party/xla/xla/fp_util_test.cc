/* Copyright 2018 The OpenXLA Authors.

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
#include "xla/fp_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/numeric/bits.h"
#include "xla/bit_cast.h"
#include "xla/test.h"
#include "xla/util.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace {

class FixedValueTest : public testing::TestWithParam<double> {};

TEST_P(FixedValueTest, DropBits) {
  double input = GetParam();
  int exponent = std::ilogb(input);
  constexpr int kMinNormalExponent =
      std::numeric_limits<float>::min_exponent - 1;
  int normalization_loss =
      std::isnormal(input) ? std::max(kMinNormalExponent - exponent, 0) : 0;
  int max_precision = std::numeric_limits<float>::digits - normalization_loss;
  for (int i = 0; i < max_precision; ++i) {
    auto result = SplitToFpPair<float>(input,
                                       /*num_high_trailing_zeros=*/i);
    auto [high_float, low_float] = result;
    if (!std::isfinite(input)) {
      EXPECT_TRUE(std::isnan(high_float));
      EXPECT_TRUE(std::isnan(low_float));
      continue;
    }
    EXPECT_FALSE(std::isnan(high_float));
    EXPECT_FALSE(std::isnan(low_float));
    EXPECT_GE(absl::countr_zero(absl::bit_cast<uint32_t>(high_float)), i);
    double sum = double{high_float} + double{low_float};
    if (input == 0.0) {
      EXPECT_EQ(high_float, 0.0f);
      EXPECT_EQ(low_float, 0.0f);
    } else {
      EXPECT_LT(std::fabs(input - double{high_float}),
                std::scalbn(input, -(max_precision - i)));
      // NOLINTNEXTLINE
      if (std::abs(input) >= std::numeric_limits<float>::min()) {
        EXPECT_LT(std::fabs(input - sum),
                  std::scalbn(std::fabs(input), -(2 * max_precision + 1 - i)));
      }
    }
    if (i == 0) {
      EXPECT_EQ(high_float + low_float, high_float);
    }
    if (input == high_float) {
      EXPECT_EQ(low_float, 0.0f);
    } else {
      EXPECT_GT(std::fabs(high_float),
                std::scalbn(low_float, max_precision - i))
          << "input: " << RoundTripFpToString(input)
          << " high_float: " << RoundTripFpToString(high_float)
          << " low_float: " << RoundTripFpToString(low_float);

      auto no_op_split = SplitToFpPair<float>(high_float,
                                              /*num_high_trailing_zeros=*/i);
      EXPECT_EQ(no_op_split.first, high_float);
      EXPECT_EQ(no_op_split.second, 0.0f);
    }
    // The sum is inexact only if the input had too many significant digits.
    if (input != sum) {
      EXPECT_LT(absl::countr_zero(absl::bit_cast<uint64_t>(input)),
                std::numeric_limits<double>::digits - (2 * max_precision + 1))
          << "input: " << RoundTripFpToString(input)
          << " high_float: " << RoundTripFpToString(high_float)
          << " low_float: " << RoundTripFpToString(low_float);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    SinglePrecisionInputs, FixedValueTest,
    testing::Values(+0.0f, -0.0f, 1.0f, static_cast<float>(M_PI),
                    static_cast<float>(M_1_PI), static_cast<float>(M_E),
                    static_cast<float>(M_LN2), static_cast<float>(M_LOG2E),
                    static_cast<float>(M_SQRT2), static_cast<float>(M_SQRT1_2),
                    static_cast<float>(M_2_SQRTPI), 0x1.555554p+1f,
                    0x1.aaaaaap+1f, 0x1.fffffcp-127f,
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()));

INSTANTIATE_TEST_SUITE_P(DoublePrecisionInputs, FixedValueTest,
                         testing::Values(+0.0, -0.0, 1.0, M_PI, M_1_PI, M_E,
                                         M_LN2, M_LOG2E, M_SQRT2, M_SQRT1_2,
                                         M_2_SQRTPI, 0x1.5555555555555p+1,
                                         0x1.aaaaaaaaaaaaap+1,
                                         0x1.fffffffffffffp-127,
                                         0x1.aaaaaaaaaaaaap-127));

// Test F8E4M3 floating-point types (F8E4M3, F8E4M3FN)
template <typename T>
class FP8E4M3DistanceTest : public ::testing::Test {};

using F8E4M3Types = ::testing::Types<tsl::float8_e4m3, tsl::float8_e4m3fn>;
TYPED_TEST_SUITE(FP8E4M3DistanceTest, F8E4M3Types);

TEST(FPDistanceTest, F8E3M4Distance) {
  // a & b are equal
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e3m4>(tsl::float8_e3m4(8.0),
                                                        tsl::float8_e3m4(8.0)),
            0);

  // a & b have the same exponents
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e3m4>(tsl::float8_e3m4(8.0),
                                                        tsl::float8_e3m4(15.5)),
            15);

  // a & b have different exponents
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e3m4>(tsl::float8_e3m4(8.0),
                                                        tsl::float8_e3m4(6)),
            8);

  // 1 from 0 in the positive direction
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e3m4>(
                std::numeric_limits<tsl::float8_e3m4>::denorm_min(),
                tsl::float8_e3m4(0)),
            1);

  // 1 from 0 in the negative direction
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e3m4>(
                -std::numeric_limits<tsl::float8_e3m4>::denorm_min(),
                tsl::float8_e3m4(0)),
            1);

  // a & b have different signs
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e3m4>(
                -std::numeric_limits<tsl::float8_e3m4>::denorm_min(),
                std::numeric_limits<tsl::float8_e3m4>::denorm_min()),
            2);

  // 1 non denorm from 0 in the positive direction
  EXPECT_EQ(
      CalculateDistanceInFloats<tsl::float8_e3m4>(
          std::numeric_limits<tsl::float8_e3m4>::min(), tsl::float8_e3m4(0)),
      16);

  // 1 non denorm from 0 in the negative direction
  EXPECT_EQ(
      CalculateDistanceInFloats<tsl::float8_e3m4>(
          -std::numeric_limits<tsl::float8_e3m4>::min(), tsl::float8_e3m4(0)),
      16);

  // a & b have different signs
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e3m4>(
                -std::numeric_limits<tsl::float8_e3m4>::min(),
                std::numeric_limits<tsl::float8_e3m4>::min()),
            32);
}

TYPED_TEST(FP8E4M3DistanceTest, F8E4M3Distance) {
  // a & b are equal, distance should be 0
  EXPECT_EQ(
      CalculateDistanceInFloats<TypeParam>(TypeParam(8.0), TypeParam(8.0)), 0);

  // a & b have the same exponents
  EXPECT_EQ(
      CalculateDistanceInFloats<TypeParam>(TypeParam(8.0), TypeParam(15.0)), 7);

  // a & b have different exponents
  EXPECT_EQ(
      CalculateDistanceInFloats<TypeParam>(TypeParam(8.0), TypeParam(6.0)), 4);

  // 1 from 0 in the positive direction
  EXPECT_EQ(CalculateDistanceInFloats<TypeParam>(
                std::numeric_limits<TypeParam>::denorm_min(), TypeParam(0)),
            1);

  // 1 from 0 in the negative direction
  EXPECT_EQ(CalculateDistanceInFloats<TypeParam>(
                -std::numeric_limits<TypeParam>::denorm_min(), TypeParam(0)),
            1);

  // a & b have different signs
  EXPECT_EQ(CalculateDistanceInFloats<TypeParam>(
                -std::numeric_limits<TypeParam>::denorm_min(),
                std::numeric_limits<TypeParam>::denorm_min()),
            2);

  // 1 non denorm from 0 in the positive direction
  EXPECT_EQ(CalculateDistanceInFloats<TypeParam>(
                std::numeric_limits<TypeParam>::min(), TypeParam(0)),
            8);

  // 1 non denorm from 0 in the negative direction
  EXPECT_EQ(CalculateDistanceInFloats<TypeParam>(
                -std::numeric_limits<TypeParam>::min(), TypeParam(0)),
            8);

  // a & b have different signs
  EXPECT_EQ(CalculateDistanceInFloats<TypeParam>(
                -std::numeric_limits<TypeParam>::min(),
                std::numeric_limits<TypeParam>::min()),
            16);
}

TEST(FPDistanceTest, F8E5M2Distance) {
  // a & b are equal
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e5m2>(tsl::float8_e5m2(8.0),
                                                        tsl::float8_e5m2(8.0)),
            0);

  // a & b have the same exponents
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e5m2>(tsl::float8_e5m2(8.0),
                                                        tsl::float8_e5m2(14)),
            3);

  // a & b have different exponents
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e5m2>(tsl::float8_e5m2(8.0),
                                                        tsl::float8_e5m2(6)),
            2);

  // 1 from 0 in the positive direction
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e5m2>(
                std::numeric_limits<tsl::float8_e5m2>::denorm_min(),
                tsl::float8_e5m2(0)),
            1);

  // 1 from 0 in the negative direction
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e5m2>(
                -std::numeric_limits<tsl::float8_e5m2>::denorm_min(),
                tsl::float8_e5m2(0)),
            1);

  // a & b have different signs
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e5m2>(
                -std::numeric_limits<tsl::float8_e5m2>::denorm_min(),
                std::numeric_limits<tsl::float8_e5m2>::denorm_min()),
            2);

  // 1 non denorm from 0 in the positive direction
  EXPECT_EQ(
      CalculateDistanceInFloats<tsl::float8_e5m2>(
          std::numeric_limits<tsl::float8_e5m2>::min(), tsl::float8_e5m2(0)),
      4);

  // 1 non denorm from 0 in the negative direction
  EXPECT_EQ(
      CalculateDistanceInFloats<tsl::float8_e5m2>(
          -std::numeric_limits<tsl::float8_e5m2>::min(), tsl::float8_e5m2(0)),
      4);

  // a & b have different signs
  EXPECT_EQ(CalculateDistanceInFloats<tsl::float8_e5m2>(
                -std::numeric_limits<tsl::float8_e5m2>::min(),
                std::numeric_limits<tsl::float8_e5m2>::min()),
            8);
}

TEST(FPDistanceTest, F64Distance) {
  // a & b are equal
  EXPECT_EQ(CalculateDistanceInFloats<double>(8.0, 8.0), 0);

  // a & b have the same exponents
  EXPECT_EQ(CalculateDistanceInFloats<double>(
                std::numeric_limits<double>::denorm_min(),
                std::nextafter(std::numeric_limits<double>::denorm_min(), 1.0)),
            1);

  // a & b have different exponents
  EXPECT_EQ(CalculateDistanceInFloats<double>(
                std::numeric_limits<double>::min(),
                std::numeric_limits<double>::denorm_min()),
            (1ULL << 52) - 1);

  // 1 from 0 in the positive direction
  EXPECT_EQ(CalculateDistanceInFloats<double>(
                std::numeric_limits<double>::denorm_min(), 0.0),
            1);

  // 1 from 0 in the negative direction
  EXPECT_EQ(CalculateDistanceInFloats<double>(
                -std::numeric_limits<double>::denorm_min(), 0.0),
            1);

  // a & b have different signs
  EXPECT_EQ(CalculateDistanceInFloats<double>(
                -std::numeric_limits<double>::denorm_min(),
                std::numeric_limits<double>::denorm_min()),
            2);

  // 1 non denorm from 0 in the positive direction
  EXPECT_EQ(CalculateDistanceInFloats<double>(
                std::numeric_limits<double>::min(), 0.0),
            1ULL << 52);

  // 1 non denorm from 0 in the negative direction
  EXPECT_EQ(CalculateDistanceInFloats<double>(
                -std::numeric_limits<double>::min(), 0.0),
            1ULL << 52);

  // a & b have different signs
  EXPECT_EQ(
      CalculateDistanceInFloats<double>(-std::numeric_limits<double>::min(),
                                        std::numeric_limits<double>::min()),
      2 * (1ULL << 52));

  // signed integer arithmetic would overflow.
  EXPECT_EQ(
      CalculateDistanceInFloats<double>(BitCast<double>(0x7fffffffffffffff),
                                        BitCast<double>(0xffffffffffffffff)),
      2);
}

}  // namespace
}  // namespace xla
