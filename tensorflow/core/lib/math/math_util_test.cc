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

#include "tensorflow/core/lib/math/math_util.h"

#include <cmath>
#include <limits>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// Number of arguments for each test of the CeilOrRatio method
const int kNumTestArguments = 4;

template <typename IntegralType, typename TestDataType>
void TestCeilOfRatio(const TestDataType test_data[][kNumTestArguments],
                     int num_tests) {
  for (int i = 0; i < num_tests; ++i) {
    const IntegralType numerator = test_data[i][0];
    const IntegralType denominator = test_data[i][1];
    const IntegralType expected_floor = test_data[i][2];
    const IntegralType expected_ceil = test_data[i][3];
    // Make sure the two ways to compute the floor return the same thing.
    IntegralType floor_1 = MathUtil::FloorOfRatio(numerator, denominator);
    IntegralType floor_2 = MathUtil::CeilOrFloorOfRatio<IntegralType, false>(
        numerator, denominator);
    EXPECT_EQ(floor_1, floor_2);
    EXPECT_EQ(expected_floor, floor_1)
        << "FloorOfRatio fails with numerator = " << numerator
        << ", denominator = " << denominator << " "
        << (8 * sizeof(IntegralType)) << " bits";
    IntegralType ceil_1 = MathUtil::CeilOfRatio(numerator, denominator);
    IntegralType ceil_2 = MathUtil::CeilOrFloorOfRatio<IntegralType, true>(
        numerator, denominator);
    EXPECT_EQ(ceil_1, ceil_2);
    EXPECT_EQ(expected_ceil, ceil_1)
        << "CeilOfRatio fails with numerator = " << numerator
        << ", denominator = " << denominator << " "
        << (8 * sizeof(IntegralType)) << " bits";
  }
}

template <typename UnsignedIntegralType>
void TestCeilOfRatioUnsigned(uint64 kMax) {
  const int kNumTests = 12;
  const uint64 kTestData[kNumTests][kNumTestArguments] = {
      // Numerator  | Denominator | Expected floor of ratio | Expected ceil of
      // ratio |
      // When numerator = 0, the result is always zero
      {0, 1, 0, 0},
      {0, 2, 0, 0},
      {0, kMax, 0, 0},
      // Try some non-extreme cases
      {1, 1, 1, 1},
      {5, 2, 2, 3},
      // Try with huge positive numerator
      {kMax, 1, kMax, kMax},
      {kMax, 2, kMax / 2, kMax / 2 + ((kMax % 2 != 0) ? 1 : 0)},
      {kMax, 3, kMax / 3, kMax / 3 + ((kMax % 3 != 0) ? 1 : 0)},
      // Try with a huge positive denominator
      {1, kMax, 0, 1},
      {2, kMax, 0, 1},
      {3, kMax, 0, 1},
      // Try with a huge numerator and a huge denominator
      {kMax, kMax, 1, 1},
  };
  TestCeilOfRatio<UnsignedIntegralType, uint64>(kTestData, kNumTests);
}

template <typename SignedInteger>
void TestCeilOfRatioSigned(int64 kMin, int64 kMax) {
  const int kNumTests = 30;
  const int64 kTestData[kNumTests][kNumTestArguments] = {
      // Numerator  | Denominator | Expected floor of ratio | Expected ceil of
      // ratio |
      // When numerator = 0, the result is always zero
      {0, 1, 0, 0},
      {0, -1, 0, 0},
      {0, 2, 0, 0},
      {0, kMin, 0, 0},
      {0, kMax, 0, 0},
      // Try all four combinations of 1 and -1
      {1, 1, 1, 1},
      {-1, 1, -1, -1},
      {1, -1, -1, -1},
      {-1, -1, 1, 1},
      // Try all four combinations of +/-5 divided by +/- 2
      {5, 2, 2, 3},
      {-5, 2, -3, -2},
      {5, -2, -3, -2},
      {-5, -2, 2, 3},
      // Try with huge positive numerator
      {kMax, 1, kMax, kMax},
      {kMax, -1, -kMax, -kMax},
      {kMax, 2, kMax / 2, kMax / 2 + ((kMax % 2 != 0) ? 1 : 0)},
      {kMax, 3, kMax / 3, kMax / 3 + ((kMax % 3 != 0) ? 1 : 0)},
      // Try with huge negative numerator
      {kMin, 1, kMin, kMin},
      {kMin, 2, kMin / 2 - ((kMin % 2 != 0) ? 1 : 0), kMin / 2},
      {kMin, 3, kMin / 3 - ((kMin % 3 != 0) ? 1 : 0), kMin / 3},
      // Try with a huge positive denominator
      {1, kMax, 0, 1},
      {2, kMax, 0, 1},
      {3, kMax, 0, 1},
      // Try with a huge negative denominator
      {1, kMin, -1, 0},
      {2, kMin, -1, 0},
      {3, kMin, -1, 0},
      // Try with a huge numerator and a huge denominator
      {kMin, kMin, 1, 1},
      {kMin, kMax, -2, -1},
      {kMax, kMin, -1, 0},
      {kMax, kMax, 1, 1},
  };
  TestCeilOfRatio<SignedInteger, int64>(kTestData, kNumTests);
}

// ------------------------------------------------------------------------ //
// Benchmarking CeilOrFloorOfRatio
//
// We compare with other implementations that are unsafe in general.
// ------------------------------------------------------------------------ //

// An implementation of CeilOfRatio that is correct for small enough values,
// and provided that the numerator and denominator are both positive
template <typename IntegralType>
static IntegralType CeilOfRatioDenomMinusOne(IntegralType numerator,
                                             IntegralType denominator) {
  const IntegralType kOne(1);
  return (numerator + denominator - kOne) / denominator;
}

// An implementation of FloorOfRatio that is correct when the denominator is
// positive and the numerator non-negative
template <typename IntegralType>
static IntegralType FloorOfRatioByDivision(IntegralType numerator,
                                           IntegralType denominator) {
  return numerator / denominator;
}

template <typename Integer, bool ComputeCeil>
static Integer CeilOrFloorOfRatioArithmetic(Integer numerator,
                                            Integer denominator) {
  if (ComputeCeil) {
    return CeilOfRatioDenomMinusOne(numerator, denominator);
  } else {
    return FloorOfRatioByDivision(numerator, denominator);
  }
}

void TestThatCeilOfRatioDenomMinusOneIsIncorrect(int64 numerator,
                                                 int64 denominator,
                                                 int64 expected_error) {
  const int64 correct_result = MathUtil::CeilOfRatio(numerator, denominator);
  const int64 result_by_denom_minus_one =
      CeilOfRatioDenomMinusOne(numerator, denominator);
  EXPECT_EQ(result_by_denom_minus_one + expected_error, correct_result)
      << "numerator = " << numerator << " denominator = " << denominator
      << " expected error = " << expected_error
      << " Actual difference: " << (correct_result - result_by_denom_minus_one);
}

// Here we demonstrate why not to use CeilOfRatioDenomMinusOne
void TestThatCeilOfRatioDenomMinusOneIsIncorrect() {
  // It does not work with negative values
  TestThatCeilOfRatioDenomMinusOneIsIncorrect(-1LL, -2LL, -1LL);

  // This would also fail if given kint64max because of signed integer overflow.
}

TEST(MathUtil, CeilOfRatio) {
  TestCeilOfRatioUnsigned<uint8>(kuint8max);
  TestCeilOfRatioUnsigned<uint16>(kuint16max);
  TestCeilOfRatioUnsigned<uint32>(kuint32max);
  TestCeilOfRatioUnsigned<uint64>(kuint64max);
  TestCeilOfRatioSigned<int8>(kint8min, kint8max);
  TestCeilOfRatioSigned<int16>(kint16min, kint16max);
  TestCeilOfRatioSigned<int32>(kint32min, kint32max);
  TestCeilOfRatioSigned<int64>(kint64min, kint64max);
#if 0
  TestThatCeilOfRatioDenomMinusOneIsIncorrect();
#endif
}

struct GCDTestCase {
  unsigned int x;
  unsigned int y;
  unsigned int gcd;
};

TEST(MathUtil, GCD) {
  std::vector<GCDTestCase> testcases({
      {10, 20, 10},  //
      {27, 8, 1},    //
      {4, 3, 1},     //
      {6, 8, 2},     //
      {5, 0, 5},     //
      {5, 5, 5},     //
      {0, 0, 0}      //
  });

  for (const auto& tc : testcases) {
    EXPECT_EQ(tc.gcd, MathUtil::GCD<uint32>(tc.x, tc.y));
    EXPECT_EQ(tc.gcd, MathUtil::GCD<uint32>(tc.y, tc.x));
    EXPECT_EQ(tc.gcd, MathUtil::GCD<uint64>(tc.x, tc.y));
    EXPECT_EQ(tc.gcd, MathUtil::GCD<uint64>(tc.y, tc.x));
  }

  const uint64 biggish_prime = 1666666667;
  EXPECT_EQ(biggish_prime,
            MathUtil::GCD<uint64>(biggish_prime * 3, biggish_prime * 4));
}

template <typename T>
void TestOneIPowN() {
  const T one{1};
  for (int i = 0; i < 1024; ++i) {
    // Computations are exact.
    EXPECT_EQ(MathUtil::IPow(one, i), one);
  }
}

template <typename T>
void TestTwoIPowN() {
  int limit = std::is_integral<T>::value ? std::numeric_limits<T>::digits : 63;
  for (int i = 0; i < limit; ++i) {
    // Computations are exact.
    EXPECT_EQ(MathUtil::IPow(T{2}, i), static_cast<T>(1ull << i));
  }
}

template <typename T>
void TestFloatIPow(const int max_exponent, const T start, const T end,
                   const T step) {
  for (T f = start; f < end; f += step) {
    for (int i = 0; i < max_exponent; ++i) {
      EXPECT_FLOAT_EQ(MathUtil::IPow(f, i), std::pow(f, i));
    }
  }
}

TEST(MathUtil, IPow) {
  TestOneIPowN<double>();
  TestOneIPowN<float>();
  TestOneIPowN<int>();
  TestOneIPowN<int64>();
  TestTwoIPowN<double>();
  TestTwoIPowN<float>();
  TestTwoIPowN<int>();
  TestTwoIPowN<int64>();

  EXPECT_EQ(MathUtil::IPow(3, 0), 1);
  EXPECT_EQ(MathUtil::IPow(3, 1), 3);
  EXPECT_EQ(MathUtil::IPow(3, 2), 9);
  EXPECT_EQ(MathUtil::IPow(3, 3), 27);
  EXPECT_EQ(MathUtil::IPow(3, 4), 81);
  EXPECT_EQ(MathUtil::IPow(3, 5), 243);

  TestFloatIPow<float>(13, -16.0f, 16.0f, 1.0f / 8);
  TestFloatIPow<double>(13, -16.0, 16.0, 1.0 / 8);

  TestFloatIPow<float>(13, -1.0f / (1 << 12), -1.0f / (1 << 12),
                       1.0f / (1 << 16));
  TestFloatIPow<double>(13, -1.0 / (1 << 12), -1.0 / (1 << 12),
                        1.0 / (1 << 16));
}

TEST(MathUtil, IPowEdgeCases) {
  constexpr const double kInf = std::numeric_limits<double>::infinity();

  EXPECT_EQ(MathUtil::IPow(-12345.0, 79), -kInf);
  EXPECT_EQ(MathUtil::IPow(-12345.0, 80), +kInf);

  // The semantics of the edge cases that follow  are defined in the standard:
  // http://en.cppreference.com/w/cpp/numeric/math/pow for a summary.

  // 1 - These edge cases apply.
  // pow(+0, exp), where exp is a positive odd integer, returns +0
  EXPECT_EQ(MathUtil::IPow(+0.0, 3), +0.0);
  // pow(-0, exp), where exp is a positive odd integer, returns -0
  EXPECT_EQ(MathUtil::IPow(-0.0, 3), -0.0);
  // pow(±0, exp), where exp is positive non-integer or a positive even integer,
  // returns +0
  EXPECT_EQ(MathUtil::IPow(+0.0, 42), +0.0);
  EXPECT_EQ(MathUtil::IPow(-0.0, 42), +0.0);
  // pow(base, ±0) returns 1 for any base, even when base is NaN
  EXPECT_EQ(MathUtil::IPow(-kInf, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(-2.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(-1.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(-0.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(+0.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(+1.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(+2.0, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(+kInf, 0.0), 1.0);
  EXPECT_EQ(MathUtil::IPow(std::numeric_limits<double>::quiet_NaN(), 0.0), 1.0);
  // pow(-∞, exp) returns -∞ if exp is a positive odd integer
  EXPECT_EQ(MathUtil::IPow(-kInf, 43), -kInf);
  // pow(-∞, exp) returns +∞ if exp is a positive non-integer or even integer
  EXPECT_EQ(MathUtil::IPow(-kInf, 42), +kInf);
  // pow(+∞, exp) returns +∞ for any positive exp
  EXPECT_EQ(MathUtil::IPow(+kInf, 42), +kInf);
  EXPECT_EQ(MathUtil::IPow(+kInf, 43), +kInf);

  // 2 - These do not apply due to the restricted exp range.
  // pow(+0, exp), where exp is a negative odd integer, returns +∞ and raises
  // FE_DIVBYZERO pow(-0, exp), where exp is a negative odd integer, returns -∞
  // and raises FE_DIVBYZERO pow(±0, exp), where exp is negative, finite, and is
  // an even integer or a non-integer, returns +∞ and raises FE_DIVBYZERO
  // pow(-1, ±∞) returns 1
  // pow(+1, exp) returns 1 for any exp, even when exp is NaN
  // pow(±0, -∞) returns +∞ and may raise FE_DIVBYZERO
  // pow(base, exp) returns NaN and raises FE_INVALID if base is finite and
  // negative and exp is finite and non-integer. pow(base, -∞) returns +∞ for
  // any |base|<1 pow(base, -∞) returns +0 for any |base|>1 pow(base, +∞)
  // returns +0 for any |base|<1 pow(base, +∞) returns +∞ for any |base|>1
  // pow(-∞, exp) returns -0 if exp is a negative odd integer
  // pow(-∞, exp) returns +0 if exp is a negative non-integer or even integer
  // pow(+∞, exp) returns +0 for any negative exp
}

}  // namespace
}  // namespace tensorflow
