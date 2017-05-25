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

#include <vector>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

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

}  // namespace tensorflow
