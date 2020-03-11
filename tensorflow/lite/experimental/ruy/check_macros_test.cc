/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/check_macros.h"

#include <gtest/gtest.h>

namespace {

#define TEST_CONDITION_FOR_FAMILY(family, vacuously_succeeds, condition) \
  do {                                                                   \
    if (vacuously_succeeds || (condition)) {                             \
      RUY_##family(condition);                                           \
    }                                                                    \
  } while (false)

#define TEST_COMPARISON_FOR_FAMILY(family, vacuously_succeeds, op_name, x, op, \
                                   y)                                          \
  do {                                                                         \
    if (vacuously_succeeds || ((x)op(y))) {                                    \
      RUY_##family##_##op_name(x, y);                                          \
    }                                                                          \
  } while (false)

#ifdef NDEBUG
#define TEST_CONDITION(condition)                       \
  do {                                                  \
    TEST_CONDITION_FOR_FAMILY(CHECK, false, condition); \
  } while (false)
#define TEST_COMPARISON(op_name, x, op, y)                       \
  do {                                                           \
    TEST_COMPARISON_FOR_FAMILY(CHECK, false, op_name, x, op, y); \
  } while (false)
#else
#define TEST_CONDITION(condition)                        \
  do {                                                   \
    TEST_CONDITION_FOR_FAMILY(CHECK, false, condition);  \
    TEST_CONDITION_FOR_FAMILY(DCHECK, false, condition); \
  } while (false)
#define TEST_COMPARISON(op_name, x, op, y)                        \
  do {                                                            \
    TEST_COMPARISON_FOR_FAMILY(CHECK, false, op_name, x, op, y);  \
    TEST_COMPARISON_FOR_FAMILY(DCHECK, false, op_name, x, op, y); \
  } while (false)

#endif

template <typename LhsType, typename RhsType>
void TestEqualityComparisons(const LhsType& lhs, const RhsType& rhs) {
  RUY_CHECK_EQ(lhs, lhs);
  TEST_COMPARISON(EQ, lhs, ==, lhs);
  RUY_CHECK_EQ(lhs, lhs);
  RUY_CHECK_EQ(lhs, lhs);
  if (lhs == rhs) {
    RUY_CHECK_EQ(lhs, rhs);
  }
  if (lhs != rhs) {
    RUY_CHECK_NE(lhs, rhs);
  }
}

template <typename LhsType, typename RhsType>
void TestComparisons(const LhsType& lhs, const RhsType& rhs) {
  TestEqualityComparisons(lhs, rhs);
  if (lhs > rhs) {
    RUY_CHECK_GT(lhs, rhs);
  }
  if (lhs >= rhs) {
    RUY_CHECK_GE(lhs, rhs);
  }
  if (lhs < rhs) {
    RUY_CHECK_LT(lhs, rhs);
  }
  if (lhs <= rhs) {
    RUY_CHECK_LE(lhs, rhs);
  }
}

TEST(CheckMacrosTest, IntInt) {
  TestComparisons(0, 0);
  TestComparisons(0, 1);
  TestComparisons(1, -1);
  TestComparisons(-1, 0);
  TestComparisons(123, -456);
  TestComparisons(std::numeric_limits<int>::min(),
                  std::numeric_limits<int>::max());
  TestComparisons(123, std::numeric_limits<int>::max());
  TestComparisons(123, std::numeric_limits<int>::min());
}

TEST(CheckMacrosTest, Uint8Uint8) {
  TestComparisons<std::uint8_t, std::uint8_t>(0, 0);
  TestComparisons<std::uint8_t, std::uint8_t>(255, 0);
  TestComparisons<std::uint8_t, std::uint8_t>(0, 255);
  TestComparisons<std::uint8_t, std::uint8_t>(12, 34);
}

TEST(CheckMacrosTest, Uint8Int) {
  TestComparisons<std::uint8_t, int>(0, std::numeric_limits<int>::min());
  TestComparisons<std::uint8_t, int>(255, std::numeric_limits<int>::min());
  TestComparisons<std::uint8_t, int>(0, std::numeric_limits<int>::max());
  TestComparisons<std::uint8_t, int>(255, std::numeric_limits<int>::max());
}

TEST(CheckMacrosTest, FloatFloat) {
  TestComparisons(0.f, 0.f);
  TestComparisons(0.f, 1.f);
  TestComparisons(1.f, -1.f);
  TestComparisons(-1.f, 0.f);
  TestComparisons(123.f, -456.f);
  TestComparisons(std::numeric_limits<float>::lowest(),
                  std::numeric_limits<float>::max());
  TestComparisons(123.f, std::numeric_limits<float>::max());
  TestComparisons(123.f, std::numeric_limits<float>::lowest());
}

TEST(CheckMacrosTest, IntFloat) {
  TestComparisons(0, 0.f);
  TestComparisons(0, 1.f);
  TestComparisons(1, -1.f);
  TestComparisons(-1, 0.f);
  TestComparisons(123, -456.f);
  TestComparisons(std::numeric_limits<int>::lowest(),
                  std::numeric_limits<float>::max());
  TestComparisons(123, std::numeric_limits<float>::max());
  TestComparisons(123, std::numeric_limits<float>::lowest());
}

TEST(CheckMacrosTest, EnumClass) {
  enum class SomeEnumClass { kA, kB, kC };
  TestEqualityComparisons(SomeEnumClass::kA, SomeEnumClass::kA);
  TestEqualityComparisons(SomeEnumClass::kA, SomeEnumClass::kB);
  TestEqualityComparisons(SomeEnumClass::kC, SomeEnumClass::kB);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
