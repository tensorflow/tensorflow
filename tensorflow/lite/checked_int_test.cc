/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/checked_int.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace tflite {
namespace {

template <typename T>
class CheckedIntTypedTest : public ::testing::Test {};

using TestTypes = ::testing::Types<
    std::pair<int8_t, int8_t>, std::pair<int8_t, int16_t>,
    std::pair<int8_t, int32_t>, std::pair<int8_t, int64_t>,
    std::pair<int8_t, uint8_t>, std::pair<int8_t, uint16_t>,
    std::pair<int8_t, uint32_t>, std::pair<int8_t, uint64_t>,
    std::pair<int16_t, int8_t>, std::pair<int16_t, int16_t>,
    std::pair<int16_t, int32_t>, std::pair<int16_t, int64_t>,
    std::pair<int16_t, uint8_t>, std::pair<int16_t, uint16_t>,
    std::pair<int16_t, uint32_t>, std::pair<int16_t, uint64_t>,
    std::pair<int32_t, int8_t>, std::pair<int32_t, int16_t>,
    std::pair<int32_t, int32_t>, std::pair<int32_t, int64_t>,
    std::pair<int32_t, uint8_t>, std::pair<int32_t, uint16_t>,
    std::pair<int32_t, uint32_t>, std::pair<int32_t, uint64_t>,
    std::pair<int64_t, int8_t>, std::pair<int64_t, int16_t>,
    std::pair<int64_t, int32_t>, std::pair<int64_t, int64_t>,
    std::pair<int64_t, uint8_t>, std::pair<int64_t, uint16_t>,
    std::pair<int64_t, uint32_t>, std::pair<int64_t, uint64_t>,
    std::pair<uint8_t, int8_t>, std::pair<uint8_t, int16_t>,
    std::pair<uint8_t, int32_t>, std::pair<uint8_t, int64_t>,
    std::pair<uint8_t, uint8_t>, std::pair<uint8_t, uint16_t>,
    std::pair<uint8_t, uint32_t>, std::pair<uint8_t, uint64_t>,
    std::pair<uint16_t, int8_t>, std::pair<uint16_t, int16_t>,
    std::pair<uint16_t, int32_t>, std::pair<uint16_t, int64_t>,
    std::pair<uint16_t, uint8_t>, std::pair<uint16_t, uint16_t>,
    std::pair<uint16_t, uint32_t>, std::pair<uint16_t, uint64_t>,
    std::pair<uint32_t, int8_t>, std::pair<uint32_t, int16_t>,
    std::pair<uint32_t, int32_t>, std::pair<uint32_t, int64_t>,
    std::pair<uint32_t, uint8_t>, std::pair<uint32_t, uint16_t>,
    std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint64_t>,
    std::pair<uint64_t, int8_t>, std::pair<uint64_t, int16_t>,
    std::pair<uint64_t, int32_t>, std::pair<uint64_t, int64_t>,
    std::pair<uint64_t, uint8_t>, std::pair<uint64_t, uint16_t>,
    std::pair<uint64_t, uint32_t>, std::pair<uint64_t, uint64_t>>;

TYPED_TEST_SUITE(CheckedIntTypedTest, TestTypes);

TYPED_TEST(CheckedIntTypedTest, ConstructorFromOtherTypeBoundsCheck) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;

  T2 x = std::numeric_limits<T2>::lowest();
  T2 y = std::numeric_limits<T2>::max();

  CheckedInt<T1> a(x);
  CheckedInt<T1> b(y);

  if constexpr (sizeof(T1) > sizeof(T2)) {
    if constexpr (std::is_signed_v<T1> || std::is_unsigned_v<T2>) {
      EXPECT_EQ(a.Value(), x);
      EXPECT_FALSE(a.Overflow());
    } else {
      EXPECT_TRUE(a.Overflow());
    }
    EXPECT_EQ(b.Value(), y);
    EXPECT_FALSE(b.Overflow());
  } else if constexpr (sizeof(T1) == sizeof(T2)) {
    if constexpr (std::is_signed_v<T1> == std::is_signed_v<T2>) {
      EXPECT_EQ(a.Value(), x);
      EXPECT_FALSE(a.Overflow());
      EXPECT_EQ(b.Value(), y);
      EXPECT_FALSE(b.Overflow());
    } else if constexpr (std::is_unsigned_v<T1>) {
      EXPECT_TRUE(a.Overflow());
      EXPECT_EQ(b.Value(), y);
      EXPECT_FALSE(b.Overflow());
    } else {  // signed T1, unsigned T2
      EXPECT_EQ(a.Value(), x);
      EXPECT_FALSE(a.Overflow());
      EXPECT_TRUE(b.Overflow());
    }
  } else {  // sizeof(T1) < sizeof(T2)
    if constexpr (std::is_signed_v<T1> && std::is_signed_v<T2>) {
      EXPECT_TRUE(a.Overflow());
      EXPECT_TRUE(b.Overflow());
    } else if constexpr (std::is_unsigned_v<T1> && std::is_unsigned_v<T2>) {
      EXPECT_EQ(a.Value(), x);
      EXPECT_FALSE(a.Overflow());
      EXPECT_TRUE(b.Overflow());
    } else if constexpr (std::is_unsigned_v<T1>) {
      EXPECT_TRUE(a.Overflow());
      EXPECT_TRUE(b.Overflow());
    } else {  // signed T1, unsigned T2
      EXPECT_EQ(a.Value(), x);
      EXPECT_FALSE(a.Overflow());
      EXPECT_TRUE(b.Overflow());
    }
  }
}

TYPED_TEST(CheckedIntTypedTest, BasicArithmetic) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;

  CheckedInt<T1> a(10);
  CheckedInt<T2> b(2);

  CheckedInt add = a + b;
  EXPECT_EQ(add.Value(), 12);
  EXPECT_FALSE(add.Overflow());

  CheckedInt sub = a - b;
  EXPECT_EQ(sub.Value(), 8);
  EXPECT_FALSE(sub.Overflow());

  CheckedInt mul = a * b;
  EXPECT_EQ(mul.Value(), 20);
  EXPECT_FALSE(mul.Overflow());

  CheckedInt div = a / b;
  EXPECT_EQ(div.Value(), 5);
  EXPECT_FALSE(div.Overflow());
}

TYPED_TEST(CheckedIntTypedTest, WorkingMultiplication) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;

  if (std::is_signed_v<T1> && std::is_signed_v<T2>) {
    {
      CheckedInt<T1> a(-1);
      CheckedInt<T2> b(5);
      CheckedInt c = a * b;
      EXPECT_EQ(c.Value(), -5);
      EXPECT_FALSE(c.Overflow());
    }
    {
      CheckedInt<T1> a(-1);
      CheckedInt<T2> b(-5);
      CheckedInt c = a * b;
      EXPECT_EQ(c.Value(), 5);
      EXPECT_FALSE(c.Overflow());
    }
    {
      CheckedInt<T1> a(1);
      CheckedInt<T2> b(-5);
      CheckedInt c = a * b;
      EXPECT_EQ(c.Value(), -5);
      EXPECT_FALSE(c.Overflow());
    }
  }
}

TYPED_TEST(CheckedIntTypedTest, OverflowPropagation) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;

  // Check overflow propagation from lhs.
  CheckedInt<T1> a_base(std::numeric_limits<T1>::max());
  CheckedInt<T1> a_add(1);
  CheckedInt a = a_base + a_add;
  ASSERT_TRUE(a.Overflow());
  CheckedInt<T2> b(2);

  EXPECT_TRUE((a + b).Overflow());
  EXPECT_TRUE((a - b).Overflow());
  EXPECT_TRUE((a * b).Overflow());
  EXPECT_TRUE((a / b).Overflow());

  // Check overflow propagation from rhs.
  CheckedInt<T1> c(10);
  CheckedInt<T2> d_base(std::numeric_limits<T2>::max());
  CheckedInt<T2> d_add(1);
  CheckedInt d = d_base + d_add;
  ASSERT_TRUE(d.Overflow());

  EXPECT_TRUE((c + d).Overflow());
  EXPECT_TRUE((c - d).Overflow());
  EXPECT_TRUE((c * d).Overflow());
  EXPECT_TRUE((c / d).Overflow());
}

TYPED_TEST(CheckedIntTypedTest, AdditionOverflow) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  using C = std::common_type_t<T1, T2>;

  CheckedInt<T1> a(std::numeric_limits<T1>::max());
  CheckedInt<T2> b(std::numeric_limits<T2>::max());
  EXPECT_EQ((a + b).Overflow(),
            sizeof(C) == sizeof(T1) || sizeof(C) == sizeof(T2));
}

TYPED_TEST(CheckedIntTypedTest, AdditionUnderflow) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  using C = std::common_type_t<T1, T2>;

  if constexpr ((std::is_signed_v<T1> && std::is_signed_v<T2> &&
                 (sizeof(C) == sizeof(T1) || sizeof(C) == sizeof(T2))) ||
                (std::is_unsigned_v<C> &&
                 (std::is_signed_v<T1> || std::is_signed_v<T2>))) {
    CheckedInt<T1> a(std::numeric_limits<T1>::lowest());
    CheckedInt<T2> b(std::numeric_limits<T2>::lowest());
    EXPECT_TRUE((a + b).Overflow()) << +a.Value() << " + " << +b.Value();
  } else {
    GTEST_SUCCEED();
  }
}

TYPED_TEST(CheckedIntTypedTest, SubtractionUnderflow) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  using C = std::common_type_t<T1, T2>;

  CheckedInt<T1> a(std::numeric_limits<T1>::lowest());
  CheckedInt<T2> b(std::numeric_limits<T2>::max());
  if (a.Value() == 0 && std::is_signed_v<C>) {
    EXPECT_FALSE((a - b).Overflow());
  } else {
    EXPECT_EQ((a - b).Overflow(),
              (sizeof(C) == sizeof(T1) || sizeof(C) == sizeof(T2)));
  }
}

TYPED_TEST(CheckedIntTypedTest, SubtractionOverflow) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  using C = std::common_type_t<T1, T2>;

  if constexpr (std::is_signed_v<T2> &&
                (sizeof(C) == sizeof(T1) || sizeof(C) == sizeof(T2))) {
    CheckedInt<T1> a(std::numeric_limits<T1>::max());
    CheckedInt<T2> b(std::numeric_limits<T2>::lowest());
    EXPECT_TRUE((a - b).Overflow()) << +a.Value() << " - " << +b.Value();
  } else {
    GTEST_SUCCEED();
  }
}

TYPED_TEST(CheckedIntTypedTest, MultiplicationOverflow) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  using C = std::common_type_t<T1, T2>;

  CheckedInt<T1> a(std::numeric_limits<T1>::max());
  CheckedInt<T2> b(std::numeric_limits<T2>::max());
  EXPECT_EQ((a * b).Overflow(),
            sizeof(C) == sizeof(T1) || sizeof(C) == sizeof(T2));
}

TYPED_TEST(CheckedIntTypedTest, MultiplicationUnderflowOppositeSign) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  using C = std::common_type_t<T1, T2>;

  if constexpr (std::is_signed_v<T1>) {
    CheckedInt<T1> a(std::numeric_limits<T1>::lowest());
    CheckedInt<T2> b(std::numeric_limits<T2>::max());
    EXPECT_EQ((a * b).Overflow(),
              sizeof(C) == sizeof(T1) || sizeof(C) == sizeof(T2));
  }
  if constexpr (std::is_signed_v<T2>) {
    CheckedInt<T1> c(std::numeric_limits<T1>::max());
    CheckedInt<T2> d(std::numeric_limits<T2>::lowest());
    EXPECT_EQ((c * d).Overflow(),
              sizeof(C) == sizeof(T1) || sizeof(C) == sizeof(T2));
  }
  if constexpr (!std::is_signed_v<T1> && !std::is_signed_v<T2>) {
    GTEST_SUCCEED();
  }
}

TYPED_TEST(CheckedIntTypedTest, MultiplicationOverflowSameSignNegative) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  using C = std::common_type_t<T1, T2>;

  if constexpr (std::is_signed_v<T1> && std::is_signed_v<T2>) {
    CheckedInt<T1> a(std::numeric_limits<T1>::lowest());
    CheckedInt<T2> b(std::numeric_limits<T2>::lowest());
    EXPECT_EQ((a * b).Overflow(),
              sizeof(C) == sizeof(T1) || sizeof(C) == sizeof(T2));
  } else {
    GTEST_SUCCEED();
  }
}

TYPED_TEST(CheckedIntTypedTest, DivisionOverflowIsDetected) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  using C = std::common_type_t<T1, T2>;

  if constexpr (std::is_signed_v<T1> && std::is_signed_v<T2> &&
                std::is_same_v<T1, C>) {
    CheckedInt<T1> a(std::numeric_limits<T1>::lowest());
    CheckedInt<T2> b(-1);
    EXPECT_TRUE((a / b).Overflow()) << +a.Value() << " / " << +b.Value();
  } else {
    GTEST_SUCCEED();
  }
}

TYPED_TEST(CheckedIntTypedTest, MixedWithStandardIntegralTypesCompiles) {
  using T1 = typename TypeParam::first_type;

  CheckedInt<T1> a(10);
  int b = 5;

  auto add1 = a + b;
  EXPECT_EQ(add1.Value(), 15);
  EXPECT_FALSE(add1.Overflow());

  auto add2 = b + a;
  EXPECT_EQ(add2.Value(), 15);
  EXPECT_FALSE(add2.Overflow());

  auto sub1 = a - b;
  EXPECT_EQ(sub1.Value(), 5);
  EXPECT_FALSE(sub1.Overflow());

  auto sub2 = b - a;
  if constexpr (std::is_signed_v<std::common_type_t<int, T1>>) {
    EXPECT_EQ(sub2.Value(), -5);
    EXPECT_FALSE(sub2.Overflow());
  } else {
    EXPECT_TRUE(sub2.Overflow());
  }

  auto mul1 = a * b;
  EXPECT_EQ(mul1.Value(), 50);
  EXPECT_FALSE(mul1.Overflow());

  auto mul2 = b * a;
  EXPECT_EQ(mul2.Value(), 50);
  EXPECT_FALSE(mul2.Overflow());

  auto div1 = a / b;
  EXPECT_EQ(div1.Value(), 2);
  EXPECT_FALSE(div1.Overflow());

  auto div2 = b / a;
  EXPECT_EQ(div2.Value(), 0);
  EXPECT_FALSE(div2.Overflow());
}

TYPED_TEST(CheckedIntTypedTest, CompoundAssignmentOperators) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;

  CheckedInt<T1> a(10);
  CheckedInt<T2> b(2);

  a += b;
  EXPECT_EQ(a.Value(), 12);
  EXPECT_FALSE(a.Overflow());

  a -= b;
  EXPECT_EQ(a.Value(), 10);
  EXPECT_FALSE(a.Overflow());

  a *= b;
  EXPECT_EQ(a.Value(), 20);
  EXPECT_FALSE(a.Overflow());

  a /= b;
  EXPECT_EQ(a.Value(), 10);
  EXPECT_FALSE(a.Overflow());

  int c = 2;
  a += c;
  EXPECT_EQ(a.Value(), 12);
  EXPECT_FALSE(a.Overflow());

  a -= c;
  EXPECT_EQ(a.Value(), 10);
  EXPECT_FALSE(a.Overflow());

  a *= c;
  EXPECT_EQ(a.Value(), 20);
  EXPECT_FALSE(a.Overflow());

  a /= c;
  EXPECT_EQ(a.Value(), 10);
  EXPECT_FALSE(a.Overflow());
}

TYPED_TEST(CheckedIntTypedTest, ComparisonOperators) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;

  CheckedInt<T1> a(10);
  CheckedInt<T2> b(10);
  CheckedInt<T2> c(12);
  CheckedInt<T1> d(8);

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_TRUE(a != c);
  EXPECT_FALSE(a != b);

  EXPECT_TRUE(a < c);
  EXPECT_FALSE(a < b);
  EXPECT_FALSE(a < d);

  EXPECT_TRUE(a <= c);
  EXPECT_TRUE(a <= b);
  EXPECT_FALSE(a <= d);

  EXPECT_TRUE(c > a);
  EXPECT_FALSE(b > a);
  EXPECT_FALSE(d > a);

  EXPECT_TRUE(c >= a);
  EXPECT_TRUE(b >= a);
  EXPECT_FALSE(d >= a);

  // Mixed type comparisons
  // NOLINTBEGIN(readability/check): We are testing the operators so we don't
  // want to proxy the check through EXPECT_EQ/NE/GT/GE/LT/LE.
  EXPECT_TRUE(a == 10);
  EXPECT_TRUE(10 == a);
  EXPECT_FALSE(a == 12);
  EXPECT_TRUE(a != 12);
  EXPECT_TRUE(a < 12);
  EXPECT_TRUE(8 < a);
  EXPECT_TRUE(a <= 10);
  EXPECT_TRUE(10 <= a);
  EXPECT_TRUE(a > 8);
  EXPECT_TRUE(12 > a);
  EXPECT_TRUE(a >= 10);
  EXPECT_TRUE(10 >= a);
  // NOLINTEND(readability/check)
}

TEST(CheckedIntSpecificTest, ConstructorMixedSignBoundsCheck) {
  // Assigning a small unsigned value to a signed type should not overflow.
  // Previously, this failed because unsigned `12` was compared to `INT_MIN`,
  // which promoted `INT_MIN` to a huge unsigned value, causing `12 <
  // INT_MIN_PROMOTED` to be true.
  unsigned int u_val = 12;
  CheckedInt<int> a(u_val);
  EXPECT_FALSE(a.Overflow());
  EXPECT_EQ(a.Value(), 12);

  // Assigning a positive signed value to an unsigned type should not overflow.
  int s_val = 12;
  CheckedInt<unsigned int> b(s_val);
  EXPECT_FALSE(b.Overflow());
  EXPECT_EQ(b.Value(), 12u);

  // Assigning a negative signed value to an unsigned type should overflow.
  int s_neg_val = -1;
  CheckedInt<unsigned int> c(s_neg_val);
  EXPECT_TRUE(c.Overflow());
}

TEST(CheckedIntSpecificTest, MultiplicationEdgeCases) {
  // These cases are edge cases for the fallback 64-bit multiplication
  // overflow checks.

  CheckedInt<int64_t> a_signed(0xFFFFFFFFLL);
  CheckedInt<int64_t> b_signed(0x100000001LL);
  EXPECT_TRUE((a_signed * b_signed).Overflow());

  CheckedInt<uint64_t> a_unsigned(0xFFFFFFFFULL);
  CheckedInt<uint64_t> b_unsigned(0x100000001ULL);
  EXPECT_FALSE((a_unsigned * b_unsigned).Overflow());
  EXPECT_EQ((a_unsigned * b_unsigned).Value(), 0xFFFFFFFFFFFFFFFFULL);

  CheckedInt<int64_t> neg_one(-1);
  CheckedInt<int64_t> five(5);
  EXPECT_FALSE((neg_one * five).Overflow());
  EXPECT_EQ((neg_one * five).Value(), -5);

  EXPECT_FALSE((five * neg_one).Overflow());
  EXPECT_EQ((five * neg_one).Value(), -5);

  CheckedInt<int64_t> int32_min(-2147483648LL);
  EXPECT_FALSE((int32_min * neg_one).Overflow());
  EXPECT_EQ((int32_min * neg_one).Value(), 2147483648LL);

  CheckedInt<int64_t> neg_two(-2);
  EXPECT_FALSE((int32_min * neg_two).Overflow());
  EXPECT_EQ((int32_min * neg_two).Value(), 4294967296LL);

  CheckedInt<int64_t> int64_max(std::numeric_limits<int64_t>::max());
  CheckedInt<int64_t> int64_min(std::numeric_limits<int64_t>::lowest());
  CheckedInt<int64_t> one(1);

  EXPECT_FALSE((int64_max * one).Overflow());
  EXPECT_TRUE((int64_max * five).Overflow());
  EXPECT_FALSE((int64_min * one).Overflow());
  EXPECT_TRUE((int64_min * neg_one).Overflow());
  EXPECT_TRUE((int64_min * five).Overflow());
}

TEST(CheckedIntSpecificTest, DivisionMixedSignOverflow) {
  CheckedInt<int32_t> a(-10);
  CheckedInt<uint32_t> b(2);
  CheckedInt c = a / b;
  EXPECT_TRUE(c.Overflow());
}

// Operations on different types overflow iff their exact result does not fit
// in the common type, like `__builtin_*_overflow`: the operands are not
// converted to the common type first.
TEST(CheckedIntSpecificTest, MixedTypesUseTheExactResult) {
  const CheckedInt<int32_t> minus_one(-1);
  const CheckedInt<uint32_t> one(1u);
  const CheckedInt<uint32_t> four(4u);

  const CheckedInt<uint32_t> sum = minus_one + one;
  EXPECT_FALSE(sum.Overflow());
  EXPECT_EQ(sum.Value(), 0u);

  const CheckedInt<uint32_t> difference = one - minus_one;
  EXPECT_FALSE(difference.Overflow());
  EXPECT_EQ(difference.Value(), 2u);

  EXPECT_TRUE((minus_one * one).Overflow());
  EXPECT_TRUE((minus_one - one).Overflow());

  const CheckedInt<uint32_t> quotient = minus_one / four;
  EXPECT_FALSE(quotient.Overflow());
  EXPECT_EQ(quotient.Value(), 0u);

  const CheckedInt<uint64_t> big = CheckedInt<int64_t>(-5) + uint64_t{7};
  EXPECT_FALSE(big.Overflow());
  EXPECT_EQ(big.Value(), 2u);

  EXPECT_TRUE((CheckedInt<int64_t>(std::numeric_limits<int64_t>::lowest()) /
               int32_t{-1})
                  .Overflow());
}

#if defined(__SIZEOF_INT128__)
// Values around the limits of `T` and of the narrower types.
template <class T>
std::vector<T> EdgeValues() {
  std::vector<T> values;
  for (__int128 v :
       {__int128{0}, __int128{1}, __int128{2}, __int128{-1}, __int128{-2},
        __int128{std::numeric_limits<int32_t>::lowest()},
        __int128{std::numeric_limits<int32_t>::max()},
        __int128{std::numeric_limits<uint32_t>::max()},
        __int128{std::numeric_limits<T>::lowest()},
        __int128{std::numeric_limits<T>::lowest()} + 1,
        __int128{std::numeric_limits<T>::max()},
        __int128{std::numeric_limits<T>::max()} - 1}) {
    if (v >= __int128{std::numeric_limits<T>::lowest()} &&
        v <= __int128{std::numeric_limits<T>::max()}) {
      values.push_back(static_cast<T>(v));
    }
  }
  return values;
}

template <class A, class B>
void ExpectExactMixedResults() {
  using C = std::common_type_t<A, B>;
  const auto fits = [](__int128 v) {
    return v >= __int128{std::numeric_limits<C>::lowest()} &&
           v <= __int128{std::numeric_limits<C>::max()};
  };
  for (A a : EdgeValues<A>()) {
    for (B b : EdgeValues<B>()) {
      SCOPED_TRACE(testing::Message()
                   << "a=" << static_cast<int64_t>(a) << " (" << sizeof(A)
                   << " bytes), b=" << static_cast<int64_t>(b) << " ("
                   << sizeof(B) << " bytes)");
      const CheckedInt<A> x(a);
      const CheckedInt<B> y(b);
      const __int128 exact[] = {__int128{a} + __int128{b},
                                __int128{a} - __int128{b},
                                __int128{a} * __int128{b}};
      const CheckedInt<C> results[] = {x + y, x - y, x * y};
      for (int op = 0; op < 3; ++op) {
        EXPECT_EQ(results[op].Overflow(), !fits(exact[op])) << "op " << op;
        if (fits(exact[op])) {
          EXPECT_EQ(results[op].Value(), static_cast<C>(exact[op]))
              << "op " << op;
        }
      }
    }
  }
}
#endif

// Operands of different signedness, where the signed operand does not fit in
// the unsigned common type, take dedicated code paths: check them against the
// exact result around the limits.
TEST(CheckedIntSpecificTest, MixedSignednessMatchesTheExactResult) {
#if defined(__SIZEOF_INT128__)
  ExpectExactMixedResults<int32_t, uint32_t>();
  ExpectExactMixedResults<uint32_t, int32_t>();
  ExpectExactMixedResults<int64_t, uint64_t>();
  ExpectExactMixedResults<uint64_t, int64_t>();
  ExpectExactMixedResults<int32_t, uint64_t>();
  ExpectExactMixedResults<uint64_t, int32_t>();
  ExpectExactMixedResults<int64_t, uint32_t>();
  ExpectExactMixedResults<uint32_t, int64_t>();
#else
  GTEST_SKIP() << "No 128-bit integer type for the reference results.";
#endif
}

}  // namespace
}  // namespace tflite
