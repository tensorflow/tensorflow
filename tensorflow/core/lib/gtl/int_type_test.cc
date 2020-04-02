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

// Unit test cases for IntType.

#include "tensorflow/core/lib/gtl/int_type.h"

#include <memory>
#include <unordered_map>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

TF_LIB_GTL_DEFINE_INT_TYPE(Int8_IT, int8);
TF_LIB_GTL_DEFINE_INT_TYPE(UInt8_IT, uint8);
TF_LIB_GTL_DEFINE_INT_TYPE(Int16_IT, int16);
TF_LIB_GTL_DEFINE_INT_TYPE(UInt16_IT, uint16);
TF_LIB_GTL_DEFINE_INT_TYPE(Int32_IT, int32);
TF_LIB_GTL_DEFINE_INT_TYPE(Int64_IT, int64);
TF_LIB_GTL_DEFINE_INT_TYPE(UInt32_IT, uint32);
TF_LIB_GTL_DEFINE_INT_TYPE(UInt64_IT, uint64);
TF_LIB_GTL_DEFINE_INT_TYPE(Long_IT, long);  // NOLINT

template <typename IntType_Type>
class IntTypeTest : public ::testing::Test {};

// All tests below will be executed on all supported IntTypes.
typedef ::testing::Types<Int8_IT, UInt8_IT, Int16_IT, UInt16_IT, Int32_IT,
                         Int64_IT, UInt64_IT, Long_IT>
    SupportedIntTypes;

TYPED_TEST_SUITE(IntTypeTest, SupportedIntTypes);

TYPED_TEST(IntTypeTest, TestInitialization) {
  constexpr TypeParam a;
  constexpr TypeParam b(1);
  constexpr TypeParam c(b);
  EXPECT_EQ(0, a);  // default initialization to 0
  EXPECT_EQ(1, b);
  EXPECT_EQ(1, c);
}

TYPED_TEST(IntTypeTest, TestOperators) {
  TypeParam a(0);
  TypeParam b(1);
  TypeParam c(2);
  constexpr TypeParam d(3);
  constexpr TypeParam e(4);

  // On all EXPECT_EQ below, we use the accessor value() as to not invoke the
  // comparison operators which must themselves be tested.

  // -- UNARY OPERATORS --------------------------------------------------------
  EXPECT_EQ(0, (a++).value());
  EXPECT_EQ(2, (++a).value());
  EXPECT_EQ(2, (a--).value());
  EXPECT_EQ(0, (--a).value());

  EXPECT_EQ(true, !a);
  EXPECT_EQ(false, !b);
  static_assert(!d == false, "Unary operator! failed");

  EXPECT_EQ(a.value(), +a);
  static_assert(+d == d.value(), "Unary operator+ failed");
  EXPECT_EQ(-a.value(), -a);
  static_assert(-d == -d.value(), "Unary operator- failed");
  EXPECT_EQ(~a.value(), ~a);  // ~zero
  EXPECT_EQ(~b.value(), ~b);  // ~non-zero
  static_assert(~d == ~d.value(), "Unary operator~ failed");

  // -- ASSIGNMENT OPERATORS ---------------------------------------------------
  // We test all assignment operators using IntType and constant as arguments.
  // We also test the return from the operators.
  // From same IntType
  c = a = b;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  // From constant
  c = b = 2;
  EXPECT_EQ(2, b.value());
  EXPECT_EQ(2, c.value());
  // From same IntType
  c = a += b;
  EXPECT_EQ(3, a.value());
  EXPECT_EQ(3, c.value());
  c = a -= b;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a *= b;
  EXPECT_EQ(2, a.value());
  EXPECT_EQ(2, c.value());
  c = a /= b;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a <<= b;
  EXPECT_EQ(4, a.value());
  EXPECT_EQ(4, c.value());
  c = a >>= b;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a %= b;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  // From constant
  c = a += 2;
  EXPECT_EQ(3, a.value());
  EXPECT_EQ(3, c.value());
  c = a -= 2;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a *= 2;
  EXPECT_EQ(2, a.value());
  EXPECT_EQ(2, c.value());
  c = a /= 2;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a <<= 2;
  EXPECT_EQ(4, a.value());
  EXPECT_EQ(4, c.value());
  c = a >>= 2;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());
  c = a %= 2;
  EXPECT_EQ(1, a.value());
  EXPECT_EQ(1, c.value());

  // -- COMPARISON OPERATORS ---------------------------------------------------
  a = 0;
  b = 1;

  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a == 0);   // NOLINT
  EXPECT_FALSE(1 == a);  // NOLINT
  static_assert(d == d, "operator== failed");
  static_assert(d == 3, "operator== failed");
  static_assert(3 == d, "operator== failed");
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a != 1);   // NOLINT
  EXPECT_FALSE(0 != a);  // NOLINT
  static_assert(d != e, "operator!= failed");
  static_assert(d != 4, "operator!= failed");
  static_assert(4 != d, "operator!= failed");
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(a < 1);   // NOLINT
  EXPECT_FALSE(0 < a);  // NOLINT
  static_assert(d < e, "operator< failed");
  static_assert(d < 4, "operator< failed");
  static_assert(3 < e, "operator< failed");
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a <= 1);  // NOLINT
  EXPECT_TRUE(0 <= a);  // NOLINT
  static_assert(d <= e, "operator<= failed");
  static_assert(d <= 4, "operator<= failed");
  static_assert(3 <= e, "operator<= failed");
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(a > 1);  // NOLINT
  EXPECT_FALSE(0 > a);  // NOLINT
  static_assert(e > d, "operator> failed");
  static_assert(e > 3, "operator> failed");
  static_assert(4 > d, "operator> failed");
  EXPECT_FALSE(a >= b);
  EXPECT_FALSE(a >= 1);  // NOLINT
  EXPECT_TRUE(0 >= a);   // NOLINT
  static_assert(e >= d, "operator>= failed");
  static_assert(e >= 3, "operator>= failed");
  static_assert(4 >= d, "operator>= failed");

  // -- BINARY OPERATORS -------------------------------------------------------
  a = 1;
  b = 3;
  EXPECT_EQ(4, (a + b).value());
  EXPECT_EQ(4, (a + 3).value());
  EXPECT_EQ(4, (1 + b).value());
  static_assert((d + e).value() == 7, "Binary operator+ failed");
  static_assert((d + 4).value() == 7, "Binary operator+ failed");
  static_assert((3 + e).value() == 7, "Binary operator+ failed");
  EXPECT_EQ(2, (b - a).value());
  EXPECT_EQ(2, (b - 1).value());
  EXPECT_EQ(2, (3 - a).value());
  static_assert((e - d).value() == 1, "Binary operator- failed");
  static_assert((e - 3).value() == 1, "Binary operator- failed");
  static_assert((4 - d).value() == 1, "Binary operator- failed");
  EXPECT_EQ(3, (a * b).value());
  EXPECT_EQ(3, (a * 3).value());
  EXPECT_EQ(3, (1 * b).value());
  static_assert((d * e).value() == 12, "Binary operator* failed");
  static_assert((d * 4).value() == 12, "Binary operator* failed");
  static_assert((3 * e).value() == 12, "Binary operator* failed");
  EXPECT_EQ(0, (a / b).value());
  EXPECT_EQ(0, (a / 3).value());
  EXPECT_EQ(0, (1 / b).value());
  static_assert((d / e).value() == 0, "Binary operator/ failed");
  static_assert((d / 4).value() == 0, "Binary operator/ failed");
  static_assert((3 / e).value() == 0, "Binary operator/ failed");
  EXPECT_EQ(8, (a << b).value());
  EXPECT_EQ(8, (a << 3).value());
  EXPECT_EQ(8, (1 << b).value());
  static_assert((d << e).value() == 48, "Binary operator<< failed");
  static_assert((d << 4).value() == 48, "Binary operator<< failed");
  static_assert((3 << e).value() == 48, "Binary operator<< failed");
  b = 8;
  EXPECT_EQ(4, (b >> a).value());
  EXPECT_EQ(4, (b >> 1).value());
  EXPECT_EQ(4, (8 >> a).value());
  static_assert((d >> e).value() == 0, "Binary operator>> failed");
  static_assert((d >> 4).value() == 0, "Binary operator>> failed");
  static_assert((3 >> e).value() == 0, "Binary operator>> failed");
  b = 3;
  a = 2;
  EXPECT_EQ(1, (b % a).value());
  EXPECT_EQ(1, (b % 2).value());
  EXPECT_EQ(1, (3 % a).value());
  static_assert((e % d).value() == 1, "Binary operator% failed");
  static_assert((e % 3).value() == 1, "Binary operator% failed");
  static_assert((4 % d).value() == 1, "Binary operator% failed");
}

TYPED_TEST(IntTypeTest, TestHashFunctor) {
  std::unordered_map<TypeParam, char, typename TypeParam::Hasher> map;
  TypeParam a(0);
  map[a] = 'c';
  EXPECT_EQ('c', map[a]);
  map[++a] = 'o';
  EXPECT_EQ('o', map[a]);

  TypeParam b(a);
  EXPECT_EQ(typename TypeParam::Hasher()(a), typename TypeParam::Hasher()(b));
}

// Tests the use of the templatized value accessor that performs static_casts.
// We use -1 to force casting in unsigned integers.
TYPED_TEST(IntTypeTest, TestValueAccessor) {
  constexpr typename TypeParam::ValueType i = -1;
  constexpr TypeParam int_type(i);
  EXPECT_EQ(i, int_type.value());
  static_assert(int_type.value() == i, "value() failed");
  // The use of the keyword 'template' (suggested by Clang) is only necessary
  // as this code is part of a template class.  Weird syntax though.  Good news
  // is that only int_type.value<int>() is needed in most code.
  EXPECT_EQ(static_cast<int>(i), int_type.template value<int>());
  EXPECT_EQ(static_cast<int8>(i), int_type.template value<int8>());
  EXPECT_EQ(static_cast<int16>(i), int_type.template value<int16>());
  EXPECT_EQ(static_cast<int32>(i), int_type.template value<int32>());
  EXPECT_EQ(static_cast<uint32>(i), int_type.template value<uint32>());
  EXPECT_EQ(static_cast<int64>(i), int_type.template value<int64>());
  EXPECT_EQ(static_cast<uint64>(i), int_type.template value<uint64>());
  EXPECT_EQ(static_cast<long>(i), int_type.template value<long>());  // NOLINT
  static_assert(int_type.template value<int>() == static_cast<int>(i),
                "value<Value>() failed");
}

TYPED_TEST(IntTypeTest, TestMove) {
  // Check that the int types have move constructor/assignment.
  // We do this by composing a struct with an int type and a unique_ptr. This
  // struct can't be copied due to the unique_ptr, so it must be moved.
  // If this compiles, it means that the int types have move operators.
  struct NotCopyable {
    TypeParam inttype;
    std::unique_ptr<int> ptr;

    static NotCopyable Make(int i) {
      NotCopyable f;
      f.inttype = TypeParam(i);
      f.ptr.reset(new int(i));
      return f;
    }
  };

  // Test move constructor.
  NotCopyable foo = NotCopyable::Make(123);
  EXPECT_EQ(123, foo.inttype);
  EXPECT_EQ(123, *foo.ptr);

  // Test move assignment.
  foo = NotCopyable::Make(321);
  EXPECT_EQ(321, foo.inttype);
  EXPECT_EQ(321, *foo.ptr);
}

}  // namespace tensorflow
