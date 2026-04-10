/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/util.h"

#include <stddef.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using testing::ElementsAreArray;

TEST(ConvertVectorToTfLiteIntArray, TestWithVector) {
  std::vector<int> input = {1, 2};
  TfLiteIntArray* output = ConvertVectorToTfLiteIntArray(input);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size, 2);
  EXPECT_EQ(output->data[0], 1);
  EXPECT_EQ(output->data[1], 2);
  TfLiteIntArrayFree(output);
}

TEST(ConvertVectorToTfLiteIntArray, TestWithEmptyVector) {
  std::vector<int> input;
  TfLiteIntArray* output = ConvertVectorToTfLiteIntArray(input);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size, 0);
  TfLiteIntArrayFree(output);
}

TEST(UtilTest, IsFlexOp) {
  EXPECT_TRUE(IsFlexOp("Flex"));
  EXPECT_TRUE(IsFlexOp("FlexOp"));
  EXPECT_FALSE(IsFlexOp("flex"));
  EXPECT_FALSE(IsFlexOp("Fle"));
  EXPECT_FALSE(IsFlexOp("OpFlex"));
  EXPECT_FALSE(IsFlexOp(nullptr));
  EXPECT_FALSE(IsFlexOp(""));
}

TEST(EqualArrayAndTfLiteIntArray, TestWithTFLiteArrayEmpty) {
  int input[] = {1, 2, 3, 4};
  EXPECT_FALSE(EqualArrayAndTfLiteIntArray(nullptr, 4, input));
}

TEST(EqualArrayAndTfLiteIntArray, TestWithTFLiteArrayWrongSize) {
  int input[] = {1, 2, 3, 4};
  TfLiteIntArray* output = ConvertArrayToTfLiteIntArray(4, input);
  EXPECT_FALSE(EqualArrayAndTfLiteIntArray(output, 3, input));
  free(output);
}

TEST(EqualArrayAndTfLiteIntArray, TestMismatch) {
  int input[] = {1, 2, 3, 4};
  TfLiteIntArray* output = ConvertVectorToTfLiteIntArray({1, 2, 2, 4});
  EXPECT_FALSE(EqualArrayAndTfLiteIntArray(output, 4, input));
  free(output);
}

TEST(EqualArrayAndTfLiteIntArray, TestMatch) {
  int input[] = {1, 2, 3, 4};
  TfLiteIntArray* output = ConvertArrayToTfLiteIntArray(4, input);
  EXPECT_TRUE(EqualArrayAndTfLiteIntArray(output, 4, input));
  free(output);
}

TEST(CombineHashes, TestHashOutputsEquals) {
  size_t output1 = CombineHashes({1, 2, 3, 4});
  size_t output2 = CombineHashes({1, 2, 3, 4});
  EXPECT_EQ(output1, output2);
}

TEST(CombineHashes, TestHashOutputsDifferent) {
  size_t output1 = CombineHashes({1, 2, 3, 4});
  size_t output2 = CombineHashes({1, 2, 2, 4});
  EXPECT_NE(output1, output2);
}

TEST(GetOpNameByRegistration, ValidBuiltinCode) {
  TfLiteRegistration registration{};
  registration.builtin_code = tflite::BuiltinOperator_ADD;
  const auto op_name = GetOpNameByRegistration(registration);
  EXPECT_EQ("ADD", op_name);
}

TEST(GetOpNameByRegistration, InvalidBuiltinCode) {
  TfLiteRegistration registration{};
  registration.builtin_code = -1;
  const auto op_name = GetOpNameByRegistration(registration);
  EXPECT_EQ("", op_name);
}

TEST(GetOpNameByRegistration, CustomName) {
  TfLiteRegistration registration{};
  registration.builtin_code = tflite::BuiltinOperator_CUSTOM;
  registration.custom_name = "TestOp";
  auto op_name = GetOpNameByRegistration(registration);
  EXPECT_EQ("CUSTOM TestOp", op_name);

  registration.builtin_code = tflite::BuiltinOperator_DELEGATE;
  registration.custom_name = "TestDelegate";
  op_name = GetOpNameByRegistration(registration);
  EXPECT_EQ("DELEGATE TestDelegate", op_name);
}

TEST(ValidationSubgraph, NameIsDetected) {
  EXPECT_FALSE(IsValidationSubgraph(nullptr));
  EXPECT_FALSE(IsValidationSubgraph(""));
  EXPECT_FALSE(IsValidationSubgraph("a name"));
  EXPECT_FALSE(IsValidationSubgraph("VALIDATIONfoo"));
  EXPECT_TRUE(IsValidationSubgraph("VALIDATION:"));
  EXPECT_TRUE(IsValidationSubgraph("VALIDATION:main"));
}

TEST(MultiplyAndCheckOverflow, Validate) {
  size_t res = 0;
  EXPECT_TRUE(MultiplyAndCheckOverflow(1, 2, &res) == kTfLiteOk);
  EXPECT_FALSE(MultiplyAndCheckOverflow(static_cast<size_t>(123456789023),
                                        1223423425, &res) == kTfLiteOk);
}

TEST(FourBitTest, BytesRequiredEven) {
  TfLiteContext context;

  int dims[] = {2, 3, 1, 5};
  const int* dims_ptr = &dims[0];
  size_t dims_size = 4;
  size_t required_bytes_four_bit;
  tflite::BytesRequired(kTfLiteInt4, dims_ptr, dims_size,
                        &required_bytes_four_bit, &context);

  ASSERT_EQ(required_bytes_four_bit, 15);
}

TEST(FourBitTest, BytesRequiredOdd) {
  TfLiteContext context;

  int dims[] = {5, 1, 1, 1};
  const int* dims_ptr = &dims[0];
  size_t dims_size = 2;
  size_t required_bytes_four_bit;
  tflite::BytesRequired(kTfLiteInt4, dims_ptr, dims_size,
                        &required_bytes_four_bit, &context);

  ASSERT_EQ(required_bytes_four_bit, 3);
}

TEST(TestMakeUniqueTensor, Valid) {
  TensorUniquePtr t = BuildTfLiteTensor(kTfLiteInt32, {2, 3}, kTfLiteDynamic);
  ASSERT_NE(t.get(), nullptr);
  ASSERT_EQ(t->buffer_handle, kTfLiteNullBufferHandle);

  EXPECT_THAT(t.get(), DimsAre({2, 3}));
  EXPECT_EQ(t->bytes, 24);

  EXPECT_EQ(t->type, kTfLiteInt32);
  EXPECT_EQ(t->allocation_type, kTfLiteDynamic);

  // Check memory has been properly allocated.
  int* data = t->data.i32;
  std::fill_n(data, 6, 0);
  ASSERT_NE(data, nullptr);
  ASSERT_THAT(std::vector<int>(data, data + 6),
              ElementsAreArray({0, 0, 0, 0, 0, 0}));
}

TEST(TestMakeUniqueTensor, NullDimsReturnsNull) {
  TensorUniquePtr t = BuildTfLiteTensor(kTfLiteInt32, nullptr, kTfLiteDynamic);
  ASSERT_EQ(t.get(), nullptr);
}

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

}  // namespace
}  // namespace tflite
