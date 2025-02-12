/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/constant_value.h"

#include <gtest/gtest.h>
#include "xla/literal_util.h"

namespace xla {
namespace {

class ConstantValueTest : public ::testing::Test {};

TEST_F(ConstantValueTest, ZeroTest32) {
  ConstantValue zero = ConstantValue::GetZero(32, /*is_signed=*/false);
  EXPECT_EQ(zero.GetSignedValue(), 0);
  EXPECT_EQ(zero.GetUnsignedValue(), 0);
  EXPECT_EQ(zero.GetBitwidth(), 32);
  EXPECT_FALSE(zero.IsSigned());

  ConstantValue zero_s = ConstantValue::GetZero(32, /*is_signed=*/true);
  EXPECT_EQ(zero_s.GetSignedValue(), 0);
  EXPECT_EQ(zero_s.GetUnsignedValue(), 0);
  EXPECT_EQ(zero_s.GetBitwidth(), 32);
  EXPECT_TRUE(zero_s.IsSigned());
}

TEST_F(ConstantValueTest, OneTest32) {
  ConstantValue one = ConstantValue::GetOne(32, /*is_signed=*/false);
  EXPECT_EQ(one.GetSignedValue(), 1);
  EXPECT_EQ(one.GetUnsignedValue(), 1);
  EXPECT_EQ(one.GetBitwidth(), 32);
  EXPECT_FALSE(one.IsSigned());

  ConstantValue one_s = ConstantValue::GetOne(32, /*is_signed=*/true);
  EXPECT_EQ(one_s.GetSignedValue(), 1);
  EXPECT_EQ(one_s.GetUnsignedValue(), 1);
  EXPECT_EQ(one_s.GetBitwidth(), 32);
  EXPECT_TRUE(one_s.IsSigned());
}

TEST_F(ConstantValueTest, Signed23) {
  // 4194303 is 2^22 - 1
  ConstantValue signed_number = ConstantValue::GetSigned(4194303, 23);
  EXPECT_EQ(signed_number.GetSignedValue(), 4194303);
  EXPECT_EQ(signed_number.GetBitwidth(), 23);
  EXPECT_TRUE(signed_number.IsSigned());

  // 4194304 is 2^22
  ConstantValue signed_number_of = ConstantValue::GetSigned(4194304, 23);
  // Verifying that if we get beyond the limit we are wrapping around.
  EXPECT_EQ(signed_number_of.GetSignedValue(), -4194304);
  EXPECT_EQ(signed_number_of.GetBitwidth(), 23);
  EXPECT_TRUE(signed_number_of.IsSigned());
}

TEST_F(ConstantValueTest, Unsigned23) {
  // 8388607 is 2^23 - 1
  ConstantValue unsigned_number = ConstantValue::GetUnsigned(8388607, 23);
  EXPECT_EQ(unsigned_number.GetUnsignedValue(), 8388607);
  EXPECT_EQ(unsigned_number.GetBitwidth(), 23);
  EXPECT_FALSE(unsigned_number.IsSigned());

  // 8388608 is 2^23
  ConstantValue unsigned_number_of = ConstantValue::GetUnsigned(8388608, 23);
  // Verifying that if we get beyond the limit we are wrapping around.
  EXPECT_EQ(unsigned_number_of.GetUnsignedValue(), 0);
  EXPECT_EQ(unsigned_number_of.GetBitwidth(), 23);
  EXPECT_FALSE(unsigned_number_of.IsSigned());
}

TEST_F(ConstantValueTest, FromLiteral) {
  auto cv_8 = ConstantValue::FromLiteral(
      LiteralUtil::CreateR0(static_cast<int8_t>(-32)));
  EXPECT_TRUE(cv_8.ok());
  EXPECT_TRUE(cv_8->IsSigned());
  EXPECT_EQ(cv_8->GetBitwidth(), 8);
  EXPECT_EQ(cv_8->GetSignedValue(), -32);

  auto cv_u8 = ConstantValue::FromLiteral(
      LiteralUtil::CreateR0(static_cast<int8_t>(32)));
  EXPECT_TRUE(cv_u8.ok());
  EXPECT_TRUE(cv_u8->IsSigned());
  EXPECT_EQ(cv_u8->GetBitwidth(), 8);
  EXPECT_EQ(cv_u8->GetUnsignedValue(), 32);

  auto cv_16 = ConstantValue::FromLiteral(
      LiteralUtil::CreateR0(static_cast<int16_t>(32000)));
  EXPECT_TRUE(cv_16.ok());
  EXPECT_TRUE(cv_16->IsSigned());
  EXPECT_EQ(cv_16->GetBitwidth(), 16);
  EXPECT_EQ(cv_16->GetSignedValue(), 32000);

  auto cv_u16 = ConstantValue::FromLiteral(
      LiteralUtil::CreateR0(static_cast<uint16_t>(33000)));
  EXPECT_TRUE(cv_u16.ok());
  EXPECT_FALSE(cv_u16->IsSigned());
  EXPECT_EQ(cv_u16->GetBitwidth(), 16);
  EXPECT_EQ(cv_u16->GetUnsignedValue(), 33000);

  auto cv_32 = ConstantValue::FromLiteral(
      LiteralUtil::CreateR0(static_cast<int32_t>(-2000000000)));
  EXPECT_TRUE(cv_32.ok());
  EXPECT_TRUE(cv_32->IsSigned());
  EXPECT_EQ(cv_32->GetBitwidth(), 32);
  EXPECT_EQ(cv_32->GetSignedValue(), -2000000000);

  auto cv_u32 = ConstantValue::FromLiteral(
      LiteralUtil::CreateR0(static_cast<uint32_t>(3000000000)));
  EXPECT_TRUE(cv_u32.ok());
  EXPECT_FALSE(cv_u32->IsSigned());
  EXPECT_EQ(cv_u32->GetBitwidth(), 32);
  EXPECT_EQ(cv_u32->GetUnsignedValue(), 3000000000);

  auto cv_64 = ConstantValue::FromLiteral(
      LiteralUtil::CreateR0(static_cast<int64_t>(3000000000)));
  EXPECT_TRUE(cv_64.ok());
  EXPECT_TRUE(cv_64->IsSigned());
  EXPECT_EQ(cv_64->GetBitwidth(), 64);
  EXPECT_EQ(cv_64->GetSignedValue(), 3000000000);

  auto cv_u64 = ConstantValue::FromLiteral(
      LiteralUtil::CreateR0(static_cast<uint64_t>(6000000000)));
  EXPECT_TRUE(cv_u64.ok());
  EXPECT_FALSE(cv_u64->IsSigned());
  EXPECT_EQ(cv_u64->GetBitwidth(), 64);
  EXPECT_EQ(cv_u64->GetUnsignedValue(), 6000000000);
}

TEST_F(ConstantValueTest, Add) {
  // 8388607 is 2^23 - 1
  ConstantValue lhs = ConstantValue::GetUnsigned(8388607, 23);
  ConstantValue rhs = ConstantValue::GetUnsigned(1, 23);
  // Result should overflow.
  ConstantValue result = lhs.add(rhs);
  EXPECT_EQ(result.GetUnsignedValue(), 0);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_FALSE(result.IsSigned());

  lhs = ConstantValue::GetUnsigned(8388600, 23);
  rhs = ConstantValue::GetUnsigned(7, 23);
  result = lhs.add(rhs);
  // Verifying some unsigned computation.
  EXPECT_EQ(result.GetUnsignedValue(), 8388607);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_FALSE(result.IsSigned());

  lhs = ConstantValue::GetSigned(-10, 23);
  rhs = ConstantValue::GetSigned(4, 23);
  result = lhs.add(rhs);
  // Verifying some signed computation.
  EXPECT_EQ(result.GetSignedValue(), -6);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_TRUE(result.IsSigned());

  lhs = ConstantValue::GetSigned(-4194304, 23);
  rhs = ConstantValue::GetSigned(-1, 23);
  result = lhs.add(rhs);
  // Verifying that if we get beyond the limit we are wrapping around.
  EXPECT_EQ(result.GetSignedValue(), 4194303);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_TRUE(result.IsSigned());
}

TEST_F(ConstantValueTest, Sub) {
  // 8388607 is 2^23 - 1
  ConstantValue lhs = ConstantValue::GetUnsigned(8388607, 23);
  ConstantValue rhs = ConstantValue::GetUnsigned(1, 23);
  // Test subtraction of unsigned numbers.
  ConstantValue result = lhs.sub(rhs);
  EXPECT_EQ(result.GetUnsignedValue(), 8388606);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_FALSE(result.IsSigned());

  lhs = ConstantValue::GetUnsigned(6, 23);
  rhs = ConstantValue::GetUnsigned(7, 23);
  result = lhs.sub(rhs);
  // Verifying some unsigned computation.
  EXPECT_EQ(result.GetUnsignedValue(), 8388607);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_FALSE(result.IsSigned());

  lhs = ConstantValue::GetSigned(-10, 23);
  rhs = ConstantValue::GetSigned(4, 23);
  result = lhs.sub(rhs);
  // Verifying some signed computation.
  EXPECT_EQ(result.GetSignedValue(), -14);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_TRUE(result.IsSigned());

  lhs = ConstantValue::GetSigned(-4194304, 23);
  rhs = ConstantValue::GetSigned(1, 23);
  result = lhs.sub(rhs);
  // Verifying that if we get beyond the limit we are wrapping around.
  EXPECT_EQ(result.GetSignedValue(), 4194303);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_TRUE(result.IsSigned());
}

TEST_F(ConstantValueTest, Div) {
  ConstantValue lhs = ConstantValue::GetUnsigned(94, 23);
  ConstantValue rhs = ConstantValue::GetUnsigned(47, 23);
  // Test division of unsigned numbers
  ConstantValue result = lhs.div(rhs);
  EXPECT_EQ(result.GetUnsignedValue(), 2);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_FALSE(result.IsSigned());

  lhs = ConstantValue::GetUnsigned(6, 23);
  rhs = ConstantValue::GetUnsigned(7, 23);
  result = lhs.div(rhs);
  // Test flooring to 0.
  EXPECT_EQ(result.GetUnsignedValue(), 0);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_FALSE(result.IsSigned());

  lhs = ConstantValue::GetSigned(-10, 23);
  rhs = ConstantValue::GetSigned(4, 23);
  result = lhs.div(rhs);
  // Test dividing signed numbers and that sign is respected.
  EXPECT_EQ(result.GetSignedValue(), -2);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_TRUE(result.IsSigned());

  lhs = ConstantValue::GetSigned(-4194304, 23);
  rhs = ConstantValue::GetSigned(2, 23);
  result = lhs.div(rhs);
  // Verifying that if we get beyond the limit we are wrapping around.
  EXPECT_EQ(result.GetSignedValue(), -2097152);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_TRUE(result.IsSigned());
}

TEST_F(ConstantValueTest, Mod) {
  ConstantValue lhs = ConstantValue::GetUnsigned(94, 23);
  ConstantValue rhs = ConstantValue::GetUnsigned(47, 23);
  // Test modulo of unsigned numbers
  ConstantValue result = lhs.mod(rhs);
  EXPECT_EQ(result.GetUnsignedValue(), 0);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_FALSE(result.IsSigned());

  lhs = ConstantValue::GetUnsigned(6, 23);
  rhs = ConstantValue::GetUnsigned(7, 23);
  result = lhs.mod(rhs);
  // Test modulo of numbers less than divisor.
  EXPECT_EQ(result.GetUnsignedValue(), 6);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_FALSE(result.IsSigned());

  lhs = ConstantValue::GetSigned(-10, 23);
  rhs = ConstantValue::GetSigned(3, 23);
  result = lhs.mod(rhs);
  // Verify that signed numbers and their sign are handled correctly.
  EXPECT_EQ(result.GetSignedValue(), -1);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_TRUE(result.IsSigned());

  lhs = ConstantValue::GetSigned(-4194304, 23);
  rhs = ConstantValue::GetSigned(1, 23);
  result = lhs.mod(rhs);
  // Verifying that if we get beyond the limit we are wrapping around.
  EXPECT_EQ(result.GetSignedValue(), 0);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_TRUE(result.IsSigned());
}

TEST_F(ConstantValueTest, Mul) {
  ConstantValue lhs = ConstantValue::GetUnsigned(94, 23);
  ConstantValue rhs = ConstantValue::GetUnsigned(47, 23);
  // Test multiply of unsigned numbers
  ConstantValue result = lhs.mul(rhs);
  EXPECT_EQ(result.GetUnsignedValue(), 4418);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_FALSE(result.IsSigned());

  lhs = ConstantValue::GetUnsigned(8388607, 23);
  rhs = ConstantValue::GetUnsigned(2, 23);
  result = lhs.mul(rhs);
  // Test multiply of numbers less than divisor.
  EXPECT_EQ(result.GetUnsignedValue(), 8388606);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_FALSE(result.IsSigned());

  lhs = ConstantValue::GetSigned(-10, 23);
  rhs = ConstantValue::GetSigned(3, 23);
  result = lhs.mul(rhs);
  // Verify that signed numbers and their sign are handled correctly.
  EXPECT_EQ(result.GetSignedValue(), -30);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_TRUE(result.IsSigned());

  lhs = ConstantValue::GetSigned(-4194304, 23);
  rhs = ConstantValue::GetSigned(2, 23);
  result = lhs.mod(rhs);
  // Verifying that if we get beyond the limit we are wrapping around.
  EXPECT_EQ(result.GetSignedValue(), 0);
  EXPECT_EQ(result.GetBitwidth(), 23);
  EXPECT_TRUE(result.IsSigned());
}

TEST_F(ConstantValueTest, LtGtEq) {
  ConstantValue lhs = ConstantValue::GetUnsigned(94, 23);
  ConstantValue rhs = ConstantValue::GetUnsigned(47, 23);
  // Test comparison of some numbers.
  EXPECT_FALSE(lhs.lt(rhs));
  EXPECT_TRUE(lhs.gt(rhs));

  lhs = ConstantValue::GetUnsigned(8388607, 23);
  rhs = ConstantValue::GetUnsigned(2, 23);
  // Test comparison with numbers at boundary of range.
  EXPECT_FALSE(lhs.lt(rhs));
  EXPECT_TRUE(lhs.gt(rhs));

  lhs = ConstantValue::GetSigned(-10, 23);
  rhs = ConstantValue::GetSigned(3, 23);
  // Test comparison with signed numbers.

  lhs = ConstantValue::GetSigned(-4194304, 23);
  rhs = ConstantValue::GetSigned(2, 23);
  // Test comparison with signed numbers at boundary of range.
  EXPECT_TRUE(lhs.lt(rhs));
  EXPECT_FALSE(lhs.gt(rhs));

  lhs = ConstantValue::GetUnsigned(43, 23);
  rhs = ConstantValue::GetUnsigned(43, 23);
  // Test equality unsigned numbers.
  EXPECT_TRUE(lhs.eq(rhs));
  EXPECT_TRUE(rhs.eq(lhs));

  lhs = ConstantValue::GetSigned(-10, 23);
  rhs = ConstantValue::GetSigned(-10, 23);
  // Test equality signed numbers.
  EXPECT_TRUE(lhs.eq(rhs));
  EXPECT_TRUE(rhs.eq(lhs));

  lhs = ConstantValue::GetUnsigned(4194304, 23);
  rhs = ConstantValue::GetUnsigned(2, 23);
  // Test inequality unsigned numbers.
  EXPECT_FALSE(lhs.eq(rhs));
  EXPECT_FALSE(rhs.eq(lhs));

  lhs = ConstantValue::GetSigned(-4194304, 23);
  rhs = ConstantValue::GetSigned(2, 23);
  // Test inequality signed numbers.
  EXPECT_FALSE(lhs.eq(rhs));
  EXPECT_FALSE(rhs.eq(lhs));
}

}  // namespace
}  // namespace xla
