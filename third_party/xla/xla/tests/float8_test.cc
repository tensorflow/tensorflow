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

#include <cmath>
#include <memory>
#include <vector>

#include "xla/client/xla_builder.h"
#include "xla/test.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

// Test FP8 floating-point types (F8E5M2, F8E4M3FN)
template <typename T>
class Float8Test : public ClientLibraryTestBase {};

using DataTypes = ::testing::Types<tsl::float8_e5m2, tsl::float8_e4m3fn>;
TYPED_TEST_SUITE(Float8Test, DataTypes);

XLA_TYPED_TEST(Float8Test, ScalarOperation) {
  XlaBuilder builder(this->TestName());
  auto x = ConstantR0<TypeParam>(&builder, static_cast<TypeParam>(2.0f));
  auto y = ConstantR0<TypeParam>(&builder, static_cast<TypeParam>(1.0f));
  Add(x, y);

  this->template ComputeAndCompareR0<TypeParam>(
      &builder, static_cast<TypeParam>(3.0f), {});
}

XLA_TYPED_TEST(Float8Test, LogOperation) {
  XlaBuilder builder(this->TestName());
  auto x = ConstantR0<TypeParam>(&builder, static_cast<TypeParam>(4.0f));
  Log(x);

  this->template ComputeAndCompareR0<TypeParam>(
      &builder, static_cast<TypeParam>(1.387f), {});
}

XLA_TYPED_TEST(Float8Test, CompareOperation) {
  XlaBuilder builder(this->TestName());
  auto x = ConstantR1<TypeParam>(&builder, {TypeParam{1.0}, TypeParam{2.0}});
  auto y = ConstantR1<TypeParam>(&builder, {TypeParam{1.0}, TypeParam{3.0}});
  Eq(x, y);
  this->template ComputeAndCompareR1<bool>(&builder, {true, false}, {});
}

XLA_TYPED_TEST(Float8Test, DotOperation) {
  XlaBuilder builder(this->TestName());
  auto x = ConstantR2<TypeParam>(&builder, {{TypeParam{0.0}, TypeParam{1.0}},
                                            {TypeParam{2.0}, TypeParam{3.0}}});
  auto y = ConstantR2<TypeParam>(&builder, {{TypeParam{3.0}, TypeParam{2.0}},
                                            {TypeParam{1.0}, TypeParam{0.0}}});
  Dot(x, y);
  this->template ComputeAndCompareR2<TypeParam>(
      &builder,
      {{TypeParam{1.0}, TypeParam{0.0}}, {TypeParam{9.0}, TypeParam{4.0}}}, {});
}

XLA_TYPED_TEST(Float8Test, NegateScalar) {
  XlaBuilder builder(this->TestName());
  Neg(ConstantR0<TypeParam>(&builder, static_cast<TypeParam>(2.0f)));

  this->template ComputeAndCompareR0<TypeParam>(
      &builder, static_cast<TypeParam>(-2.0f), {});
}

}  // namespace
}  // namespace xla
