/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/result_caster.h"

#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

class ResultCasterTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType, PrimitiveType>> {};

TEST_P(ResultCasterTest, CastResultWhenNeeded) {
  PrimitiveType lhs_type, rhs_type, result_type;
  std::tie(lhs_type, rhs_type, result_type) = GetParam();
  absl::string_view module_tmpl = R"(
  HloModule module

  ENTRY main {
    p0 = $0[2,3]{1,0} parameter(0)
    p1 = $1[3,2]{1,0} parameter(1)
    ROOT dot = $2[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
  })";
  auto module_string = absl::Substitute(
      module_tmpl, primitive_util::LowercasePrimitiveTypeName(lhs_type),
      primitive_util::LowercasePrimitiveTypeName(rhs_type),
      primitive_util::LowercasePrimitiveTypeName(result_type));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool casted, ResultCaster().Run(module.get()));
  const PrimitiveType accumulation_type =
      primitive_util::HigherPrecisionType(lhs_type, rhs_type);
  const bool should_cast = result_type != accumulation_type;
  EXPECT_EQ(casted, should_cast);
  auto lhs = op::Parameter(0);
  auto rhs = op::Parameter(1);
  auto original_shape_str = absl::Substitute(
      "$0[2,2]{1,0}", primitive_util::LowercasePrimitiveTypeName(result_type));
  auto accumulation_shape_str = absl::Substitute(
      "$0[2,2]{1,0}",
      primitive_util::LowercasePrimitiveTypeName(accumulation_type));
  if (should_cast) {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                AllOf(op::Convert(AllOf(op::Dot(lhs, rhs),
                                        op::Shape(accumulation_shape_str))),
                      op::Shape(original_shape_str)));
  } else {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                AllOf(op::Dot(lhs, rhs), op::Shape(original_shape_str)));
  }
}

INSTANTIATE_TEST_SUITE_P(All, ResultCasterTest,
                         ::testing::Values(std::make_tuple(BF16, BF16, S32),
                                           std::make_tuple(F32, F32, S32),
                                           std::make_tuple(F32, BF16, F32)));

}  // namespace
}  // namespace xla
