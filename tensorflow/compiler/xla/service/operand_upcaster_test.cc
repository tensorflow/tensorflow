/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/operand_upcaster.h"

#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

class OperandUpcasterTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType, PrimitiveType>> {};

bool ShouldUpcast(PrimitiveType operand_type, PrimitiveType result_type) {
  return primitive_util::BitWidth(operand_type) <
         primitive_util::BitWidth(result_type);
}

TEST_P(OperandUpcasterTest, ConvertInserted) {
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
  TF_ASSERT_OK_AND_ASSIGN(bool upcasted, OperandUpcaster().Run(module.get()));
  EXPECT_EQ(upcasted, ShouldUpcast(lhs_type, result_type) ||
                          ShouldUpcast(rhs_type, result_type));
  auto original_lhs = op::Parameter(0);
  auto original_rhs = op::Parameter(1);
  auto upcasted_lhs =
      ShouldUpcast(lhs_type, result_type)
          ? AllOf(op::Convert(original_lhs),
                  op::Shape(absl::Substitute(
                      "$0[2,3]{1,0}",
                      primitive_util::LowercasePrimitiveTypeName(result_type))))
          : original_lhs;
  auto upcasted_rhs =
      ShouldUpcast(rhs_type, result_type)
          ? AllOf(op::Convert(original_rhs),
                  op::Shape(absl::Substitute(
                      "$0[3,2]{1,0}",
                      primitive_util::LowercasePrimitiveTypeName(result_type))))
          : original_rhs;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      AllOf(op::Dot(upcasted_lhs, upcasted_rhs),
            op::Shape(absl::Substitute(
                "$0[2,2]{1,0}",
                primitive_util::LowercasePrimitiveTypeName(result_type)))));
}

INSTANTIATE_TEST_SUITE_P(S16U16, OperandUpcasterTest,
                         ::testing::Values(std::make_tuple(S8, S8, S16),
                                           std::make_tuple(U8, U8, U16)));

INSTANTIATE_TEST_SUITE_P(S32, OperandUpcasterTest,
                         ::testing::Combine(::testing::Values(S8, S16),
                                            ::testing::Values(S8, S16),
                                            ::testing::Values(S32)));

INSTANTIATE_TEST_SUITE_P(U32, OperandUpcasterTest,
                         ::testing::Combine(::testing::Values(U8, U16),
                                            ::testing::Values(U8, U16),
                                            ::testing::Values(U32)));

INSTANTIATE_TEST_SUITE_P(F32, OperandUpcasterTest,
                         ::testing::Combine(::testing::Values(BF16, F16),
                                            ::testing::Values(BF16, F16),
                                            ::testing::Values(F32)));

}  // namespace

}  // namespace xla
