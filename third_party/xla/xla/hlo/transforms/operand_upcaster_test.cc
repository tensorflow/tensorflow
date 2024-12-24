/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/transforms/operand_upcaster.h"

#include <memory>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

class OperandUpcasterTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType, PrimitiveType>> {};

bool ShouldUpcast(PrimitiveType operand_type, PrimitiveType result_type) {
  return operand_type != result_type &&
         primitive_util::HigherPrecisionType(operand_type, result_type) ==
             result_type;
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
                         ::testing::Combine(::testing::Values(S8, U8, S16),
                                            ::testing::Values(S8, U8, S16),
                                            ::testing::Values(S32)));

INSTANTIATE_TEST_SUITE_P(U32, OperandUpcasterTest,
                         ::testing::Combine(::testing::Values(U8, U16),
                                            ::testing::Values(U8, U16),
                                            ::testing::Values(U32)));

INSTANTIATE_TEST_SUITE_P(BF16, OperandUpcasterTest,
                         ::testing::Combine(::testing::Values(BF16, S8, U8),
                                            ::testing::Values(BF16, S8, U8),
                                            ::testing::Values(BF16)));

INSTANTIATE_TEST_SUITE_P(F32, OperandUpcasterTest,
                         ::testing::Combine(::testing::Values(BF16, F16),
                                            ::testing::Values(BF16, F16),
                                            ::testing::Values(F32)));

INSTANTIATE_TEST_SUITE_P(NoUpcast, OperandUpcasterTest,
                         ::testing::Values(std::make_tuple(F32, F32, BF16),
                                           std::make_tuple(S32, S32, U32)));

TEST_F(OperandUpcasterTest, SparseDot) {
  absl::string_view kHlo = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[2,16]{1,0} parameter(0)
    p1 = bf16[32,2]{1,0} parameter(1)
    meta = u16[2,2]{1,0} parameter(2)
    ROOT dot = f32[2,2]{1,0} dot(p0, p1, meta),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(bool upcasted, OperandUpcaster().Run(module.get()));
  EXPECT_TRUE(upcasted);
  auto upcasted_lhs =
      AllOf(op::Convert(op::Parameter(0)), op::Shape("f32[2,16]{1,0}"));
  auto upcasted_rhs =
      AllOf(op::Convert(op::Parameter(1)), op::Shape("f32[32,2]{1,0}"));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(::testing::MakeMatcher(new ::xla::testing::HloMatcher(
                        HloOpcode::kDot,
                        {upcasted_lhs, upcasted_rhs, op::Parameter(2)})),
                    op::Shape("f32[2,2]{1,0}")));
}

}  // namespace

}  // namespace xla
