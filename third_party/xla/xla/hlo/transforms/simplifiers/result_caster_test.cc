/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/result_caster.h"

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

class ResultCasterTest
    : public HloHardwareIndependentTestBase,
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
  const bool should_cast =
      result_type != accumulation_type &&
      primitive_util::HigherPrecisionType(accumulation_type, result_type) ==
          accumulation_type;
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
                                           std::make_tuple(F32, BF16, F32),
                                           std::make_tuple(BF16, F32, F64)));

TEST_F(ResultCasterTest, SparseDot) {
  absl::string_view kHlo = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[2,16]{1,0} parameter(0)
    p1 = f32[32,2]{1,0} parameter(1)
    meta = u16[2,2]{1,0} parameter(2)
    ROOT dot = bf16[2,2]{1,0} dot(p0, p1, meta),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(bool casted, ResultCaster().Run(module.get()));
  EXPECT_TRUE(casted);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Convert(::testing::MakeMatcher(new ::xla::testing::HloMatcher(
                  HloOpcode::kDot,
                  {op::Parameter(0), op::Parameter(1), op::Parameter(2)}))));
}

}  // namespace
}  // namespace xla
