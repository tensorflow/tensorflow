/* Copyright 2025 The OpenXLA Authors.
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
#include "xla/backends/gpu/transforms/scaled_dot_rewriter.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

struct ScaledDotRewriterTestCase {
  PrimitiveType operand_type;
  PrimitiveType scale_type;
};

using ScaledDotRewriterTest =
    ::testing::WithParamInterface<ScaledDotRewriterTestCase>;

class ScaledDotRewriterTestFixture : public HloTestBase,
                                     public ScaledDotRewriterTest {
 public:
  void SetUp() override {}
};

TEST_P(ScaledDotRewriterTestFixture, ScaledDot) {
  const ScaledDotRewriterTestCase& test_case = GetParam();

  // lhs_scale should have two dim
  const std::string hlo_string = absl::Substitute(
      R"(
        HloModule module

        ENTRY main {
          lhs = $0[1024,512] parameter(0)
          rhs = $0[64,512] parameter(1)
          lhs_scale = $1[32,2] parameter(2)
          rhs_scale = $1[64,2] parameter(3)
          ROOT dot = f32[1024,64] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
            lhs_contracting_dims={1},
            rhs_contracting_dims={1}
        }
      )",
      absl::AsciiStrToLower(PrimitiveType_Name(test_case.operand_type)),
      absl::AsciiStrToLower(PrimitiveType_Name(test_case.scale_type)));
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  ScaledDotRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify that the module is still valid after the rewrite.
  auto status_or_module = ParseAndReturnVerifiedModule(module->ToString());
  EXPECT_TRUE(status_or_module.status().ok()) << status_or_module.status();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kDot);
  for (const HloInstruction* operand : root->operands()) {
    std::vector<HloOpcode> actual_op_codes{};
    while (operand->opcode() != HloOpcode::kParameter) {
      actual_op_codes.push_back(operand->opcode());
      if (operand->opcode() == HloOpcode::kMultiply) {
        operand = operand->operand(1);
      } else {
        operand = operand->operand(0);
      }
    }
    actual_op_codes = std::vector<HloOpcode>(actual_op_codes.rbegin(),
                                             actual_op_codes.rend());

    if (test_case.scale_type == PrimitiveType::BF16) {
      const std::vector<HloOpcode> expected_op_codes{
          HloOpcode::kBroadcast, HloOpcode::kReshape, HloOpcode::kMultiply};
      EXPECT_THAT(actual_op_codes, expected_op_codes);
    } else {
      const std::vector<HloOpcode> expected_op_codes_with_convert{
          HloOpcode::kConvert, HloOpcode::kBroadcast, HloOpcode::kReshape,
          HloOpcode::kMultiply};
      EXPECT_THAT(actual_op_codes, expected_op_codes_with_convert);
    }
  }

  EXPECT_TRUE(RunAndCompare(std::move(module),
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

INSTANTIATE_TEST_SUITE_P(
    ScaledDotRewriterTests, ScaledDotRewriterTestFixture,
    ::testing::ValuesIn<ScaledDotRewriterTestCase>({
        {PrimitiveType::F8E4M3FN, PrimitiveType::F8E8M0FNU},
        {PrimitiveType::F8E5M2, PrimitiveType::F8E8M0FNU},
        {PrimitiveType::BF16, PrimitiveType::BF16},
        {PrimitiveType::S4, PrimitiveType::BF16},
    }),
    [](const ::testing::TestParamInfo<ScaledDotRewriterTest::ParamType>& info) {
      return absl::StrCat(PrimitiveType_Name(info.param.operand_type), "_",
                          PrimitiveType_Name(info.param.scale_type));
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
