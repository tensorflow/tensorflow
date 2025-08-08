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
#include "xla/service/gpu/transforms/scaled_dot_rewriter.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
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

  const std::string hlo_string = absl::Substitute(
      R"(
    HloModule module

    ENTRY main {
      lhs = $0[8,16] parameter(0)
      lhs_scale = $1[] parameter(1)
      rhs = $0[16,32] parameter(2)
      rhs_scale = $1[] parameter(3)
      ROOT dot = f32[8,32] scaled-dot(lhs, lhs_scale, rhs, rhs_scale), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )",
      absl::AsciiStrToLower(PrimitiveType_Name(test_case.operand_type)),
      absl::AsciiStrToLower(PrimitiveType_Name(test_case.scale_type)));
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ScaledDotRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  LOG(ERROR) << "MODULE after rewriter: " << module->ToString();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kDot);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kMultiply);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kMultiply);

  EXPECT_EQ(root->operand(0)->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(root->operand(0)->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(root->operand(0)->operand(1)->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(root->operand(0)->operand(1)->operand(0)->opcode(),
            HloOpcode::kConvert);
  EXPECT_EQ(root->operand(0)->operand(1)->operand(0)->shape().element_type(),
            BF16);

  EXPECT_EQ(root->operand(1)->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(root->operand(1)->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(root->operand(1)->operand(1)->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(root->operand(1)->operand(1)->operand(0)->opcode(),
            HloOpcode::kConvert);
  EXPECT_EQ(root->operand(1)->operand(1)->operand(0)->shape().element_type(),
            BF16);
}

INSTANTIATE_TEST_SUITE_P(
    ScaledDotRewriterTests, ScaledDotRewriterTestFixture,
    ::testing::ValuesIn<ScaledDotRewriterTestCase>({
        {PrimitiveType::F8E4M3FN, PrimitiveType::F8E8M0FNU},
        {PrimitiveType::F8E5M2, PrimitiveType::F8E8M0FNU},
        {PrimitiveType::BF16, PrimitiveType::F8E8M0FNU},
        {PrimitiveType::BF16, PrimitiveType::BF16},
        {PrimitiveType::S4, PrimitiveType::BF16},
        {PrimitiveType::F32, PrimitiveType::F8E8M0FNU},
    }),
    [](const ::testing::TestParamInfo<ScaledDotRewriterTest::ParamType>& info) {
      return absl::StrCat(PrimitiveType_Name(info.param.operand_type), "_",
                          PrimitiveType_Name(info.param.scale_type));
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
