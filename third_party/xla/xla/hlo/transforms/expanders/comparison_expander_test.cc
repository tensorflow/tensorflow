/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/comparison_expander.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/substitute.h"
#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ComparisonExpanderTest = HloHardwareIndependentTestBase;

TEST_F(ComparisonExpanderTest, ExpandWeakOrderFloatComparisonMatchesEvaluator) {
  const float inf = std::numeric_limits<float>::infinity();
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const float neg_qnan = std::copysign(qnan, -1.0f);
  const std::vector<float> values = {-inf,  -1.0f, -0.0f,    +0.0f,
                                     +1.0f, +inf,  neg_qnan, qnan};
  std::vector<float> lhs_vec;
  std::vector<float> rhs_vec;
  for (float a : values) {
    for (float b : values) {
      lhs_vec.push_back(a);
      rhs_vec.push_back(b);
    }
  }
  Literal lhs_lit = LiteralUtil::CreateR1<float>(lhs_vec);
  Literal rhs_lit = LiteralUtil::CreateR1<float>(rhs_vec);

  for (ComparisonDirection dir :
       {ComparisonDirection::kEq, ComparisonDirection::kNe,
        ComparisonDirection::kLt, ComparisonDirection::kLe,
        ComparisonDirection::kGt, ComparisonDirection::kGe}) {
    std::string hlo_text = absl::Substitute(
        R"(
HloModule test
ENTRY main {
  p0 = f32[64] parameter(0)
  p1 = f32[64] parameter(1)
  ROOT cmp = pred[64] compare(p0, p1), direction=$0, order=WEAK
}
)",
        ComparisonDirectionToString(dir));

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text));

    HloEvaluator evaluator;
    TF_ASSERT_OK_AND_ASSIGN(Literal unexpanded_result,
                            evaluator.Evaluate(*module, {&lhs_lit, &rhs_lit}));

    ComparisonExpander expander;
    TF_ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
    EXPECT_TRUE(changed);

    // Verify no WEAK comparisons remain.
    for (const HloInstruction* inst :
         module->entry_computation()->instructions()) {
      if (inst->opcode() == HloOpcode::kCompare) {
        EXPECT_EQ(inst->comparison_order(), ComparisonOrder::kPartial);
      }
    }

    TF_ASSERT_OK_AND_ASSIGN(Literal expanded_result,
                            evaluator.Evaluate(*module, {&lhs_lit, &rhs_lit}));
    EXPECT_TRUE(LiteralTestUtil::Equal(unexpanded_result, expanded_result))
        << "Mismatch for direction " << ComparisonDirectionToString(dir);
  }
}

TEST_F(ComparisonExpanderTest, ExpandWeakOrderFloatWithoutNaN) {
  constexpr absl::string_view kHloText = R"(
HloModule test
ENTRY main {
  p0 = f4e2m1fn[8] parameter(0)
  p1 = f4e2m1fn[8] parameter(1)
  ROOT cmp = pred[8] compare(p0, p1), direction=LT, order=WEAK
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  ComparisonExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCompare);
  EXPECT_EQ(root->comparison_direction(), ComparisonDirection::kLt);
  EXPECT_EQ(root->comparison_order(), ComparisonOrder::kPartial);
}

}  // namespace
}  // namespace xla
