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

#include "xla/service/control_dep_rewriter.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ControlDepRewriterTest = HloHardwareIndependentTestBase;

TEST_F(ControlDepRewriterTest, RewriteControlDep) {
  // Parse the module.
  const char* const hlo_string = R"(
HloModule ControlDepModule

ENTRY entry {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  exp = f32[] exponential(p0)
  cos = f32[] cosine(p1)
  control_dep = () custom-call(exp, cos), custom_call_target="control_dep"
  ROOT root = tuple(exp, cos)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Run the ControlDepRewriter pass.
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ControlDepRewriter{}.Run(module.get()));
  ASSERT_TRUE(changed);

  // Verify that the correct control dependencies were added.
  HloInstruction* exp = FindInstruction(module.get(), HloOpcode::kExp);
  HloInstruction* cos = FindInstruction(module.get(), HloOpcode::kCos);
  ASSERT_NE(exp, nullptr);
  ASSERT_NE(cos, nullptr);
  EXPECT_THAT(exp->control_successors(), ::testing::ElementsAre(cos));
  EXPECT_THAT(cos->control_predecessors(), ::testing::ElementsAre(exp));

  // Verify that the "control_dep" custom call was removed.
  EXPECT_EQ(FindInstruction(module.get(), HloOpcode::kCustomCall), nullptr);
}

}  // namespace
}  // namespace xla
