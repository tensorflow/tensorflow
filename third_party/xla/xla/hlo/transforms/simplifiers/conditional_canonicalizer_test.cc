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

#include "xla/hlo/transforms/simplifiers/conditional_canonicalizer.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class ConditionalCanonicalizerTest : public HloHardwareIndependentTestBase {
 protected:
  ConditionalCanonicalizerTest() {}
};

TEST_F(ConditionalCanonicalizerTest, DenseArrayConditionalRewrite) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule _
true_branch {
  true_param = (s32[3,2]) parameter(0)
  ROOT root = s32[] constant(0)
}

false_branch {
  false_param = (s32[3,2]) parameter(0)
  ROOT root = s32[] constant(1)
}

ENTRY entry {
  param0 = s32[3,2] parameter(0)
  branch = pred[] constant(false)
  param_tuple = (s32[3 ,2]) tuple(param0)
  ROOT conditional = s32[] conditional(branch, param_tuple, param_tuple),
    true_computation=true_branch, false_computation=false_branch
}
)")
                    .value();
  ConditionalCanonicalizer pass;
  EXPECT_TRUE(pass.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement(op::Conditional()));
}

TEST_F(ConditionalCanonicalizerTest, SharedBranches) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule _
true_branch {
  true_param = (s32[3,2]) parameter(0)
  ROOT root = s32[] constant(0)
}

false_branch {
  false_param = (s32[3,2]) parameter(0)
  ROOT root = s32[] constant(1)
}

ENTRY entry {
  param0 = s32[3,2] parameter(0)
  branch = pred[] constant(false)
  param_tuple = (s32[3 ,2]) tuple(param0)
  conditional = s32[] conditional(branch, param_tuple, param_tuple),
    true_computation=true_branch, false_computation=false_branch
  conditional2 = s32[] conditional(branch, param_tuple, param_tuple),
    true_computation=true_branch, false_computation=false_branch
  ROOT tuple = (s32[],s32[]) tuple(conditional, conditional2)
}
)")
                    .value();
  ConditionalCanonicalizer pass;
  EXPECT_TRUE(pass.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Conditional()),
                        op::GetTupleElement(op::Conditional())));
}

}  // namespace
}  // namespace xla
