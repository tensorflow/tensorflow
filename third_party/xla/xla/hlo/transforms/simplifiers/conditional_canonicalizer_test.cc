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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/types.h"
#include "xla/util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

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

TEST_F(ConditionalCanonicalizerTest, ArgumentRewrite) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule CanonicalizeArgumentsToTuples
true_branch {
  true_param = s32[3,2] parameter(0)
  ROOT root = s32[] constant(0)
}

false_branch {
  false_param = s32[3,2] parameter(0)
  ROOT root = s32[] constant(1)
}

ENTRY entry {
  param0 = s32[3,2] parameter(0)
  branch = pred[] parameter(1)
  ROOT conditional = s32[] conditional(branch, param0, param0),
    true_computation=true_branch, false_computation=false_branch
}
)")
                    .value();
  ConditionalCanonicalizer pass;
  EXPECT_TRUE(pass.Run(module.get()).value());

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0)->users(),
              ElementsAre(op::Tuple(), op::Tuple()));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1)->users(),
              ElementsAre(op::Conditional()));
  for (auto* computation : module->computations()) {
    if (computation == module->entry_computation()) continue;
    EXPECT_TRUE(computation->parameter_instruction(0)->shape().IsTuple());
    EXPECT_THAT(computation->root_instruction(), op::Tuple());
  }
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement(op::Conditional()));
}

TEST_F(ConditionalCanonicalizerTest, MultipleConditionalRewrite) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule CanonicalizeArgumentsToTuples
true_branch {
  true_param = s32[3,2] parameter(0)
  ROOT root = s32[] constant(0)
}

false_branch {
  false_param = s32[3,2] parameter(0)
  ROOT root = s32[] constant(1)
}

ENTRY entry {
  param0 = s32[3,2] parameter(0)
  branch = pred[] parameter(1)
  conditional.0 = s32[] conditional(branch, param0, param0),
    true_computation=true_branch, false_computation=false_branch
  conditional.1 = s32[] conditional(branch, param0, param0),
    true_computation=true_branch, false_computation=false_branch
  ROOT root = tuple(conditional.0, conditional.1)
}
)")
                    .value();
  ConditionalCanonicalizer pass;
  EXPECT_TRUE(pass.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0)->users(),
              ElementsAre(op::Tuple(), op::Tuple(), op::Tuple(), op::Tuple()));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1)->users(),
              ElementsAre(op::Conditional(), op::Conditional()));
  for (auto* computation : module->computations()) {
    if (computation == module->entry_computation()) {
      continue;
    }
    EXPECT_TRUE(computation->parameter_instruction(0)->shape().IsTuple());
    EXPECT_THAT(computation->root_instruction(), op::Tuple());
  }
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Conditional()),
                        op::GetTupleElement(op::Conditional())));
}

TEST_F(ConditionalCanonicalizerTest, ConditionalAndCallRewrite) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule CanonicalizeArgumentsToTuples
true_branch {
  true_param = s32[3,2] parameter(0)
  ROOT root = s32[] constant(0)
}

false_branch {
  false_param = s32[3,2] parameter(0)
  ROOT root = s32[] constant(1)
}

ENTRY entry {
  param0 = s32[3,2] parameter(0)
  branch = pred[] parameter(1)
  conditional = s32[] conditional(branch, param0, param0),
    true_computation=true_branch, false_computation=false_branch
  call = s32[] call(param0), to_apply=false_branch
  ROOT root = tuple(conditional, call)
}
)")
                    .value();
  ConditionalCanonicalizer pass;
  EXPECT_TRUE(pass.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0)->users(),
              UnorderedElementsAre(op::Tuple(), op::Tuple(), op::Call()));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1)->users(),
              ElementsAre(op::Conditional()));

  for (auto* computation : module->computations()) {
    if (computation == module->entry_computation()) {
      continue;
    }
    EXPECT_EQ(computation->caller_instructions().size(), 1);
    const auto* caller = computation->caller_instructions()[0];
    if (caller->opcode() == HloOpcode::kConditional) {
      EXPECT_TRUE(computation->parameter_instruction(0)->shape().IsTuple());
      EXPECT_THAT(computation->root_instruction(), op::Tuple());
    } else {
      EXPECT_TRUE(caller->opcode() == HloOpcode::kCall);
      EXPECT_FALSE(computation->parameter_instruction(0)->shape().IsTuple());
      EXPECT_THAT(computation->root_instruction(), op::Constant());
    }
  }
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Conditional()), op::Call()));
}

}  // namespace
}  // namespace xla
