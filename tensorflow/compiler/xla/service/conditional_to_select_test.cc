/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/conditional_to_select.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using ConditionalToSelectTest = HloTestBase;
using ::testing::_;

// Test that a conditional of simple constants is transformed to a select
TEST_F(ConditionalToSelectTest, MapConditionalConstants) {
  const string hlo_text = R"(
HloModule MapConditionalConstants

if {
  %pif = () parameter(0)
  ROOT %cif = f32[] constant(0)
}

else {
  %pelse = () parameter(0)
  ROOT %celse = f32[] constant(1)
}

mapped {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  %lt = pred[] compare(%a, %b), direction=LT
  %t = () tuple()
  ROOT %conditional = f32[] conditional(%lt, %t, %t), true_computation=if, false_computation=else
}

ENTRY comp {
  %p1 = f32[1000]{0} parameter(0)
  %p2 = f32[1000]{0} parameter(1)
  ROOT %mapped = f32[1000]{0} map(%p1, %p2), dimensions={0}, to_apply=mapped
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  ConditionalToSelect pass;
  ASSERT_TRUE(pass.Run(&*module).ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kMap);
  HloComputation* mapped = root->called_computations()[0];
  EXPECT_THAT(mapped->root_instruction(),
              op::Select(op::Lt(op::Parameter(0), op::Parameter(1)),
                         op::Constant(), op::Constant()));
}

// Test that the condition gets broadcasted for feeding into
// select when the output is non-scalar.
TEST_F(ConditionalToSelectTest, MapConditionalNonScalar) {
  const string hlo_text = R"(
HloModule MapConditionalNonScalar

if {
  %pif = () parameter(0)
  %zero = f32[] constant(0)
  ROOT %zero_broadcasted = f32[2,2]{1,0} broadcast(%zero), dimensions={}
}

else {
  %pelse = () parameter(0)
  %one = f32[] constant(0)
  ROOT %one_broadcasted = f32[2,2]{1,0} broadcast(%one), dimensions={}
}

add {
  %add_lhs = f32[] parameter(0)
  %add_rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%add_lhs, %add_rhs)
}

mapped {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  %lt = pred[] compare(%a, %b), direction=LT
  %t = () tuple()
  %conditional = f32[2,2]{1,0} conditional(%lt, %t, %t), true_computation=if, false_computation=else
  %zero = f32[] constant(0)
  ROOT %reduced = f32[] reduce(%conditional, %zero), dimensions={0,1}, to_apply=add
}

ENTRY comp {
  %p1 = f32[1000]{0} parameter(0)
  %p2 = f32[1000]{0} parameter(1)
  ROOT %mapped = f32[1000]{0} map(%p1, %p2), dimensions={0}, to_apply=mapped
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  ConditionalToSelect pass;
  ASSERT_TRUE(pass.Run(&*module).ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kMap);
  HloComputation* mapped = root->called_computations()[0];
  EXPECT_THAT(
      mapped->root_instruction(),
      op::Reduce(
          op::Select(op::Broadcast(op::Lt(op::Parameter(0), op::Parameter(1))),
                     _, _),
          _));
}

// Test that conditionals of tuple type get turned into kTupleSelect
TEST_F(ConditionalToSelectTest, MapConditionalTuples) {
  const string hlo_text = R"(
HloModule MapConditionalTuples

if {
  %pif = () parameter(0)
  %zero = f32[] constant(0)
  ROOT %tup = (f32[],f32[]) tuple(%zero, %zero)
}

else {
  %pelse = () parameter(0)
  %one = f32[] constant(0)
  ROOT %tup = (f32[],f32[]) tuple(%one, %one)
}

add {
  %add_lhs = f32[] parameter(0)
  %add_rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%add_lhs, %add_rhs)
}

mapped {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  %lt = pred[] compare(%a, %b), direction=LT
  %t = () tuple()
  %conditional = (f32[], f32[]) conditional(%lt, %t, %t), true_computation=if, false_computation=else
  %el1 = f32[] get-tuple-element(%conditional), index=0
  %el2 = f32[] get-tuple-element(%conditional), index=1
  %reduced = f32[] add(%el1, %el2)
}

ENTRY comp {
  %p1 = f32[1000]{0} parameter(0)
  %p2 = f32[1000]{0} parameter(1)
  ROOT %mapped = f32[1000]{0} map(%p1, %p2), dimensions={0}, to_apply=mapped
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  ConditionalToSelect pass;
  ASSERT_TRUE(pass.Run(&*module).ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kMap);
  HloComputation* mapped = root->called_computations()[0];
  EXPECT_THAT(mapped->root_instruction(),
              op::Add(op::GetTupleElement(op::TupleSelect(_, _, _)), _));
}
}  // namespace
}  // namespace xla
