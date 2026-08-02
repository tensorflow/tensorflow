/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/while_loop_constant_sinking.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::_;
using WhileLoopConstantSinkingTest = HloHardwareIndependentTestBase;

TEST_F(WhileLoopConstantSinkingTest, SinkOneConstant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  add.0 = f32[2] add(p_body.0, p_body.1)
  ROOT root = (f32[2],f32[2]) tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[2],f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})
  while_init = (f32[2],f32[2]) tuple(const_0, const_1)
  ROOT while = (f32[2],f32[2]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      WhileLoopConstantSinking(/*sink_broadcast_of_constants=*/false,
                               /*sink_only_scalar_constants=*/true)
          .Run(module.get()));
  ASSERT_FALSE(changed);

  TF_ASSERT_OK_AND_ASSIGN(
      changed, WhileLoopConstantSinking(/*sink_broadcast_of_constants=*/false,
                                        /*sink_only_scalar_constants=*/false)
                   .Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body.sunk");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(_, op::Constant()), _));
}

TEST_F(WhileLoopConstantSinkingTest, SinkOneConstantWithOriginalValue) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  add.0 = f32[2] add(p_body.0, p_body.1)
  ROOT root = (f32[2],f32[2]) tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[2],f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})
  while_init = (f32[2],f32[2]) tuple(const_0, const_1)
  ROOT while = (f32[2],f32[2]) while(while_init), condition=condition, body=body, origin={({"a"},{"b"})}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      WhileLoopConstantSinking(/*sink_broadcast_of_constants=*/false,
                               /*sink_only_scalar_constants=*/false)
          .Run(module.get()));
  ASSERT_TRUE(changed);

  auto while_instr = module->entry_computation()->root_instruction();
  ASSERT_NE(while_instr->original_value(), nullptr);
  EXPECT_TRUE(
      while_instr->original_value()->IsCompatibleWith(while_instr->shape()));
}

TEST_F(WhileLoopConstantSinkingTest, SinkBroadcastOfConstant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[16],f32[16]) parameter(0)
  p_body.0 = get-tuple-element(p_body), index=0
  p_body.1 = get-tuple-element(p_body), index=1

  add.0 = add(p_body.0, p_body.1)
  ROOT root = tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[16],f32[16]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[] constant(1)
  const_1 = f32[] constant(2)
  broadcast_0 = f32[16] broadcast(const_0), dimensions={}
  broadcast_1 = f32[16] broadcast(const_1), dimensions={}
  while_init = tuple(broadcast_0, broadcast_1)
  ROOT while = while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      WhileLoopConstantSinking(/*sink_broadcast_of_constants=*/false)
          .Run(module.get()));
  ASSERT_FALSE(changed);

  TF_ASSERT_OK_AND_ASSIGN(
      changed, WhileLoopConstantSinking(/*sink_broadcast_of_constants=*/true)
                   .Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body.sunk");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(_, op::Broadcast(op::Constant())), _));
}

TEST_F(WhileLoopConstantSinkingTest, SinkBroadcastOfConstantWithOriginalValue) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[16],f32[16]) parameter(0)
  p_body.0 = get-tuple-element(p_body), index=0
  p_body.1 = get-tuple-element(p_body), index=1

  add.0 = add(p_body.0, p_body.1)
  ROOT root = tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[16],f32[16]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[] constant(1)
  const_1 = f32[] constant(2)
  broadcast_0 = f32[16] broadcast(const_0), dimensions={}
  broadcast_1 = f32[16] broadcast(const_1), dimensions={}
  while_init = tuple(broadcast_0, broadcast_1)
  ROOT while = while(while_init), condition=condition, body=body, origin={({"a"},{"b"})}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      WhileLoopConstantSinking(/*sink_broadcast_of_constants=*/true)
          .Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_instr = module->entry_computation()->root_instruction();
  ASSERT_NE(while_instr->original_value(), nullptr);
  EXPECT_TRUE(
      while_instr->original_value()->IsCompatibleWith(while_instr->shape()));
}

TEST_F(WhileLoopConstantSinkingTest, KeepConstantsLoopInvariant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[2],f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_body), index=1
  p_body.2 = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_body), index=2

  add.0 = f32[2] add(p_body.1, p_body.2)
  ROOT root = (f32[2],f32[2],f32[2]) tuple(add.0, p_body.1, p_body.2)
}

condition {
  p_cond = (f32[2],f32[2],f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})
  const_2 = f32[2] constant({3, 1})
  while_init = (f32[2],f32[2],f32[2]) tuple(const_0, const_1, const_2)
  ROOT while = (f32[2],f32[2],f32[2]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body.sunk");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(op::Constant(), op::Constant()),
                        op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0))));
}

TEST_F(WhileLoopConstantSinkingTest, TupleShapedConstants) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_b = (f32[2],(f32[2],f32[2])) parameter(0)
  p_b.0 = f32[2] get-tuple-element((f32[2],(f32[2],f32[2])) p_b), index=0
  p_b.1 = (f32[2],f32[2]) get-tuple-element((f32[2],(f32[2],f32[2])) p_b), index=1

  p_b.1.1 = f32[2] get-tuple-element(p_b.1), index=0

  ROOT root = (f32[2],(f32[2],f32[2])) tuple(p_b.1.1, p_b.1)
}

condition {
  p_cond = (f32[2],(f32[2],f32[2])) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = (f32[2], f32[2]) constant(({2, 1},{3,1}))
  while_init = (f32[2],(f32[2],f32[2])) tuple(const_0, const_1)
  ROOT while = (f32[2],(f32[2],f32[2])) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body.sunk");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Constant(), 0),
                        op::GetTupleElement(op::Parameter(0))));
}

TEST_F(WhileLoopConstantSinkingTest, TupleShapedConstantsWithOriginalValue) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_b = (f32[2],(f32[2],f32[2])) parameter(0)
  p_b.0 = f32[2] get-tuple-element((f32[2],(f32[2],f32[2])) p_b), index=0
  p_b.1 = (f32[2],f32[2]) get-tuple-element((f32[2],(f32[2],f32[2])) p_b), index=1

  p_b.1.1 = f32[2] get-tuple-element(p_b.1), index=0

  ROOT root = (f32[2],(f32[2],f32[2])) tuple(p_b.1.1, p_b.1)
}

condition {
  p_cond = (f32[2],(f32[2],f32[2])) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = (f32[2], f32[2]) constant(({2, 1},{3,1}))
  while_init = (f32[2],(f32[2],f32[2])) tuple(const_0, const_1)
  ROOT while = (f32[2],(f32[2],f32[2])) while(while_init), condition=condition, body=body, origin={({"a"},({"b"},{"c"}))}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_instr = module->entry_computation()->root_instruction();
  ASSERT_NE(while_instr->original_value(), nullptr);
  EXPECT_TRUE(
      while_instr->original_value()->IsCompatibleWith(while_instr->shape()));
}

TEST_F(WhileLoopConstantSinkingTest, DuplicateGTEs) {
  // This test shows that the pass fails to optimize non-canonical IR.
  //
  // Even though the input IR has a constant value for p_b.2.dup,
  // WhileLoopConstantSinking doesn't try to detect this.  Instead, it relies on
  // prior runs of HLO CSE to have commoned these identical GTE instructions.

  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_b = (f32[2],f32[2],f32[2]) parameter(0)

  p_b.1     = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_b), index=1
  p_b.2     = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_b), index=2
  p_b.2.dup = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_b), index=2

  add.0 = f32[2] add(p_b.1, p_b.2.dup)
  ROOT root = (f32[2],f32[2],f32[2]) tuple(add.0, p_b.1, p_b.2)
}

condition {
  p_cond = (f32[2],f32[2],f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})
  const_2 = f32[2] constant({3, 1})
  while_init = (f32[2],f32[2],f32[2]) tuple(const_0, const_1, const_2)
  ROOT while = (f32[2],f32[2],f32[2]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body.sunk");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(op::Constant(), ::testing::Not(op::Constant())),
                        op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0))));
}

TEST_F(WhileLoopConstantSinkingTest, DontCreateDeadConstant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  token0 = token[] after-all()
  outfeed = token[] outfeed(p_body.0, token0)
  ROOT root = (f32[2],f32[2],f32[2]) tuple(p_body.0, p_body.1, p_body.1)
}

condition {
  p_cond = (f32[2],f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})
  while_init = (f32[2],f32[2]) tuple(const_0, const_1)
  ROOT while = (f32[2],f32[2],f32[2]) while(while_init), condition=condition,
                                      body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body.sunk");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                        op::GetTupleElement()));
  for (const HloInstruction* inst : while_body->instructions()) {
    if (inst->opcode() == HloOpcode::kConstant) {
      EXPECT_GT(inst->user_count(), 0);
    }
  }
}

TEST_F(WhileLoopConstantSinkingTest, ConditionalSinkConstant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[],f32[]) parameter(0)
  p_body.0 = f32[] get-tuple-element((f32[],f32[]) p_body), index=0
  const = f32[] constant(1)
  add = f32[] add(p_body.0, const)
  p_body.1 = f32[] get-tuple-element((f32[],f32[]) p_body), index=1
  ROOT root = (f32[],f32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (f32[],f32[]) parameter(0)
  p_cond.0 = f32[] get-tuple-element((f32[],f32[]) p_cond), index=0
  p_cond.1 = f32[] get-tuple-element((f32[],f32[]) p_cond), index=1
  ROOT result = pred[] compare(p_cond.0, p_cond.1), direction=LT
}

ENTRY entry {
  const_0 = f32[] constant(0)
  const_1 = f32[] constant(10)
  while_init = (f32[],f32[]) tuple(const_0, const_1)
  ROOT while = (f32[],f32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_condition = module->GetComputationWithName("condition.sunk");
  EXPECT_THAT(while_condition->root_instruction(), op::Lt(_, op::Constant()));
}

TEST_F(WhileLoopConstantSinkingTest, ConditionalSinkConstantWithOriginalValue) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[],f32[]) parameter(0)
  p_body.0 = f32[] get-tuple-element((f32[],f32[]) p_body), index=0
  const = f32[] constant(1)
  add = f32[] add(p_body.0, const)
  p_body.1 = f32[] get-tuple-element((f32[],f32[]) p_body), index=1
  ROOT root = (f32[],f32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (f32[],f32[]) parameter(0)
  p_cond.0 = f32[] get-tuple-element((f32[],f32[]) p_cond), index=0
  p_cond.1 = f32[] get-tuple-element((f32[],f32[]) p_cond), index=1
  ROOT result = pred[] compare(p_cond.0, p_cond.1), direction=LT
}

ENTRY entry {
  const_0 = f32[] constant(0)
  const_1 = f32[] constant(10)
  while_init = (f32[],f32[]) tuple(const_0, const_1)
  ROOT while = (f32[],f32[]) while(while_init), condition=condition, body=body, origin={({"a"},{"b"})}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_instr = module->entry_computation()->root_instruction();
  ASSERT_NE(while_instr->original_value(), nullptr);
  EXPECT_TRUE(
      while_instr->original_value()->IsCompatibleWith(while_instr->shape()));
}

TEST_F(WhileLoopConstantSinkingTest, ConditionalTupleShapedConstants) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_b = (f32[],(f32[],f32[])) parameter(0)
  p_b.0 = f32[] get-tuple-element((f32[],(f32[],f32[])) p_b), index=0
  p_b.1 = (f32[],f32[]) get-tuple-element((f32[],(f32[],f32[])) p_b), index=1
  p_b.1.0 = f32[] get-tuple-element((f32[],f32[]) p_b.1), index=0
  add = f32[] add(p_b.0, p_b.1.0)
  ROOT root = (f32[],(f32[],f32[])) tuple(add, p_b.1)
}

condition {
  p_c = (f32[],(f32[],f32[])) parameter(0)
  p_c.0 = f32[] get-tuple-element((f32[],(f32[],f32[])) p_c), index=0
  p_c.1 = (f32[],f32[]) get-tuple-element((f32[],(f32[],f32[])) p_c), index=1
  p_c.1.1 = f32[] get-tuple-element((f32[],f32[]) p_c.1), index=1
  ROOT result = pred[] compare(p_c.0, p_c.1.1), direction=LT
}

ENTRY entry {
  const_0 = f32[] constant(0)
  const_1 = (f32[], f32[]) constant((1, 10))
  while_init = (f32[],(f32[],f32[])) tuple(const_0, const_1)
  ROOT while = (f32[],(f32[],f32[])) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_condition = module->GetComputationWithName("condition.sunk");
  EXPECT_THAT(while_condition->root_instruction(),
              op::Lt(_, op::GetTupleElement(op::Constant())));
}

TEST_F(WhileLoopConstantSinkingTest, ConditionalDontCreateDeadConstant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[],f32[],f32[]) parameter(0)
  p_body.0 = f32[] get-tuple-element((f32[],f32[],f32[]) p_body), index=0
  const = f32[] constant(1)
  add = f32[] add(p_body.0, const)
  p_body.1 = f32[] get-tuple-element((f32[],f32[],f32[]) p_body), index=1
  p_body.2 = f32[] get-tuple-element((f32[],f32[],f32[]) p_body), index=2
  ROOT root = (f32[],f32[],f32[]) tuple(add, p_body.1, p_body.2)
}

condition {
  p_cond = (f32[],f32[],f32[]) parameter(0)
  p_cond.0 = f32[] get-tuple-element((f32[],f32[],f32[]) p_cond), index=0
  p_cond.1 = f32[] get-tuple-element((f32[],f32[],f32[]) p_cond), index=1
  p_cond.2 = f32[] get-tuple-element((f32[],f32[],f32[]) p_cond), index=2
  ROOT result = pred[] compare(p_cond.0, p_cond.1), direction=LT
}

ENTRY entry {
  const_0 = f32[] constant(0)
  const_1 = f32[] constant(10)
  const_2 = f32[] constant(12)
  while_init = (f32[],f32[],f32[]) tuple(const_0, const_1, const_2)
  ROOT while = (f32[],f32[],f32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_condition = module->GetComputationWithName("condition.sunk");
  EXPECT_THAT(while_condition->root_instruction(), op::Lt(_, op::Constant()));
  for (const HloInstruction* inst : while_condition->instructions()) {
    if (inst->opcode() == HloOpcode::kConstant) {
      EXPECT_GT(inst->user_count(), 0);
    }
  }
}

TEST_F(WhileLoopConstantSinkingTest, ConditionalMultipleSameIndexGTEs) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[],f32[],f32[]) parameter(0)
  p_body.0 = f32[] get-tuple-element((f32[],f32[],f32[]) p_body), index=0
  const = f32[] constant(1)
  add.0 = f32[] add(p_body.0, const)
  p_body.1 = f32[] get-tuple-element((f32[],f32[],f32[]) p_body), index=1
  add.1 = f32[] add(p_body.1, const)
  p_body.2 = f32[] get-tuple-element((f32[],f32[],f32[]) p_body), index=2
  ROOT root = (f32[],f32[],f32[]) tuple(add.0, add.1, p_body.2)
}

condition {
  p_cond = (f32[],f32[],f32[]) parameter(0)
  p_cond.0 = f32[] get-tuple-element((f32[],f32[],f32[]) p_cond), index=0
  p_cond.2 = f32[] get-tuple-element((f32[],f32[],f32[]) p_cond), index=2
  lt.0 = pred[] compare(p_cond.0, p_cond.2), direction=LT
  p_cond.1 = f32[] get-tuple-element((f32[],f32[],f32[]) p_cond), index=1
  p_cond.2.c = f32[] get-tuple-element((f32[],f32[],f32[]) p_cond), index=2
  lt.1 = pred[] compare(p_cond.1, p_cond.2.c), direction=LT
  ROOT result = pred[] and(lt.0, lt.1)
}

ENTRY entry {
  const_0 = f32[] constant(0)
  const_1 = f32[] constant(0)
  const_2 = f32[] constant(12)
  while_init = (f32[],f32[],f32[]) tuple(const_0, const_1, const_2)
  ROOT while = (f32[],f32[],f32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_condition = module->GetComputationWithName("condition.sunk");
  EXPECT_THAT(while_condition->root_instruction(),
              op::And(op::Lt(_, op::Constant()), op::Lt(_, op::Constant())));
}

TEST_F(WhileLoopConstantSinkingTest, SinkWithSharedBody) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  add.0 = f32[2] add(p_body.0, p_body.1)
  ROOT root = (f32[2],f32[2]) tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[2],f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})
  while_init = (f32[2],f32[2]) tuple(const_0, const_1)
  while = (f32[2],f32[2]) while(while_init), condition=condition, body=body
  while_init2 = (f32[2],f32[2]) tuple(const_1, const_0)
  while2 = (f32[2],f32[2]) while(while_init2), condition=condition, body=body
  ROOT tuple = ((f32[2],f32[2]),(f32[2],f32[2])) tuple(while, while2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      WhileLoopConstantSinking(/*sink_broadcast_of_constants=*/false,
                               /*sink_only_scalar_constants=*/false)
          .Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body.sunk");
  EXPECT_THAT(
      while_body->root_instruction(),
      op::Tuple(op::Add(_, op::Constant(LiteralUtil::CreateR1<float>({2, 1}))),
                _));
  while_body = module->GetComputationWithName("body.sunk.1");
  EXPECT_THAT(
      while_body->root_instruction(),
      op::Tuple(op::Add(_, op::Constant(LiteralUtil::CreateR1<float>({1, 2}))),
                _));
}

TEST_F(WhileLoopConstantSinkingTest, NoIncidentalChanges) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  add.0 = f32[2] add(p_body.0, p_body.1)
  ROOT root = (f32[2],f32[2]) tuple(add.0, p_body.1)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})
  ROOT add = f32[2] add(const_0, const_1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      WhileLoopConstantSinking(/*sink_broadcast_of_constants=*/false,
                               /*sink_only_scalar_constants=*/false)
          .Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->computation_count(), 2);
}

TEST_F(WhileLoopConstantSinkingTest, SinkWithThreeLoopsSharedCall) {
  const char* const hlo_string = R"(
HloModule SinkWithThreeLoopsSharedCall

shared_comp {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  ROOT add = f32[2] add(p0, p1)
}

// CHECK-LABEL: %body1.sunk
// CHECK:         [[P_BODY:%.*]] = (f32[2]{0}, f32[2]{0}) parameter(0)
// CHECK-DAG:     [[GTE0:%.*]] = f32[2]{0} get-tuple-element([[P_BODY]]), index=0
// CHECK-DAG:     [[CONST:%.*]] = f32[2]{0} constant({1, 2})
// CHECK:         [[CALL:%.*]] = f32[2]{0} call([[GTE0]], [[CONST]]), to_apply=%shared_comp
// CHECK:       }
body1 {
  p_body1 = (f32[2], f32[2]) parameter(0)
  p_body1.0 = f32[2] get-tuple-element(p_body1), index=0
  p_body1.1 = f32[2] get-tuple-element(p_body1), index=1

  call1 = f32[2] call(p_body1.0, p_body1.1), to_apply=shared_comp
  ROOT root1 = (f32[2], f32[2]) tuple(call1, p_body1.1)
}

// CHECK-LABEL: %body2.sunk
// CHECK:         [[P_BODY:%.*]] = (f32[2]{0}, f32[2]{0}) parameter(0)
// CHECK-DAG:     [[GTE0:%.*]] = f32[2]{0} get-tuple-element([[P_BODY]]), index=0
// CHECK-DAG:     [[CONST:%.*]] = f32[2]{0} constant({2, 1})
// CHECK:         [[CALL:%.*]] = f32[2]{0} call([[GTE0]], [[CONST]]), to_apply=%shared_comp
// CHECK:       }
body2 {
  p_body2 = (f32[2], f32[2]) parameter(0)
  p_body2.0 = f32[2] get-tuple-element(p_body2), index=0
  p_body2.1 = f32[2] get-tuple-element(p_body2), index=1

  call2 = f32[2] call(p_body2.0, p_body2.1), to_apply=shared_comp
  ROOT root2 = (f32[2], f32[2]) tuple(call2, p_body2.1)
}

// CHECK-LABEL: %body3
// CHECK:         [[P_BODY:%.*]] = (f32[2]{0}, f32[2]{0}) parameter(0)
// CHECK-DAG:     [[GTE0:%.*]] = f32[2]{0} get-tuple-element([[P_BODY]]), index=0
// CHECK-DAG:     [[GTE1:%.*]] = f32[2]{0} get-tuple-element([[P_BODY]]), index=1
// CHECK:         [[CALL:%.*]] = f32[2]{0} call([[GTE0]], [[GTE1]]), to_apply=%shared_comp
// CHECK:       }
body3 {
  p_body3 = (f32[2], f32[2]) parameter(0)
  p_body3.0 = f32[2] get-tuple-element(p_body3), index=0
  p_body3.1 = f32[2] get-tuple-element(p_body3), index=1

  call3 = f32[2] call(p_body3.0, p_body3.1), to_apply=shared_comp
  ROOT root3 = (f32[2], f32[2]) tuple(call3, p_body3.1)
}

condition {
  p_cond = (f32[2], f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})

  param_x = f32[2] parameter(0)
  param_y = f32[2] parameter(1)
  param_z = f32[2] parameter(2)
  param_w = f32[2] parameter(3)

  while1_init = (f32[2], f32[2]) tuple(param_x, const_0)
  while1 = (f32[2], f32[2]) while(while1_init), condition=condition, body=body1

  while2_init = (f32[2], f32[2]) tuple(param_y, const_1)
  while2 = (f32[2], f32[2]) while(while2_init), condition=condition, body=body2

  while3_init = (f32[2], f32[2]) tuple(param_z, param_w)
  while3 = (f32[2], f32[2]) while(while3_init), condition=condition, body=body3

  ROOT tuple = ((f32[2], f32[2]), (f32[2], f32[2]), (f32[2], f32[2])) tuple(while1, while2, while3)
}
)";

  RunAndFilecheckHloRewrite(hlo_string,
                            WhileLoopConstantSinking(
                                /*sink_broadcast_of_constants=*/false,
                                /*sink_only_scalar_constants=*/false));
}

TEST_F(WhileLoopConstantSinkingTest, SinkWithNestedLoopsSharedCall) {
  const char* const hlo_string = R"(
HloModule SinkWithNestedLoopsSharedCall

shared_comp {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

// CHECK-LABEL: %body_inner.sunk
// CHECK:         [[P_INNER:%.*]] = (f32[], f32[]) parameter(0)
// CHECK-DAG:     [[GTE0_INNER:%.*]] = f32[] get-tuple-element([[P_INNER]]), index=0
// CHECK-DAG:     [[CONST_INNER:%.*]] = f32[] constant(1)
// CHECK:         [[CALL_INNER:%.*]] = f32[] call([[GTE0_INNER]], [[CONST_INNER]]), to_apply=%shared_comp
// CHECK:       }
body_inner {
  p_inner = (f32[], f32[]) parameter(0)
  g0_inner = f32[] get-tuple-element(p_inner), index=0
  g1_inner = f32[] get-tuple-element(p_inner), index=1

  call_inner = f32[] call(g0_inner, g1_inner), to_apply=shared_comp

  ROOT root_inner = (f32[], f32[]) tuple(call_inner, g1_inner)
}

cond_inner {
  p_cond_inner = (f32[], f32[]) parameter(0)
  ROOT result = pred[] constant(true)
}

// CHECK-LABEL: %body_outer.sunk
// CHECK:         [[P_OUTER:%.*]] = (f32[], f32[]) parameter(0)
// CHECK-DAG:     [[GTE0_OUTER:%.*]] = f32[] get-tuple-element([[P_OUTER]]), index=0
// CHECK-DAG:     [[CONST_OUTER:%.*]] = f32[] constant(1)
// CHECK:         [[CALL_OUTER:%.*]] = f32[] call([[GTE0_OUTER]], [[CONST_OUTER]]), to_apply=%shared_comp
// CHECK:         [[INIT_INNER:%.*]] = (f32[], f32[]) tuple([[GTE0_OUTER]], [[CONST_OUTER]])
// CHECK:         [[WHILE_INNER:%.*]] = (f32[], f32[]) while([[INIT_INNER]]), condition=%cond_inner, body=%body_inner.sunk
// CHECK:       }
body_outer {
  p_outer = (f32[], f32[]) parameter(0)
  g0_outer = f32[] get-tuple-element(p_outer), index=0
  g1_outer = f32[] get-tuple-element(p_outer), index=1

  call_outer = f32[] call(g0_outer, g1_outer), to_apply=shared_comp

  while_inner_init = (f32[], f32[]) tuple(g0_outer, g1_outer)
  while_inner = (f32[], f32[]) while(while_inner_init), condition=cond_inner, body=body_inner

  g0_inner_out = f32[] get-tuple-element(while_inner), index=0

  ROOT root_outer = (f32[], f32[]) tuple(g0_inner_out, g1_outer)
}

cond_outer {
  p_cond_outer = (f32[], f32[]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[] constant(1)
  param_x = f32[] parameter(0)

  while_outer_init = (f32[], f32[]) tuple(param_x, const_0)
  ROOT while_outer = (f32[], f32[]) while(while_outer_init), condition=cond_outer, body=body_outer
}
)";

  RunAndFilecheckHloRewrite(hlo_string,
                            WhileLoopConstantSinking(
                                /*sink_broadcast_of_constants=*/false,
                                /*sink_only_scalar_constants=*/false));
}

}  // namespace
}  // namespace xla
