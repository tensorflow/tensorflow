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

#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::_;
using WhileLoopConstantSinkingTest = HloTestBase;

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

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(_, op::Constant()), _));
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

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(_, op::Broadcast(op::Constant())), _));
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

  auto* while_body = module->GetComputationWithName("body");
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

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Constant(), 0),
                        op::GetTupleElement(op::Parameter(0))));
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

  auto* while_body = module->GetComputationWithName("body");
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

  auto* while_body = module->GetComputationWithName("body");
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

  auto* while_condition = module->GetComputationWithName("condition");
  EXPECT_THAT(while_condition->root_instruction(), op::Lt(_, op::Constant()));
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

  auto* while_condition = module->GetComputationWithName("condition");
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

  auto* while_condition = module->GetComputationWithName("condition");
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

  auto* while_condition = module->GetComputationWithName("condition");
  EXPECT_THAT(while_condition->root_instruction(),
              op::And(op::Lt(_, op::Constant()), op::Lt(_, op::Constant())));
}
}  // namespace
}  // namespace xla
