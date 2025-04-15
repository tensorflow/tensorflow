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

#include "xla/service/while_loop_expensive_invariant_code_motion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using WhileLoopExpensiveInvariantCodeMotionTest = HloTestBase;
namespace op = xla::testing::opcode_matchers;

constexpr char kModuleWithNonInflatingInvariantDot[] = R"(
HloModule ModuleWithWhile

mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}

body {
  p_body = (f32[], f32[16, 8]) parameter(0)
  b = get-tuple-element(p_body), index=1
  const = f32[] constant(1.0)
  lhs = f32[8, 16] broadcast(const), dimensions={}
  dot = dot(lhs, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  reduced = reduce(dot, const), dimensions={0, 1}, to_apply=mul
  a = get-tuple-element(p_body), index=0
  add = add(reduced, a)
  ROOT root = tuple(add, b)
}

condition {
  p_cond = (f32[], f32[16, 8]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  param0 = f32[] parameter(0)
  param1 = f32[16, 8] parameter(1)
  while_init = tuple(param0, param1)
  ROOT while = while(while_init), condition=condition, body=body
}
)";

TEST_F(WhileLoopExpensiveInvariantCodeMotionTest,
       HoistsGroupOfAllowedNonInflating) {
  auto m =
      ParseAndReturnVerifiedModule(kModuleWithNonInflatingInvariantDot).value();

  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopExpensiveInvariantCodeMotion(
          /*worth_hoisting_individually=*/HloPredicateIsOp<HloOpcode::kDot>)
          .Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloComputation* while_body = m->GetComputationWithName("wide.body");
  ASSERT_NE(while_body, nullptr);
  EXPECT_THAT(while_body->instructions(), Not(Contains(op::Dot())));
  // kReduce not in the allow list.
  EXPECT_THAT(while_body->instructions(), Contains(op::Reduce()));
}

TEST_F(WhileLoopExpensiveInvariantCodeMotionTest,
       HoistsGroupOfAllNonInflating) {
  auto m =
      ParseAndReturnVerifiedModule(kModuleWithNonInflatingInvariantDot).value();

  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopExpensiveInvariantCodeMotion(
          /*worth_hoisting_individually=*/HloPredicateIsOp<HloOpcode::kDot,
                                                           HloOpcode::kReduce>)
          .Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloComputation* while_body = m->GetComputationWithName("wide.body");
  ASSERT_NE(while_body, nullptr);
  EXPECT_THAT(while_body->instructions(), Not(Contains(op::Dot())));
  EXPECT_THAT(while_body->instructions(), Not(Contains(op::Reduce())));
}

TEST_F(WhileLoopExpensiveInvariantCodeMotionTest,
       DoesNotHoistsUnallowedInstructions) {
  auto m =
      ParseAndReturnVerifiedModule(kModuleWithNonInflatingInvariantDot).value();

  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopExpensiveInvariantCodeMotion(
                              /*worth_hoisting_individually=*/HloPredicateFalse)
                              .Run(m.get()));
  EXPECT_FALSE(simplified_loop);
}

constexpr char kModuleWithInflatingInvariantDot[] = R"(
HloModule ModuleWithWhile

mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}

body {
  p_body = (f32[], f32[16, 4]) parameter(0)
  b = get-tuple-element(p_body), index=1
  const = f32[] constant(1.0)
  lhs = f32[4, 16] broadcast(const), dimensions={}
  dot = dot(lhs, b), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  reduced = reduce(dot, const), dimensions={0, 1}, to_apply=mul
  a = get-tuple-element(p_body), index=0
  add = add(reduced, a)
  ROOT root = tuple(add, b)
}

condition {
  p_cond = (f32[], f32[16, 4]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  param0 = f32[] parameter(0)
  param1 = f32[16, 4] parameter(1)
  while_init = tuple(param0, param1)
  ROOT while = while(while_init), condition=condition, body=body
}
)";

TEST_F(WhileLoopExpensiveInvariantCodeMotionTest, DoesNotHoistsInflating) {
  auto m =
      ParseAndReturnVerifiedModule(kModuleWithInflatingInvariantDot).value();

  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopExpensiveInvariantCodeMotion(
          /*worth_hoisting_individually=*/HloPredicateIsOp<HloOpcode::kDot>)
          .Run(m.get()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(WhileLoopExpensiveInvariantCodeMotionTest,
       HoistsGroupOfNonInflatingWithInflatingIntermediate) {
  auto m =
      ParseAndReturnVerifiedModule(kModuleWithInflatingInvariantDot).value();

  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopExpensiveInvariantCodeMotion(
          /*worth_hoisting_individually=*/HloPredicateIsOp<HloOpcode::kDot,
                                                           HloOpcode::kReduce>)
          .Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloComputation* while_body = m->GetComputationWithName("wide.body");
  ASSERT_NE(while_body, nullptr);
  EXPECT_THAT(while_body->instructions(), Not(Contains(op::Dot())));
  EXPECT_THAT(while_body->instructions(), Not(Contains(op::Reduce())));
}

TEST_F(WhileLoopExpensiveInvariantCodeMotionTest,
       HoistsOpWithDuplicateOperands) {
  constexpr char kModuleWithDuplicateOperands[] = R"(
HloModule ModuleWithWhile

mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}

body {
  p_body = (f32[4, 4], f32[4, 4]) parameter(0)
  a = get-tuple-element(p_body), index=0
  dot = dot(a, a), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  b = get-tuple-element(p_body), index=1
  add = add(b, dot)
  ROOT root = tuple(a, add)
}

condition {
  p_cond = (f32[4, 4], f32[4, 4]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  param0 = f32[4, 4] parameter(0)
  param1 = f32[4, 4] parameter(1)
  while_init = tuple(param0, param1)
  ROOT while = while(while_init), condition=condition, body=body
}
)";
  auto m = ParseAndReturnVerifiedModule(kModuleWithDuplicateOperands).value();

  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopExpensiveInvariantCodeMotion(
          /*worth_hoisting_individually=*/HloPredicateIsOp<HloOpcode::kDot>)
          .Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloComputation* while_body = m->GetComputationWithName("wide.body");
  ASSERT_NE(while_body, nullptr);
  EXPECT_THAT(while_body->instructions(), Not(Contains(op::Dot())));
}

TEST_F(WhileLoopExpensiveInvariantCodeMotionTest,
       DoesNotHoistShardingCustomCalls) {
  constexpr char kModuleWithShardingCustomCalls[] = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[4, 4], f32[4, 4]) parameter(0)
  a = f32[4, 4] get-tuple-element(p_body), index=0
  custom-call.1 = f32[4, 4] custom-call(a), custom_call_target="Sharding", sharding={devices=[4,1]0,1,2,3}
  custom-call.2 = f32[4, 4] custom-call(custom-call.1), custom_call_target="SPMDFullToShardShape", sharding={manual}
  dot = f32[4, 4] dot(a, a), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  b = f32[4, 4] get-tuple-element(p_body), index=1
  add = f32[4, 4] add(b, dot)
  custom-call.3 = f32[4, 4] custom-call(add), custom_call_target="Sharding", sharding={manual}
  custom-call.4 = f32[4, 4] custom-call(custom-call.3), custom_call_target="SPMDShardToFullShape", sharding={devices=[4,1]0,1,2,3}
  ROOT root = (f32[4, 4], f32[4, 4]) tuple(a, custom-call.4)
}

condition {
  p_cond = (f32[4, 4], f32[4, 4]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  param0 = f32[4, 4] parameter(0)
  param1 = f32[4, 4] parameter(1)
  while_init = (f32[4, 4], f32[4, 4]) tuple(param0, param1)
  ROOT while = (f32[4, 4], f32[4, 4]) while(while_init), condition=condition, body=body
}
)";
  auto m = ParseAndReturnVerifiedModule(kModuleWithShardingCustomCalls).value();

  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopExpensiveInvariantCodeMotion(
          /*worth_hoisting_individually=*/HloPredicateIsOp<HloOpcode::kDot>)
          .Run(m.get()));
  EXPECT_FALSE(simplified_loop);
}

}  // namespace
}  // namespace xla
