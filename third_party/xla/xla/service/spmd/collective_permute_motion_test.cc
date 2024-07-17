/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/spmd/collective_permute_motion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using CollectivePermuteMotionTest = HloTestBase;
namespace op = xla::testing::opcode_matchers;

TEST_F(CollectivePermuteMotionTest, SimpleMove) {
  absl::string_view hlo_string = R"(
  HloModule test
  body {
    loop_var = (s32[], f32[4,4]) parameter(0)
    constant.1 = s32[] constant(1)
    gte0 = s32[] get-tuple-element(loop_var), index=0
    add = s32[] add(gte0, constant.1)
    gte1 = f32[4,4] get-tuple-element(loop_var), index=1
    mul = f32[4,4] multiply(gte1, gte1)
    cp = f32[4,4] collective-permute(mul), source_target_pairs={{0,1},{1,2}}
    ROOT tuple = (s32[], f32[4,4]) tuple(add, cp)
  }
  cond {
    loop_var = (s32[], f32[4,4]) parameter(0)
    gte.cond = s32[] get-tuple-element(loop_var), index=0
    constant.3 = s32[] constant(5)
    ROOT lt = pred[] compare(gte.cond, constant.3), direction=LT
  }
  ENTRY  main {
    constant.2 = s32[] constant(0)
    param = f32[4,4] parameter(0)
    tuple.1 = (s32[], f32[4,4]) tuple(constant.2, param)
    while = (s32[], f32[4,4]) while(tuple.1), condition=cond, body=body
    ROOT result = s32[] get-tuple-element(while), index=0
  }
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CollectivePermuteMotion pass;
  ASSERT_TRUE(pass.Run(&*module).value());

  VLOG(1) << module->ToString();
  const HloInstruction* loop = FindInstruction(module.get(), "while");
  // Check if the operands are reshaped.
  const HloInstruction* output =
      loop->while_body()->root_instruction()->operand(1);
  auto input =
      AllOf(op::Shape("f32[4,4]"), op::GetTupleElement(op::Parameter(0)));
  auto cp = op::CollectivePermute(input);
  auto select = op::Select(op::Broadcast(op::Compare()), input, cp);
  EXPECT_THAT(output, op::Multiply(select, select));
}

TEST_F(CollectivePermuteMotionTest, NoCollectivePermute) {
  absl::string_view hlo_string = R"(
  HloModule test
  body {
    loop_var = (s32[], f32[], f32[]) parameter(0)
    constant.1 = s32[] constant(1)
    gte0 = s32[] get-tuple-element(loop_var), index=0
    add = s32[] add(gte0, constant.1)
    gte1 = f32[] get-tuple-element(loop_var), index=1
    constant.4 = f32[] constant(4.0)
    ROOT tuple = (s32[], f32[], f32[]) tuple(add, constant.4, gte1)
  }
  cond {
    loop_var = (s32[], f32[], f32[]) parameter(0)
    gte.cond = s32[] get-tuple-element(loop_var), index=0
    constant.3 = s32[] constant(5)
    ROOT lt = pred[] compare(gte.cond, constant.3), direction=LT
  }
  ENTRY  main {
    constant.2 = s32[] constant(0)
    param = f32[] parameter(0)
    param.1 = f32[] parameter(1)
    tuple.1 = (s32[], f32[], f32[]) tuple(constant.2, param, param.1)
    while = (s32[], f32[], f32[]) while(tuple.1), condition=cond, body=body
    ROOT result = s32[] get-tuple-element(while), index=0
  }
)";
  // Test that the pass does not crash if there is no collective permute
  // (but other conditions in FindMovableClusterAtBodyRoot are satisfied).
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CollectivePermuteMotion pass;
  ASSERT_FALSE(pass.Run(&*module).value());
}

TEST_F(CollectivePermuteMotionTest, MoveWithElementwise) {
  absl::string_view hlo_string = R"(
  HloModule test
  body {
    loop_var = (s32[], f32[4,4]) parameter(0)
    constant.1 = s32[] constant(1)
    gte0 = s32[] get-tuple-element(loop_var), index=0
    add = s32[] add(gte0, constant.1)
    gte1 = f32[4,4] get-tuple-element(loop_var), index=1
    mul = f32[4,4] multiply(gte1, gte1)
    cp = f32[4,4] collective-permute(mul), source_target_pairs={{0,1},{1,2}}
    constant.4 = f32[] constant(1)
    broadcast = f32[4,4] broadcast(constant.4), dimensions={}
    add1 = f32[4,4] add(cp, broadcast)
    ROOT tuple = (s32[], f32[4,4]) tuple(add, add1)
  }
  cond {
    loop_var = (s32[], f32[4,4]) parameter(0)
    gte.cond = s32[] get-tuple-element(loop_var), index=0
    constant.3 = s32[] constant(5)
    ROOT lt = pred[] compare(gte.cond, constant.3), direction=LT
  }
  ENTRY  main {
    constant.2 = s32[] constant(0)
    param = f32[4,4] parameter(0)
    tuple.1 = (s32[], f32[4,4]) tuple(constant.2, param)
    while = (s32[], f32[4,4]) while(tuple.1), condition=cond, body=body
    ROOT result = s32[] get-tuple-element(while), index=0
  }
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CollectivePermuteMotion pass;
  ASSERT_TRUE(pass.Run(&*module).value());

  VLOG(1) << module->ToString();
  const HloInstruction* loop = FindInstruction(module.get(), "while");
  // Check if the operands are reshaped.
  const HloInstruction* output =
      loop->while_body()->root_instruction()->operand(1);
  auto input =
      AllOf(op::Shape("f32[4,4]"), op::GetTupleElement(op::Parameter(0)));
  auto moved =
      op::Add(op::CollectivePermute(input), op::Broadcast(op::Constant()));
  auto select = op::Select(op::Broadcast(op::Compare()), input, moved);
  EXPECT_THAT(output, op::Multiply(select, select));
}

TEST_F(CollectivePermuteMotionTest, DoNotMoveWithNonConstElementwise) {
  absl::string_view hlo_string = R"(
  HloModule test
  body {
    loop_var = (s32[], f32[4,4]) parameter(0)
    constant.1 = s32[] constant(1)
    gte0 = s32[] get-tuple-element(loop_var), index=0
    add = s32[] add(gte0, constant.1)
    gte1 = f32[4,4] get-tuple-element(loop_var), index=1
    mul = f32[4,4] multiply(gte1, gte1)
    cp = f32[4,4] collective-permute(mul), source_target_pairs={{0,1},{1,2}}
    constant.4 = f32[] constant(1)
    nonconst = f32[4,4] custom-call(), custom_call_target="unknown"
    add1 = f32[4,4] add(cp, nonconst)
    ROOT tuple = (s32[], f32[4,4]) tuple(add, add1)
  }
  cond {
    loop_var = (s32[], f32[4,4]) parameter(0)
    gte.cond = s32[] get-tuple-element(loop_var), index=0
    constant.3 = s32[] constant(5)
    ROOT lt = pred[] compare(gte.cond, constant.3), direction=LT
  }
  ENTRY  main {
    constant.2 = s32[] constant(0)
    param = f32[4,4] parameter(0)
    tuple.1 = (s32[], f32[4,4]) tuple(constant.2, param)
    while = (s32[], f32[4,4]) while(tuple.1), condition=cond, body=body
    ROOT result = s32[] get-tuple-element(while), index=0
  }
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CollectivePermuteMotion pass;
  ASSERT_FALSE(pass.Run(&*module).value());
}

TEST_F(CollectivePermuteMotionTest, DoNotMoveIfOutputUsed) {
  absl::string_view hlo_string = R"(
  HloModule test
  body {
    loop_var = (s32[], f32[4,4]) parameter(0)
    constant.1 = s32[] constant(1)
    gte0 = s32[] get-tuple-element(loop_var), index=0
    add = s32[] add(gte0, constant.1)
    gte1 = f32[4,4] get-tuple-element(loop_var), index=1
    mul = f32[4,4] multiply(gte1, gte1)
    cp = f32[4,4] collective-permute(mul), source_target_pairs={{0,1},{1,2}}
    ROOT tuple = (s32[], f32[4,4]) tuple(add, cp)
  }
  cond {
    loop_var = (s32[], f32[4,4]) parameter(0)
    gte.cond = s32[] get-tuple-element(loop_var), index=0
    constant.3 = s32[] constant(5)
    ROOT lt = pred[] compare(gte.cond, constant.3), direction=LT
  }
  ENTRY  main {
    constant.2 = s32[] constant(0)
    param = f32[4,4] parameter(0)
    tuple.1 = (s32[], f32[4,4]) tuple(constant.2, param)
    while = (s32[], f32[4,4]) while(tuple.1), condition=cond, body=body
    ROOT result = f32[4,4] get-tuple-element(while), index=1
  }
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CollectivePermuteMotion pass;
  ASSERT_FALSE(pass.Run(&*module).value());
}

TEST_F(CollectivePermuteMotionTest, DoNotMoveIfIndictionVarUnknown) {
  absl::string_view hlo_string = R"(
  HloModule test
  body {
    loop_var = (s32[], f32[4,4]) parameter(0)
    constant.1 = s32[] constant(1)
    gte0 = s32[] get-tuple-element(loop_var), index=0
    custom = s32[] custom-call(gte0, constant.1), custom_call_target="unknown"
    gte1 = f32[4,4] get-tuple-element(loop_var), index=1
    mul = f32[4,4] multiply(gte1, gte1)
    cp = f32[4,4] collective-permute(mul), source_target_pairs={{0,1},{1,2}}
    ROOT tuple = (s32[], f32[4,4]) tuple(custom, cp)
  }
  cond {
    loop_var = (s32[], f32[4,4]) parameter(0)
    gte.cond = s32[] get-tuple-element(loop_var), index=0
    constant.3 = s32[] constant(5)
    ROOT lt = pred[] compare(gte.cond, constant.3), direction=LT
  }
  ENTRY  main {
    constant.2 = s32[] constant(0)
    param = f32[4,4] parameter(0)
    tuple.1 = (s32[], f32[4,4]) tuple(constant.2, param)
    while = (s32[], f32[4,4]) while(tuple.1), condition=cond, body=body
    ROOT result = s32[] get-tuple-element(while), index=0
  }
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CollectivePermuteMotion pass;
  ASSERT_FALSE(pass.Run(&*module).value());
}

TEST_F(CollectivePermuteMotionTest, DoNotMoveIfMultiOutput) {
  absl::string_view hlo_string = R"(
  HloModule test
  body {
    loop_var = (s32[], f32[4,4], f32[4,4]) parameter(0)
    constant.1 = s32[] constant(1)
    gte0 = s32[] get-tuple-element(loop_var), index=0
    add = s32[] add(gte0, constant.1)
    gte1 = f32[4,4] get-tuple-element(loop_var), index=1
    mul = f32[4,4] multiply(gte1, gte1)
    cp = f32[4,4] collective-permute(mul), source_target_pairs={{0,1},{1,2}}
    ROOT tuple = (s32[], f32[4,4], f32[4,4]) tuple(add, cp, cp)
  }
  cond {
    loop_var = (s32[], f32[4,4], f32[4,4]) parameter(0)
    gte.cond = s32[] get-tuple-element(loop_var), index=0
    constant.3 = s32[] constant(5)
    ROOT lt = pred[] compare(gte.cond, constant.3), direction=LT
  }
  ENTRY  main {
    constant.2 = s32[] constant(0)
    param = f32[4,4] parameter(0)
    tuple.1 = (s32[], f32[4,4], f32[4,4]) tuple(constant.2, param, param)
    while = (s32[], f32[4,4], f32[4,4]) while(tuple.1),
      condition=cond, body=body
    ROOT result = s32[] get-tuple-element(while), index=0
  }
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CollectivePermuteMotion pass;
  ASSERT_FALSE(pass.Run(&*module).value());
}

}  // namespace
}  // namespace xla
