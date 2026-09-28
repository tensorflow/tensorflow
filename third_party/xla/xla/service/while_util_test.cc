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

#include "xla/service/while_util.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/shape.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

class WhileUtilTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> GetParsedModule(
      HloComputation** entry_computation, HloInstruction** param0,
      HloInstruction** param1, HloInstruction** param2) {
    const char* const hlo_string = R"(
HloModule ModuleWithWhile

while_body {
  ROOT p_body = (f32[32,32]{1,0}, f32[32,32]{1,0}) parameter(0)
}

while_condition {
  p_cond = (f32[32,32]{1,0}, f32[32,32]{1,0}) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  p_entry_0 = f32[32,32]{1,0} parameter(0)
  p_entry_1 = s32[32,32]{1,0} parameter(1)
  p_entry_2 = s64[32,32]{1,0} parameter(2)
  while_init = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(p_entry_0, p_entry_0)
  ROOT while = (f32[32,32]{1,0}, f32[32,32]{1,0}) while(while_init), condition=while_condition, body=while_body
}
)";

    ABSL_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_string));

    *entry_computation = module->entry_computation();
    *param0 = (*entry_computation)->parameter_instruction(0);
    *param1 = (*entry_computation)->parameter_instruction(1);
    *param2 = (*entry_computation)->parameter_instruction(2);

    return std::move(module);
  }
};

TEST_F(WhileUtilTest, MakeZeroInstructionsLiveOp) {
  HloInstruction *param0, *param1, *param2;
  HloComputation* entry_computation;

  ASSERT_OK_AND_ASSIGN(auto module, GetParsedModule(&entry_computation, &param0,
                                                    &param1, &param2));

  HloInstruction* while_instr = entry_computation->root_instruction();
  ASSERT_EQ(while_instr->opcode(), HloOpcode::kWhile);

  ASSERT_OK_AND_ASSIGN(
      WhileUtil::MakeInstructionsLiveInResult make_live_in_result,
      WhileUtil::MakeInstructionsLiveIn(while_instr, /*instructions=*/{}));

  HloInstruction* new_while_instr = make_live_in_result.new_while_instr;

  EXPECT_THAT(
      entry_computation->root_instruction(),
      op::Tuple(op::GetTupleElement(::testing::Eq(new_while_instr), 0),
                op::GetTupleElement(::testing::Eq(new_while_instr), 1)));

  auto param_reconstructed =
      op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                op::GetTupleElement(op::Parameter(0), 1));

  EXPECT_THAT(new_while_instr->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(param_reconstructed, 0),
                        op::GetTupleElement(param_reconstructed, 1)));
}

TEST_F(WhileUtilTest, MakeTwoInstructionsLive) {
  HloInstruction *param0, *param1, *param2;
  HloComputation* entry_computation;

  ASSERT_OK_AND_ASSIGN(auto module, GetParsedModule(&entry_computation, &param0,
                                                    &param1, &param2));

  HloInstruction* while_instr = entry_computation->root_instruction();
  ASSERT_EQ(while_instr->opcode(), HloOpcode::kWhile);

  ASSERT_OK_AND_ASSIGN(
      WhileUtil::MakeInstructionsLiveInResult make_live_in_result,
      WhileUtil::MakeInstructionsLiveIn(while_instr,
                                        /*instructions=*/{param0, param1}));

  HloInstruction* new_while_instr = make_live_in_result.new_while_instr;

  XLA_VLOG_LINES(3, module->ToString());

  EXPECT_THAT(
      entry_computation->root_instruction(),
      op::Tuple(op::GetTupleElement(::testing::Eq(new_while_instr), 0),
                op::GetTupleElement(::testing::Eq(new_while_instr), 1)));

  auto first_half_param_reconstructed =
      op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                op::GetTupleElement(op::Parameter(0), 1));

  EXPECT_THAT(new_while_instr->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(first_half_param_reconstructed, 0),
                        op::GetTupleElement(first_half_param_reconstructed, 1),
                        op::GetTupleElement(op::Parameter(0), 2),
                        op::GetTupleElement(op::Parameter(0), 3)));
}

TEST_F(WhileUtilTest, MakeInstructionLiveInWithOriginalValue) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

while_body {
  ROOT p_body = (s32[], s32[]) parameter(0)
}

while_condition {
  p_cond = (s32[], s32[]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  p_entry_0 = s32[] parameter(0), origin={{"p0"}}
  p_entry_1 = s32[] parameter(1), origin={{"p1"}}
  live_in_1 = f32[] parameter(2), origin={{"live_in_1"}}
  live_in_2 = f32[] parameter(3)
  live_tuple = (s32[], s32[]) tuple(p_entry_0, p_entry_1), origin={({"tuple_1"}, {"tuple_2"})}
  while_init = (s32[], s32[]) tuple(p_entry_0, p_entry_1)
  ROOT while0 = (s32[], s32[]) while(while_init), condition=while_condition, body=while_body, origin={({"p0_while"},{"p1_while"})}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* entry_computation = module->entry_computation();
  HloInstruction* while_instr = entry_computation->root_instruction();
  ASSERT_EQ(while_instr->opcode(), HloOpcode::kWhile);
  HloInstruction* live_in_1 = entry_computation->parameter_instruction(2);
  HloInstruction* live_in_2 = entry_computation->parameter_instruction(3);
  HloInstruction* live_tuple =
      entry_computation->GetInstructionWithName("live_tuple");

  ASSERT_OK_AND_ASSIGN(
      WhileUtil::MakeInstructionsLiveInResult make_live_in_result,
      WhileUtil::MakeInstructionsLiveIn(
          while_instr,
          /*instructions=*/{live_in_1, live_in_2, live_tuple}));

  HloInstruction* new_while_instr = make_live_in_result.new_while_instr;
  ASSERT_NE(new_while_instr->original_value(), nullptr);
  EXPECT_EQ(new_while_instr->original_value()->ToString(),
            "({\"p0_while\"}, {\"p1_while\"}, {\"live_in_1\"}, {}, "
            "({\"tuple_1\"}, {\"tuple_2\"}))");
}

TEST_F(WhileUtilTest, GetInvariantGTEsForWhileBody) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  param.b = (s32[], s32[]) parameter(0)
  gte.0 = s32[] get-tuple-element(param.b), index=0
  gte.1 = s32[] get-tuple-element(param.b), index=1
  add = s32[] add(gte.0, gte.1)
  ROOT tuple = (s32[], s32[]) tuple(gte.0, add)
}

cond {
  param.c = (s32[], s32[]) parameter(0)
  ROOT constant = pred[] constant(true)
}

ENTRY main {
  init = (s32[], s32[]) parameter(0)
  ROOT while = (s32[], s32[]) while(init), condition=cond, body=body
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  HloComputation* while_body = module->GetComputationWithName("body");

  ASSERT_NE(while_body, nullptr)
      << "Expected exactly one while_body computation";

  std::vector<HloInstruction*> gte_list =
      WhileUtil::GetInvariantGTEsForWhileBody(*while_body);

  ASSERT_EQ(gte_list.size(), 1);
  EXPECT_EQ((*gte_list.begin())->name(), "gte.0");
}

TEST_F(WhileUtilTest, AlwaysRemovePreviousWhileBody) {
  const char* const hlo_string = R"(
HloModule WhileWithSideEffects

body {
  param.b = (s32[], s32[]) parameter(0)
  gte.0 = s32[] get-tuple-element(param.b), index=0
  gte.1 = s32[] get-tuple-element(param.b), index=1
  add = s32[] add(gte.0, gte.1)
  ROOT tuple = (s32[], s32[]) tuple(gte.0, add)
}

cond {
  param.c = (s32[], s32[]) parameter(0)
  token0 = token[] after-all()
  infeed = (pred[], token[]) infeed(token0)
  ROOT condition = pred[] get-tuple-element(infeed), index=0
}

ENTRY main {
  init = (s32[], s32[]) parameter(0)
  to_make_live_in = f32[100] parameter(1)
  ROOT while = (s32[], s32[]) while(init), condition=cond, body=body
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  HloComputation* main = module->GetComputationWithName("main");
  HloInstruction* while_instr = main->root_instruction();
  HloInstruction* to_make_live_in = main->parameter_instruction(1);

  ASSERT_OK_AND_ASSIGN(
      WhileUtil::MakeInstructionsLiveInResult make_live_in_result,
      WhileUtil::MakeInstructionsLiveIn(while_instr,
                                        /*instructions=*/{to_make_live_in}));

  auto is_while = [](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kWhile;
  };
  EXPECT_EQ(absl::c_count_if(main->instructions(), is_while), 1);
}

TEST_F(WhileUtilTest, TryIncrementNonCounterTripCount) {
  constexpr absl::string_view hlo = R"(
HloModule main

body {
  param.0 = (s32[], s32[]) parameter(0)
  gte.0 = get-tuple-element(param.0), index=0
  gte.1 = get-tuple-element(param.0), index=1
  one.0 = s32[] constant(2)
  add.0 = s32[] add(gte.0, one.0)
  ROOT tuple.0 = (s32[], s32[]) tuple(add.0, gte.1)
}

cond {
  param.0 = (s32[], s32[]) parameter(0)
  gte.0 = get-tuple-element(param.0), index=0
  gte.1 = get-tuple-element(param.0), index=1
  minus-one.0 = s32[] constant(-1)
  add.0 = add(gte.1, minus-one.0)
  ROOT compare.0 = compare(gte.0, add.0), direction=LT
}

ENTRY main {
  param.0 = (s32[], s32[]) parameter(0)
  ROOT while = while(param.0), condition=cond, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* main = module->GetComputationWithName("main");
  const HloInstruction* while_instr = main->root_instruction();
  // Loop body increments induction variable by 2, in this case we should fail.
  EXPECT_FALSE(
      WhileUtil::IncrementWhileLoopTripCount(*while_instr, /*increment=*/1)
          .ok());
}

TEST_F(WhileUtilTest, TryIncrementNonConstantTripCount) {
  constexpr absl::string_view hlo = R"(
HloModule main

body {
  param.0 = (s32[], s32[]) parameter(0)
  gte.0 = get-tuple-element(param.0), index=0
  gte.1 = get-tuple-element(param.0), index=1
  one.0 = s32[] constant(1)
  add.0 = s32[] add(gte.0, one.0)
  add.1 = s32[] add(gte.1, one.0)
  ROOT tuple.0 = (s32[], s32[]) tuple(add.0, add.1)
}

cond {
  param.0 = (s32[], s32[]) parameter(0)
  gte.0 = get-tuple-element(param.0), index=0
  gte.1 = get-tuple-element(param.0), index=1
  minus-one.0 = s32[] constant(-1)
  add.0 = add(gte.1, minus-one.0)
  ROOT compare.0 = compare(gte.0, add.0), direction=LT
}

ENTRY main {
  param.0 = (s32[], s32[]) parameter(0)
  ROOT while = while(param.0), condition=cond, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* main = module->GetComputationWithName("main");
  const HloInstruction* while_instr = main->root_instruction();
  // Loop body increments trip count, in this case we should fail.
  EXPECT_FALSE(
      WhileUtil::IncrementWhileLoopTripCount(*while_instr, /*increment=*/1)
          .ok());
}

TEST_F(WhileUtilTest, TryIncrementSideEffecting) {
  constexpr absl::string_view hlo = R"(
HloModule main

body {
  param.0 = (s32[], s32[]) parameter(0)
  gte.0 = get-tuple-element(param.0), index=0
  gte.1 = get-tuple-element(param.0), index=1
  one.0 = s32[] constant(1)
  add.0 = s32[] add(gte.0, one.0)
  ROOT tuple.0 = (s32[], s32[]) tuple(add.0, gte.1)
}

cond {
  param.0 = (s32[], s32[]) parameter(0)
  gte.0 = get-tuple-element(param.0), index=0
  gte.1 = get-tuple-element(param.0), index=1
  minus-one.0 = s32[] constant(-1)
  add.0 = s32[] custom-call(gte.1, minus-one.0), custom_call_target="add", custom_call_has_side_effect=true
  ROOT compare.0 = compare(gte.0, add.0), direction=LT
}

ENTRY main {
  param.0 = (s32[], s32[]) parameter(0)
  ROOT while = while(param.0), condition=cond, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* main = module->GetComputationWithName("main");
  const HloInstruction* while_instr = main->root_instruction();
  // The trip count is modified with a side effecting op, in this case we
  // should fail.
  EXPECT_FALSE(
      WhileUtil::IncrementWhileLoopTripCount(*while_instr, /*increment=*/1)
          .ok());
}

TEST_F(WhileUtilTest, IncrementTripCountLt) {
  constexpr absl::string_view hlo = R"(
HloModule main

body {
  param.0 = (s32[], s32[]) parameter(0)
  gte.0 = get-tuple-element(param.0), index=0
  gte.1 = get-tuple-element(param.0), index=1
  one.0 = s32[] constant(1)
  add.0 = s32[] add(gte.0, one.0)
  ROOT tuple.0 = (s32[], s32[]) tuple(add.0, gte.1)
}

cond {
  param.0 = (s32[], s32[]) parameter(0)
  gte.0 = get-tuple-element(param.0), index=0
  gte.1 = get-tuple-element(param.0), index=1
  minus-one.0 = s32[] constant(-1)
  add.0 = add(gte.1, minus-one.0)
  ROOT compare.0 = compare(gte.0, add.0), direction=LT
}

ENTRY main {
  param.0 = (s32[], s32[]) parameter(0)
  ROOT while = while(param.0), condition=cond, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* main = module->GetComputationWithName("main");
  const HloInstruction* while_instr = main->root_instruction();
  EXPECT_OK(
      WhileUtil::IncrementWhileLoopTripCount(*while_instr, /*increment=*/1));

  const HloComputation* cond = module->GetComputationWithName("cond");
  EXPECT_THAT(cond->root_instruction()->operand(0),
              op::Add(op::GetTupleElement(), op::Constant()));
}

TEST_F(WhileUtilTest, IncrementTripCountGt) {
  constexpr absl::string_view hlo = R"(
HloModule main

body {
  param.0 = (s32[], s32[]) parameter(0)
  gte.0 = get-tuple-element(param.0), index=0
  gte.1 = get-tuple-element(param.0), index=1
  one.0 = s32[] constant(1)
  add.0 = s32[] add(gte.1, one.0)
  ROOT tuple.0 = (s32[], s32[]) tuple(gte.0, add.0)
}

cond {
  param.0 = (s32[], s32[]) parameter(0)
  gte.0 = get-tuple-element(param.0), index=0
  gte.1 = get-tuple-element(param.0), index=1
  minus-one.0 = s32[] constant(-1)
  add.0 = add(gte.0, minus-one.0)
  ROOT compare.0 = compare(add.0, gte.1), direction=GT
}

ENTRY main {
  param.0 = (s32[], s32[]) parameter(0)
  ROOT while = while(param.0), condition=cond, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* main = module->GetComputationWithName("main");
  const HloInstruction* while_instr = main->root_instruction();
  EXPECT_OK(
      WhileUtil::IncrementWhileLoopTripCount(*while_instr, /*increment=*/1));

  const HloComputation* cond = module->GetComputationWithName("cond");
  EXPECT_THAT(cond->root_instruction()->operand(1),
              op::Add(op::GetTupleElement(), op::Constant()));
}

TEST_F(WhileUtilTest, IsUpdatedBufferWriteOnly) {
  const char* const hlo_string = R"(
HloModule ModuleWithWriteOnly

ENTRY entry {
  p_entry_0 = f32[32,32]{1,0} parameter(0)
  p_entry_1 = f32[32,32]{1,0} parameter(1)

  zero = s32[] constant(0)
  update_slice = f32[1,32]{1,0} constant({ {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0} })

  // Base DUS
  dus1 = f32[32,32]{1,0} dynamic-update-slice(p_entry_0, update_slice, zero, zero)

  // Chained DUS (Safe)
  dus2 = f32[32,32]{1,0} dynamic-update-slice(dus1, update_slice, zero, zero)

  // Unsafe read
  slice1 = f32[1,32]{1,0} dynamic-slice(dus1, zero, zero), dynamic_slice_sizes={1,32}

  // Unsafe usage as payload
  dus3 = f32[32,32]{1,0} dynamic-update-slice(p_entry_1, dus1, zero, zero)

  // Dead code DUS (Unsafe because update is discarded)
  dus4 = f32[32,32]{1,0} dynamic-update-slice(p_entry_1, update_slice, zero, zero)

  ROOT root = tuple(dus2, slice1, dus3)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* dus1 = FindInstruction(module.get(), "dus1");
  HloInstruction* dus2 = FindInstruction(module.get(), "dus2");
  HloInstruction* dus4 = FindInstruction(module.get(), "dus4");

  ASSERT_NE(dus1, nullptr);
  ASSERT_NE(dus2, nullptr);
  ASSERT_NE(dus4, nullptr);

  // dus1 is NOT write-only because it feeds slice1 (a read) and dus3 (as
  // payload)
  EXPECT_FALSE(WhileUtil::IsUpdatedBufferWriteOnly(dus1));

  // dus2 IS write-only because it ONLY feeds the root instruction
  EXPECT_TRUE(WhileUtil::IsUpdatedBufferWriteOnly(dus2));

  // dus4 is NOT write-only because it has 0 users (dead code)
  EXPECT_FALSE(WhileUtil::IsUpdatedBufferWriteOnly(dus4));
}

TEST_F(WhileUtilTest,
       AppendToWhileLoopOriginalValueUpdatesAllLoopInstructions) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

while_body {
  ROOT p_body = (f32[], f32[]) parameter(0), origin={({"p0"}, {"p1"})}
}

while_condition {
  p_cond = (f32[], f32[]) parameter(0), origin={({"p0"}, {"p1"})}
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  p0 = f32[] parameter(0), origin={{"p0"}}
  p1 = f32[] parameter(1), origin={{"p1"}}
  p2 = f32[] parameter(2), origin={{"p2"}}
  init = (f32[], f32[]) tuple(p0, p1), origin={({"p0"}, {"p1"})}
  while = (f32[], f32[]) while(init), condition=while_condition, body=while_body, origin={({"p0"}, {"p1"}),["while#$"]}
  gte0 = f32[] get-tuple-element(while), index=0
  ROOT root = (f32[]) tuple(gte0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* while_instr = FindInstruction(module.get(), "while");
  HloInstruction* p2 = module->entry_computation()->parameter_instruction(2);

  HloComputation* body = while_instr->while_body();
  HloComputation* cond = while_instr->while_condition();

  Shape widened_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeScalarShape(F32), ShapeUtil::MakeScalarShape(F32),
       ShapeUtil::MakeScalarShape(F32)});
  *while_instr->mutable_shape() = widened_shape;
  *body->parameter_instruction(0)->mutable_shape() = widened_shape;
  *cond->parameter_instruction(0)->mutable_shape() = widened_shape;
  *body->root_instruction()->mutable_shape() = widened_shape;

  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  HloInstruction* p1 = module->entry_computation()->parameter_instruction(1);
  HloInstruction* new_init = module->entry_computation()->AddInstruction(
      HloInstruction::CreateTuple({p0, p1, p2}));
  ASSERT_OK(while_instr->ReplaceOperandWithDifferentShape(0, new_init));

  AppendToWhileLoopOriginalValue(while_instr, {p2});

  ASSERT_NE(while_instr->original_value(), nullptr);
  EXPECT_TRUE(while_instr->original_value()->IsCompatibleWith(widened_shape));
  EXPECT_THAT(while_instr->original_value()->original_array({2}),
              ::testing::Optional(::testing::Eq(OriginalArray{"p2"})));
  ASSERT_TRUE(while_instr->original_value()->call_hierarchy().has_value());
  EXPECT_EQ(*while_instr->original_value()->call_hierarchy(), "while#$");

  ASSERT_NE(body->parameter_instruction(0)->original_value(), nullptr);
  EXPECT_TRUE(
      body->parameter_instruction(0)->original_value()->IsCompatibleWith(
          widened_shape));
  EXPECT_THAT(
      body->parameter_instruction(0)->original_value()->original_array({2}),
      ::testing::Optional(::testing::Eq(OriginalArray{"p2"})));

  ASSERT_NE(cond->parameter_instruction(0)->original_value(), nullptr);
  EXPECT_TRUE(
      cond->parameter_instruction(0)->original_value()->IsCompatibleWith(
          widened_shape));
  EXPECT_THAT(
      cond->parameter_instruction(0)->original_value()->original_array({2}),
      ::testing::Optional(::testing::Eq(OriginalArray{"p2"})));

  ASSERT_NE(body->root_instruction()->original_value(), nullptr);
  EXPECT_TRUE(body->root_instruction()->original_value()->IsCompatibleWith(
      widened_shape));
}

TEST_F(WhileUtilTest,
       AppendToWhileLoopOriginalValueIncompatibleSubtreeCrashes) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

while_body {
  ROOT p_body = (f32[], f32[]) parameter(0), origin={({"p0"}, {"p1"})}
}

while_condition {
  p_cond = (f32[], f32[]) parameter(0), origin={({"p0"}, {"p1"})}
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  p0 = f32[] parameter(0), origin={{"p0"}}
  p1 = f32[] parameter(1), origin={{"p1"}}
  p2 = (f32[], f32[]) parameter(2), origin={({"p2_0"}, {"p2_1"})}
  init = (f32[], f32[]) tuple(p0, p1), origin={({"p0"}, {"p1"})}
  while = (f32[], f32[]) while(init), condition=while_condition, body=while_body, origin={({"p0"}, {"p1"}),["while#$"]}
  gte0 = f32[] get-tuple-element(while), index=0
  ROOT root = (f32[]) tuple(gte0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* while_instr = FindInstruction(module.get(), "while");
  HloInstruction* p2 = module->entry_computation()->parameter_instruction(2);

  EXPECT_DEATH(
      {
        Shape widened_shape = ShapeUtil::MakeTupleShape(
            {ShapeUtil::MakeScalarShape(F32), ShapeUtil::MakeScalarShape(F32),
             ShapeUtil::MakeScalarShape(F32)});
        *while_instr->mutable_shape() = widened_shape;
        AppendToWhileLoopOriginalValue(while_instr, {p2});
      },
      "Incompatible OriginalValue subtree for appended while input element 0");
}

TEST_F(WhileUtilTest, MakeInstructionsLiveInPropagatesOriginalValue) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

while_body {
  ROOT p_body = (f32[], f32[]) parameter(0), origin={({"p0"}, {"p1"})}
}

while_condition {
  p_cond = (f32[], f32[]) parameter(0), origin={({"p0"}, {"p1"})}
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  p0 = f32[] parameter(0), origin={{"p0"}}
  p1 = f32[] parameter(1), origin={{"p1"}}
  p2 = f32[] parameter(2), origin={{"p2"}}
  init = (f32[], f32[]) tuple(p0, p1), origin={({"p0"}, {"p1"})}
  while = (f32[], f32[]) while(init), condition=while_condition, body=while_body, origin={({"p0"}, {"p1"}),["while#$"]}
  gte0 = f32[] get-tuple-element(while), index=0
  ROOT root = (f32[]) tuple(gte0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* while_instr = FindInstruction(module.get(), "while");
  HloInstruction* p2 = module->entry_computation()->parameter_instruction(2);

  ASSERT_OK_AND_ASSIGN(WhileUtil::MakeInstructionsLiveInResult result,
                       WhileUtil::MakeInstructionsLiveIn(while_instr, {p2}));

  HloInstruction* new_while = result.new_while_instr;
  ASSERT_NE(new_while->original_value(), nullptr);
  EXPECT_TRUE(
      new_while->original_value()->IsCompatibleWith(new_while->shape()));
  EXPECT_THAT(new_while->original_value()->original_array({2}),
              ::testing::Optional(::testing::Eq(OriginalArray{"p2"})));

  ASSERT_NE(new_while->while_init()->original_value(), nullptr);
  EXPECT_TRUE(new_while->while_init()->original_value()->IsCompatibleWith(
      new_while->while_init()->shape()));
  EXPECT_THAT(new_while->while_init()->original_value()->original_array({2}),
              ::testing::Optional(::testing::Eq(OriginalArray{"p2"})));

  ASSERT_NE(new_while->while_body()->parameter_instruction(0)->original_value(),
            nullptr);
  EXPECT_TRUE(
      new_while->while_body()
          ->parameter_instruction(0)
          ->original_value()
          ->IsCompatibleWith(
              new_while->while_body()->parameter_instruction(0)->shape()));
  EXPECT_THAT(new_while->while_body()
                  ->parameter_instruction(0)
                  ->original_value()
                  ->original_array({2}),
              ::testing::Optional(::testing::Eq(OriginalArray{"p2"})));

  ASSERT_NE(
      new_while->while_condition()->parameter_instruction(0)->original_value(),
      nullptr);
  EXPECT_TRUE(
      new_while->while_condition()
          ->parameter_instruction(0)
          ->original_value()
          ->IsCompatibleWith(
              new_while->while_condition()->parameter_instruction(0)->shape()));

  ASSERT_NE(new_while->while_body()->root_instruction()->original_value(),
            nullptr);
  EXPECT_TRUE(new_while->while_body()
                  ->root_instruction()
                  ->original_value()
                  ->IsCompatibleWith(
                      new_while->while_body()->root_instruction()->shape()));

  ASSERT_NE(result.replacement_instr->original_value(), nullptr);
  EXPECT_TRUE(result.replacement_instr->original_value()->IsCompatibleWith(
      result.replacement_instr->shape()));
  ASSERT_EQ(result.while_body_live_in_values.size(), 1);
  ASSERT_NE(result.while_body_live_in_values[0]->original_value(), nullptr);
  EXPECT_THAT(
      result.while_body_live_in_values[0]->original_value()->original_array({}),
      ::testing::Optional(::testing::Eq(OriginalArray{"p2"})));
}

}  // namespace
}  // namespace xla
