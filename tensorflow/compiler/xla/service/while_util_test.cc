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

#include "tensorflow/compiler/xla/service/while_util.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

StatusOr<std::unique_ptr<HloModule>> GetParsedModule(
    HloComputation** entry_computation, HloInstruction** param0,
    HloInstruction** param1, HloInstruction** param2) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

while_body {
  ROOT p_body = (f32[32,32]{1,0}, f32[32,32]{1,0}) parameter(0)
}

while_condition {
  p_cond = f32[32,32]{1,0} parameter(0)
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

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      tools::Parse(hlo_string));

  *entry_computation = module->entry_computation();
  *param0 = (*entry_computation)->parameter_instruction(0);
  *param1 = (*entry_computation)->parameter_instruction(1);
  *param2 = (*entry_computation)->parameter_instruction(2);

  return std::move(module);
}

TEST(WhileUtil, MakeZeroInstructionsLiveOp) {
  HloInstruction *param0, *param1, *param2;
  HloComputation* entry_computation;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      GetParsedModule(&entry_computation, &param0, &param1, &param2));

  HloInstruction* while_instr = entry_computation->root_instruction();
  ASSERT_EQ(while_instr->opcode(), HloOpcode::kWhile);

  TF_ASSERT_OK_AND_ASSIGN(
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

TEST(WhileUtilTest, MakeTwoInstructionsLive) {
  HloInstruction *param0, *param1, *param2;
  HloComputation* entry_computation;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      GetParsedModule(&entry_computation, &param0, &param1, &param2));

  HloInstruction* while_instr = entry_computation->root_instruction();
  ASSERT_EQ(while_instr->opcode(), HloOpcode::kWhile);

  TF_ASSERT_OK_AND_ASSIGN(
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

}  // namespace
}  // namespace xla
