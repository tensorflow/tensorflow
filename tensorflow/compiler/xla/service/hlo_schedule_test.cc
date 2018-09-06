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

#include "tensorflow/compiler/xla/service/hlo_schedule.h"

#include <memory>
#include <string>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_scheduling.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class HloScheduleTest : public HloTestBase {};

TEST_F(HloScheduleTest, UpdateScheduleUnchangedModule) {
  // Updating the schedule of an unchanged HLO module should not affect the
  // schedule at all.
  const string module_str = R"(
HloModule UpdateScheduleUnchanged

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] constant(42.0)
  sum = f32[] add(a, b)
  neg = f32[] negate(c)
  ROOT root = f32[] multiply(sum, neg)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(*module, [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));
  const std::vector<const HloInstruction*>& entry_schedule =
      schedule.sequence(module->entry_computation()).instructions();

  EXPECT_EQ(entry_schedule.size(), 6);

  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(entry_schedule,
            schedule.sequence(module->entry_computation()).instructions());
}

TEST_F(HloScheduleTest, UpdateScheduleWithNewInstructions) {
  // Add some additional instructions to a module and verify the schedule can be
  // updated.
  const string module_str = R"(
HloModule UpdateScheduleWithNewInstructions

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] constant(42.0)
  sum = f32[] add(a, b)
  neg = f32[] negate(c)
  ROOT root = f32[] multiply(sum, neg)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(*module, [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));

  HloComputation* entry = module->entry_computation();
  const Shape shape = entry->root_instruction()->shape();
  HloInstruction* constant = entry->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  HloInstruction* sub = entry->AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, constant, entry->root_instruction()));
  entry->set_root_instruction(sub);

  auto in_schedule = [&](const HloInstruction* hlo) {
    return absl::c_linear_search(schedule.sequence(entry).instructions(), hlo);
  };

  EXPECT_EQ(schedule.sequence(entry).size(), 6);
  EXPECT_FALSE(in_schedule(constant));
  EXPECT_FALSE(in_schedule(sub));

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(schedule.sequence(entry).size(), 8);
  EXPECT_TRUE(in_schedule(constant));
  EXPECT_TRUE(in_schedule(sub));
}

TEST_F(HloScheduleTest, UpdateScheduleWithAddedAndDeletedInstruction) {
  // Add and delete some instructions from a module and verify that the schedule
  // can be updated successfully.
  const string module_str = R"(
HloModule UpdateScheduleWithAddedAndDeletedInstruction

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] constant(42.0)
  sum = f32[] add(a, b)
  neg = f32[] negate(c)
  ROOT root = f32[] multiply(sum, neg)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(*module, [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));

  // Set the entry root to some expression containing just a parameter and a
  // constant.
  HloComputation* entry = module->entry_computation();
  HloInstruction* constant = entry->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  HloInstruction* new_root = entry->AddInstruction(
      HloInstruction::CreateBinary(constant->shape(), HloOpcode::kSubtract,
                                   constant, entry->parameter_instruction(0)));
  entry->set_root_instruction(new_root);

  // DCE should remove everything but the parameters and the newly added code.
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()).status());

  EXPECT_EQ(schedule.sequence(entry).size(), 6);

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(schedule.sequence(entry).size(), 4);
}

TEST_F(HloScheduleTest, UpdateScheduleWithCompletelyReplacedModule) {
  // Completely replace a module with an entirely new set of instructions and
  // verify that the schedule can be updated successfully.
  const string module_str = R"(
HloModule UpdateScheduleWithCompletelyReplacedModule

ENTRY main {
  a = f32[] constant(42.0)
  b = f32[] constant(123.0)
  ROOT sum = f32[] add(a, b)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(*module, [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));

  // Replace the entry computation with the negation of a constant.
  HloComputation* entry = module->entry_computation();
  HloInstruction* constant = entry->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* new_root = entry->AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kNegate, constant));
  entry->set_root_instruction(new_root);

  // DCE the old instructions.
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()).status());

  EXPECT_EQ(schedule.sequence(entry).size(), 3);

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(schedule.sequence(entry).size(), 2);
}

TEST_F(HloScheduleTest, UpdateScheduleWithMultipleComputations) {
  // Create changes to more than one computation in an HLO module and verify
  // that the schedule can be updated.
  const string module_str = R"(
HloModule UpdateScheduleWithMultipleComputations

%Body (param.1: (s32[], token[])) -> (s32[], token[]) {
  %param.1 = (s32[], token[]) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], token[]) %param.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = token[] get-tuple-element((s32[], token[]) %param.1), index=1
  %after-all = token[] after-all(token[] %get-tuple-element.2)
  ROOT %tuple = (s32[], token[]) tuple(s32[] %add, token[] %after-all)
}

%Cond (param: (s32[], token[])) -> pred[] {
  %param = (s32[], token[]) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], token[]) %param), index=0
  %constant = s32[] constant(42)
  ROOT %less-than = pred[] less-than(s32[] %get-tuple-element, s32[] %constant)
}

ENTRY %WhileLoop () -> s32[] {
  %zero = s32[] constant(0)
  %init_token = token[] after-all()
  %init_tuple = (s32[], token[]) tuple(s32[] %zero, token[] %init_token)
  %while = (s32[], token[]) while((s32[], token[]) %init_tuple), condition=%Cond, body=%Body
  ROOT %root = s32[] get-tuple-element((s32[], token[]) %while), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(*module, [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(),
                                     /*pointer_size=*/sizeof(void*));
      }));

  const HloInstruction* xla_while =
      module->entry_computation()->root_instruction()->operand(0);
  HloComputation* body = xla_while->while_body();
  HloComputation* cond = xla_while->while_condition();

  // Negate the root of the cond.
  cond->set_root_instruction(cond->AddInstruction(
      HloInstruction::CreateUnary(ShapeUtil::MakeShape(PRED, {}),
                                  HloOpcode::kNot, cond->root_instruction())));

  // Replace the body with a computation which just passes through its
  // parameter.
  body->set_root_instruction(body->parameter_instruction(0));

  // DCE the dead code in the body.
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()).status());

  EXPECT_EQ(schedule.sequence(body).size(), 7);
  EXPECT_EQ(schedule.sequence(cond).size(), 4);

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(schedule.sequence(body).size(), 1);
  EXPECT_EQ(schedule.sequence(cond).size(), 5);
}

TEST_F(HloScheduleTest, UpdateScheduleComputationRemoved) {
  // Remove computations from a module and verify the schedule can be updated.
  const string module_str = R"(
HloModule UpdateScheduleWithMultipleComputations

%Body (param.1: (s32[], token[])) -> (s32[], token[]) {
  %param.1 = (s32[], token[]) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], token[]) %param.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = token[] get-tuple-element((s32[], token[]) %param.1), index=1
  %after-all = token[] after-all(token[] %get-tuple-element.2)
  ROOT %tuple = (s32[], token[]) tuple(s32[] %add, token[] %after-all)
}

%Cond (param: (s32[], token[])) -> pred[] {
  %param = (s32[], token[]) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], token[]) %param), index=0
  %constant = s32[] constant(42)
  ROOT %less-than = pred[] less-than(s32[] %get-tuple-element, s32[] %constant)
}

ENTRY %WhileLoop () -> s32[] {
  %zero = s32[] constant(0)
  %init_token = token[] after-all()
  %init_tuple = (s32[], token[]) tuple(s32[] %zero, token[] %init_token)
  %while = (s32[], token[]) while((s32[], token[]) %init_tuple), condition=%Cond, body=%Body
  ROOT %root = s32[] get-tuple-element((s32[], token[]) %while), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(*module, [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(),
                                     /*pointer_size=*/sizeof(void*));
      }));

  HloInstruction* xla_while =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  HloInstruction* init = xla_while->mutable_operand(0);

  // Replace the while with its init value. The conditional and body
  // computations should then be dead.
  TF_ASSERT_OK(xla_while->ReplaceAllUsesWith(init));

  // DCE the dead code in the body.
  HloDCE dce;
  ASSERT_EQ(module->computation_count(), 3);
  TF_ASSERT_OK(dce.Run(module.get()).status());
  ASSERT_EQ(module->computation_count(), 1);

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update());
  TF_ASSERT_OK(schedule.Verify());
}

}  // namespace
}  // namespace xla
