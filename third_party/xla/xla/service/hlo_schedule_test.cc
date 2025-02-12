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

#include "xla/hlo/ir/hlo_schedule.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class HloScheduleTest : public HloTestBase {};

TEST_F(HloScheduleTest, UpdateScheduleUnchangedModule) {
  // Updating the schedule of an unchanged HLO module should not affect the
  // schedule at all.
  const std::string module_str = R"(
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
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));
  const auto& entry_schedule =
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
  const std::string module_str = R"(
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
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
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
  const std::string module_str = R"(
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
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
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
  const std::string module_str = R"(
HloModule UpdateScheduleWithCompletelyReplacedModule

ENTRY main {
  a = f32[] constant(42.0)
  b = f32[] constant(123.0)
  ROOT sum = f32[] add(a, b)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
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
  const std::string module_str = R"(
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
  ROOT %less-than = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
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
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
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
  const std::string module_str = R"(
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
  ROOT %less-than = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
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
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue& buffer) {
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

TEST_F(HloScheduleTest, UpdateScheduleComputationRemovedWithMultiThreads) {
  // Remove computations from a module main thread and verify the schedule can
  // be updated while the other threads are remaining unchanged.
  const std::string module_str = R"(
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
  ROOT %less-than = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
}

%async_builder {
  %p0 = f32[10] parameter(0)
  %p1 = f32[10] parameter(1)
  ROOT %foo = add(%p0, %p1)
}, execution_thread="parallel_thread"

ENTRY %WhileLoop () -> (s32[], f32[10]) {
  %p0 = f32[10] parameter(0)
  %p1 = f32[10] parameter(1)
  %zero = s32[] constant(0)
  %init_token = token[] after-all()
  %init_tuple = (s32[], token[]) tuple(s32[] %zero, token[] %init_token)
  %while = (s32[], token[]) while((s32[], token[]) %init_tuple), condition=%Cond, body=%Body
  %async-start = ((f32[10], f32[10]), f32[10], s32[]) async-start(f32[10] %p0, f32[10] %p1), async_execution_thread="parallel_thread",calls=%async_builder
  %async-done = f32[10]{0} async-done(((f32[10], f32[10]), f32[10], s32[]) %async-start), async_execution_thread="parallel_thread", calls=%async_builder
  %main_res = s32[] get-tuple-element((s32[], token[]) %while), index=0
  ROOT %res = tuple(%main_res, %async-done)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(),
                     [](const BufferValue& buffer) {
                       return ShapeUtil::ByteSizeOf(
                           buffer.shape(),
                           /*pointer_size=*/sizeof(void*));
                     },
                     /*algorithm=*/{}, {HloInstruction::kMainExecutionThread}));

  HloInstruction* xla_while = module->entry_computation()
                                  ->root_instruction()
                                  ->mutable_operand(0)
                                  ->mutable_operand(0);
  HloInstruction* init = xla_while->mutable_operand(0);

  // Replace the while with its init value. The conditional and body
  // computations should then be dead.
  TF_ASSERT_OK(xla_while->ReplaceAllUsesWith(init));

  // DCE the dead code in the body.
  HloDCE dce;
  ASSERT_EQ(module->computation_count(), 4);
  TF_ASSERT_OK(dce.Run(module.get()).status());
  ASSERT_EQ(module->computation_count(), 2);

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update({HloInstruction::kMainExecutionThread}));
  TF_ASSERT_OK(schedule.Verify());

  ASSERT_EQ(module->MakeNonfusionComputations({"parallel_thread"}).size(), 1);
  ASSERT_FALSE(schedule.is_computation_scheduled(
      module->MakeNonfusionComputations({"parallel_thread"}).front()));
}

TEST_F(HloScheduleTest, UpdateScheduleAddComputation) {
  // Add a computation from a module main thread and verify the schedule can
  // be updated.
  const std::string module_str = R"(
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
  ROOT %less-than = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
}

%async_builder {
  %p0 = f32[10] parameter(0)
  %p1 = f32[10] parameter(1)
  ROOT %foo = add(%p0, %p1)
}, execution_thread="parallel_thread"

ENTRY %WhileLoop () -> (s32[], f32[10]) {
  %p0 = f32[10] parameter(0)
  %p1 = f32[10] parameter(1)
  %zero = s32[] constant(0)
  %init_token = token[] after-all()
  %init_tuple = (s32[], token[]) tuple(s32[] %zero, token[] %init_token)
  %while = (s32[], token[]) while((s32[], token[]) %init_tuple), condition=%Cond, body=%Body
  %async-start = ((f32[10], f32[10]), f32[10], s32[]) async-start(f32[10] %p0, f32[10] %p1), async_execution_thread="parallel_thread",calls=%async_builder
  %async-done = f32[10]{0} async-done(((f32[10], f32[10]), f32[10], s32[]) %async-start), async_execution_thread="parallel_thread", calls=%async_builder
  %main_res = s32[] get-tuple-element((s32[], token[]) %while), index=0
  ROOT %res = tuple(%main_res, %async-done)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(),
                     [](const BufferValue& buffer) {
                       return ShapeUtil::ByteSizeOf(
                           buffer.shape(),
                           /*pointer_size=*/sizeof(void*));
                     },
                     /*algorithm=*/{}, {HloInstruction::kMainExecutionThread}));

  HloComputation* entry_computation = module->entry_computation();
  // Insert computation
  HloComputation::Builder comp_builder("fusion_computation");
  HloInstruction* entry_comp_parameter_0 =
      entry_computation->parameter_instruction(0);
  HloInstruction* entry_comp_parameter_1 =
      entry_computation->parameter_instruction(1);

  std::vector<HloInstruction*> instructions_in_new_computation;

  HloInstruction* added_instruction =
      entry_computation->AddInstruction(HloInstruction::CreateBinary(
          entry_comp_parameter_0->shape(), HloOpcode::kMultiply,
          entry_comp_parameter_0, entry_comp_parameter_1));
  instructions_in_new_computation.push_back(added_instruction);

  HloInstruction* call =
      entry_computation->CreateCallInstruction(instructions_in_new_computation);

  Shape completion_sflag_shape = ShapeUtil::MakeScalarShape(U32);
  TF_ASSERT_OK_AND_ASSIGN(
      HloInstruction * async_done,
      entry_computation->CreateAsyncInstructions(
          call, {completion_sflag_shape}, entry_computation->execution_thread(),
          /*replace=*/true, /*override_names=*/true));

  HloInstruction* result_2 =
      entry_computation->root_instruction()->mutable_operand(1);
  HloInstruction* modified_result_2 =
      entry_computation->AddInstruction(HloInstruction::CreateBinary(
          result_2->shape(), HloOpcode::kAdd, async_done, result_2));

  TF_ASSERT_OK(result_2->ReplaceAllUsesWith(modified_result_2));

  auto added_computation_name =
      async_done->operand(0)->called_computations()[0]->name();
  ASSERT_FALSE(schedule.is_computation_scheduled(
      module->GetComputationWithName(added_computation_name)));

  ASSERT_IS_NOT_OK(schedule.Verify());
  TF_ASSERT_OK(schedule.Update({HloInstruction::kMainExecutionThread}));
  TF_ASSERT_OK(schedule.Verify());

  ASSERT_TRUE(schedule.is_computation_scheduled(
      module->GetComputationWithName(added_computation_name)));
}

}  // namespace
}  // namespace xla
