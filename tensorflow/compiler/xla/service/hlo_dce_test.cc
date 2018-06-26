/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_dce.h"

#include <memory>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class HloDceTest : public HloTestBase {
 protected:
  HloDceTest() {}

  // Returns whether the given instruction exists in the given computation.
  bool HasInstruction(const HloComputation& computation,
                      const HloInstruction* instruction) {
    return std::find(computation.instructions().begin(),
                     computation.instructions().end(),
                     instruction) != computation.instructions().end();
  }
};

TEST_F(HloDceTest, NoDeadCode) {
  // Verify that no dead code is removed from a computation with no dead code.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(123.0f)));
  builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(3, computation->instruction_count());
}

TEST_F(HloDceTest, InstructionsWithSideEffect) {
  // Verify that side-effect instructions (Send in this test) are not removed.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateSend(constant, /*channel_id=*/0));
  builder.AddInstruction(HloInstruction::CreateTuple({}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(3, computation->instruction_count());
}

TEST_F(HloDceTest, DeadParameters) {
  // Verify that dead parameters are not removed, but use of the dead parameters
  // are.
  auto builder = HloComputation::Builder(TestName());
  auto live_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "live_param"));
  auto dead_param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {}), "dead_param1"));
  builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(F32, {}), "dead_param2"));

  // This is a dead negate instruction.
  builder.AddInstruction(HloInstruction::CreateUnary(
      dead_param1->shape(), HloOpcode::kNegate, dead_param1));

  // This negate is not dead because it is the root.
  builder.AddInstruction(HloInstruction::CreateUnary(
      live_param->shape(), HloOpcode::kNegate, live_param));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_EQ(1, dead_param1->user_count());

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_EQ(0, dead_param1->user_count());
}

TEST_F(HloDceTest, ControlDependencies) {
  // Verify that instructions with control dependencies are not removed.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(123.0f)));

  // Create two dead instructions: a negate and an add.
  auto dead_negate = builder.AddInstruction(HloInstruction::CreateUnary(
      constant1->shape(), HloOpcode::kNegate, constant1));
  auto dead_add = builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  // Create the same two instructions again, but these will have a control
  // dependency added.
  auto dead_negate_with_control_dep =
      builder.AddInstruction(HloInstruction::CreateUnary(
          constant1->shape(), HloOpcode::kNegate, constant1));
  auto dead_add_with_control_dep =
      builder.AddInstruction(HloInstruction::CreateBinary(
          constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  // Create a root so the previously added instruction is dead.
  builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Add a control dependency between two instructions.
  TF_ASSERT_OK(dead_negate_with_control_dep->AddControlDependencyTo(
      dead_add_with_control_dep));

  EXPECT_EQ(7, computation->instruction_count());
  EXPECT_TRUE(HasInstruction(*computation, dead_negate));
  EXPECT_TRUE(HasInstruction(*computation, dead_add));
  EXPECT_TRUE(HasInstruction(*computation, dead_negate_with_control_dep));
  EXPECT_TRUE(HasInstruction(*computation, dead_add_with_control_dep));

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_FALSE(HasInstruction(*computation, dead_negate));
  EXPECT_FALSE(HasInstruction(*computation, dead_add));
  EXPECT_TRUE(HasInstruction(*computation, dead_negate_with_control_dep));
  EXPECT_TRUE(HasInstruction(*computation, dead_add_with_control_dep));
}

// Tests that a dead call instruction is removed.
TEST_F(HloDceTest, DeadInstructionWithCalledComputation) {
  auto module = CreateNewModule();
  Shape shape = ShapeUtil::MakeShape(F32, {});

  // Called computation for the call instruction.
  auto callee_builder = HloComputation::Builder(TestName() + "-callee");
  {
    auto param = callee_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    callee_builder.AddInstruction(
        HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  }
  auto called_computation =
      module->AddEmbeddedComputation(callee_builder.Build());

  // Entry computation with a call instruction.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto dead_call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {param}, called_computation));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, dead_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, dead_call));

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(1, param->user_count());
  EXPECT_FALSE(HasInstruction(*computation, dead_call));
}

// Tests that a while instruction with an infeed (effectul instruction) in its
// body is not removed, even its user count is 0.
TEST_F(HloDceTest, CalledComputationWithSideEffect) {
  auto module = CreateNewModule();
  Shape shape = ShapeUtil::MakeShape(F32, {});

  // Condition computation of a while instruction.
  auto cond_builder = HloComputation::Builder(TestName() + "-cond");
  {
    auto param = cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "cond_param"));
    auto constant = cond_builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
    cond_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, param, constant));
  }
  auto cond_computation = module->AddEmbeddedComputation(cond_builder.Build());

  // Body computation of a while instruction.
  auto body_builder = HloComputation::Builder(TestName() + "-body");
  {
    auto param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    auto token =
        body_builder.AddInstruction(HloInstruction::CreateAfterAll({}));
    auto infeed = body_builder.AddInstruction(
        HloInstruction::CreateInfeed(shape, token, ""));
    body_builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param, infeed));
  }
  auto body_computation = module->AddEmbeddedComputation(body_builder.Build());

  // Entry computation with a while instruction and a negate on the parameter.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto live_while = builder.AddInstruction(HloInstruction::CreateWhile(
      shape, cond_computation, body_computation, param));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  auto computation = module->AddEntryComputation(builder.Build());

  // Check the while instruction is not removed even if its user count is 0.
  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_while->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_while));

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_while->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_while));
}

// Tests that a nested call instruction with a side effect is not removed.
TEST_F(HloDceTest, CalledComputationWithNestedSideEffect) {
  auto module = CreateNewModule();
  Shape shape = ShapeUtil::MakeShape(F32, {});

  // Nested called computation with a side effect.
  auto nested_callee_builder =
      HloComputation::Builder(TestName() + "-nested_callee");
  {
    auto param = nested_callee_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    auto token = nested_callee_builder.AddInstruction(
        HloInstruction::CreateAfterAll({}));
    nested_callee_builder.AddInstruction(
        HloInstruction::CreateOutfeed(shape, param, token, ""));
  }
  auto nested_called_computation =
      module->AddEmbeddedComputation(nested_callee_builder.Build());

  // Outer called computation that calls the nested computation.
  auto callee_builder = HloComputation::Builder(TestName() + "-callee");
  {
    auto param = callee_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    callee_builder.AddInstruction(
        HloInstruction::CreateCall(shape, {param}, nested_called_computation));
  }
  auto called_computation =
      module->AddEmbeddedComputation(callee_builder.Build());

  // Entry computation with a call instruction.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto live_call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {param}, called_computation));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_call));

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_call));
}

TEST_F(HloDceTest, RemoveDeadSubcomputation) {
  auto module = CreateNewModule();
  HloComputation::Builder builder(TestName());

  HloComputation::Builder subcomp_builder("reduction_subcomp");
  {
    auto* param0 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "param0"));
    auto* param1 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "param1"));
    subcomp_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, param0, param1));
  }
  auto reduce_subcomp = module->AddEmbeddedComputation(subcomp_builder.Build());

  // Create a dead reduce instruction.
  builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(F32, {1}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {100}), "param0")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<float>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  // Add another instruction as the root of the computation.
  builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0)));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  // We should have DCE'ed the reduction computation along with the reduction
  // instruction.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 1);
}

TEST_F(HloDceTest, KeepUsedSubcomputation) {
  auto module = CreateNewModule();
  HloComputation::Builder builder(TestName());

  HloComputation::Builder subcomp_builder("reduction_subcomp");
  {
    auto* param0 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "param0"));
    auto* param1 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "param1"));
    subcomp_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, param0, param1));
  }
  auto reduce_subcomp = module->AddEmbeddedComputation(subcomp_builder.Build());

  // Create a dead reduce instruction.
  builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(F32, {1}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {100}), "param0")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<float>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  // Add another instruction as the root of the computation that also uses
  // reduce_subcomp.
  builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(F32, {1}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {100}), "param1")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<float>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  // We shouldn't have DCE'ed reduce_subcomp, even though we removed one of
  // its users.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);
}

}  // namespace
}  // namespace xla
