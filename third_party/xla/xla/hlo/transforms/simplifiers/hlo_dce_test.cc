/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/hlo_dce.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = ::xla::match;

class HloDceTest : public HloHardwareIndependentTestBase {
 protected:
  HloDceTest() {}

  // Returns whether the given instruction exists in the given computation.
  bool HasInstruction(const HloComputation& computation,
                      const HloInstruction* instruction) {
    return absl::c_linear_search(computation.instructions(), instruction);
  }
};

TEST_F(HloDceTest, NoDeadCode) {
  // Verify that no dead code is removed from a computation with no dead code.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(123.0f)));
  builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
}

TEST_F(HloDceTest, InstructionsWithSideEffect) {
  // Verify that side-effect instructions (Send in this test) are not removed.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto send = builder.AddInstruction(HloInstruction::CreateSend(
      constant, token, /*channel_id=*/0, /*is_host_transfer=*/false));
  builder.AddInstruction(HloInstruction::CreateSendDone(
      send, send->channel_id(), /*is_host_transfer=*/false));
  builder.AddInstruction(HloInstruction::CreateTuple({}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).value());

  EXPECT_EQ(5, computation->instruction_count());
}

TEST_F(HloDceTest, CustomCallInstructionsWithSideEffect) {
  // Verify that custom call instruction with side-effect is not removed.
  auto builder = HloComputation::Builder(TestName());
  auto instr = Cast<HloCustomCallInstruction>(builder.AddInstruction(
      HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                       /*operands=*/{},
                                       /*custom_call_target=*/"foo")));
  instr->set_custom_call_has_side_effect(true);
  builder.AddInstruction(HloInstruction::CreateTuple({}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&dce, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloDceTest, AsyncCustomCallInstructionsWithSideEffect) {
  // Verify that custom call instruction with side-effect is not removed.
  auto builder = HloComputation::Builder(TestName());
  auto instr = Cast<HloCustomCallInstruction>(builder.AddInstruction(
      HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                       /*operands=*/{},
                                       /*custom_call_target=*/"foo")));
  instr->set_custom_call_has_side_effect(true);
  builder.AddInstruction(HloInstruction::CreateTuple({}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN([[maybe_unused]] HloInstruction * async_done,
                          module->entry_computation()->CreateAsyncInstructions(
                              instr, {{ShapeUtil::MakeScalarShape(U32)}},
                              HloInstruction::kMainExecutionThread,
                              /*replace=*/true, /*override_names=*/true));

  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&dce, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloDceTest, CustomCallInstructionsWithoutSideEffect) {
  // Verify that custom call instruction without side-effect is removed.
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(
      HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                       /*operands=*/{},
                                       /*custom_call_target=*/"foo"));
  builder.AddInstruction(HloInstruction::CreateTuple({}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
}

TEST_F(HloDceTest, AsyncCustomCallInstructionsWithoutSideEffect) {
  // Verify that custom call instruction without side-effect is removed.
  auto builder = HloComputation::Builder(TestName());
  auto instr = Cast<HloCustomCallInstruction>(builder.AddInstruction(
      HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                       /*operands=*/{},
                                       /*custom_call_target=*/"foo")));
  instr->set_custom_call_has_side_effect(false);
  builder.AddInstruction(HloInstruction::CreateTuple({}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN([[maybe_unused]] HloInstruction * async_done,
                          module->entry_computation()->CreateAsyncInstructions(
                              instr, {{ShapeUtil::MakeScalarShape(U32)}},
                              HloInstruction::kMainExecutionThread,
                              /*replace=*/true, /*override_names=*/true));

  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
}

TEST_F(HloDceTest, ShardingCustomCallInstruction) {
  // Verify that sharding custom call instruction is not removed.
  auto builder = HloComputation::Builder(TestName());
  auto p0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {10, 10}), "p0"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(p0->shape(), HloOpcode::kAdd, p0, p0));
  // This is a dangling sharding custom-call to annotate add without any users.
  auto dangling_sharding = builder.AddInstruction(
      HloInstruction::CreateCustomCall(p0->shape(),
                                       /*operands=*/{add},
                                       /*custom_call_target=*/"Sharding"));
  dangling_sharding->set_sharding(HloSharding::Tile(TileAssignment({2, 1})));
  builder.AddInstruction(HloInstruction::CreateBinary(
      p0->shape(), HloOpcode::kMultiply, add, add));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&dce, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloDceTest, ShardingCustomCallInstructionWithDeadOperand) {
  // Verify that sharding custom call instruction is removed if its operand is
  // already dead.
  auto builder = HloComputation::Builder(TestName());
  auto p0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {10, 10}), "p0"));
  // This add is dead.
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(p0->shape(), HloOpcode::kAdd, p0, p0));
  // This is a dangling sharding custom-call to annotate add without any users.
  auto dangling_sharding = builder.AddInstruction(
      HloInstruction::CreateCustomCall(p0->shape(),
                                       /*operands=*/{add},
                                       /*custom_call_target=*/"Sharding"));
  dangling_sharding->set_sharding(HloSharding::Tile(TileAssignment({2, 1})));
  builder.AddInstruction(
      HloInstruction::CreateBinary(p0->shape(), HloOpcode::kMultiply, p0, p0));
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).value());

  EXPECT_EQ(2, computation->instruction_count());
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

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_EQ(1, dead_param1->user_count());

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).value());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_EQ(0, dead_param1->user_count());
}

TEST_F(HloDceTest, ControlDependencies) {
  // Verify that instructions with control dependencies are not removed.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(123.0f)));

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

  auto module = CreateNewVerifiedModule();
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
  EXPECT_TRUE(dce.Run(module.get()).value());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_FALSE(HasInstruction(*computation, dead_negate));
  EXPECT_FALSE(HasInstruction(*computation, dead_add));
  EXPECT_TRUE(HasInstruction(*computation, dead_negate_with_control_dep));
  EXPECT_TRUE(HasInstruction(*computation, dead_add_with_control_dep));
}

// Tests that a dead call instruction is removed.
TEST_F(HloDceTest, DeadInstructionWithCalledComputation) {
  auto module = CreateNewVerifiedModule();
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
  EXPECT_TRUE(dce.Run(module.get()).value());

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(1, param->user_count());
  EXPECT_FALSE(HasInstruction(*computation, dead_call));
}

// Tests that a while instruction with an infeed (effectul instruction) in its
// body is not removed, even its user count is 0.
TEST_F(HloDceTest, CalledComputationWithSideEffect) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {});

  // Condition computation of a while instruction.
  auto cond_builder = HloComputation::Builder(TestName() + "-cond");
  {
    auto param = cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "cond_param"));
    auto constant = cond_builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
    cond_builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param,
                                      constant, ComparisonDirection::kLt));
  }
  auto cond_computation = module->AddEmbeddedComputation(cond_builder.Build());

  // Body computation of a while instruction.
  auto body_builder = HloComputation::Builder(TestName() + "-body");
  {
    auto param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    auto token = body_builder.AddInstruction(HloInstruction::CreateToken());
    auto infeed = body_builder.AddInstruction(
        HloInstruction::CreateInfeed(shape, token, ""));
    auto infeed_data = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(shape, infeed, 0));
    body_builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, param, infeed_data));
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
  EXPECT_FALSE(dce.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_while->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_while));
}

// Tests that a nested call instruction with a side effect is not removed.
TEST_F(HloDceTest, CalledComputationWithNestedSideEffect) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {});

  // Nested called computation with a side effect.
  auto nested_callee_builder =
      HloComputation::Builder(TestName() + "-nested_callee");
  {
    auto param = nested_callee_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    auto token =
        nested_callee_builder.AddInstruction(HloInstruction::CreateToken());
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
    callee_builder.AddInstruction(HloInstruction::CreateCall(
        ShapeUtil::MakeTokenShape(), {param}, nested_called_computation));
  }
  auto called_computation =
      module->AddEmbeddedComputation(callee_builder.Build());

  // Entry computation with a call instruction.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto live_call = builder.AddInstruction(HloInstruction::CreateCall(
      ShapeUtil::MakeTokenShape(), {param}, called_computation));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(1, param->user_count());
  EXPECT_EQ(0, live_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_call));

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).value());

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(1, param->user_count());
  EXPECT_EQ(0, live_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_call));
}

TEST_F(HloDceTest, RemoveDeadSubcomputation) {
  auto module = CreateNewVerifiedModule();
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
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  // Add another instruction as the root of the computation.
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).value());

  // We should have DCE'ed the reduction computation along with the reduction
  // instruction.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 1);
}

TEST_F(HloDceTest, KeepUsedSubcomputation) {
  auto module = CreateNewVerifiedModule();
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
      ShapeUtil::MakeShape(F32, {}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {100}), "param0")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  // Add another instruction as the root of the computation that also uses
  // reduce_subcomp.
  builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(F32, {}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {100}), "param1")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).value());

  // We shouldn't have DCE'ed reduce_subcomp, even though we removed one of
  // its users.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);
}

TEST_F(HloDceTest, RemovedNestedDeadComputations) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {});

  HloComputation::Builder called_subcomp_builder("called_dead_add");
  {
    auto* param0 =
        called_subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/0, shape, "param0"));
    auto* param1 =
        called_subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/1, shape, "param1"));
    called_subcomp_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, param0, param1));
  }
  auto called_subcomp =
      module->AddEmbeddedComputation(called_subcomp_builder.Build());

  // Creates a module with unflattened control flow with two dead computations
  // that both call the same subcomputation, which becomes dead after the two
  // callers are removed.
  {
    HloComputation::Builder dead_subcomp_builder("dead_caller0");
    auto* param0 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param0"));
    auto* param1 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "param1"));
    dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateCall(shape, {param0, param1}, called_subcomp));
    module->AddEmbeddedComputation(dead_subcomp_builder.Build());
  }

  {
    HloComputation::Builder dead_subcomp_builder("dead_caller1");
    auto* param0 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param0"));
    auto* param1 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "param1"));
    dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateCall(shape, {param0, param1}, called_subcomp));
    module->AddEmbeddedComputation(dead_subcomp_builder.Build());
  }

  HloComputation::Builder builder(TestName());

  // Adds a constant instruction as the root of the computation.
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 4);

  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);

  // Only the entry computation should be left after eliminating the dead caller
  // and callee subcomputations.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 1);
}

TEST_F(HloDceTest, MultiOutputFusionRemoveUnusedTupleElementsRemoveTuple) {
  constexpr char kHloString[] = R"(
  HloModule test_module
  fused_add {
    p0 = f32[32,32]{1,0} parameter(0)
    p1 = f32[32,32]{1,0} parameter(1)
    p2 = f32[32,32]{1,0} parameter(2) // becomes dead
    add = f32[32,32]{1,0} add(p0, p1)
    ROOT res = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(p2, add)
  }

  ENTRY reduce {
    param0 = f32[32,32]{1,0} parameter(0)
    param1 = f32[32,32]{1,0} parameter(1)
    param2 = f32[32,32]{1,0} parameter(2)
    fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(param0, param1, param2), kind=kLoop, calls=fused_add
    gte.0 = f32[32,32]{1,0} get-tuple-element(fusion), index=0  // dead
    ROOT gte.1 = f32[32,32]{1,0} get-tuple-element(fusion), index=1
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);
  HloInstruction* root = module->entry_computation()->root_instruction();
  // We expect that the dead parameter and the dead tuple entry are removed.
  EXPECT_THAT(root, GmockMatch(m::Fusion(m::Parameter(0), m::Parameter(1))
                                   .WithShape(F32, {32, 32})));
  EXPECT_THAT(
      root->fused_expression_root(),
      GmockMatch(
          m::Add(m::Parameter(0), m::Parameter(1)).WithShape(F32, {32, 32})));
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);
}

TEST_F(
    HloDceTest,
    MultiOutputFusionRemoveUnusedTupleElementsRemoveTupleMultiUsersPerOutput) {
  constexpr char kHloString[] = R"(
  HloModule test_module
  fused_add {
    p0 = f32[32,32]{1,0} parameter(0)
    p1 = f32[32,32]{1,0} parameter(1)
    p2 = f32[32,32]{1,0} parameter(2) // becomes dead
    add = f32[32,32]{1,0} add(p0, p1)
    ROOT res = (f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(p2, add, p2)
  }

  ENTRY reduce {
    param0 = f32[32,32]{1,0} parameter(0)
    param1 = f32[32,32]{1,0} parameter(1)
    param2 = f32[32,32]{1,0} parameter(2)
    fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(param0, param1, param2), kind=kLoop, calls=fused_add
    gte.1 = f32[32,32]{1,0} get-tuple-element(fusion), index=1
    gte.1.again = f32[32,32]{1,0} get-tuple-element(fusion), index=1
    ROOT res = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(gte.1, gte.1.again)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);

  HloInstruction* gte_0 = FindInstruction(module.get(), "gte.0");
  EXPECT_EQ(gte_0, nullptr);
  HloInstruction* gte_1 = FindInstruction(module.get(), "gte.1");
  EXPECT_EQ(gte_1, nullptr);
  HloInstruction* gte_1_again = FindInstruction(module.get(), "gte.1.again");
  EXPECT_EQ(gte_1_again, nullptr);

  HloInstruction* fusion = FindInstruction(module.get(), "fusion");
  ASSERT_NE(fusion, nullptr);
  EXPECT_FALSE(fusion->shape().IsTuple());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand_count(), 2);
  EXPECT_EQ(root->operand(0), fusion);
  EXPECT_EQ(root->operand(1), fusion);
}

TEST_F(
    HloDceTest,
    MultiOutputFusionRemoveUnusedTupleElementsRemoveTupleNonContiguousRemoval) {
  constexpr char kHloString[] = R"(
  HloModule test_module
  fused_add {
    p0 = f32[32,32]{1,0} parameter(0)
    p1 = f32[32,32]{1,0} parameter(1)
    p2 = f32[32,32]{1,0} parameter(2) // becomes dead
    add = f32[32,32]{1,0} add(p0, p1)
    ROOT res = (f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(p2, add, p2, p2)
  }

  ENTRY reduce {
    param0 = f32[32,32]{1,0} parameter(0)
    param1 = f32[32,32]{1,0} parameter(1)
    param2 = f32[32,32]{1,0} parameter(2)
    fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(param0, param1, param2), kind=kLoop, calls=fused_add
    gte.0 = f32[32,32]{1,0} get-tuple-element(fusion), index=0  // dead
    gte.1 = f32[32,32]{1,0} get-tuple-element(fusion), index=1
    gte.1.again = f32[32,32]{1,0} get-tuple-element(fusion), index=1
    gte.3 = f32[32,32]{1,0} get-tuple-element(fusion), index=3
    ROOT res = (f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(gte.1, gte.1.again, gte.3)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);

  // We expect that the dead parameter and the dead tuple entry are removed.
  HloInstruction* gte_0 = FindInstruction(module.get(), "gte.0");
  EXPECT_EQ(gte_0, nullptr);
  HloInstruction* gte_1 = FindInstruction(module.get(), "gte.1");
  EXPECT_NE(gte_1, nullptr);
  EXPECT_EQ(static_cast<HloGetTupleElementInstruction*>(gte_1)->tuple_index(),
            0);
  HloInstruction* gte_1_again = FindInstruction(module.get(), "gte.1.again");
  EXPECT_EQ(
      static_cast<HloGetTupleElementInstruction*>(gte_1_again)->tuple_index(),
      0);
  EXPECT_NE(gte_1_again, nullptr);
  HloInstruction* gte_3 = FindInstruction(module.get(), "gte.3");
  EXPECT_NE(gte_3, nullptr);
  EXPECT_EQ(static_cast<HloGetTupleElementInstruction*>(gte_3)->tuple_index(),
            1);

  HloInstruction* fusion = FindInstruction(module.get(), "fusion");
  ASSERT_NE(fusion, nullptr);
  EXPECT_TRUE(fusion->shape().IsTuple());
  EXPECT_EQ(fusion->shape().tuple_shapes_size(), 2);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand_count(), 3);
  EXPECT_EQ(root->operand(0), gte_1);
  EXPECT_EQ(root->operand(1), gte_1_again);
  EXPECT_EQ(root->operand(2), gte_3);
}

TEST_F(HloDceTest, MultiOutputFusionRemoveUnusedTupleElementAdjustTuple) {
  constexpr char kHloString[] = R"(
  HloModule test_module
  fused_add {
    p0 = f32[32,32]{1,0} parameter(0)
    p1 = f32[32,32]{1,0} parameter(1)
    add = f32[32,32]{1,0} add(p0, p1)
    neg = f32[32,32]{1,0} negate(add)
    ROOT res = (f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(neg, p0, add)
  }

  ENTRY reduce {
    param0 = f32[32,32]{1,0} parameter(0)
    param1 = f32[32,32]{1,0} parameter(1)
    fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(param0, param1), kind=kLoop, calls=fused_add
    gte.0 = f32[32,32]{1,0} get-tuple-element(fusion), index=0
    gte.1 = f32[32,32]{1,0} get-tuple-element(fusion), index=1
    gte.2 = f32[32,32]{1,0} get-tuple-element(fusion), index=2
    ROOT add = f32[32,32]{1,0} add(gte.0, gte.2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);
  Shape shape = ShapeUtil::MakeShape(F32, {32, 32});
  Shape expected_shape = ShapeUtil::MakeTupleShape({shape, shape});
  HloInstruction* fusion;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Add(
                  m::GetTupleElement(
                      m::Fusion(&fusion).WithShapeEqualTo(&expected_shape), 0),
                  m::GetTupleElement(m::Fusion(), 1))));
  EXPECT_THAT(
      fusion->fused_expression_root(),
      GmockMatch(
          m::Tuple(m::Negate(), m::Add()).WithShapeEqualTo(&expected_shape)));
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);
}
TEST_F(HloDceTest,
       MultiOutputFusionRemoveUnusedTupleElementWithControlAdjustTupleAndDep) {
  constexpr char kHloString[] = R"(
  HloModule test_module
  fused_add {
    p0 = f32[32,32]{1,0} parameter(0)
    p1 = f32[32,32]{1,0} parameter(1)
    add = f32[32,32]{1,0} add(p0, p1)
    ROOT res = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(p0, add)
  }

  ENTRY reduce {
    param0 = f32[32,32]{1,0} parameter(0)
    param1 = f32[32,32]{1,0} parameter(1)
    fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(param0, param1), kind=kLoop, calls=fused_add
    gte.1 = f32[32,32]{1,0} get-tuple-element(fusion), index=1
    add.2 = f32[32,32]{1,0} add(param0, param1), control-predecessors={gte.1}
    ROOT add = f32[32,32]{1,0} add(add.2, gte.1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);
  HloInstruction* fusion;
  HloInstruction* add2;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::Add(&add2, m::Parameter(), m::Parameter()),
                                m::Fusion(&fusion))));
  EXPECT_EQ(add2->control_predecessors().size(), 1);
  EXPECT_EQ(add2->control_predecessors()[0], fusion);
}
}  // namespace
}  // namespace xla
