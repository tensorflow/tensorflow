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

#include "tensorflow/compiler/xla/service/instruction_fusion.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

namespace op = xla::testing::opcode_matchers;

using InstructionFusionTest = HloTestBase;

// Subclass of InstructionFusion exposing the protected methods Fuse and
// FuseIntoMultiOutput for testing.
class InstructionFusionForTesting : public InstructionFusion {
 public:
  explicit InstructionFusionForTesting()
      : InstructionFusion(InstructionFusion::IsExpensive) {}

  HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer,
                       HloComputation* computation) override {
    return InstructionFusion::Fuse(producer, consumer, computation);
  }

  HloInstruction* FuseIntoMultiOutput(HloInstruction* producer,
                                      HloInstruction* consumer,
                                      HloComputation* computation) override {
    return InstructionFusion::FuseIntoMultiOutput(producer, consumer,
                                                  computation);
  }
};

TEST_F(InstructionFusionTest, FuseInstructions) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY entry_computation {
    p0 = f32[4,3]{1,0} parameter(0)
    add = f32[4,3]{1,0} add(p0, p0)
    ROOT sub = f32[4,3]{1,0} subtract(add, p0)
  })")
                    .value();
  HloInstruction* sub = module->entry_computation()->root_instruction();
  HloInstruction* add = sub->mutable_operand(0);
  HloInstruction* fusion =
      InstructionFusionForTesting().Fuse(add, sub, module->entry_computation());

  ASSERT_THAT(fusion, op::Fusion()) << module->ToString();
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Subtract(op::Add(), op::Parameter()))
      << module->ToString();
}

TEST_F(InstructionFusionTest, FuseIntoFusionInstruction) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  fused_computation {
    p1 = f32[4,3] parameter(0)
    add = f32[4,3] add(p1, p1)
  }
  ENTRY entry_computation {
    p0 = f32[4,3] parameter(0)
    abs = f32[4,3] abs(p0)
    ROOT fusion = f32[4,3] fusion(abs), kind=kLoop, calls=fused_computation
  })")
                    .value();
  HloInstruction* root = module->entry_computation()->root_instruction();
  HloInstruction* abs = root->mutable_operand(0);
  HloInstruction* fusion = InstructionFusionForTesting().Fuse(
      abs, root, module->entry_computation());

  ASSERT_THAT(fusion, op::Fusion()) << module->ToString();
  EXPECT_THAT(fusion->fused_expression_root(), op::Add(op::Abs(), op::Abs()))
      << module->ToString();
}

TEST_F(InstructionFusionTest, FuseInstructionsIntoMultiOutput) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY entry_computation {
    p0 = f32[4,3]{1,0} parameter(0)
    abs = f32[4,3]{1,0} abs(p0)
    tanh = f32[4,3]{1,0} tanh(abs)
    ROOT add = f32[4,3]{1,0} add(abs, tanh)
  })")
                    .value();
  HloInstruction* root = module->entry_computation()->root_instruction();
  HloInstruction* abs = root->mutable_operand(0);
  HloInstruction* tanh = root->mutable_operand(1);
  HloInstruction* fusion = InstructionFusionForTesting().FuseIntoMultiOutput(
      abs, tanh, module->entry_computation());

  ASSERT_THAT(fusion, op::Fusion()) << module->ToString();
  EXPECT_THAT(fusion->fused_expression_root(), op::Tuple(op::Tanh(), op::Abs()))
      << module->ToString();
}

TEST_F(InstructionFusionTest, AvoidDuplicationIfNotAllFusible) {
  HloComputation::Builder builder(TestName());
  auto shape = ShapeUtil::MakeShape(F32, {16, 16});
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "1"));
  HloInstruction* binary1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto send =
      builder.AddInstruction(HloInstruction::CreateSend(binary1, token, 0));
  builder.AddInstruction(HloInstruction::CreateSendDone(send));
  HloInstruction* unary = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, binary1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(unary, computation->root_instruction());
  EXPECT_FALSE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .value())
      << module->ToString();
}

// Counts the number of HLO ops with a given op code in the specified module.
static int Count(const HloModule& module, HloOpcode op) {
  int count = 0;
  for (const auto* computation : module.computations()) {
    for (const auto* instruction : computation->instructions()) {
      if (instruction->opcode() == op) {
        ++count;
      }
    }
  }
  return count;
}

TEST_F(InstructionFusionTest, FuseCheapNonDuplicatableOps) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[4,3]{1,0} parameter(0)
    add = f32[4,3]{1,0} add(p0, p0)
    ROOT root = f32[4,3]{1,0} subtract(add, add)
  })")
                    .value();
  // Expect the add and subtraction to be fused.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .value())
      << module->ToString();
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1) << module->ToString();

  // Make sure the add hasn't been duplicated.
  EXPECT_EQ(Count(*module, HloOpcode::kAdd), 1) << module->ToString();
}

TEST_F(InstructionFusionTest, AvoidDuplicationIfNotAllFusibleRecursively) {
  // Make sure we do not duplicate the add, as we cannot fuse through the rng.
  //
  // (p0, p1) -> add -------------------------> sub
  //                 \-> abs1 -> rng -> abs2 -/
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    add = f32[] add(p0, p1)
    abs1 = f32[] abs(add)
    rng = f32[] rng(p1, abs1), distribution=rng_uniform
    abs2 = f32[] abs(rng)
    abs3 = f32[] abs(rng)
    ROOT root = f32[] subtract(abs2, add)
  })")
                    .value();
  // We expect abs2 to be fused into root.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .value())
      << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Subtract(op::Abs(op::Parameter()), op::Parameter()))
      << module->ToString();

  // Make sure the add hasn't been duplicated.
  EXPECT_EQ(Count(*module, HloOpcode::kAdd), 1) << module->ToString();

  // Use a log node with a second consumer to break the fusion.
  //
  // (p0, p1) -> add -------------------------> sub
  //                 \-> abs1 -> log -> abs2 -/
  //                                 \-> send -> send-done
  module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[4,3]{1,0} parameter(0)
    p1 = f32[4,3]{1,0} parameter(1)
    add = f32[4,3]{1,0} add(p0, p1)
    abs1 = f32[4,3]{1,0} abs(add)
    log = f32[4,3]{1,0} log(abs1)
    token0 = token[] after-all()
    send = f32[4,3]{1,0} send(log, token0), channel_id=1
    send-done = token[] send-done(send), channel_id=1
    abs2 = f32[4,3]{1,0} abs(log)
    ROOT root = f32[4,3]{1,0} subtract(abs2, add)
  })")
               .value();

  // We expect abs2 to be fused into root and abs1 to be fused into log.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .value())
      << module->ToString();
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 2) << module->ToString();

  // Make sure the add hasn't been duplicated.
  EXPECT_EQ(Count(*module, HloOpcode::kAdd), 1) << module->ToString();

  // Make sure we still fuse ops where one operand in the chain to the producer
  // can't be fused.
  //
  // (p0, p1) ---> add1 -----------> sub
  //          \         \-> add2 -/
  //           \-> log -/
  //                   \-> send -> send-done
  module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[4,3]{1,0} parameter(0)
    p1 = f32[4,3]{1,0} parameter(1)
    add1 = f32[4,3]{1,0} add(p0, p1)
    log = f32[4,3]{1,0} log(p0)
    token0 = token[] after-all()
    send = f32[4,3]{1,0} send(log, token0), channel_id=1
    send-done = token[] send-done(send), channel_id=1
    add2 = f32[4,3]{1,0} add(log, add1)
    ROOT root = f32[4,3]{1,0} subtract(add1, add2)
  })")
               .value();

  // Expect the add1 and add2 to be fused into root.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .value())
      << module->ToString();
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1) << module->ToString();

  // Make sure we didn't duplicate any adds.
  EXPECT_EQ(Count(*module, HloOpcode::kAdd), 2) << module->ToString();

  // A variant of the above that allows the algorithm to put add2 into the set
  // of unfusible ops to short-circuit the decision whether add1 should be fused
  // into sub2.
  //
  //             /---------------\
  // (p0, p1) ---> add1 ---> add2 ------> sub2
  //                             \------> sub1
  //                              log -/
  //                                  \-> send
  module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[4,3]{1,0} parameter(0)
    p1 = f32[4,3]{1,0} parameter(1)
    add1 = f32[4,3]{1,0} add(p0, p1)
    add2 = f32[4,3]{1,0} add(add1, p1)
    log = f32[4,3]{1,0} log(add2)
    token0 = token[] after-all()
    send = f32[4,3]{1,0} send(log, token0), channel_id=1
    send-done = token[] send-done(send), channel_id=1
    sub1 = f32[4,3]{1,0} subtract(log, add2)
    sub2 = f32[4,3]{1,0} subtract(add2, add1)
    ROOT root = (f32[4,3]{1,0}, f32[4,3]{1,0}) tuple(sub1, sub2)
  })")
               .value();

  // Expect sub1 and sub2 to be fused into root.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .value())
      << module->ToString();
  root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Tuple(op::Subtract(op::Parameter(), op::Parameter()),
                        op::Subtract(op::Parameter(), op::Parameter())))
      << module->ToString();

  // Make sure we didn't duplicate any adds.
  EXPECT_EQ(Count(*module, HloOpcode::kAdd), 2) << module->ToString();
}

TEST_F(InstructionFusionTest, AllowUnaryDuplication) {
  HloComputation::Builder builder(TestName());
  auto shape = ShapeUtil::MakeShape(F32, {16, 16});
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "0"));
  HloInstruction* unary1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kFloor, param0));
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto send =
      builder.AddInstruction(HloInstruction::CreateSend(unary1, token, 0));
  builder.AddInstruction(HloInstruction::CreateSendDone(send));
  HloInstruction* unary2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, unary1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(unary2, computation->root_instruction());
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .value());
}

TEST_F(InstructionFusionTest, AllowEffectiveUnaryDuplication) {
  auto shape = ShapeUtil::MakeShape(F32, {16, 16});
  auto small_shape = ShapeUtil::MakeShape(F32, {16});
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, small_shape, "0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "1"));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(shape, param0, {0}));
  HloInstruction* binary1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, broadcast, param1));
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto send =
      builder.AddInstruction(HloInstruction::CreateSend(binary1, token, 0));
  builder.AddInstruction(HloInstruction::CreateSendDone(send));
  HloInstruction* unary = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, binary1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(unary, computation->root_instruction());
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .value());
}

TEST_F(InstructionFusionTest, AllowBinarySameValueOperandsDuplication) {
  // Make sure we do duplicate the add of the same values, even though we cannot
  // fuse through the rng.
  //
  // p0 -> add -------------------------> sub
  //           \-> abs1 -> rng -> abs2 -/
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[] parameter(0)
    add = f32[] add(p0, p0)
    abs1 = f32[] abs(add)
    rng = f32[] rng(p0, abs1), distribution=rng_uniform
    abs2 = f32[] abs(rng)
    abs3 = f32[] abs(rng)
    ROOT root = f32[] subtract(abs2, add)
  })")
                    .value();
  // We expect abs2 to be fused into root.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .value())
      << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Subtract(op::Abs(op::Parameter()),
                           op::Add(op::Parameter(), op::Parameter())))
      << module->ToString();

  // Make sure the add has been duplicated.
  EXPECT_EQ(Count(*module, HloOpcode::kAdd), 2) << module->ToString();
}

TEST_F(InstructionFusionTest, FuseDiamondGraphsNoDuplication) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY Test {
    p0 = f32[100] parameter(0)
    p1 = f32[100] parameter(1)
    add = f32[100] add(p0, p1)
    slice1 = f32[99] slice(add), slice={[0:99:1]}
    slice2 = f32[99] slice(add), slice={[1:100:1]}
    ROOT add2 = f32[99] add(slice1, slice2)
  })")
                    .value();
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  // 'add' would originally need to be duplicated if fused. However after its
  // two users 'slice1' and 'slice2' are fused into 'add2', 'add' has only one
  // user and can now be also fused.
  EXPECT_THAT(root, op::Fusion(op::Parameter(), op::Parameter()));
}

TEST_F(InstructionFusionTest, FuseDiamondGraphsAllowDuplication) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY Test {
    p0 = f32[100] parameter(0)
    p1 = f32[100] parameter(1)
    add = f32[100] add(p0, p1)
    slice1 = f32[99] slice(add), slice={[0:99:1]}
    slice2 = f32[99] slice(add), slice={[1:100:1]}
    ROOT add2 = f32[99] add(slice1, slice2)
  })")
                    .value();
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .value())
      << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  // 'add' would originally need to be duplicated if fused. However after its
  // two users 'slice1' and 'slice2' are fused into 'add2', 'add' has only one
  // user and can now be also fused.
  EXPECT_THAT(root, op::Fusion(op::Parameter(), op::Parameter()));
}

TEST_F(InstructionFusionTest,
       WideningConvertsAreAlwaysDuplicableIntoConsumers) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY Test {
    p0 = f16[100] parameter(0)
    c = f32[100] convert(p0)
    add = f32[100] add(c, c)
    ROOT mul = f32[100] multiply(c, c)
  })")
                    .value();

  // The convert should be fused into the add and mul, even though may_duplicate
  // is false, because it's always beneficial to fuse/duplicate widening
  // converts into consumers.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion(op::Parameter()));
}

TEST_F(InstructionFusionTest, BroadcastsAreAlwaysDuplicableIntoConsumers) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY Test {
    p0 = f16[100] parameter(0)
    c = f32[100,100] broadcast(p0), dimensions={0}
    add = f32[100,100] add(c, c)
    ROOT mul = f32[100,100] multiply(c, c)
  })")
                    .value();

  // The broadcast should be fused into the add and mul, even though
  // may_duplicate is false, because it's always beneficial to fuse/duplicate
  // broadcasts into consumers.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion(op::Parameter()));
}

TEST_F(InstructionFusionTest,
       InPlaceOpShouldNotFuseWithNonElementwiseSharedOperand) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY Test {
    parameter.1 = f32[8] parameter(0)
    slice.19 = f32[7] slice(parameter.1), slice={[0:7]}
    constant.7 = f32[] constant(1)
    broadcast.8 = f32[7] broadcast(constant.7), dimensions={}
    add.9 = f32[7] add(slice.19, broadcast.8)
    constant.10 = s32[] constant(1)
    ROOT dynamic-update-slice.1 = f32[8] dynamic-update-slice(parameter.1, add.9, constant.10)
  })")
                    .value();
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();
  // Verify that we don't fuse dynamic-update-slice and slice together since
  // dynamic-update-slice modifies the input buffer in-place, which is also used
  // as slice's input.
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion(op::Parameter(), op::Slice()));
}

TEST_F(InstructionFusionTest, InPlaceOpShouldFuseWithSliceSameIndex) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY Test {
    parameter.1 = f32[8] parameter(0)
    slice.19 = f32[7] slice(parameter.1), slice={[1:8]}
    constant.7 = f32[] constant(1)
    broadcast.8 = f32[7] broadcast(constant.7), dimensions={}
    add.9 = f32[7] add(slice.19, broadcast.8)
    constant.10 = s32[] constant(1)
    ROOT dynamic-update-slice.1 = f32[8] dynamic-update-slice(parameter.1, add.9, constant.10)
  })")
                    .value();
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();
  // Verify that we fuse dynamic-update-slice and slice together because they
  // have the same index.
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion(op::Parameter()));
}

TEST_F(InstructionFusionTest, InPlaceOpShouldNotFuseWithUnknownDynamicSlice) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY Test {
    parameter.0 = f32[8] parameter(0)
    parameter.1 = s32[] parameter(1)
    dynamic-slice = f32[7] dynamic-slice(parameter.0, parameter.1), dynamic_slice_sizes={7}
    constant.7 = f32[] constant(1)
    broadcast.8 = f32[7] broadcast(constant.7), dimensions={}
    add.9 = f32[7] add(dynamic-slice, broadcast.8)
    constant.10 = s32[] constant(1)
    ROOT dynamic-update-slice.1 = f32[8] dynamic-update-slice(parameter.0, add.9, constant.10)
  })")
                    .value();
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion(op::Parameter(), op::DynamicSlice()));
}

TEST_F(InstructionFusionTest, InPlaceOpShouldFuseWithSameDynamicSlice) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY Test {
    parameter.0 = f32[8] parameter(0)
    parameter.1 = s32[] parameter(1)
    dynamic-slice = f32[7] dynamic-slice(parameter.0, parameter.1), dynamic_slice_sizes={7}
    constant.7 = f32[] constant(1)
    broadcast.8 = f32[7] broadcast(constant.7), dimensions={}
    add.9 = f32[7] add(dynamic-slice, broadcast.8)
    ROOT dynamic-update-slice.1 = f32[8] dynamic-update-slice(parameter.0, add.9, parameter.1)
  })")
                    .value();
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion(op::Parameter(), op::Parameter()));
}

TEST_F(InstructionFusionTest, InPlaceOpShouldNotFuseWithSliceSameIndex) {
  // Test case for b/223895450. Even though the indices for slice and DUS match,
  // fusing the two will cause reverse to share operand with the DUS in-place
  // buffer.
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY Test {
    parameter.1 = f32[8] parameter(0)
    slice.19 = f32[7] slice(parameter.1), slice={[1:8]}
    constant.7 = f32[] constant(1)
    broadcast.8 = f32[7] broadcast(constant.7), dimensions={}
    add.9 = f32[7] add(slice.19, broadcast.8)
    reverse = f32[7] reverse(add.9), dimensions={0}
    constant.10 = s32[] constant(1)
    ROOT dynamic-update-slice.1 = f32[8] dynamic-update-slice(parameter.1, reverse, constant.10)
  })")
                    .value();
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();
  // Verify that the slice is not fused because that would fuse with a
  // non-elementwise op (reverse) in between the slice and DUS.
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion(op::Parameter(), op::Slice()));
}

TEST_F(InstructionFusionTest,
       InPlaceOpShouldFuseWithSliceSameIndexWithoutUnsafeNonelementwise) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY Test {
    parameter.1 = f32[8] parameter(0)
    parameter.2 = f32[7] parameter(1)
    slice.19 = f32[7] slice(parameter.1), slice={[1:8]}
    reverse = f32[7] reverse(parameter.2), dimensions={0}
    add.9 = f32[7] add(slice.19, reverse)
    constant.10 = s32[] constant(1)
    ROOT dynamic-update-slice.1 = f32[8] dynamic-update-slice(parameter.1, add.9, constant.10)
  })")
                    .value();
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion(op::Parameter(), op::Parameter()));
}

TEST_F(InstructionFusionTest, InPlaceOpShouldNotBeFusedIfItSharesOperand) {
  // Test case for b/223896048. In-place operations that have an additional
  // operand that has the same value as the in-place buffer should not be fused.
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  update_s32 {
    lhs = s32[] parameter(0)
    ROOT rhs = s32[] parameter(1)
  }

  ENTRY main {
    arg0 = s32[9] parameter(0)
    iota = s32[9] iota(), iota_dimension=0
    indices = s32[9] reverse(iota), dimensions={0}
    ROOT scatter = s32[9] scatter(arg0, indices, arg0), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=update_s32
  }
  )")
                    .value();
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Scatter());
}

TEST_F(InstructionFusionTest, DontFuseAcrossRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY entry_computation {
    p0 = f32[4,3]{1,0} parameter(0)
    mul = f32[4,3]{1,0} multiply(p0, p0)
    ROOT add = f32[4,3]{1,0} add(mul, p0)
    sub = f32[4,3]{1,0} subtract(p0, add)
  })")
                    .value();
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/false)
          .Run(module.get())
          .value())
      << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion(op::Parameter()));
  EXPECT_THAT(
      root->fused_expression_root(),
      op::Add(op::Multiply(op::Parameter(), op::Parameter()), op::Parameter()));
}

class FusionDecisionTest : public HloTestBase {};

TEST_F(FusionDecisionTest, NotFusionPossibleDisjunction) {
  FusionDecision a = {};
  FusionDecision b = "not possible";
  EXPECT_TRUE(!a || !b);
  EXPECT_EQ((!(!a || !b)).Explain(), "not possible");

  a = "not possible";
  b = {};
  EXPECT_TRUE(!a || !b);
  EXPECT_EQ((!(!a || !b)).Explain(), "not possible");

  a = "impossible";
  b = "very impossible";
  EXPECT_TRUE(!a || !b);
  EXPECT_EQ((!(!a || !b)).Explain(), "impossible");

  a = {};
  b = {};
  EXPECT_FALSE(!a || !b);
}

}  // namespace xla
