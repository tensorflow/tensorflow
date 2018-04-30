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
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"

namespace xla {

using InstructionFusionTest = HloTestBase;

TEST_F(InstructionFusionTest, PotentialBitcastReshapeOfParameterUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "0"));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {1, 1}), param0));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(reshape1, computation->root_instruction());
  EXPECT_FALSE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

TEST_F(InstructionFusionTest, PotentialBitcastSimpleReshapeOfParameterUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "0"));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {1, 1}), param0));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(reshape1, computation->root_instruction());
  EXPECT_FALSE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

TEST_F(InstructionFusionTest, PotentialBitcastTransposeOfParameterUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "0"));
  auto transpose1 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(S32, {}), param0, {}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(transpose1, computation->root_instruction());
  EXPECT_FALSE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

TEST_F(InstructionFusionTest, AvoidDuplicationIfNotAllFusable) {
  HloComputation::Builder builder(TestName());
  auto shape = ShapeUtil::MakeShape(F32, {16, 16});
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "1"));
  HloInstruction* binary1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  builder.AddInstruction(HloInstruction::CreateSend(binary1, 0));
  HloInstruction* unary = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, binary1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(unary, computation->root_instruction());
  EXPECT_FALSE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
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
  auto module = tools::Parse(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[4,3]{1,0} parameter(0)
    add = f32[4,3]{1,0} add(p0, p0)
    ROOT root = f32[4,3]{1,0} subtract(add, add)
  })")
                    .ValueOrDie();
  // Expect the add and subtraction to be fused.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie())
      << module->ToString();
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1) << module->ToString();

  // Make sure the add hasn't been duplicated.
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1) << module->ToString();
}

TEST_F(InstructionFusionTest, AvoidDuplicationIfNotAllFusableRecursively) {
  // Make sure we do not duplicate the add, as we cannot fuse through the rng.
  //
  // p0 -> add -------------------------> sub
  //           \-> abs1 -> rng -> abs2 -/
  auto module = tools::Parse(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[4,3]{1,0} parameter(0)
    add = f32[4,3]{1,0} add(p0, p0)
    abs1 = f32[4,3]{1,0} abs(add)
    rng = f32[4,3]{1,0} rng(abs1), distribution=rng_uniform
    abs2 = f32[4,3]{1,0} abs(rng)
    ROOT root = f32[4,3]{1,0} subtract(abs2, add)
  })")
                    .ValueOrDie();
  // We expect abs2 to be fused into root.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie())
      << module->ToString();
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1) << module->ToString();

  // Make sure the add hasn't been duplicated.
  EXPECT_EQ(Count(*module, HloOpcode::kAdd), 1) << module->ToString();

  // Use a log node with a second consumer to break the fusion.
  //
  // p0 -> add -------------------------> sub
  //           \-> abs1 -> log -> abs2 -/
  //                           \-> send
  module = tools::Parse(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[4,3]{1,0} parameter(0)
    add = f32[4,3]{1,0} add(p0, p0)
    abs1 = f32[4,3]{1,0} abs(add)
    log = f32[4,3]{1,0} log(abs1)
    send = f32[4,3]{1,0} send(log), channel_id=0
    abs2 = f32[4,3]{1,0} abs(log)
    ROOT root = f32[4,3]{1,0} subtract(abs2, add)
  })")
               .ValueOrDie();

  // We expect abs2 to be fused into root and abs1 to be fused into log.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie())
      << module->ToString();
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 2) << module->ToString();

  // Make sure the add hasn't been duplicated.
  EXPECT_EQ(Count(*module, HloOpcode::kAdd), 1) << module->ToString();

  // Make sure we still fuse ops where one operand in the chain to the producer
  // can't be fused.
  //
  // p0 ---> add1 -----------> sub
  //    \         \-> add2 -/
  //     \-> log -/
  //             \-> send
  module = tools::Parse(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[4,3]{1,0} parameter(0)
    add1 = f32[4,3]{1,0} add(p0, p0)
    log = f32[4,3]{1,0} log(p0)
    send = f32[4,3]{1,0} send(log), channel_id=0
    add2 = f32[4,3]{1,0} add(log, add1)
    ROOT root = f32[4,3]{1,0} subtract(add1, add2)
  })")
               .ValueOrDie();

  // Expect the add1 and add2 to be fused into root.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie())
      << module->ToString();
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1) << module->ToString();

  // Make sure we didn't duplicate any adds.
  EXPECT_EQ(Count(*module, HloOpcode::kAdd), 2) << module->ToString();

  // A variant of the above that allows the algorithm to put add2 into the set
  // of unfusable ops to short-circuit the decision whether add1 should be fused
  // into sub2.
  //
  //             /---------------\
  // p0 ---> add1 ---> add2 ------> sub2
  //                       \------> sub1
  //                        log -/
  //                            \-> send
  module = tools::Parse(R"(
  HloModule test_module
  ENTRY OutputFusion {
    p0 = f32[4,3]{1,0} parameter(0)
    add1 = f32[4,3]{1,0} add(p0, p0)
    add2 = f32[4,3]{1,0} add(add1, add1)
    log = f32[4,3]{1,0} log(add2)
    send = f32[4,3]{1,0} send(log), channel_id=0
    sub1 = f32[4,3]{1,0} subtract(log, add2)
    sub2 = f32[4,3]{1,0} subtract(add2, add1)
    ROOT root = (f32[4,3]{1,0}, f32[4,3]{1,0}) tuple(sub1, sub2)
  })")
               .ValueOrDie();

  // Expect sub1 and sub2 to be fused into root.
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie())
      << module->ToString();
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1) << module->ToString();

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
  builder.AddInstruction(HloInstruction::CreateSend(unary1, 0));
  HloInstruction* unary2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, unary1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(unary2, computation->root_instruction());
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

TEST_F(InstructionFusionTest, AllowEffectiveUnaryDuplication) {
  auto shape = ShapeUtil::MakeShape(F32, {16, 16});
  auto small_shape = ShapeUtil::MakeShape(F32, {16});
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, small_shape, "0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "1"));
  HloInstruction* binary1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  builder.AddInstruction(HloInstruction::CreateSend(binary1, 0));
  HloInstruction* unary = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, binary1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(unary, computation->root_instruction());
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

}  // namespace xla
