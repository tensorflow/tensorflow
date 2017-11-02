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

#include "tensorflow/compiler/xla/service/defuser.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class DefuserTest : public HloVerifiedTestBase {
 protected:
  // Returns the number of fusion instructions in the module.
  int FusionCount() {
    int count = 0;
    for (HloComputation* computation : module().computations()) {
      if (computation->IsFusionComputation()) {
        count++;
      }
    }
    return count;
  }

  Defuser defuser_;
  const Shape shape_ = ShapeUtil::MakeShape(F32, {2, 2});
};

TEST_F(DefuserTest, NoFusionInstruction) {
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));

  module().AddEntryComputation(builder.Build());
  EXPECT_EQ(0, FusionCount());

  EXPECT_FALSE(defuser_.Run(&module()).ValueOrDie());
}

TEST_F(DefuserTest, TrivialFusionInstructionAsRoot) {
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));

  auto computation = module().AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({add},
                                       HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(1, FusionCount());
  EXPECT_TRUE(defuser_.Run(&module()).ValueOrDie());
  EXPECT_EQ(0, FusionCount());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Parameter(), op::Parameter()));
}

TEST_F(DefuserTest, TrivialFusionInstructionNotAsRoot) {
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape_, HloOpcode::kNegate, add));

  auto computation = module().AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({add},
                                       HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Negate(op::Fusion()));

  EXPECT_EQ(1, FusionCount());
  EXPECT_TRUE(defuser_.Run(&module()).ValueOrDie());
  EXPECT_EQ(0, FusionCount());

  EXPECT_THAT(computation->root_instruction(),
              op::Negate(op::Add(op::Parameter(), op::Parameter())));
}

TEST_F(DefuserTest, NonTrivialFusionInstruction) {
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto param3 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape_, "p2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(shape_, HloOpcode::kNegate, add));
  auto sub = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kSubtract, add, negate));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kMultiply, sub, param3));
  auto div = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kDivide, mul, param3));
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, constant, div));

  auto computation = module().AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction(
      {add2, constant, div, mul, sub, negate, add},
      HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(1, FusionCount());
  EXPECT_TRUE(defuser_.Run(&module()).ValueOrDie());
  EXPECT_EQ(0, FusionCount());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Constant(), op::Divide()));
}

TEST_F(DefuserTest, MultipleFusionInstructions) {
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto param3 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape_, "p2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(shape_, HloOpcode::kNegate, add));
  auto sub = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kSubtract, add, negate));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kMultiply, sub, param3));
  auto div = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kDivide, mul, param3));
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, constant, div));

  auto computation = module().AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({add2, constant, div, mul},
                                       HloInstruction::FusionKind::kLoop);
  computation->CreateFusionInstruction({sub, negate, add},
                                       HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(2, FusionCount());
  EXPECT_TRUE(defuser_.Run(&module()).ValueOrDie());
  EXPECT_EQ(0, FusionCount());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Constant(), op::Divide()));
}

TEST_F(DefuserTest, NestedFusionInstructions) {
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(shape_, HloOpcode::kNegate, add));

  auto computation = module().AddEntryComputation(builder.Build());
  auto outer_fusion = computation->CreateFusionInstruction(
      {negate, add}, HloInstruction::FusionKind::kLoop);
  HloInstruction* fused_negate = outer_fusion->fused_expression_root();
  ASSERT_EQ(fused_negate->opcode(), HloOpcode::kNegate);
  outer_fusion->fused_instructions_computation()->CreateFusionInstruction(
      {fused_negate}, HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(2, FusionCount());
  EXPECT_TRUE(defuser_.Run(&module()).ValueOrDie());
  EXPECT_EQ(0, FusionCount());

  EXPECT_THAT(computation->root_instruction(), op::Negate(op::Add()));
}

}  // namespace
}  // namespace xla
