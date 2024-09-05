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

#include "xla/service/defuser.h"

#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class DefuserTest : public HloTestBase {
 protected:
  // Returns the number of fusion instructions in the module.
  int FusionCount(const HloModule* m) {
    int count = 0;
    for (HloComputation* computation : m->computations()) {
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
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));

  m->AddEntryComputation(builder.Build());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_FALSE(defuser_.Run(m.get()).value());
}

TEST_F(DefuserTest, TrivialFusionInstructionAsRoot) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));

  auto computation = m->AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({add},
                                       HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(1, FusionCount(m.get()));
  EXPECT_TRUE(defuser_.Run(m.get()).value());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Parameter(), op::Parameter()));
}

TEST_F(DefuserTest, TrivialFusionInstructionNotAsRoot) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape_, HloOpcode::kNegate, add));

  auto computation = m->AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({add},
                                       HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Negate(op::Fusion()));

  EXPECT_EQ(1, FusionCount(m.get()));
  EXPECT_TRUE(defuser_.Run(m.get()).value());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_THAT(computation->root_instruction(),
              op::Negate(op::Add(op::Parameter(), op::Parameter())));
}

TEST_F(DefuserTest, NonTrivialFusionInstruction) {
  auto m = CreateNewVerifiedModule();
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
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, constant, div));

  auto computation = m->AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction(
      {add2, constant, div, mul, sub, negate, add},
      HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(1, FusionCount(m.get()));
  EXPECT_TRUE(defuser_.Run(m.get()).value());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Constant(), op::Divide()));
}

TEST_F(DefuserTest, MultipleFusionInstructions) {
  auto m = CreateNewVerifiedModule();
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
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, constant, div));

  auto computation = m->AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({add2, constant, div, mul},
                                       HloInstruction::FusionKind::kLoop);
  computation->CreateFusionInstruction({sub, negate, add},
                                       HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(2, FusionCount(m.get()));
  EXPECT_TRUE(defuser_.Run(m.get()).value());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Constant(), op::Divide()));
}

TEST_F(DefuserTest, NestedFusionInstructions) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape_, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape_, "p1"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_, HloOpcode::kAdd, param0, param1));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(shape_, HloOpcode::kNegate, add));

  auto computation = m->AddEntryComputation(builder.Build());
  auto outer_fusion = computation->CreateFusionInstruction(
      {negate, add}, HloInstruction::FusionKind::kLoop);
  HloInstruction* fused_negate = outer_fusion->fused_expression_root();
  ASSERT_EQ(fused_negate->opcode(), HloOpcode::kNegate);
  outer_fusion->fused_instructions_computation()->CreateFusionInstruction(
      {fused_negate}, HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(), op::Fusion());

  EXPECT_EQ(2, FusionCount(m.get()));
  EXPECT_TRUE(defuser_.Run(m.get()).value());
  EXPECT_EQ(0, FusionCount(m.get()));

  EXPECT_THAT(computation->root_instruction(), op::Negate(op::Add()));
}

}  // namespace
}  // namespace xla
