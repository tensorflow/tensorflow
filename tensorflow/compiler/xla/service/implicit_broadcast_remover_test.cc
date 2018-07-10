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

#include "tensorflow/compiler/xla/service/implicit_broadcast_remover.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class ImplicitBroadcastRemoverTest : public HloVerifiedTestBase {
 protected:
  ImplicitBroadcastRemover remover_;
};

TEST_F(ImplicitBroadcastRemoverTest, NoImplicitBroadcast) {
  auto builder = HloComputation::Builder(TestName());

  const Shape shape = ShapeUtil::MakeShape(F32, {2, 4});
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));

  HloComputation* computation = module().AddEntryComputation(builder.Build());

  EXPECT_FALSE(remover_.Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Parameter(), op::Parameter()));
}

TEST_F(ImplicitBroadcastRemoverTest, ScalarBroadcast) {
  auto builder = HloComputation::Builder(TestName());

  const Shape shape = ShapeUtil::MakeShape(F32, {2, 4});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "scalar_param"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kPower, param0, param1));

  HloComputation* computation = module().AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();

  EXPECT_FALSE(ShapeUtil::Compatible(root->shape(), root->operand(0)->shape()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(1)->shape()));

  EXPECT_TRUE(remover_.Run(&module()).ValueOrDie());
  root = computation->root_instruction();

  EXPECT_THAT(root, op::Power(op::Broadcast(op::Parameter()), op::Parameter()));

  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(0)->shape()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(1)->shape()));
}

TEST_F(ImplicitBroadcastRemoverTest, DegenerateDimensionBroadcast) {
  auto builder = HloComputation::Builder(TestName());

  const Shape shape = ShapeUtil::MakeShape(F32, {2, 4, 6});
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 4, 1}), "p1"));
  builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, param0, param1));

  HloComputation* computation = module().AddEntryComputation(builder.Build());

  EXPECT_TRUE(remover_.Run(&module()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Subtract(op::Parameter(),
                                 op::Broadcast(op::Reshape(op::Parameter()))));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(0)->shape()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(1)->shape()));
}

TEST_F(ImplicitBroadcastRemoverTest, ScalarBroadcastToDegenerateDimensions) {
  auto builder = HloComputation::Builder(TestName());

  const Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 1});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "scalar_param"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, param0, param1));

  HloComputation* computation = module().AddEntryComputation(builder.Build());

  EXPECT_TRUE(remover_.Run(&module()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root,
              op::Subtract(op::Broadcast(op::Parameter()), op::Parameter()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(0)->shape()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(1)->shape()));
}

TEST_F(ImplicitBroadcastRemoverTest, TernaryDegenerateDimensionBroadcast) {
  auto builder = HloComputation::Builder(TestName());

  const Shape shape = ShapeUtil::MakeShape(F32, {2, 4, 6, 8});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 4, 1, 8}), "p0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 1, 6, 8}), "p1"));
  auto param2 = builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(F32, {2, 1, 6, 8}), "p2"));
  builder.AddInstruction(HloInstruction::CreateTernary(shape, HloOpcode::kClamp,
                                                       param0, param1, param2));

  HloComputation* computation = module().AddEntryComputation(builder.Build());

  EXPECT_TRUE(remover_.Run(&module()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Clamp(op::Broadcast(op::Reshape(op::Parameter())),
                              op::Broadcast(op::Reshape(op::Parameter())),
                              op::Broadcast(op::Reshape(op::Parameter()))));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(0)->shape()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(1)->shape()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(2)->shape()));
}

TEST_F(ImplicitBroadcastRemoverTest,
       TernaryScalarAndDegenerateDimensionBroadcast) {
  auto builder = HloComputation::Builder(TestName());

  const Shape shape = ShapeUtil::MakeShape(F32, {2, 4, 6});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 4, 6}), "p1"));
  auto param2 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape, "p2"));
  builder.AddInstruction(HloInstruction::CreateTernary(shape, HloOpcode::kClamp,
                                                       param0, param1, param2));

  HloComputation* computation = module().AddEntryComputation(builder.Build());

  EXPECT_TRUE(remover_.Run(&module()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Clamp(op::Broadcast(op::Parameter()),
                              op::Broadcast(op::Reshape(op::Parameter())),
                              op::Parameter()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(0)->shape()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(1)->shape()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), root->operand(2)->shape()));
}

}  // namespace
}  // namespace xla
