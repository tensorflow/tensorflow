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

#include "tensorflow/compiler/xla/service/reshape_mover.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {
using ReshapeMoverTest = HloTestBase;

TEST_F(ReshapeMoverTest, ReshapesWithDifferentInputShapesNotMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 8, 7, 1}), "param1"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, reshape1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), op::Reshape(param1)));

  EXPECT_FALSE(ReshapeMover().Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), op::Reshape(param1)));
}

TEST_F(ReshapeMoverTest, ScalarReshapesNotMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 1, 1}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 1, 1}), "param1"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, reshape1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), op::Reshape(param1)));

  EXPECT_FALSE(ReshapeMover().Run(module.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Add(op::Reshape(op::Parameter()), op::Reshape(op::Parameter())));
}

TEST_F(ReshapeMoverTest, EquivalentReshapesMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param1"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, reshape1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), op::Reshape(param1)));
  EXPECT_TRUE(ReshapeMover().Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Add(param0, param1)));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

TEST_F(ReshapeMoverTest, ConstantAndReshapeMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {2, 3});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 3, 1, 2}), "param0"));
  auto const1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1, 2, 3}, {4, 5, 6}})));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, const1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), const1));

  EXPECT_TRUE(ReshapeMover().Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Add(param0, op::Reshape(const1))));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

TEST_F(ReshapeMoverTest, EquivalentReshapesMovedAcrossFusion) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param1"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, reshape1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->AddInstruction(HloInstruction::CreateFusion(
      add->shape(), HloInstruction::FusionKind::kLoop, add));
  TF_CHECK_OK(computation->ReplaceInstruction(add, fusion));

  EXPECT_THAT(computation->root_instruction(),
              op::Fusion(op::Reshape(param0), op::Reshape(param1)));

  EXPECT_TRUE(ReshapeMover().Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Fusion(param0, param1)));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

TEST_F(ReshapeMoverTest, EquivalentReshapesMovedAcrossSelect) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  auto pred_shape = ShapeUtil::MakeShape(PRED, {8, 7});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param1"));
  auto pred = builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(PRED, {1, 8, 1, 7}), "pred"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  auto reshape_pred =
      builder.AddInstruction(HloInstruction::CreateReshape(pred_shape, pred));
  builder.AddInstruction(HloInstruction::CreateTernary(
      root_shape, HloOpcode::kSelect, reshape_pred, reshape0, reshape1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Select(op::Reshape(pred), op::Reshape(param0), op::Reshape(param1)));

  EXPECT_TRUE(ReshapeMover().Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Select(pred, param0, param1)));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

TEST_F(ReshapeMoverTest, ScalarReshapeNotMovedAcrossSelect) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {});
  auto pred_shape = ShapeUtil::MakeShape(PRED, {});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {}), "param1"));
  auto pred = builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(PRED, {1, 1, 1}), "pred"));
  auto reshape_pred =
      builder.AddInstruction(HloInstruction::CreateReshape(pred_shape, pred));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      root_shape, HloOpcode::kSelect, reshape_pred, param0, param1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              op::Select(op::Reshape(pred), param0, param1));

  EXPECT_FALSE(ReshapeMover().Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Select(op::Reshape(pred), param0, param1));
  EXPECT_EQ(select, computation->root_instruction());
}

// Tree looks like this:
//
// add1
// |
// +- reshape2 - param2
// |
// +- reshape3 - add0
//               |
//               + reshape0 - param0
//               |
//               + reshape1 - param1
//
// We expect reshape{0,1} AND reshape{2,3} to be lifted.
TEST_F(ReshapeMoverTest, MultiplePasses) {
  auto shape1 = ShapeUtil::MakeShape(F32, {1, 8, 1, 7});
  auto shape2 = ShapeUtil::MakeShape(F32, {8, 7, 1});
  auto shape3 = ShapeUtil::MakeShape(F32, {8, 7});
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape1, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape1, "param1"));
  auto param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, shape2, "param2"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(shape2, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(shape2, param1));
  auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
      shape2, HloOpcode::kAdd, reshape0, reshape1));
  auto reshape2 =
      builder.AddInstruction(HloInstruction::CreateReshape(shape3, param2));
  auto reshape3 =
      builder.AddInstruction(HloInstruction::CreateReshape(shape3, add0));
  builder.AddInstruction(HloInstruction::CreateBinary(shape3, HloOpcode::kAdd,
                                                      reshape2, reshape3));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Add(op::Reshape(param2),
              op::Reshape(op::Add(op::Reshape(param0), op::Reshape(param1)))));

  EXPECT_TRUE(ReshapeMover().Run(module.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Reshape(op::Add(param2, op::Reshape(op::Add(param0, param1)))));
}

}  // namespace
}  // namespace xla
