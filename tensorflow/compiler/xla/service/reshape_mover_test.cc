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
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace {
using ReshapeMoverTest = HloTestBase;

TEST_F(ReshapeMoverTest, ReshapesWithNonSameInputShapesNotMoved) {
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 8, 7, 1}), "param0"));
  auto reshape2 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape3 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  auto add4 = builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape2, reshape3));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(add4, computation->root_instruction());
  EXPECT_FALSE(ReshapeMover().Run(module.get()).ValueOrDie());
  EXPECT_EQ(add4, computation->root_instruction());
}

TEST_F(ReshapeMoverTest, EquivalentReshapesMovedAcrossFusion) {
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto reshape2 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape3 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  auto add4 = builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape2, reshape3));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->AddInstruction(HloInstruction::CreateFusion(
      add4->shape(), HloInstruction::FusionKind::kLoop, add4));
  TF_CHECK_OK(computation->ReplaceInstruction(add4, fusion));
  EXPECT_EQ(fusion, computation->root_instruction());
  EXPECT_TRUE(ReshapeMover().Run(module.get()).ValueOrDie());
  EXPECT_NE(fusion, computation->root_instruction());
  EXPECT_EQ(HloOpcode::kReshape, computation->root_instruction()->opcode());
}

}  // namespace
}  // namespace xla
