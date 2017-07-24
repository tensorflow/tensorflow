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

#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using FuseOpsTest = HloTestBase;

TEST_F(FuseOpsTest, DynamicUpdateSlice) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {10, 10});
  Shape update_shape = ShapeUtil::MakeShape(F32, {2, 3});

  auto builder = HloComputation::Builder(TestName());
  auto operand = builder.AddInstruction(
          HloInstruction::CreateParameter(0, input_shape, "operand"));
  auto update = builder.AddInstruction(
          HloInstruction::CreateParameter(1, update_shape, "update"));
  auto indices = builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR1<int64>({1, 2})));
  auto slice = builder.AddInstruction(
          HloInstruction::CreateDynamicUpdateSlice(
                  input_shape, operand, update, indices));

  builder.AddInstruction(
          HloInstruction::CreateTuple({slice}));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 5);

  FuseOps fuser;
  EXPECT_TRUE(fuser.Run(hlo_module.get()).ValueOrDie());

  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 4);
}

TEST_F(FuseOpsTest, Relu) {
  Shape shape = ShapeUtil::MakeShape(F32, {2});

  auto builder = HloComputation::Builder(TestName());
  auto i1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "operand"));
  auto c1 = builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR1<float>({0, 0})));
  auto m1 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape, HloOpcode::kMaximum, i1, c1));

  builder.AddInstruction(
          HloInstruction::CreateTuple({m1}));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 4);

  FuseOps fuser;
  EXPECT_TRUE(fuser.Run(hlo_module.get()).ValueOrDie());

  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 3);
}


}
}
}
