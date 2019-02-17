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

#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_wide_const.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using FuseWideConstTest = HloTestBase;

TEST_F(FuseWideConstTest, ReplaceWithWideConstant) {
  Shape vector = ShapeUtil::MakeShape(F32, {2});

  auto builder = HloComputation::Builder(TestName());
  auto c1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1)));
  builder.AddInstruction(HloInstruction::CreateBroadcast(vector, c1, {}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  EXPECT_THAT(hlo_module->computation_count(), 1);
  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 2);

  CompilerAnnotations annotations(hlo_module.get());
  FuseWideConst fwc(annotations);
  EXPECT_TRUE(fwc.Run(hlo_module.get()).ValueOrDie());
  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 1);

  HloInstruction* inst = hlo_module->entry_computation()->root_instruction();
  EXPECT_THAT(inst->opcode(), HloOpcode::kFusion);
  EXPECT_THAT(inst->fused_instructions_computation()->name(),
              "_pop_op_wide_const");
  EXPECT_TRUE(ShapeUtil::Equal(inst->shape(), vector));
  EXPECT_THAT(inst->operand_count(), 0);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
