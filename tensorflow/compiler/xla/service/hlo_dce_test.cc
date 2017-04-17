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

#include "tensorflow/compiler/xla/service/hlo_dce.h"

#include <memory>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class HloDceTest : public HloTestBase {
 protected:
  HloDceTest() {}
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

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(3, computation->instruction_count());
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

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_EQ(1, dead_param1->user_count());

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_EQ(0, dead_param1->user_count());
}

}  // namespace
}  // namespace xla
