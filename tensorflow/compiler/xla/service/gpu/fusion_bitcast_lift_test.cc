/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/fusion_bitcast_lift.h"

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

class FusionBitcastLiftTest : public HloTestBase {};

// Tests that we lift bitcast outside the fusion.
//
// This test MultiOutputFusion, multiple consecutive lift, bitcast
// with multiple users and bitcast that are used many time by the same
// user. This is a real kernel from Efficient Net, but with smaller
// shape to speed up tests.
//
// Input graph:
// Fusion 4d input, 2 1d output
//
// After optimization, the graph is:
// Bitcast 4d -> 2d
//   |
// Fusion 2d input, 2x1d outputs.
TEST_F(FusionBitcastLiftTest, NoBroadcastTest) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}

%fused_computation.21_4d (param_0.59: f16[2,14,14,672]) -> (f32[672], f32[672]) {
  %param_0.59 = f16[2,14,14,672] parameter(0)
  %convert.21 = f32[2,14,14,672] convert(%param_0.59)
  %bitcast.25 = f32[392,672] bitcast(%convert.21)
  %constant_84 = f32[] constant(0)
  %reduce.12 = f32[672]{0} reduce(%bitcast.25, %constant_84), dimensions={0}, to_apply=%scalar_add_computation
  %multiply.43.clone.1 = f32[2,14,14,672] multiply(%convert.21, %convert.21)
  %bitcast.24.clone.1 = f32[392,672] bitcast(%multiply.43.clone.1)
  %reduce.11.clone.1 = f32[672]{0} reduce(%bitcast.24.clone.1, %constant_84), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple.13 = (f32[672]{0}, f32[672]{0}) tuple(%reduce.12, %reduce.11.clone.1)
}

ENTRY main {
  %param_0.59 = f16[2,14,14,672] parameter(0)
  ROOT %fusion.21_4d = (f32[672]{0}, f32[672]{0}) fusion(%param_0.59), kind=kInput, calls=%fused_computation.21_4d
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_TRUE(FusionBitcastLift().Run(module.get()).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(HloOpcode::kFusion, root->opcode());

  // The fusion should have 1 input and it should be a bitcast with 2d output.
  EXPECT_EQ(1, root->operands().size());
  EXPECT_EQ(HloOpcode::kBitcast, root->operand(0)->opcode());
  EXPECT_EQ(2, root->operand(0)->shape().rank());

  // No bitcast should be left inside the fusion.
  for (HloInstruction* instr : root->fused_instructions()) {
    EXPECT_NE(HloOpcode::kBitcast, instr->opcode());
  }

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

// Tests that we lift bitcast outside the fusion when scalar broadcasting are
// present.
//
// Input graph:
// Fusion 1x4d and 1x0d inputs, 2 1d output
//
// After optimization, the graph is:
// Bitcast 4d -> 2d
//   |
// Fusion 1x2d and 1x0d inputs, 2 1d output
//   Inside the fusion, there is a bitcast left after the broadcast.
TEST_F(FusionBitcastLiftTest, ScalarBroadcastTest) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}

%fused_computation.21_4d (param_0.59: f16[2,14,14,672], param_1: f32[]) -> (f32[672], f32[672]) {
  %param_0.59 = f16[2,14,14,672] parameter(0)
  %convert.21 = f32[2,14,14,672] convert(%param_0.59)
  %bitcast.25 = f32[392,672] bitcast(%convert.21)
  %constant_84 = f32[] constant(0)
  %reduce.12 = f32[672]{0} reduce(%bitcast.25, %constant_84), dimensions={0}, to_apply=%scalar_add_computation
  %param_1 = f32[] parameter(1)
  %broadcast = f32[2,14,14,672] broadcast(%param_1), dimensions={}
  %multiply.43.clone.1 = f32[2,14,14,672] multiply(%convert.21, %broadcast)
  %bitcast.24.clone.1 = f32[392,672] bitcast(%multiply.43.clone.1)
  %reduce.11.clone.1 = f32[672]{0} reduce(%bitcast.24.clone.1, %constant_84), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple.13 = (f32[672]{0}, f32[672]{0}) tuple(%reduce.12, %reduce.11.clone.1)
}

ENTRY main {
  %param_0.59 = f16[2,14,14,672] parameter(0)
  %param_1 = f32[] parameter(1)
  ROOT %fusion.21_4d = (f32[672]{0}, f32[672]{0}) fusion(%param_0.59, %param_1), kind=kInput, calls=%fused_computation.21_4d
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_TRUE(FusionBitcastLift().Run(module.get()).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(HloOpcode::kFusion, root->opcode());

  // The fusion should have 2 inputs and the first one should be a
  // bitcast with 2d output.
  EXPECT_EQ(2, root->operands().size());
  EXPECT_EQ(HloOpcode::kBitcast, root->operand(0)->opcode());
  EXPECT_EQ(2, root->operand(0)->shape().rank());

  // Inside the fusion, there is 1 bitcast left after the broadcast.
  EXPECT_EQ(HloOpcode::kBroadcast, root->fused_instructions_computation()
                                       ->parameter_instruction(1)
                                       ->users()[0]
                                       ->opcode());
  EXPECT_EQ(HloOpcode::kBitcast, root->fused_instructions_computation()
                                     ->parameter_instruction(1)
                                     ->users()[0]
                                     ->users()[0]
                                     ->opcode());

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FusionBitcastLiftTest, RowBroadcastTest) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}

%fused_computation.21_4d (param_0.59: f16[2,14,14,672], param_1: f32[672]) -> (f32[672], f32[672]) {
  %param_0.59 = f16[2,14,14,672] parameter(0)
  %convert.21 = f32[2,14,14,672] convert(%param_0.59)
  %bitcast.25 = f32[392,672] bitcast(%convert.21)
  %constant_84 = f32[] constant(0)
  %reduce.12 = f32[672]{0} reduce(%bitcast.25, %constant_84), dimensions={0}, to_apply=%scalar_add_computation
  %param_1 = f32[672] parameter(1)
  %broadcast = f32[2,14,14,672] broadcast(%param_1), dimensions={3}
  %multiply.43.clone.1 = f32[2,14,14,672] multiply(%convert.21, %broadcast)
  %bitcast.24.clone.1 = f32[392,672] bitcast(%multiply.43.clone.1)
  %reduce.11.clone.1 = f32[672]{0} reduce(%bitcast.24.clone.1, %constant_84), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple.13 = (f32[672]{0}, f32[672]{0}) tuple(%reduce.12, %reduce.11.clone.1)
}

ENTRY main {
  %param_0.59 = f16[2,14,14,672] parameter(0)
  %param_1 = f32[672] parameter(1)
  ROOT %fusion.21_4d = (f32[672]{0}, f32[672]{0}) fusion(%param_0.59, %param_1), kind=kInput, calls=%fused_computation.21_4d
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_TRUE(FusionBitcastLift().Run(module.get()).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(HloOpcode::kFusion, root->opcode());

  // The fusion should have 2 inputs and the first one should be a
  // bitcast with 2d output.
  EXPECT_EQ(2, root->operands().size());
  EXPECT_EQ(HloOpcode::kBitcast, root->operand(0)->opcode());
  EXPECT_EQ(2, root->operand(0)->shape().rank());

  // Inside the fusion, there is 1 bitcast left after the broadcast.
  EXPECT_EQ(HloOpcode::kBroadcast, root->fused_instructions_computation()
                                       ->parameter_instruction(1)
                                       ->users()[0]
                                       ->opcode());
  EXPECT_EQ(HloOpcode::kBitcast, root->fused_instructions_computation()
                                     ->parameter_instruction(1)
                                     ->users()[0]
                                     ->users()[0]
                                     ->opcode());

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
