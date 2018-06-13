/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"

#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;

class FusionMergerTest : public HloTestBase {};

// Tests that we can merge a fusion instruction that is below threshold.
//
// Computation after fusion merger pass (Fusion2 is merged into Fusion0 and
// Fusion1):
//                   Param
//                 /   |   \
//          Fusion3 Fusion0 Fusion1
//                 \   |   /
//                   Tuple
//
TEST_F(FusionMergerTest, MergeSharedFusionInstruction) {
  auto module = ParseHloString(R"(
HloModule MergeSharedFusionInstruction

comp.3 {
  constant.param_0 = f32[4]{0} parameter(0)
  param.param_1.2 = (f32[4]{0}, f32[4]{0}, f32[4]{0}) parameter(1)
  get-tuple-element.6 = f32[4]{0} get-tuple-element(param.param_1.2), index=0
  ROOT add.7 = f32[4]{0} add(constant.param_0, get-tuple-element.6)
}

comp.2 {
  param.param_1.1 = (f32[4]{0}, f32[4]{0}, f32[4]{0}) parameter(0)
  get-tuple-element.4 = f32[4]{0} get-tuple-element(param.param_1.1), index=1
  get-tuple-element.5 = f32[4]{0} get-tuple-element(param.param_1.1), index=2
  ROOT add.6 = f32[4]{0} add(get-tuple-element.4, get-tuple-element.5)
}

comp.1 {
  add.1.param_1.1 = f32[4]{0} parameter(1)
  constant.param_1.3 = f32[4]{0} parameter(0)
  add.5 = f32[4]{0} add(add.1.param_1.1, constant.param_1.3)
  ROOT multiply.3 = f32[4]{0} multiply(add.5, constant.param_1.3)
}

comp {
  add.1.param_1 = f32[4]{0} parameter(1)
  constant.param_1.1 = f32[4]{0} parameter(0)
  multiply.2 = f32[4]{0} multiply(add.1.param_1, constant.param_1.1)
  ROOT add.4 = f32[4]{0} add(multiply.2, constant.param_1.1)
}

ENTRY MergeSharedFusionInstruction.Computation0 {
  constant = f32[4]{0} constant({1, 1, 1, 1})
  param = (f32[4]{0}, f32[4]{0}, f32[4]{0}) parameter(0)
  fusion.3 = f32[4]{0} fusion(constant, param), kind=kLoop, calls=comp.3
  fusion.4 = f32[4]{0} fusion(param), kind=kLoop, calls=comp.2
  fusion.5 = f32[4]{0} fusion(constant, fusion.4), kind=kLoop, calls=comp.1
  fusion.6 = f32[4]{0} fusion(constant, fusion.4), kind=kLoop, calls=comp
  ROOT tuple = (f32[4]{0}, f32[4]{0}, f32[4]{0}) tuple(fusion.3, fusion.5, fusion.6)
})")
                    .ValueOrDie();
  EXPECT_TRUE(FusionMerger().Run(module.get()).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(HloOpcode::kTuple, root->opcode());
  // Check operand 0 (not merged). Should have 4 instructions.
  auto* operand0 = root->operand(0);
  EXPECT_EQ(HloOpcode::kFusion, operand0->opcode());
  EXPECT_EQ(4, operand0->fused_instruction_count());
  // Check operand 1 (should have merged in its operand fusion instruction).
  auto* operand1 = root->operand(1);
  EXPECT_EQ(HloOpcode::kFusion, operand1->opcode());
  EXPECT_EQ(7, operand1->fused_instruction_count());
  // Check operand 2 (should have merged in its operand fusion instruction).
  auto* operand2 = root->operand(2);
  EXPECT_EQ(HloOpcode::kFusion, operand2->opcode());
  EXPECT_EQ(7, operand2->fused_instruction_count());
}

// Tests that we do not merge a fusion instruction that above flops to bytes
// threshold.
//
// Fusion2 is not merged because it exceeds the threshold flops-to-bytes ratio.
TEST_F(FusionMergerTest, FlopsToBytesRatioThresholdExceeded) {
  auto module = ParseHloString(R"(
HloModule FlopsToBytesRatioThresholdExceeded

comp.2 {
  state.param_1.1 = (f32[4]{0}, f32[4]{0}) parameter(0)
  get-tuple-element.3 = f32[4]{0} get-tuple-element(state.param_1.1), index=0
  get-tuple-element.4 = f32[4]{0} get-tuple-element(state.param_1.1), index=2
  multiply.29 = f32[4]{0} multiply(get-tuple-element.3, get-tuple-element.4)
  multiply.30 = f32[4]{0} multiply(get-tuple-element.3, multiply.29)
  multiply.31 = f32[4]{0} multiply(get-tuple-element.3, multiply.30)
  multiply.32 = f32[4]{0} multiply(get-tuple-element.3, multiply.31)
  multiply.33 = f32[4]{0} multiply(get-tuple-element.3, multiply.32)
  multiply.34 = f32[4]{0} multiply(get-tuple-element.3, multiply.33)
  multiply.35 = f32[4]{0} multiply(get-tuple-element.3, multiply.34)
  multiply.36 = f32[4]{0} multiply(get-tuple-element.3, multiply.35)
  multiply.37 = f32[4]{0} multiply(get-tuple-element.3, multiply.36)
  multiply.38 = f32[4]{0} multiply(get-tuple-element.3, multiply.37)
  multiply.39 = f32[4]{0} multiply(get-tuple-element.3, multiply.38)
  multiply.40 = f32[4]{0} multiply(get-tuple-element.3, multiply.39)
  ROOT multiply.41 = f32[4]{0} multiply(get-tuple-element.3, multiply.40)
}

comp.1 {
  multiply.12.param_1.1 = f32[4]{0} parameter(1)
  constant.param_1.3 = f32[4]{0} parameter(0)
  add.3 = f32[4]{0} add(multiply.12.param_1.1, constant.param_1.3)
  ROOT multiply.16 = f32[4]{0} multiply(add.3, constant.param_1.3)
}

comp {
  multiply.12.param_1 = f32[4]{0} parameter(1)
  constant.param_1.1 = f32[4]{0} parameter(0)
  multiply.15 = f32[4]{0} multiply(multiply.12.param_1, constant.param_1.1)
  ROOT add.2 = f32[4]{0} add(multiply.15, constant.param_1.1)
}

ENTRY FlopsToBytesRatioThresholdExceeded.Computation1 {
  constant = f32[4]{0} constant({1, 1, 1, 1})
  state = (f32[4]{0}, f32[4]{0}) parameter(0)
  fusion.2 = f32[4]{0} fusion(state), kind=kLoop, calls=comp.2
  fusion.3 = f32[4]{0} fusion(constant, fusion.2), kind=kLoop, calls=comp.1
  fusion.4 = f32[4]{0} fusion(constant, fusion.2), kind=kLoop, calls=comp
  ROOT tuple = (f32[4]{0}, f32[4]{0}) tuple(fusion.3, fusion.4)
})")
                    .ValueOrDie();
  // Run fusion merger pass, which should detect that the flops/bytes of the
  // shared fusion instruction exceeds the threshold ratio, and therefore
  // cannot be merged with other fusion instructions.
  EXPECT_FALSE(FusionMerger().Run(module.get()).ValueOrDie());
}

// Tests that threshold for bytes transferred if merged is exceeded.
//
// Fusion2 is not merged because it exceeds the threshold bytes transferred.
// This is because the bytes read by Fusion2 (when replicated if the instruction
// is merged into Fusion0 and Fusion1) would exceed the bytes transferred
// threshold.
TEST_F(FusionMergerTest, BytesTransferredThresholdExeceeded) {
  auto module = ParseHloString(R"(
HloModule BytesTransferredThresholdExeceeded

comp.2 {
  state.param_1.1 = (f32[4]{0}, f32[4]{0}, f32[4]{0}, f32[4]{0}) parameter(0)
  get-tuple-element.7 = f32[4]{0} get-tuple-element(state.param_1.1), index=0
  get-tuple-element.8 = f32[4]{0} get-tuple-element(state.param_1.1), index=1
  add.9 = f32[4]{0} add(get-tuple-element.7, get-tuple-element.8)
  get-tuple-element.9 = f32[4]{0} get-tuple-element(state.param_1.1), index=2
  add.10 = f32[4]{0} add(add.9, get-tuple-element.9)
  get-tuple-element.10 = f32[4]{0} get-tuple-element(state.param_1.1), index=3
  ROOT add.11 = f32[4]{0} add(add.10, get-tuple-element.10)
}

comp.1 {
  add.2.param_1.1 = f32[4]{0} parameter(1)
  constant.param_1.3 = f32[4]{0} parameter(0)
  add.6 = f32[4]{0} add(add.2.param_1.1, constant.param_1.3)
  ROOT multiply.3 = f32[4]{0} multiply(add.6, constant.param_1.3)
}

comp {
  add.2.param_1 = f32[4]{0} parameter(1)
  constant.param_1.1 = f32[4]{0} parameter(0)
  multiply.2 = f32[4]{0} multiply(add.2.param_1, constant.param_1.1)
  ROOT add.5 = f32[4]{0} add(multiply.2, constant.param_1.1)
}

ENTRY BytesTransferredThresholdExeceeded.Computation2 {
  constant = f32[4]{0} constant({1, 1, 1, 1})
  state = (f32[4]{0}, f32[4]{0}, f32[4]{0}, f32[4]{0}) parameter(0)
  fusion.2 = f32[4]{0} fusion(state), kind=kLoop, calls=comp.2
  fusion.3 = f32[4]{0} fusion(constant, fusion.2), kind=kLoop, calls=comp.1
  fusion.4 = f32[4]{0} fusion(constant, fusion.2), kind=kLoop, calls=comp
  ROOT tuple = (f32[4]{0}, f32[4]{0}) tuple(fusion.3, fusion.4)
})")
                    .ValueOrDie();
  // Run fusion merger pass, which should detect that the net bytes transferred
  // (if merged) would increase.
  EXPECT_FALSE(FusionMerger().Run(module.get()).ValueOrDie());
}

// Tests that threshold for bytes transferred if merged is not exceeded.
//
// Fusion2 is merged into Fusion0 and Fusion1, because bytes read from Param by
// Fusion2 is reduced for this test which makes the merge operation into its
// operand below the bytes transferred threshold.
TEST_F(FusionMergerTest, BytesTransferredThresholdNotExeceeded) {
  auto module = ParseHloString(R"(
HloModule BytesTransferredThresholdNotExeceeded

comp.2 {
  state.param_1.1 = (f32[4]{0}, f32[4]{0}, f32[4]{0}) parameter(0)
  get-tuple-element.5 = f32[4]{0} get-tuple-element(state.param_1.1), index=0
  get-tuple-element.6 = f32[4]{0} get-tuple-element(state.param_1.1), index=1
  add.7 = f32[4]{0} add(get-tuple-element.5, get-tuple-element.6)
  get-tuple-element.7 = f32[4]{0} get-tuple-element(state.param_1.1), index=2
  ROOT add.8 = f32[4]{0} add(add.7, get-tuple-element.7)
}

comp.1 {
  add.1.param_1.1 = f32[4]{0} parameter(1)
  constant.param_1.3 = f32[4]{0} parameter(0)
  add.5 = f32[4]{0} add(add.1.param_1.1, constant.param_1.3)
  ROOT multiply.3 = f32[4]{0} multiply(add.5, constant.param_1.3)
}

comp {
  add.1.param_1 = f32[4]{0} parameter(1)
  constant.param_1.1 = f32[4]{0} parameter(0)
  multiply.2 = f32[4]{0} multiply(add.1.param_1, constant.param_1.1)
  ROOT add.4 = f32[4]{0} add(multiply.2, constant.param_1.1)
}

ENTRY BytesTransferredThresholdNotExeceeded.Computation2 {
  constant = f32[4]{0} constant({1, 1, 1, 1})
  state = (f32[4]{0}, f32[4]{0}, f32[4]{0}) parameter(0)
  fusion.2 = f32[4]{0} fusion(state), kind=kLoop, calls=comp.2
  fusion.3 = f32[4]{0} fusion(constant, fusion.2), kind=kLoop, calls=comp.1
  fusion.4 = f32[4]{0} fusion(constant, fusion.2), kind=kLoop, calls=comp
  ROOT tuple = (f32[4]{0}, f32[4]{0}) tuple(fusion.3, fusion.4)
})")
                    .ValueOrDie();
  // Run fusion merger pass, which should detect that the net bytes transferred
  // (if merged) would not increase.
  EXPECT_TRUE(FusionMerger().Run(module.get()).ValueOrDie());
}

// Check that we're willing to merge f1_computation into f2_computation, even
// though f2 is an input fusion node.
TEST_F(FusionMergerTest, WillMergeIntoInputFusion) {
  auto module = ParseHloString(R"(
    HloModule m

    f1_computation {
      f1_p0 = f32[10]{0} parameter(0)
      ROOT f1_root = f32[10]{0} add(f1_p0, f1_p0)
    }

    add_computation {
      add_lhs = f32[] parameter(0)
      add_rhs = f32[] parameter(1)
      ROOT add_root = f32[] add(add_lhs, add_rhs)
    }

    f2_computation {
      f2_p0 = f32[10]{0} parameter(0)
      f2_mul = f32[10]{0} multiply(f2_p0, f2_p0)
      f2_zero = f32[] constant(0)
      ROOT f2_root = f32[] reduce(f2_mul, f2_zero), dimensions={0},
             to_apply=add_computation
    }

    ENTRY entry {
      p0 = f32[10]{0} parameter(0)
      f1 = f32[10]{0} fusion(p0), kind=kLoop, calls=f1_computation
      ROOT f2 = f32[] fusion(f1), kind=kInput, calls=f2_computation
    })")
                    .ValueOrDie();
  EXPECT_TRUE(FusionMerger().Run(module.get()).ValueOrDie());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Fusion(op::Parameter()));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
