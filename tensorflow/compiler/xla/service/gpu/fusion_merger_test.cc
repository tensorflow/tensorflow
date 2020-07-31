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

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
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
  auto module = ParseAndReturnVerifiedModule(R"(
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

// Tests that threshold for bytes transferred if merged is exceeded.
//
// Fusion2 is not merged because it exceeds the threshold bytes transferred.
// This is because the bytes read by Fusion2 (when replicated if the instruction
// is merged into Fusion0 and Fusion1) would exceed the bytes transferred
// threshold.
TEST_F(FusionMergerTest, BytesTransferredThresholdExceeded) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule BytesTransferredThresholdExceeded

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

ENTRY BytesTransferredThresholdExceeded.Computation2 {
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
TEST_F(FusionMergerTest, BytesTransferredThresholdNotExceeded) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule BytesTransferredThresholdNotExceeded

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

ENTRY BytesTransferredThresholdNotExceeded.Computation2 {
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
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m

    f1_computation {
      f1_p0 = f32[32]{0} parameter(0)
      ROOT f1_root = f32[32]{0} add(f1_p0, f1_p0)
    }

    add_computation {
      add_lhs = f32[] parameter(0)
      add_rhs = f32[] parameter(1)
      ROOT add_root = f32[] add(add_lhs, add_rhs)
    }

    f2_computation {
      f2_p0 = f32[32]{0} parameter(0)
      f2_mul = f32[32]{0} multiply(f2_p0, f2_p0)
      f2_zero = f32[] constant(0)
      ROOT f2_root = f32[] reduce(f2_mul, f2_zero), dimensions={0},
             to_apply=add_computation
    }

    ENTRY entry {
      p0 = f32[32]{0} parameter(0)
      f1 = f32[32]{0} fusion(p0), kind=kLoop, calls=f1_computation
      ROOT f2 = f32[] fusion(f1), kind=kInput, calls=f2_computation
    })")
                    .ValueOrDie();
  EXPECT_TRUE(FusionMerger().Run(module.get()).ValueOrDie());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Fusion(op::Parameter()));
}

TEST_F(FusionMergerTest, WillNotMergeReduceUnfriendlyLayouts) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m

    f1_computation {
      f1_p0 = f32[16,16,256]{0,1,2} parameter(0)
      add = f32[16,16,256]{0,1,2} add(f1_p0, f1_p0)
      // Note that the copy changes the layout from {0,1,2} to {2,1,0}.
      ROOT f1_root = f32[16,16,256]{2,1,0} copy(add)
    }

    add_computation {
      add_lhs = f32[] parameter(0)
      add_rhs = f32[] parameter(1)
      ROOT add_root = f32[] add(add_lhs, add_rhs)
    }

    f2_computation {
      f2_p0 = f32[16,16,256]{2,1,0} parameter(0)
      f2_zero = f32[] constant(0)
      ROOT f2_root = f32[] reduce(f2_p0, f2_zero), dimensions={0,1,2},
             to_apply=add_computation
    }

    ENTRY entry {
      p0 = f32[16,16,256]{0,1,2} parameter(0)
      f1 = f32[16,16,256]{2,1,0} fusion(p0), kind=kLoop, calls=f1_computation
      ROOT f2 = f32[] fusion(f1), kind=kInput, calls=f2_computation
    })")
                    .ValueOrDie();
  EXPECT_FALSE(FusionMerger().Run(module.get()).ValueOrDie());
}

// Check that we limit the number of operands to fusions we create.
TEST_F(FusionMergerTest, AvoidsLargeFusion) {
  constexpr int64 kNumParams = kMaxOperandsAndOutputsPerFusion + 1;

  // Compute
  //   p0 + p1 + p2 + ... + pn,
  // Use so many parameters that they do not fit into one fusion.
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {10, 100});

  std::vector<HloInstruction*> entry_params;

  for (int64 i = 0; i < kNumParams; ++i) {
    entry_params.push_back(
        b.AddInstruction(HloInstruction::CreateParameter(i, shape, "p")));
  }
  auto make_fusion = [&](absl::Span<HloInstruction* const> params) {
    // Build a fusion computation for calculating the sum of all parameters.
    HloComputation::Builder sub_builder("subcomp");
    HloInstruction* sum = nullptr;
    for (int64 i = 0; i < params.size(); ++i) {
      auto p = sub_builder.AddInstruction(
          HloInstruction::CreateParameter(i, shape, "p"));
      if (sum == nullptr) {
        sum = p;
      } else {
        sum = sub_builder.AddInstruction(
            HloInstruction::CreateBinary(shape, HloOpcode::kAdd, sum, p));
      }
    }
    HloComputation* subcomp =
        module->AddEmbeddedComputation(sub_builder.Build());
    return HloInstruction::CreateFusion(
        shape, HloInstruction::FusionKind::kLoop, params, subcomp);
  };
  auto fusion = b.AddInstruction(
      make_fusion(absl::MakeSpan(entry_params)
                      .subspan(0, kMaxOperandsAndOutputsPerFusion)));
  b.AddInstruction(make_fusion({entry_params.back(), fusion}));
  module->AddEntryComputation(b.Build());
  EXPECT_FALSE(FusionMerger().Run(module.get()).ValueOrDie());
}

// TODO(b/119692968): Remove this test once fusion emitter is fixed.
TEST_F(FusionMergerTest, WillNotMergeIfFusionEmitterIsInefficient) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m

    %fused_computation (param_0.10: f32[6]) -> f32[1] {
      %param_0.10 = f32[6]{0} parameter(0)
      %add.7 = f32[6]{0} add(%param_0.10, %param_0.10)
      %slice.21 = f32[5]{0} slice(%add.7), slice={[0:5]}
      %slice.18 = f32[5]{0} slice(%add.7), slice={[1:6]}
      %add.5 = f32[5]{0} add(%slice.21, %slice.18)
      %slice.15 = f32[4]{0} slice(%add.5), slice={[0:4]}
      %slice.12 = f32[4]{0} slice(%add.5), slice={[1:5]}
      %add.4 = f32[4]{0} add(%slice.15, %slice.12)
      %slice.9 = f32[3]{0} slice(%add.4), slice={[0:3]}
      %slice.6 = f32[3]{0} slice(%add.4), slice={[1:4]}
      %add.2 = f32[3]{0} add(%slice.9, %slice.6)
      %slice.3 = f32[2]{0} slice(%add.2), slice={[0:2]}
      %slice.2 = f32[2]{0} slice(%add.2), slice={[1:3]}
      %add.1 = f32[2]{0} add(%slice.3, %slice.2)
      %slice.1 = f32[1]{0} slice(%add.1), slice={[0:1]}
      %slice.0 = f32[1]{0} slice(%add.1), slice={[1:2]}
      ROOT %add.0 = f32[1]{0} add(%slice.1, %slice.0)
    }

    %fused_computation.1 (param_0.21: f32[11], param_1.21: f32[11]) -> f32[6] {
      %param_0.21 = f32[11]{0} parameter(0)
      %param_1.21 = f32[11]{0} parameter(1)
      %add.16 = f32[11]{0} add(%param_0.21, %param_1.21)
      %slice.51 = f32[10]{0} slice(%add.16), slice={[0:10]}
      %slice.48 = f32[10]{0} slice(%add.16), slice={[1:11]}
      %add.14 = f32[10]{0} add(%slice.51, %slice.48)
      %slice.45 = f32[9]{0} slice(%add.14), slice={[0:9]}
      %slice.42 = f32[9]{0} slice(%add.14), slice={[1:10]}
      %add.13 = f32[9]{0} add(%slice.45, %slice.42)
      %slice.39 = f32[8]{0} slice(%add.13), slice={[0:8]}
      %slice.36 = f32[8]{0} slice(%add.13), slice={[1:9]}
      %add.11 = f32[8]{0} add(%slice.39, %slice.36)
      %slice.33 = f32[7]{0} slice(%add.11), slice={[0:7]}
      %slice.30 = f32[7]{0} slice(%add.11), slice={[1:8]}
      %add.10 = f32[7]{0} add(%slice.33, %slice.30)
      %slice.27 = f32[6]{0} slice(%add.10), slice={[0:6]}
      %slice.24 = f32[6]{0} slice(%add.10), slice={[1:7]}
      ROOT %add.8 = f32[6]{0} add(%slice.27, %slice.24)
    }

    ENTRY entry {
      p0 = f32[11]{0} parameter(0)
      p1 = f32[11]{0} parameter(1)
      f1 = f32[6]{0} fusion(p0, p1), kind=kLoop, calls=%fused_computation.1
      ROOT f2 = f32[1] fusion(f1), kind=kLoop, calls=%fused_computation
    })")
                    .ValueOrDie();
  EXPECT_FALSE(FusionMerger().Run(module.get()).ValueOrDie());
}

TEST_F(FusionMergerTest, WillMergeExpensiveFusionsIfSavesMemory) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m

    %f_a (p: f32[]) -> f32[1024,1024,1024] {
      %p = f32[] parameter(0)
      %b = f32[1024,1024,1024] broadcast(%p), dimensions={}
      ROOT %t = f32[1024,1024,1024] tanh(%b)
    }

    %f_b (p: f32[1024,1024,1024]) -> f32[1024,1024,1024] {
      %p = f32[1024,1024,1024] parameter(0)
      ROOT %t = f32[1024,1024,1024] tanh(%p)
    }

    %f_c (p: f32[1024,1024,1024]) -> f32[1024,1024,1024] {
      %p = f32[1024,1024,1024] parameter(0)
      ROOT %t = f32[1024,1024,1024] tanh(%p)
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      f1 = f32[1024,1024,1024] fusion(p0), kind=kLoop, calls=%f_a
      f2 = f32[1024,1024,1024] fusion(f1), kind=kLoop, calls=%f_b
      f3 = f32[1024,1024,1024] fusion(f1), kind=kLoop, calls=%f_c
      ROOT f4 = f32[1024,1024,1024] add(f2, f3)
    })")
                    .ValueOrDie();
  EXPECT_TRUE(FusionMerger().Run(module.get()).ValueOrDie());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
