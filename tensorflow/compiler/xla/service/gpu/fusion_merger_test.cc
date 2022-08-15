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

TEST_F(FusionMergerTest, WillMergeIntoUnfusedConsumer) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule jit_matmul.36

    max (parameter.13: f32[], parameter.14: f32[]) -> f32[] {
      parameter.13 = f32[] parameter(0)
      parameter.14 = f32[] parameter(1)
      ROOT maximum.15 = f32[] maximum(f32[] parameter.13, f32[] parameter.14)
    }

    add (parameter.29: f32[], parameter.30: f32[]) -> f32[] {
      parameter.29 = f32[] parameter(0)
      parameter.30 = f32[] parameter(1)
      ROOT add.31 = f32[] add(f32[] parameter.29, f32[] parameter.30)
    }

    fused_computation.1 (param_1.4: f32[200,200,200], param_2.1: f32[200,200]) -> f32[200,200] {
      param_1.4 = f32[200,200,200]{2,1,0} parameter(0)
      param_2.1 = f32[200,200]{1,0} parameter(1)
      broadcast.3 = f32[200,200,200]{2,1,0} broadcast(f32[200,200]{1,0} param_2.1), dimensions={0,2}
      subtract.0 = f32[200,200,200]{2,1,0} subtract(f32[200,200,200]{2,1,0} param_1.4, f32[200,200,200]{2,1,0} broadcast.3)
      exponential.0 = f32[200,200,200]{2,1,0} exponential(f32[200,200,200]{2,1,0} subtract.0)
      constant.27 = f32[] constant(0)
      ROOT reduce.0 = f32[200,200]{1,0} reduce(f32[200,200,200]{2,1,0} exponential.0, f32[] constant.27), dimensions={1}, to_apply=add
    }

    fused_computation.3 (param_0.7: f32[200,200], param_1.9: f32[200,200]) -> f32[200,200,200] {
      param_1.9 = f32[200,200]{1,0} parameter(1)
      broadcast.10 = f32[200,200,200]{2,1,0} broadcast(f32[200,200]{1,0} param_1.9), dimensions={0,1}
      param_0.7 = f32[200,200]{1,0} parameter(0)
      broadcast.8 = f32[200,200,200]{2,1,0} broadcast(f32[200,200]{1,0} param_0.7), dimensions={1,2}
      ROOT add.1 = f32[200,200,200]{2,1,0} add(f32[200,200,200]{2,1,0} broadcast.10, f32[200,200,200]{2,1,0} broadcast.8)
    }

    ENTRY entry (parameter.1: f32[200,200], parameter.2: f32[200,200]) -> f32[200,200] {
      parameter.2 = f32[200,200]{1,0} parameter(1)
      parameter.1 = f32[200,200]{1,0} parameter(0)
      fusion.3 = f32[200,200,200]{2,1,0} fusion(f32[200,200]{1,0} parameter.2, f32[200,200]{1,0} parameter.1), kind=kLoop, calls=fused_computation.3
      constant.11 = f32[] constant(-inf)
      reduce.16 = f32[200,200]{1,0} reduce(f32[200,200,200]{2,1,0} fusion.3, f32[] constant.11), dimensions={1}, to_apply=max
      ROOT fusion.1 = f32[200,200]{1,0} fusion(f32[200,200,200]{2,1,0} fusion.3, f32[200,200]{1,0} reduce.16), kind=kInput, calls=fused_computation.1
    })")
                    .ValueOrDie();
  EXPECT_TRUE(FusionMerger().Run(module.get()).ValueOrDie());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Fusion(op::Fusion(), op::Parameter(), op::Parameter()));
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
  constexpr int64_t kNumParams = MaxOperandsAndOutputsPerFusion() + 1;

  // Compute
  //   p0 + p1 + p2 + ... + pn,
  // Use so many parameters that they do not fit into one fusion.
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {10, 100});

  std::vector<HloInstruction*> entry_params;

  for (int64_t i = 0; i < kNumParams; ++i) {
    entry_params.push_back(
        b.AddInstruction(HloInstruction::CreateParameter(i, shape, "p")));
  }
  auto make_fusion = [&](absl::Span<HloInstruction* const> params) {
    // Build a fusion computation for calculating the sum of all parameters.
    HloComputation::Builder sub_builder("subcomp");
    HloInstruction* sum = nullptr;
    for (int64_t i = 0; i < params.size(); ++i) {
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
                      .subspan(0, MaxOperandsAndOutputsPerFusion())));
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

TEST_F(FusionMergerTest, WillMergeExpensiveFusionsWithSingleConsumer) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m

    %f_b (p: f32[1024,1024,1024]) -> f32[1024,1024,1024] {
      %p = f32[1024,1024,1024] parameter(0)
      ROOT %t = f32[1024,1024,1024] tanh(%p)
    }

    %f_c (p: f32[1024,1024,1024]) -> f32[1024,1024,1024] {
      %p = f32[1024,1024,1024] parameter(0)
      ROOT %t = f32[1024,1024,1024] add(%p, %p)
    }

    ENTRY entry {
      p0 = f32[1024,1024,1024] parameter(0)
      f1 = f32[1024,1024,1024] fusion(p0), kind=kLoop, calls=%f_b
      ROOT f2 = f32[1024,1024,1024] fusion(f1), kind=kLoop, calls=%f_c
    })")
                    .ValueOrDie();
  EXPECT_TRUE(FusionMerger().Run(module.get()).ValueOrDie());
}

TEST_F(FusionMergerTest, WillNotMergeExpensiveFusionsWithReusingConsumer) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m

    %f_b (p: f32[1024,1024,1024]) -> f32[1024,1024,1024] {
      %p = f32[1024,1024,1024] parameter(0)
      ROOT %t = f32[1024,1024,1024] tanh(%p)
    }

    %f_c (p: f32[1024,1024,1024]) -> f32[1024,1024,1024,2] {
      %p = f32[1024,1024,1024] parameter(0)
      ROOT %t = f32[1024,1024,1024,2] broadcast(%p), dimensions={0,1,2}
    }

    ENTRY entry {
      p0 = f32[1024,1024,1024] parameter(0)
      f1 = f32[1024,1024,1024] fusion(p0), kind=kLoop, calls=%f_b
      ROOT f2 = f32[1024,1024,1024,2] fusion(f1), kind=kLoop, calls=%f_c
    })")
                    .ValueOrDie();
  EXPECT_FALSE(FusionMerger().Run(module.get()).ValueOrDie());
}

TEST_F(FusionMergerTest, NoMergeBecauseOfTwoUsers) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule module

fused_computation.1 {
  param_1.917 = f32[64,4]{1,0} parameter(1)
  broadcast.1982 = f32[1,6400,64,4,1]{3,2,1,4,0} broadcast(param_1.917), dimensions={2,3}
  param_0.776 = f32[64,4]{1,0} parameter(0)
  broadcast.1981 = f32[1,6400,64,4,1]{3,2,1,4,0} broadcast(param_0.776), dimensions={2,3}
  ROOT concatenate.91 = f32[1,6400,64,4,2]{3,2,1,4,0} concatenate(broadcast.1982, broadcast.1981), dimensions={4}
}

fused_computation.2 {
  param_0 = f32[1,6400,64,4,2]{3,2,1,4,0} parameter(0)
  sqrt = f32[1,6400,64,4,2]{3,2,1,4,0} sqrt(param_0)
  log = f32[1,6400,64,4,2]{3,2,1,4,0} log(param_0)
  ROOT add = f32[1,6400,64,4,2]{3,2,1,4,0} add(sqrt, log)
}

ENTRY main {
  param_0.776 = f32[64,4]{1,0} parameter(0)
  param_1.917 = f32[64,4]{1,0} parameter(1)
  param_2 = f32[1,6400,64,4,2]{3,2,1,4,0} parameter(2)
  fusion.1 = f32[1,6400,64,4,2]{3,2,1,4,0} fusion(param_0.776, param_1.917), kind=kLoop, calls=fused_computation.1
  ROOT fusion.2 = f32[1,6400,64,4,2]{3,2,1,4,0} fusion(fusion.1), kind=kLoop, calls=fused_computation.2
}
)")
                    .ValueOrDie();
  EXPECT_FALSE(FusionMerger().Run(module.get()).ValueOrDie());
}

TEST_F(FusionMergerTest, AllowMergeBecauseUsersAreInDifferentFusions) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule module

fused_computation.1 {
  param_1.917 = f32[64,4]{1,0} parameter(1)
  broadcast.1982 = f32[1,6400,64,4,1]{3,2,1,4,0} broadcast(param_1.917), dimensions={2,3}
  param_0.776 = f32[64,4]{1,0} parameter(0)
  broadcast.1981 = f32[1,6400,64,4,1]{3,2,1,4,0} broadcast(param_0.776), dimensions={2,3}
  ROOT concatenate.91 = f32[1,6400,64,4,2]{3,2,1,4,0} concatenate(broadcast.1982, broadcast.1981), dimensions={4}
}

fused_computation.2 {
  param_0 = f32[1,6400,64,4,2]{3,2,1,4,0} parameter(0)
  sqrt = f32[1,6400,64,4,2]{3,2,1,4,0} sqrt(param_0)
  ROOT log = f32[1,6400,64,4,2]{3,2,1,4,0} log(sqrt)
}

ENTRY main {
  param_0.776 = f32[64,4]{1,0} parameter(0)
  param_1.917 = f32[64,4]{1,0} parameter(1)
  param_2 = f32[1,6400,64,4,2]{3,2,1,4,0} parameter(2)
  fusion.1 = f32[1,6400,64,4,2]{3,2,1,4,0} fusion(param_0.776, param_1.917), kind=kLoop, calls=fused_computation.1
  fusion.2 = f32[1,6400,64,4,2]{3,2,1,4,0} fusion(fusion.1), kind=kLoop, calls=fused_computation.2
  ROOT add = f32[1,6400,64,4,2]{3,2,1,4,0} add(fusion.1, param_2)
}
)")
                    .ValueOrDie();
  EXPECT_TRUE(FusionMerger().Run(module.get()).ValueOrDie());
}

TEST_F(FusionMergerTest, NoMergeBecauseCodeDuplication) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule module

and.reduce_sub_computation {
  x = pred[] parameter(0)
  y = pred[] parameter(1)
  ROOT and = pred[] and(x, y)
}

fused_computation.1 {
  param_4.658 = f32[2,20,256]{2,0,1} parameter(4)
  slice.1385 = f32[2,1,256]{2,0,1} slice(param_4.658), slice={[0:2], [11:12], [0:256]}
  constant.6847 = s32[] constant(0)
  broadcast.4823 = s32[3]{0} broadcast(constant.6847), dimensions={}
  param_9.415 = s32[3]{0} parameter(9)
  compare.700 = pred[3]{0} compare(broadcast.4823, param_9.415), direction=LE
  constant.6846 = pred[] constant(true)
  reduce.221 = pred[] reduce(compare.700, constant.6846), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2933 = pred[2,1,256]{2,0,1} broadcast(reduce.221), dimensions={}
  param_5.528 = f32[2,512]{1,0} parameter(5)
  slice.1384 = f32[2,256]{1,0} slice(param_5.528), slice={[0:2], [0:256]}
  bitcast.341 = f32[2,1,256]{2,0,1} bitcast(slice.1384)
  constant.5418 = f32[] constant(0)
  broadcast.3227 = f32[2,1,256]{2,0,1} broadcast(constant.5418), dimensions={}
  select.173 = f32[2,1,256]{2,0,1} select(broadcast.2933, bitcast.341, broadcast.3227)
  add.573 = f32[2,1,256]{2,0,1} add(slice.1385, select.173)
  param_0.299 = s32[] parameter(0)
  constant.5157 = s32[] constant(11)
  dynamic-update-slice.189 = f32[2,20,256]{2,0,1} dynamic-update-slice(param_4.658, add.573, param_0.299, constant.5157, param_0.299)
  slice.1383 = f32[2,1,256]{2,0,1} slice(dynamic-update-slice.189), slice={[0:2], [10:11], [0:256]}
  constant.6800 = s32[] constant(0)
  broadcast.4803 = s32[3]{0} broadcast(constant.6800), dimensions={}
  param_8.484 = s32[3]{0} parameter(8)
  compare.681 = pred[3]{0} compare(broadcast.4803, param_8.484), direction=LE
  constant.6798 = pred[] constant(true)
  reduce.203 = pred[] reduce(compare.681, constant.6798), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2932 = pred[2,1,256]{2,0,1} broadcast(reduce.203), dimensions={}
  param_3.1169 = f32[2,512]{1,0} parameter(3)
  slice.1382 = f32[2,256]{1,0} slice(param_3.1169), slice={[0:2], [0:256]}
  bitcast.340 = f32[2,1,256]{2,0,1} bitcast(slice.1382)
  select.172 = f32[2,1,256]{2,0,1} select(broadcast.2932, bitcast.340, broadcast.3227)
  add.572 = f32[2,1,256]{2,0,1} add(slice.1383, select.172)
  constant.5154 = s32[] constant(10)
  dynamic-update-slice.188 = f32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.189, add.572, param_0.299, constant.5154, param_0.299)
  slice.1381 = f32[2,1,256]{2,0,1} slice(dynamic-update-slice.188), slice={[0:2], [9:10], [0:256]}
  constant.6794 = s32[] constant(0)
  broadcast.4801 = s32[3]{0} broadcast(constant.6794), dimensions={}
  param_7.478 = s32[3]{0} parameter(7)
  compare.679 = pred[3]{0} compare(broadcast.4801, param_7.478), direction=LE
  constant.6793 = pred[] constant(true)
  reduce.201 = pred[] reduce(compare.679, constant.6793), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2930 = pred[2,1,256]{2,0,1} broadcast(reduce.201), dimensions={}
  param_2.1685 = f32[2,512]{1,0} parameter(2)
  slice.1380 = f32[2,256]{1,0} slice(param_2.1685), slice={[0:2], [0:256]}
  bitcast.339 = f32[2,1,256]{2,0,1} bitcast(slice.1380)
  select.171 = f32[2,1,256]{2,0,1} select(broadcast.2930, bitcast.339, broadcast.3227)
  add.571 = f32[2,1,256]{2,0,1} add(slice.1381, select.171)
  constant.5153 = s32[] constant(9)
  dynamic-update-slice.187 = f32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.188, add.571, param_0.299, constant.5153, param_0.299)
  slice.1379 = f32[2,1,256]{2,0,1} slice(dynamic-update-slice.187), slice={[0:2], [8:9], [0:256]}
  constant.6788 = s32[] constant(0)
  broadcast.4799 = s32[3]{0} broadcast(constant.6788), dimensions={}
  param_6.495 = s32[3]{0} parameter(6)
  compare.677 = pred[3]{0} compare(broadcast.4799, param_6.495), direction=LE
  constant.6786 = pred[] constant(true)
  reduce.199 = pred[] reduce(compare.677, constant.6786), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2929 = pred[2,1,256]{2,0,1} broadcast(reduce.199), dimensions={}
  param_1.1408 = f32[2,512]{1,0} parameter(1)
  slice.1378 = f32[2,256]{1,0} slice(param_1.1408), slice={[0:2], [0:256]}
  bitcast.338 = f32[2,1,256]{2,0,1} bitcast(slice.1378)
  select.170 = f32[2,1,256]{2,0,1} select(broadcast.2929, bitcast.338, broadcast.3227)
  add.570 = f32[2,1,256]{2,0,1} add(slice.1379, select.170)
  constant.5152 = s32[] constant(8)
  ROOT dynamic-update-slice.186 = f32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.187, add.570, param_0.299, constant.5152, param_0.299)
}

fused_computation.2 {
  param_4.655 = f32[2,20,256]{2,0,1} parameter(4)
  slice.1369 = f32[2,1,256]{2,0,1} slice(param_4.655), slice={[0:2], [7:8], [0:256]}
  param_6.483 = pred[] parameter(6)
  broadcast.2927 = pred[2,1,256]{2,0,1} broadcast(param_6.483), dimensions={}
  param_5.525 = f32[2,512]{1,0} parameter(5)
  slice.1368 = f32[2,256]{1,0} slice(param_5.525), slice={[0:2], [0:256]}
  bitcast.333 = f32[2,1,256]{2,0,1} bitcast(slice.1368)
  constant.5415 = f32[] constant(0)
  broadcast.3225 = f32[2,1,256]{2,0,1} broadcast(constant.5415), dimensions={}
  select.161 = f32[2,1,256]{2,0,1} select(broadcast.2927, bitcast.333, broadcast.3225)
  add.549 = f32[2,1,256]{2,0,1} add(slice.1369, select.161)
  param_0.265 = s32[] parameter(0)
  constant.5151 = s32[] constant(7)
  dynamic-update-slice.185 = f32[2,20,256]{2,0,1} dynamic-update-slice(param_4.655, add.549, param_0.265, constant.5151, param_0.265)
  slice.1367 = f32[2,1,256]{2,0,1} slice(dynamic-update-slice.185), slice={[0:2], [6:7], [0:256]}
  constant.6782 = s32[] constant(0)
  broadcast.4797 = s32[3]{0} broadcast(constant.6782), dimensions={}
  param_9.391 = s32[3]{0} parameter(9)
  compare.675 = pred[3]{0} compare(broadcast.4797, param_9.391), direction=LE
  constant.6781 = pred[] constant(true)
  reduce.197 = pred[] reduce(compare.675, constant.6781), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2926 = pred[2,1,256]{2,0,1} broadcast(reduce.197), dimensions={}
  param_3.1167 = f32[2,512]{1,0} parameter(3)
  slice.1366 = f32[2,256]{1,0} slice(param_3.1167), slice={[0:2], [0:256]}
  bitcast.332 = f32[2,1,256]{2,0,1} bitcast(slice.1366)
  select.160 = f32[2,1,256]{2,0,1} select(broadcast.2926, bitcast.332, broadcast.3225)
  add.548 = f32[2,1,256]{2,0,1} add(slice.1367, select.160)
  constant.5150 = s32[] constant(6)
  dynamic-update-slice.184 = f32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.185, add.548, param_0.265, constant.5150, param_0.265)
  slice.1365 = f32[2,1,256]{2,0,1} slice(dynamic-update-slice.184), slice={[0:2], [5:6], [0:256]}
  constant.6776 = s32[] constant(0)
  broadcast.4794 = s32[3]{0} broadcast(constant.6776), dimensions={}
  param_8.464 = s32[3]{0} parameter(8)
  compare.673 = pred[3]{0} compare(broadcast.4794, param_8.464), direction=LE
  constant.6775 = pred[] constant(true)
  reduce.195 = pred[] reduce(compare.673, constant.6775), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2925 = pred[2,1,256]{2,0,1} broadcast(reduce.195), dimensions={}
  param_2.1684 = f32[2,512]{1,0} parameter(2)
  slice.1364 = f32[2,256]{1,0} slice(param_2.1684), slice={[0:2], [0:256]}
  bitcast.331 = f32[2,1,256]{2,0,1} bitcast(slice.1364)
  select.159 = f32[2,1,256]{2,0,1} select(broadcast.2925, bitcast.331, broadcast.3225)
  add.547 = f32[2,1,256]{2,0,1} add(slice.1365, select.159)
  constant.5149 = s32[] constant(5)
  dynamic-update-slice.183 = f32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.184, add.547, param_0.265, constant.5149, param_0.265)
  slice.1363 = f32[2,1,256]{2,0,1} slice(dynamic-update-slice.183), slice={[0:2], [4:5], [0:256]}
  constant.6770 = s32[] constant(0)
  broadcast.4792 = s32[3]{0} broadcast(constant.6770), dimensions={}
  param_7.458 = s32[3]{0} parameter(7)
  compare.671 = pred[3]{0} compare(broadcast.4792, param_7.458), direction=LE
  constant.6769 = pred[] constant(true)
  reduce.193 = pred[] reduce(compare.671, constant.6769), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2924 = pred[2,1,256]{2,0,1} broadcast(reduce.193), dimensions={}
  param_1.1405 = f32[2,512]{1,0} parameter(1)
  slice.1362 = f32[2,256]{1,0} slice(param_1.1405), slice={[0:2], [0:256]}
  bitcast.330 = f32[2,1,256]{2,0,1} bitcast(slice.1362)
  select.158 = f32[2,1,256]{2,0,1} select(broadcast.2924, bitcast.330, broadcast.3225)
  add.546 = f32[2,1,256]{2,0,1} add(slice.1363, select.158)
  constant.5148 = s32[] constant(4)
  ROOT dynamic-update-slice.182 = f32[2,20,256]{2,0,1} dynamic-update-slice(dynamic-update-slice.183, add.546, param_0.265, constant.5148, param_0.265)
}

ENTRY main {
  param_0.0 = s32[] parameter(0)
  param_1.0 = f32[2,512]{1,0} parameter(1)
  param_2.0 = f32[2,512]{1,0} parameter(2)
  param_3.0 = f32[2,512]{1,0} parameter(3)
  param_4.0 = f32[2,20,256]{2,1,0} parameter(4)
  param_5.0 = f32[2,512]{1,0} parameter(5)
  param_6.0 = s32[3]{0} parameter(6)
  param_7.0 = s32[3]{0} parameter(7)
  param_8.0 = s32[3]{0} parameter(8)
  param_9.0 = s32[3]{0} parameter(9)
  fusion.1 = f32[2,20,256]{2,0,1} fusion(param_0.0, param_1.0, param_2.0, param_3.0, param_4.0, param_5.0, param_6.0, param_7.0, param_8.0, param_9.0), kind=kLoop, calls=fused_computation.1
  param_10 = pred[] parameter(10)
  ROOT fusion.2 = f32[2,20,256]{2,0,1} fusion(param_0.0, param_1.0, param_2.0, param_3.0, fusion.1, param_5.0, param_10, param_7.0, param_8.0, param_9.0), kind=kLoop, calls=fused_computation.2
}
  )")
                    .ValueOrDie();
  EXPECT_FALSE(FusionMerger().Run(module.get()).ValueOrDie());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
