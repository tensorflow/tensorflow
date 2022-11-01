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

class FusionMergerTest : public HloTestBase {
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

 public:
  FusionMerger fusion_merger_{ShapeSizeBytesFunction()};
  FusionMergerTest() : HloTestBase() {}
};

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
                    .value();
  EXPECT_TRUE(fusion_merger_.Run(module.get()).value());

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
                    .value();
  // Run fusion merger pass, which should detect that the net bytes transferred
  // (if merged) would increase.
  EXPECT_FALSE(fusion_merger_.Run(module.get()).value());
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
                    .value();
  // Run fusion merger pass, which should detect that the net bytes transferred
  // (if merged) would not increase.
  EXPECT_TRUE(fusion_merger_.Run(module.get()).value());
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
                    .value();
  EXPECT_TRUE(fusion_merger_.Run(module.get()).value());
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
                    .value();
  EXPECT_TRUE(fusion_merger_.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Fusion(op::Fusion(), op::Parameter(), op::Parameter()));
}

TEST_F(FusionMergerTest, WillNotMergeReduceUnfriendlyLayouts) {
  // TODO(b/247762001): the case here does not represent the problem -
  // profiling shows that it works faster if merged (even on larger dimensions).
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
                    .value();
  EXPECT_FALSE(fusion_merger_.Run(module.get()).value());
}

TEST_F(FusionMergerTest, WillMergeReduceNotTooUnfriendlyLayouts) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m

    f1_computation {
      f1_p0 = f32[16,16,256]{0,1,2} parameter(0)
      slice1 = f32[5,16,256]{0,1,2} slice(f1_p0), slice={[0:5], [0:16], [0:256]}
      // Here the copy changes the layout only of a part of the data.
      f1_copy = f32[5,16,256]{2,1,0} copy(slice1)
      slice2 = f32[11,16,256]{0,1,2} slice(f1_p0), slice={[0:11], [0:16], [0:256]}
      bitcast = f32[11,16,256]{2,1,0} bitcast(slice2)
      ROOT f1_root = f32[16,16,256]{2,1,0} concatenate(f1_copy, bitcast), dimensions={0}
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
                    .value();
  EXPECT_TRUE(fusion_merger_.Run(module.get()).value());
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
  EXPECT_FALSE(fusion_merger_.Run(module.get()).value());
}

// TODO(b/119692968): Remove this test once fusion emitter is fixed.
TEST_F(FusionMergerTest, WillNotMergeIfFusionEmitterIsInefficient) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

f1 {
  Arg_0.5 = f32[200000] parameter(0)
  slice.7 = f32[100000] slice(Arg_0.5), slice={[0:199999:2]}
  slice.8 = f32[100000] slice(Arg_0.5), slice={[1:200000:2]}
  add.9 = f32[100000] add(slice.7, slice.8)
  slice.10 = f32[50000] slice(add.9), slice={[0:99999:2]}
  slice.11 = f32[50000] slice(add.9), slice={[1:100000:2]}
  add.12 = f32[50000] add(slice.10, slice.11)
  slice.13 = f32[25000] slice(add.12), slice={[0:49999:2]}
  slice.14 = f32[25000] slice(add.12), slice={[1:50000:2]}
  add.15 = f32[25000] add(slice.13, slice.14)
  slice.16 = f32[12500] slice(add.15), slice={[0:24999:2]}
  slice.17 = f32[12500] slice(add.15), slice={[1:25000:2]}
  add.18 = f32[12500] add(slice.16, slice.17)
  slice.19 = f32[6250] slice(add.18), slice={[0:12499:2]}
  slice.20 = f32[6250] slice(add.18), slice={[1:12500:2]}
  add.21 = f32[6250] add(slice.19, slice.20)
  slice.22 = f32[3125] slice(add.21), slice={[0:6249:2]}
  slice.23 = f32[3125] slice(add.21), slice={[1:6250:2]}
  ROOT add.24 = f32[3125] add(slice.22, slice.23)
}

f2 {
  Arg_0 = f32[3125] parameter(0)
  slice.25 = f32[1562] slice(Arg_0), slice={[0:3124:2]}
  slice.26 = f32[1562] slice(Arg_0), slice={[1:3125:2]}
  add.27 = f32[1562] add(slice.25, slice.26)
  slice.28 = f32[781] slice(add.27), slice={[0:1561:2]}
  slice.29 = f32[781] slice(add.27), slice={[1:1562:2]}
  add.30 = f32[781] add(slice.28, slice.29)
  slice.31 = f32[390] slice(add.30), slice={[0:780:2]}
  slice.32 = f32[390] slice(add.30), slice={[1:781:2]}
  add.33 = f32[390] add(slice.31, slice.32)
  slice.34 = f32[195] slice(add.33), slice={[0:389:2]}
  slice.35 = f32[195] slice(add.33), slice={[1:390:2]}
  add.36 = f32[195] add(slice.34, slice.35)
  slice.37 = f32[97] slice(add.36), slice={[0:194:2]}
  slice.38 = f32[97] slice(add.36), slice={[1:195:2]}
  add.39 = f32[97] add(slice.37, slice.38)
  slice.40 = f32[48] slice(add.39), slice={[0:96:2]}
  slice.41 = f32[48] slice(add.39), slice={[1:97:2]}
  ROOT add.42 = f32[48] add(slice.40, slice.41)
}

ENTRY e {
  p0 = f32[200000] parameter(0)
  f1 = f32[3125] fusion(p0), kind=kLoop, calls=f1
  ROOT r = f32[48] fusion(f1), kind=kLoop, calls=f2
})")
                    .value();
  EXPECT_FALSE(fusion_merger_.Run(module.get()).value());
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
                    .value();
  EXPECT_TRUE(fusion_merger_.Run(module.get()).value());
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
                    .value();
  EXPECT_TRUE(fusion_merger_.Run(module.get()).value());
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
                    .value();
  EXPECT_FALSE(fusion_merger_.Run(module.get()).value());
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
                    .value();
  EXPECT_FALSE(fusion_merger_.Run(module.get()).value());
}

TEST_F(FusionMergerTest, NoMergeBecauseTooManyBasicBlockSplits) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

region_6.97 {
  Arg_0.98 = pred[] parameter(0)
  Arg_1.99 = pred[] parameter(1)
  ROOT or.100 = pred[] or(Arg_0.98, Arg_1.99)
}

region_4.50 {
  Arg_0.51 = f64[] parameter(0)
  Arg_1.52 = f64[] parameter(1)
  ROOT add.53 = f64[] add(Arg_0.51, Arg_1.52)
}

f2 {
  param_0 = s64[1]{0} parameter(0)
  constant_70 = f64[] constant(0)
  convert.41.clone.1 = f64[1]{0} convert(param_0)
  ROOT pad.99.clone.1 = f64[3]{0} pad(convert.41.clone.1, constant_70), padding=0_2
}

f1 {
  param_0.361 = pred[5]{0} parameter(0)
  broadcast.107 = pred[10,5]{1,0} broadcast(param_0.361), dimensions={1}
  param_6.244 = pred[5]{0} parameter(6)
  broadcast.111.clone.1 = pred[10,5]{1,0} broadcast(param_6.244), dimensions={1}
  param_1.450 = f64[10,5]{1,0} parameter(1)
  constant_294_clone_1 = f64[] constant(1)
  broadcast.153.clone.1 = f64[10,5]{1,0} broadcast(constant_294_clone_1), dimensions={}
  compare.22.clone.1 = pred[10,5]{1,0} compare(param_1.450, broadcast.153.clone.1), direction=GE
  constant_75_clone_1 = f64[] constant(-1)
  broadcast.109.clone.1 = f64[10,5]{1,0} broadcast(constant_75_clone_1), dimensions={}
  add.34.clone.1 = f64[10,5]{1,0} add(param_1.450, broadcast.109.clone.1)
  param_5.322 = f64[10,5,4]{1,0,2} parameter(5)
  slice.45.clone.1 = f64[10,5,1]{1,0,2} slice(param_5.322), slice={[0:10], [0:5], [3:4]}
  bitcast.94.clone.1 = f64[10,5]{1,0} bitcast(slice.45.clone.1)
  divide.7.clone.1 = f64[10,5]{1,0} divide(add.34.clone.1, bitcast.94.clone.1)
  add.33.clone.1 = f64[10,5]{1,0} add(divide.7.clone.1, broadcast.153.clone.1)
  constant_70 = f64[] constant(0)
  broadcast.157.clone.1 = f64[10,5]{1,0} broadcast(constant_70), dimensions={}
  compare.26.clone.1 = pred[10,5]{1,0} compare(param_1.450, broadcast.157.clone.1), direction=LE
  slice.46.clone.1 = f64[10,5,1]{1,0,2} slice(param_5.322), slice={[0:10], [0:5], [0:1]}
  bitcast.93.clone.1 = f64[10,5]{1,0} bitcast(slice.46.clone.1)
  divide.6.clone.1 = f64[10,5]{1,0} divide(param_1.450, bitcast.93.clone.1)
  broadcast.295.clone.1 = f64[10,5,3]{1,0,2} broadcast(param_1.450), dimensions={0,1}
  param_4.368 = f64[10,5,2]{1,0,2} parameter(4)
  pad.103.clone.1 = f64[10,5,3]{1,0,2} pad(param_4.368, constant_70), padding=0_0x0_0x1_0
  compare.121.clone.1 = pred[10,5,3]{1,0,2} compare(broadcast.295.clone.1, pad.103.clone.1), direction=GE
  pad.102.clone.1 = f64[10,5,3]{1,0,2} pad(param_4.368, constant_294_clone_1), padding=0_0x0_0x0_1
  compare.120.clone.1 = pred[10,5,3]{1,0,2} compare(broadcast.295.clone.1, pad.102.clone.1), direction=LT
  and.39.clone.1 = pred[10,5,3]{1,0,2} and(compare.121.clone.1, compare.120.clone.1)
  transpose.9 = pred[3,10,5]{2,1,0} transpose(and.39.clone.1), dimensions={2,0,1}
  constant_296_clone_1 = pred[] constant(false)
  reduce.91.clone.1 = pred[10,5]{1,0} reduce(transpose.9, constant_296_clone_1), dimensions={0}, to_apply=region_6.97
  broadcast.294.clone.1 = pred[10,5,3]{1,0,2} broadcast(reduce.91.clone.1), dimensions={0,1}
  pad.99.clone.1 = f64[3]{0} parameter(3)
  broadcast.292.clone.1 = f64[3]{0} broadcast(constant_70), dimensions={}
  compare.117.clone.1 = pred[3]{0} compare(pad.99.clone.1, broadcast.292.clone.1), direction=NE
  broadcast.290.clone.1 = pred[10,5,3]{1,0,2} broadcast(compare.117.clone.1), dimensions={2}
  select.67.clone.1 = pred[10,5,3]{1,0,2} select(broadcast.294.clone.1, and.39.clone.1, broadcast.290.clone.1)
  convert.40.clone.1 = f64[10,5,3]{1,0,2} convert(select.67.clone.1)
  broadcast.288.clone.1 = f64[10,5,3,3]{1,0,2,3} broadcast(convert.40.clone.1), dimensions={0,1,2}
  param_2.361 = f64[10,5,4,3]{1,0,2,3} parameter(2)
  slice.114.clone.1 = f64[10,5,3,3]{1,0,2,3} slice(param_2.361), slice={[0:10], [0:5], [1:4], [0:3]}
  multiply.53.clone.1 = f64[10,5,3,3]{1,0,2,3} multiply(broadcast.288.clone.1, slice.114.clone.1)
  transpose.10 = f64[3,3,10,5]{3,2,1,0} transpose(multiply.53.clone.1), dimensions={3,2,0,1}
  reduce.90.clone.1 = f64[3,10,5]{2,1,0} reduce(transpose.10, constant_70), dimensions={1}, to_apply=region_4.50
  transpose.11 = f64[10,5,3]{1,0,2} transpose(reduce.90.clone.1), dimensions={1,2,0}
  slice.28.clone.1 = f64[10,5,1]{1,0,2} slice(transpose.11), slice={[0:10], [0:5], [0:1]}
  bitcast.99.clone.1 = f64[10,5]{1,0} bitcast(slice.28.clone.1)
  slice.108.clone.1 = f64[10,5,3,3]{1,0,2,3} slice(param_2.361), slice={[0:10], [0:5], [0:3], [0:3]}
  multiply.49.clone.1 = f64[10,5,3,3]{1,0,2,3} multiply(broadcast.288.clone.1, slice.108.clone.1)
  transpose.12 = f64[3,3,10,5]{3,2,1,0} transpose(multiply.49.clone.1), dimensions={3,2,0,1}
  reduce.82.clone.1 = f64[3,10,5]{2,1,0} reduce(transpose.12, constant_70), dimensions={1}, to_apply=region_4.50
  transpose.13 = f64[10,5,3]{1,0,2} transpose(reduce.82.clone.1), dimensions={1,2,0}
  slice.107.clone.1 = f64[10,5,1]{1,0,2} slice(transpose.13), slice={[0:10], [0:5], [0:1]}
  bitcast.240.clone.1 = f64[10,5]{1,0} bitcast(slice.107.clone.1)
  subtract.27.clone.1 = f64[10,5]{1,0} subtract(bitcast.99.clone.1, bitcast.240.clone.1)
  slice.27.clone.1 = f64[10,5,1]{1,0,2} slice(transpose.13), slice={[0:10], [0:5], [2:3]}
  bitcast.98.clone.1 = f64[10,5]{1,0} bitcast(slice.27.clone.1)
  slice.26.clone.1 = f64[10,5,1]{1,0,2} slice(transpose.11), slice={[0:10], [0:5], [2:3]}
  bitcast.97.clone.1 = f64[10,5]{1,0} bitcast(slice.26.clone.1)
  add.36.clone.1 = f64[10,5]{1,0} add(bitcast.97.clone.1, bitcast.98.clone.1)
  slice.24.clone.1 = f64[10,5,1]{1,0,2} slice(transpose.11), slice={[0:10], [0:5], [1:2]}
  bitcast.95.clone.1 = f64[10,5]{1,0} bitcast(slice.24.clone.1)
  slice.121.clone.1 = f64[10,5,1]{1,0,2} slice(transpose.13), slice={[0:10], [0:5], [1:2]}
  bitcast.274.clone.1 = f64[10,5]{1,0} bitcast(slice.121.clone.1)
  subtract.26.clone.1 = f64[10,5]{1,0} subtract(bitcast.95.clone.1, bitcast.274.clone.1)
  divide.21 = f64[10,5]{1,0} divide(subtract.26.clone.1, subtract.27.clone.1)
  constant_77_clone_1 = f64[] constant(2)
  broadcast.117.clone.1 = f64[10,5]{1,0} broadcast(constant_77_clone_1), dimensions={}
  multiply.37.clone.1 = f64[10,5]{1,0} multiply(divide.21, broadcast.117.clone.1)
  subtract.25.clone.1 = f64[10,5]{1,0} subtract(add.36.clone.1, multiply.37.clone.1)
  subtract.24.clone.1 = f64[10,5]{1,0} subtract(param_1.450, bitcast.274.clone.1)
  divide.9.clone.1 = f64[10,5]{1,0} divide(subtract.24.clone.1, subtract.26.clone.1)
  clamp.7.clone.1 = f64[10,5]{1,0} clamp(broadcast.157.clone.1, divide.9.clone.1, broadcast.153.clone.1)
  multiply.36.clone.1 = f64[10,5]{1,0} multiply(subtract.25.clone.1, clamp.7.clone.1)
  subtract.23.clone.1 = f64[10,5]{1,0} subtract(bitcast.98.clone.1, multiply.36.clone.1)
  compare.13.clone.1 = pred[10,5]{1,0} compare(subtract.23.clone.1, broadcast.157.clone.1), direction=GE
  negate.19.clone.1 = f64[10,5]{1,0} negate(divide.21)
  multiply.35.clone.1 = f64[10,5]{1,0} multiply(negate.19.clone.1, clamp.7.clone.1)
  multiply.34.clone.1 = f64[10,5]{1,0} multiply(multiply.35.clone.1, broadcast.117.clone.1)
  negate.18.clone.1 = f64[10,5]{1,0} negate(subtract.23.clone.1)
  multiply.33.clone.1 = f64[10,5]{1,0} multiply(subtract.23.clone.1, subtract.23.clone.1)
  subtract.22.clone.1 = f64[10,5]{1,0} subtract(divide.21, subtract.23.clone.1)
  constant_78_clone_1 = f64[] constant(4)
  broadcast.113.clone.1 = f64[10,5]{1,0} broadcast(constant_78_clone_1), dimensions={}
  multiply.32.clone.1 = f64[10,5]{1,0} multiply(subtract.22.clone.1, broadcast.113.clone.1)
  multiply.31.clone.1 = f64[10,5]{1,0} multiply(multiply.32.clone.1, multiply.35.clone.1)
  subtract.21.clone.1 = f64[10,5]{1,0} subtract(multiply.33.clone.1, multiply.31.clone.1)
  compare.12.clone.1 = pred[10,5]{1,0} compare(subtract.21.clone.1, broadcast.157.clone.1), direction=GT
  constant_79_clone_1 = f64[] constant(2.2250738585072014e-308)
  broadcast.112.clone.1 = f64[10,5]{1,0} broadcast(constant_79_clone_1), dimensions={}
  maximum.18.clone.1 = f64[10,5]{1,0} maximum(broadcast.112.clone.1, subtract.21.clone.1)
  sqrt.1.clone.1 = f64[10,5]{1,0} sqrt(maximum.18.clone.1)
  select.47.clone.1 = f64[10,5]{1,0} select(compare.12.clone.1, sqrt.1.clone.1, broadcast.157.clone.1)
  add.35.clone.1 = f64[10,5]{1,0} add(negate.18.clone.1, select.47.clone.1)
  select.46.clone.1 = f64[10,5]{1,0} select(compare.13.clone.1, multiply.34.clone.1, add.35.clone.1)
  subtract.20.clone.1 = f64[10,5]{1,0} subtract(negate.18.clone.1, select.47.clone.1)
  multiply.30.clone.1 = f64[10,5]{1,0} multiply(subtract.22.clone.1, broadcast.117.clone.1)
  select.45.clone.1 = f64[10,5]{1,0} select(compare.13.clone.1, subtract.20.clone.1, multiply.30.clone.1)
  divide.8.clone.1 = f64[10,5]{1,0} divide(select.46.clone.1, select.45.clone.1)
  clamp.6.clone.1 = f64[10,5]{1,0} clamp(broadcast.157.clone.1, divide.8.clone.1, broadcast.153.clone.1)
  multiply.29.clone.1 = f64[10,5]{1,0} multiply(subtract.27.clone.1, clamp.6.clone.1)
  add.32.clone.1 = f64[10,5]{1,0} add(multiply.29.clone.1, bitcast.240.clone.1)
  select.44.clone.1 = f64[10,5]{1,0} select(compare.26.clone.1, divide.6.clone.1, add.32.clone.1)
  select.43.clone.1 = f64[10,5]{1,0} select(compare.22.clone.1, add.33.clone.1, select.44.clone.1)
  select.42.clone.1 = f64[10,5]{1,0} select(broadcast.111.clone.1, param_1.450, select.43.clone.1)
  select.41 = f64[10,5]{1,0} select(broadcast.107, select.42.clone.1, broadcast.157.clone.1)
  ROOT tuple.14 = (f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}) tuple(select.41, select.42.clone.1, clamp.6.clone.1, subtract.25.clone.1, bitcast.97.clone.1, multiply.37.clone.1, bitcast.98.clone.1, divide.21)
}

ENTRY e {
  p3 = s64[1]{0} parameter(3)
  f2 = f64[3]{0} fusion(p3), kind=kLoop, calls=f2

  p0 = pred[5]{0} parameter(0)
  p1 = f64[10,5]{1,0} parameter(1)
  p2 = f64[10,5,4,3]{1,0,2,3} parameter(2)
  p4 = f64[10,5,2]{1,0,2} parameter(4)
  p5 = f64[10,5,4]{1,0,2} parameter(5)
  p6 = pred[5]{0} parameter(6)
  ROOT ret = (f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}, f64[10,5]{1,0}) fusion(p0, p1, p2, f2, p4, p5, p6), kind=kLoop, calls=f1
}
  )")
                    .value();
  EXPECT_FALSE(fusion_merger_.Run(module.get()).value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
