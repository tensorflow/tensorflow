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
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info_for_tests.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
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
  FusionMerger fusion_merger_{TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                              ShapeSizeBytesFunction()};
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

TEST_F(FusionMergerTest, MoreMemoryAccessIfFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

f32add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT _ = f32[] add(x, y)
}

comp0 {
  p = (f32[100000000], f32[100000000], f32[100000000], f32[100000000]) parameter(0)
  gte0 = f32[100000000] get-tuple-element(p), index=0
  gte1 = f32[100000000] get-tuple-element(p), index=1
  add.9 = f32[100000000] add(gte0, gte1)
  gte2 = f32[100000000] get-tuple-element(p), index=2
  add.10 = f32[100000000] add(add.9, gte2)
  gte3 = f32[100000000] get-tuple-element(p), index=3
  add.11 = f32[100000000] add(add.10, gte3)
  p1 = (f32[100000000], f32[100000000], f32[100000000], f32[100000000]) parameter(1)
  gte4 = f32[100000000] get-tuple-element(p1), index=0
  gte5 = f32[100000000] get-tuple-element(p1), index=1
  add.12 = f32[100000000] add(gte4, gte5)
  gte6 = f32[100000000] get-tuple-element(p1), index=2
  add.13 = f32[100000000] add(add.12, gte6)
  gte7 = f32[100000000] get-tuple-element(p1), index=3
  add.14 = f32[100000000] add(add.13, gte7)
  ROOT r = f32[100000000] add(add.14, add.11)
}

comp1 {
  p = f32[100000000] parameter(0)
  c0 = f32[] constant(0)
  ROOT r = f32[] reduce(p, c0), dimensions={0}, to_apply=f32add
}

comp2 {
  p = f32[100000000] parameter(0)
  c0 = f32[] constant(0)
  r = f32[] reduce(p, c0), dimensions={0}, to_apply=f32add
  ROOT n = f32[] negate(r)
}

ENTRY m.Computation2 {
  p0 = (f32[100000000], f32[100000000], f32[100000000], f32[100000000]) parameter(0)
  p1 = (f32[100000000], f32[100000000], f32[100000000], f32[100000000]) parameter(1)
  fusion.0 = f32[100000000] fusion(p0, p1), kind=kLoop, calls=comp0
  fusion.1 = f32[] fusion(fusion.0), kind=kLoop, calls=comp1
  fusion.2 = f32[] fusion(fusion.0), kind=kLoop, calls=comp2
  ROOT tuple = (f32[], f32[]) tuple(fusion.1, fusion.2)
}
)")
                    .value();
  EXPECT_FALSE(fusion_merger_.Run(module.get()).value());
}

TEST_F(FusionMergerTest, LessMemoryAccessIfFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

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

ENTRY m.Computation2 {
  constant = f32[4]{0} constant({1, 1, 1, 1})
  state = (f32[4]{0}, f32[4]{0}, f32[4]{0}) parameter(0)
  fusion.2 = f32[4]{0} fusion(state), kind=kLoop, calls=comp.2
  fusion.3 = f32[4]{0} fusion(constant, fusion.2), kind=kLoop, calls=comp.1
  fusion.4 = f32[4]{0} fusion(constant, fusion.2), kind=kLoop, calls=comp
  ROOT tuple = (f32[4]{0}, f32[4]{0}) tuple(fusion.3, fusion.4)
})")
                    .value();
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

TEST_F(FusionMergerTest, WillMergeSliceIntoReusingConsumer) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

f1 {
  p01 = s8[1000000] parameter(0)
  ROOT s0 = s8[10] slice(p01), slice={[0:10]}
}

f2 {
  p02 = s8[10] parameter(0)
  ROOT b0 = s8[10,1000000] broadcast(p02), dimensions={0}
}

ENTRY e {
  p0 = s8[1000000] parameter(0)
  f1 = s8[10] fusion(p0), kind=kLoop, calls=f1
  ROOT r = s8[10,1000000] fusion(f1), kind=kLoop, calls=f2
})")
                    .value();
  EXPECT_TRUE(fusion_merger_.Run(module.get()).value());
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

    %f_b {
      %p = f32[1024,1024,1024] parameter(0)
      %t1 = f32[1024,1024,1024] tanh(%p)
      %t2 = f32[1024,1024,1024] tanh(%t1)
      %t3 = f32[1024,1024,1024] tanh(%t2)
      %t4 = f32[1024,1024,1024] tanh(%t3)
      %t5 = f32[1024,1024,1024] tanh(%t4)
      %t6 = f32[1024,1024,1024] tanh(%t5)
      %t7 = f32[1024,1024,1024] tanh(%t6)
      %t8 = f32[1024,1024,1024] tanh(%t7)
      ROOT %t9 = f32[1024,1024,1024] tanh(%t8)
    }

    %f_c {
      %p = f32[1024,1024,1024] parameter(0)
      ROOT %t = f32[1024,1024,1024,2048] broadcast(%p), dimensions={0,1,2}
    }

    ENTRY entry {
      p0 = f32[1024,1024,1024] parameter(0)
      f1 = f32[1024,1024,1024] fusion(p0), kind=kLoop, calls=%f_b
      ROOT f2 = f32[1024,1024,1024,2048] fusion(f1), kind=kLoop, calls=%f_c
    })")
                    .value();
  EXPECT_FALSE(fusion_merger_.Run(module.get()).value());
}

TEST_F(FusionMergerTest, NoMergeWithBitcast) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

f32add {
  x.634 = f32[] parameter(0)
  y.635 = f32[] parameter(1)
  ROOT add.636 = f32[] add(x.634, y.635)
}

fused_computation.103 {
  param_0.310 = f16[1,8,512,1536]{2,3,1,0} parameter(0)
  param_1.420 = f32[8,512]{1,0} parameter(1)
  bitcast.1144 = f32[1,8,512]{2,1,0} bitcast(param_1.420)
  convert.252 = f16[1,8,512]{2,1,0} convert(bitcast.1144)
  bitcast.1143 = f16[8,512]{1,0} bitcast(convert.252)
  broadcast.481 = f16[1,8,512,1536]{2,3,1,0} broadcast(bitcast.1143), dimensions={1,2}
  divide.15 = f16[1,8,512,1536]{2,3,1,0} divide(param_0.310, broadcast.481)
  ROOT bitcast.1142 = f16[8,512,1536]{1,2,0} bitcast(divide.15)
}

fused_computation.105 {
  param_1.426 = f16[8,1536,512]{2,1,0} parameter(1)
  bitcast.1896 = f16[1,8,1536,512]{3,2,1,0} bitcast(param_1.426)
  transpose.238 = f16[1,8,512,1536]{2,3,1,0} transpose(bitcast.1896), dimensions={0,1,3,2}
  param_0.315 = f16[8,512]{1,0} parameter(0)
  broadcast.482 = f16[1,8,512,1536]{2,3,1,0} broadcast(param_0.315), dimensions={1,2}
  subtract.22 = f16[1,8,512,1536]{2,3,1,0} subtract(transpose.238, broadcast.482) 
  ROOT exponential.15 = f16[1,8,512,1536]{2,3,1,0} exponential(subtract.22)
}

fused_computation.104 {
  param_0.1000 = f16[8,1536,512]{2,1,0} parameter(0)
  convert.652 = f32[8,1536,512]{2,1,0} convert(param_0.1000)
  constant_752 = f32[] constant(-0)
  ROOT reduce.232 = f32[8,512]{1,0} reduce(convert.652, constant_752),
  dimensions={1}, to_apply=f32add
}

ENTRY entry {
  p0 = f16[8,1536,512]{2,1,0} parameter(0)
  p1 = f16[8,512]{1,0} parameter(1)
  fusion.105 = f16[1,8,512,1536]{2,3,1,0} fusion(p1, p0), kind=kLoop, calls=fused_computation.105
  bitcast.1787 = f16[8,1536,512]{2,1,0} bitcast(fusion.105)
  fusion.104 = f32[8,512]{1,0} fusion(bitcast.1787), kind=kInput, calls=fused_computation.104
  ROOT fusion.103 = f16[8,512,1536]{1,2,0} fusion(fusion.105, fusion.104), kind=kLoop, calls=fused_computation.103
}
    )")
                    .value();
  EXPECT_FALSE(fusion_merger_.Run(module.get()).value());
}

TEST_F(FusionMergerTest, CostBasedMerge) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

fused_computation.45 {
  param_1.194 = f16[8,1536,512]{2,1,0} parameter(1)
  bitcast.1042 = f16[1,8,512,1536]{2,3,1,0} bitcast(param_1.194)
  param_0.135 = f16[8,512]{1,0} parameter(0)
  broadcast.391 = f16[1,8,512,1536]{2,3,1,0} broadcast(param_0.135), dimensions={1,2}
  subtract.6 = f16[1,8,512,1536]{2,3,1,0} subtract(bitcast.1042, broadcast.391)
  ROOT exponential.11 = f16[1,8,512,1536]{2,3,1,0} exponential(subtract.6)
}

f32add {
  x.634 = f32[] parameter(0)
  y.635 = f32[] parameter(1)
  ROOT add.636 = f32[] add(x.634, y.635)
}

fused_computation.44 {
  param_0.869 = f16[1,8,512,1536]{2,3,1,0} parameter(0)
  convert.221 = f32[1,8,512,1536]{2,3,1,0} convert(param_0.869)
  transpose.212 = f32[1,8,1536,512]{3,2,1,0} transpose(convert.221), dimensions={0,1,3,2}
  bitcast.1041 = f32[8,1536,512]{2,1,0} bitcast(transpose.212)
  constant_429 = f32[] constant(0)
  ROOT reduce.149 = f32[8,512]{1,0} reduce(bitcast.1041, constant_429), dimensions={1}, to_apply=f32add
}

fused_computation.43 {
  param_0.130 = f16[1,8,512,1536]{2,3,1,0} parameter(0)
  param_1.188 = f32[8,512]{1,0} parameter(1)
  bitcast.1040 = f32[1,8,512]{2,1,0} bitcast(param_1.188)
  convert.220 = f16[1,8,512]{2,1,0} convert(bitcast.1040)
  bitcast.1039 = f16[8,512]{1,0} bitcast(convert.220)
  broadcast.390 = f16[1,8,512,1536]{2,3,1,0} broadcast(bitcast.1039), dimensions={1,2}
  divide.11 = f16[1,8,512,1536]{2,3,1,0} divide(param_0.130, broadcast.390)
  ROOT bitcast.1038 = f16[8,512,1536]{1,2,0} bitcast(divide.11)
}

ENTRY entry {
  p0 = f16[8,1536,512]{2,1,0} parameter(0)
  p1 = f16[8,512]{1,0} parameter(1)
  fusion.45 = f16[1,8,512,1536]{2,3,1,0} fusion(p1, p0), kind=kLoop, calls=fused_computation.45
  fusion.44 = f32[8,512]{1,0} fusion(fusion.45), kind=kInput, calls=fused_computation.44
  ROOT fusion.43 = f16[8,512,1536]{1,2,0} fusion(fusion.45, fusion.44), kind=kLoop, calls=fused_computation.43
}
    )")
                    .value();
  EXPECT_TRUE(fusion_merger_.Run(module.get()).value());
}

// Outputs of fusions 66 and 67 here are heavily reused by fusion 59 - so
// it is better to not merge here.
TEST_F(FusionMergerTest, CostBasedNoMerge) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

add_float_.56 {
  x.57 = f32[] parameter(0)
  y.58 = f32[] parameter(1)
  ROOT add.59 = f32[] add(x.57, y.58)
}

fused_computation.66 {
  constant.635 = f32[] constant(0)
  broadcast.257 = f32[459,3]{1,0} broadcast(constant.635), dimensions={}
  constant.641 = f32[] constant(1)
  broadcast.256 = f32[459,3]{1,0} broadcast(constant.641), dimensions={}
  broadcast.255 = f32[459]{0} broadcast(constant.635), dimensions={}
  iota.28 = f32[459]{0} iota(), iota_dimension=0
  constant.629 = f32[] constant(1.49891067)
  broadcast.253 = f32[459]{0} broadcast(constant.629), dimensions={}
  multiply.39 = f32[459]{0} multiply(iota.28, broadcast.253)
  constant.633 = f32[] constant(-1)
  broadcast.252 = f32[459]{0} broadcast(constant.633), dimensions={}
  add.31 = f32[459]{0} add(multiply.39, broadcast.252)
  ceil.11 = f32[459]{0} ceil(add.31)
  constant.630 = f32[] constant(685)
  broadcast.251 = f32[459]{0} broadcast(constant.630), dimensions={}
  clamp.49 = f32[459]{0} clamp(broadcast.255, ceil.11, broadcast.251)
  subtract.11 = f32[459]{0} subtract(clamp.49, multiply.39)
  broadcast.249 = f32[459,3]{1,0} broadcast(subtract.11), dimensions={0}
  iota.26 = f32[459,3]{1,0} iota(), iota_dimension=1
  add.30 = f32[459,3]{1,0} add(broadcast.249, iota.26)
  abs.3 = f32[459,3]{1,0} abs(add.30)
  subtract.10 = f32[459,3]{1,0} subtract(broadcast.256, abs.3)
  maximum.6 = f32[459,3]{1,0} maximum(broadcast.257, subtract.10)
  ROOT reduce.3 = f32[459]{0} reduce(maximum.6, constant.635), dimensions={1}, to_apply=add_float_.56
}

fused_computation.67 {
  constant.684 = f32[] constant(0)
  broadcast.296 = f32[1130,3]{1,0} broadcast(constant.684), dimensions={}
  constant.685 = f32[] constant(1)
  broadcast.295 = f32[1130,3]{1,0} broadcast(constant.685), dimensions={}
  broadcast.294 = f32[1130]{0} broadcast(constant.684), dimensions={}
  iota.41 = f32[1130]{0} iota(), iota_dimension=0
  constant.675 = f32[] constant(1.34513271)
  broadcast.293 = f32[1130]{0} broadcast(constant.675), dimensions={}
  multiply.47 = f32[1130]{0} multiply(iota.41, broadcast.293)
  constant.677 = f32[] constant(-1)
  broadcast.290 = f32[1130]{0} broadcast(constant.677), dimensions={}
  add.39 = f32[1130]{0} add(multiply.47, broadcast.290)
  ceil.15 = f32[1130]{0} ceil(add.39)
  constant.676 = f32[] constant(1517)
  broadcast.289 = f32[1130]{0} broadcast(constant.676), dimensions={}
  clamp.53 = f32[1130]{0} clamp(broadcast.294, ceil.15, broadcast.289)
  subtract.19 = f32[1130]{0} subtract(clamp.53, multiply.47)
  broadcast.287 = f32[1130,3]{1,0} broadcast(subtract.19), dimensions={0}
  iota.39 = f32[1130,3]{1,0} iota(), iota_dimension=1
  add.38 = f32[1130,3]{1,0} add(broadcast.287, iota.39)
  abs.7 = f32[1130,3]{1,0} abs(add.38)
  subtract.18 = f32[1130,3]{1,0} subtract(broadcast.295, abs.7)
  maximum.10 = f32[1130,3]{1,0} maximum(broadcast.296, subtract.18)
  ROOT reduce.4 = f32[1130]{0} reduce(maximum.10, constant.684), dimensions={1}, to_apply=add_float_.56
}

fused_computation.59 {
  constant.532 = f32[] constant(0)
  broadcast.316 = f32[1130,3]{1,0} broadcast(constant.532), dimensions={}
  constant.663 = f32[] constant(1)
  broadcast.315 = f32[1130,3]{1,0} broadcast(constant.663), dimensions={}
  broadcast.314 = f32[1130]{0} broadcast(constant.532), dimensions={}
  iota.47 = f32[1130]{0} iota(), iota_dimension=0
  constant.579 = f32[] constant(1.34513271)
  broadcast.311 = f32[1130]{0} broadcast(constant.579), dimensions={}
  multiply.51 = f32[1130]{0} multiply(iota.47, broadcast.311)
  constant.578 = f32[] constant(-1)
  broadcast.310 = f32[1130]{0} broadcast(constant.578), dimensions={}
  add.43 = f32[1130]{0} add(multiply.51, broadcast.310)
  ceil.17 = f32[1130]{0} ceil(add.43)
  constant.576 = f32[] constant(1517)
  broadcast.309 = f32[1130]{0} broadcast(constant.576), dimensions={}
  clamp.55 = f32[1130]{0} clamp(broadcast.314, ceil.17, broadcast.309)
  subtract.24 = f32[1130]{0} subtract(clamp.55, multiply.51)
  broadcast.306 = f32[1130,3]{1,0} broadcast(subtract.24), dimensions={0}
  iota.45 = f32[1130,3]{1,0} iota(), iota_dimension=1
  add.42 = f32[1130,3]{1,0} add(broadcast.306, iota.45)
  abs.9 = f32[1130,3]{1,0} abs(add.42)
  subtract.23 = f32[1130,3]{1,0} subtract(broadcast.315, abs.9)
  maximum.12 = f32[1130,3]{1,0} maximum(broadcast.316, subtract.23)
  param_2.183 = f32[1130]{0} parameter(2)
  broadcast.172 = f32[1130,3]{1,0} broadcast(param_2.183), dimensions={0}
  divide.3 = f32[1130,3]{1,0} divide(maximum.12, broadcast.172)
  bitcast.53 = f32[3390]{0} bitcast(divide.3)
  broadcast.171 = f32[3390,1377]{1,0} broadcast(bitcast.53), dimensions={0}
  broadcast.276 = f32[459,3]{1,0} broadcast(constant.532), dimensions={}
  broadcast.275 = f32[459,3]{1,0} broadcast(constant.663), dimensions={}
  broadcast.274 = f32[459]{0} broadcast(constant.532), dimensions={}
  iota.35 = f32[459]{0} iota(), iota_dimension=0
  constant.614 = f32[] constant(1.49891067)
  broadcast.273 = f32[459]{0} broadcast(constant.614), dimensions={}
  multiply.43 = f32[459]{0} multiply(iota.35, broadcast.273)
  broadcast.272 = f32[459]{0} broadcast(constant.578), dimensions={}
  add.35 = f32[459]{0} add(multiply.43, broadcast.272)
  ceil.13 = f32[459]{0} ceil(add.35)
  constant.611 = f32[] constant(685)
  broadcast.269 = f32[459]{0} broadcast(constant.611), dimensions={}
  clamp.51 = f32[459]{0} clamp(broadcast.274, ceil.13, broadcast.269)
  subtract.15 = f32[459]{0} subtract(clamp.51, multiply.43)
  broadcast.267 = f32[459,3]{1,0} broadcast(subtract.15), dimensions={0}
  iota.33 = f32[459,3]{1,0} iota(), iota_dimension=1
  add.34 = f32[459,3]{1,0} add(broadcast.267, iota.33)
  abs.5 = f32[459,3]{1,0} abs(add.34)
  subtract.14 = f32[459,3]{1,0} subtract(broadcast.275, abs.5)
  maximum.8 = f32[459,3]{1,0} maximum(broadcast.276, subtract.14)
  param_1.177 = f32[459]{0} parameter(1)
  broadcast.170 = f32[459,3]{1,0} broadcast(param_1.177), dimensions={0}
  divide.2 = f32[459,3]{1,0} divide(maximum.8, broadcast.170)
  bitcast.52 = f32[1377]{0} bitcast(divide.2)
  broadcast.169 = f32[3390,1377]{1,0} broadcast(bitcast.52), dimensions={1}
  multiply.15 = f32[3390,1377]{1,0} multiply(broadcast.171, broadcast.169)
  bitcast.61 = f32[1130,3,459,3]{3,2,1,0} bitcast(multiply.15)
  transpose.68 = f32[459,1130,3,3]{2,0,3,1} transpose(bitcast.61), dimensions={2,0,3,1}
  copy.1 = f32[459,1130,3,3]{3,2,1,0} copy(transpose.68)
  bitcast.50 = f32[1130,459,9]{2,1,0} bitcast(copy.1)
  broadcast.168 = f32[1130,459,6,9]{3,2,1,0} broadcast(bitcast.50), dimensions={0,1,3}
  param_0.171 = u8[1,688,1520,6]{3,2,1,0} parameter(0)
  bitcast.49 = u8[688,1520,1,6]{3,1,0,2} bitcast(param_0.171)
  convert.175 = f32[688,1520,1,6]{3,1,0,2} convert(bitcast.49)
  broadcast.167 = f32[459,1130,1]{2,1,0} broadcast(clamp.51), dimensions={0}
  broadcast.166 = f32[459,1130,1]{2,1,0} broadcast(clamp.55), dimensions={1}
  concatenate.3 = f32[459,1130,2]{2,1,0} concatenate(broadcast.167, broadcast.166), dimensions={2}
  convert.174 = s32[459,1130,2]{2,1,0} convert(concatenate.3)
  bitcast.48 = s32[518670,2]{1,0} bitcast(convert.174)
  gather.1 = f32[518670,3,3,1,6]{2,1,4,0,3} gather(convert.175, bitcast.48), offset_dims={1,2,3,4}, collapsed_slice_dims={}, start_index_map={0,1}, index_vector_dim=1, slice_sizes={3,3,1,6}
  transpose.69 = f32[1,518670,6,3,3]{4,3,2,1,0} transpose(gather.1), dimensions={3,0,4,1,2}
  bitcast.47 = f32[1130,459,6,9]{3,2,1,0} bitcast(transpose.69)
  multiply.14 = f32[1130,459,6,9]{3,2,1,0} multiply(broadcast.168, bitcast.47)
  reduce.2 = f32[1130,459,6]{2,1,0} reduce(multiply.14, constant.532), dimensions={3}, to_apply=add_float_.56
  convert.173 = f16[1130,459,6]{2,1,0} convert(reduce.2)
  bitcast.46 = f16[1,459,1130,6]{3,2,1,0} bitcast(convert.173)
  constant.533 = f16[] constant(0)
  pad.9 = f16[1,480,1130,6]{3,2,1,0} pad(bitcast.46, constant.533), padding=0_0x0_21x0_0x0_0
  pad.8 = f16[1,480,1152,6]{3,2,1,0} pad(pad.9, constant.533), padding=0_0x0_0x0_22x0_0
  constant.532f16 = f16[] constant(0)
  ROOT pad.7 = f16[1,485,1157,6]{3,2,1,0} pad(pad.8, constant.532f16), padding=0_0x2_3x2_3x0_0
}

ENTRY e {
  arg0.1 = u8[1,688,1520,6]{3,2,1,0} parameter(0), parameter_replication={false}
  fusion.66 = f32[459]{0} fusion(), kind=kLoop, calls=fused_computation.66
  fusion.67 = f32[1130]{0} fusion(), kind=kLoop, calls=fused_computation.67
  ROOT fusion.59 = f16[1,485,1157,6]{2,1,3,0} fusion(arg0.1, fusion.66, fusion.67), kind=kLoop, calls=fused_computation.59
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

TEST_F(FusionMergerTest, CommonElementwiseUsedParameter) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m

    p {
      p0 = f32[10000000] parameter(0)
      p1 = f32[10000000] parameter(1)
      p2 = f32[10000000] parameter(2)
      p3 = f32[10000000] parameter(3)
      a0 = f32[10000000] add(p1, p2)
      a1 = f32[10000000] add(a0, p3)
      ROOT _ = add(p0, a1)
    }

    c1 {
      p0 = f32[10000000] parameter(0)
      p1 = f32[10000000] parameter(1)
      ROOT _ = add(p0, p1)
    }

    c2 {
      p0 = f32[10000000] parameter(0)
      p1 = f32[10000000] parameter(1)
      ROOT _ = multiply(p0, p1)
    }

    ENTRY entry {
      p0 = f32[10000000] parameter(0)
      p1 = f32[10000000] parameter(1)
      p2 = f32[10000000] parameter(2)
      p3 = f32[10000000] parameter(3)
      f = f32[10000000] fusion(p0, p1, p2, p3), kind=kLoop, calls=p
      f1 = f32[10000000] fusion(p0, f), kind=kLoop, calls=c1
      f2 = f32[10000000] fusion(p1, f), kind=kLoop, calls=c2
      ROOT _ = (f32[10000000], f32[10000000]) tuple(f1, f2)
    }
    )")
                    .value();
  EXPECT_TRUE(fusion_merger_.Run(module.get()).value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
