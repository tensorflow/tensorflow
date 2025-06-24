/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/multi_output_fusion.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

namespace m = ::xla::match;

class MultiOutputFusionTest : public HloHardwareIndependentTestBase {
 public:
  MultiOutputFusion mof_{TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                         HloCostAnalysis::DefaultShapeSize};

  void CheckMultiOutputFusion(absl::string_view hlo,
                              std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(
        hlo,
        MultiOutputFusion{TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                          HloCostAnalysis::DefaultShapeSize},
        expected);
  }
};

const char kModulePrefix[] = R"(
    HloModule test_module

    scalar_add_computation {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      ROOT add.0 = f32[] add(scalar_lhs.0, scalar_rhs.0)
    }
    scalar_mul_computation {
      scalar_lhs.1 = f32[] parameter(0)
      scalar_rhs.1 = f32[] parameter(1)
      ROOT mul.1 = f32[] multiply(scalar_lhs.1, scalar_rhs.1)
    })";

static int64_t CountMultiOutputFusions(const HloModule* module) {
  int multi_output_fusion_count = 0;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instr : computation->instructions()) {
      if (instr->IsMultiOutputFusion()) {
        multi_output_fusion_count++;
      }
    }
  }
  return multi_output_fusion_count;
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingReduceAndReduceFusion) {
  // Fusion with reduce instruction root and a sibling reduce instruction
  // sharing the same input param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[512]{0} reduce(mul, const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      const.2 = f32[] constant(1)
      fusion = f32[512] fusion(p0, p1), kind=kInput, calls=fused_computation
      reduce.2 = f32[512]{0} reduce(p1, const.2), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT root = (f32[512]{0}, f32[512]{0}) tuple(fusion, reduce.2)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce())));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDifferentReduceInputShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[6400]{0} parameter(1)
      mul = f32[6400]{0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[] reduce(mul, const.1), dimensions={0}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = f32[6400]{0} parameter(1)
      r1 = f32[64,100]{0,1} reshape(p1.2)
      const.2 = f32[] parameter(0)
      ROOT reduce.2 = f32[] reduce(r1, const.2), dimensions={1,0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[6400]{0} parameter(1)
      fusion.1 = f32[] fusion(p0, p1), kind=kInput, calls=fused_computation_1
      fusion.2 = f32[] fusion(p0, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[], f32[]) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, ReduceMofDifferentTypes) {
  // Fusion with reduce instruction root and a sibling reduce instruction
  // sharing the same input param.
  const char* hlo = R"(
HloModule module

scalar_add_computation {
  scalar_lhs.1 = f32[] parameter(0)
  scalar_rhs.1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(scalar_lhs.1, scalar_rhs.1)
}

scalar_add_computation_f16 {
  scalar_lhs.0 = f16[] parameter(0)
  scalar_rhs.0 = f16[] parameter(1)
  ROOT add.0 = f16[] add(scalar_lhs.0, scalar_rhs.0)
}

fused_computation {
  param_0.2 = f32[128,512,28,28]{3,2,1,0} parameter(0)
  c.1 = f16[128,512,28,28]{3,2,1,0} convert(param_0.2)
  const.0 = f16[] constant(0)
  ROOT reduce.0 = f16[512]{0} reduce(c.1, const.0), dimensions={0,2,3}, to_apply=scalar_add_computation_f16
}

ENTRY entry {
  p0 = f32[] parameter(0)
  p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
  const.2 = f32[] constant(0)
  reduce.1 = f32[512]{0} reduce(p1, const.2), dimensions={0,2,3}, to_apply=scalar_add_computation
  fusion = f16[512]{0} fusion(p1), kind=kInput, calls=fused_computation
  ROOT root = (f32[512]{0}, f16[512]{0}) tuple(reduce.1, fusion)
})";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation
// CHECK-NEXT:   [[param_0_2_0:%[^ ]+]] = f32[128,512,28,28]{3,2,1,0} parameter(0)
// CHECK-NEXT:   [[c_1_1:%[^ ]+]] = f16[128,512,28,28]{3,2,1,0} convert([[param_0_2_0]])
// CHECK-NEXT:   [[const_0_2:%[^ ]+]] = f16[] constant(0)
// CHECK-NEXT:   [[reduce_0_3:%[^ ]+]] = f16[512]{0} reduce([[c_1_1]], [[const_0_2]]), dimensions={0,2,3}, to_apply=[[scalar_add_computation_f16_4:%[^ ]+]]
// CHECK-NEXT:   [[param_1_5:%[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:   [[reduce_2_6:%[^ ]+]] = f32[512]{0} reduce([[param_0_2_0]], [[param_1_5]]), dimensions={0,2,3}, to_apply=[[scalar_add_computation_7:%[^ ]+]]
// CHECK-NEXT:   ROOT [[tuple_8:%[^ ]+]] = (f16[512]{0}, f32[512]{0}) tuple([[reduce_0_3]], [[reduce_2_6]])
// CHECK:   [[fusion_9:%[^ ]+]] = (f16[512]{0}, f32[512]{0}) fusion([[p1_10:%[^ ]+]], [[const_2_11:%[^ ]+]]), kind=kInput, calls=[[fused_computation_12:%[^ ]+]]
)");
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDifferentReduceOutputShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[10,10]{1,0} parameter(1)
      mul = f32[10,10]{1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[] reduce(mul, const.1), dimensions={0,1}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = f32[10,10]{1,0} parameter(1)
      const.2 = f32[] parameter(0)
      ROOT reduce.2 = f32[10]{0} reduce(p1.2, const.2), dimensions={0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1.3 = f32[10,10]{1,0} parameter(1)
      fusion.1 = f32[] fusion(p0, p1.3), kind=kInput, calls=fused_computation_1
      p2 = f32[] parameter(2)
      fusion.2 = f32[10]{0} fusion(p2, p1.3), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[], f32[10]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingReduceFusions) {
  // Two sibling fusions with reduce instruction roots sharing the same input
  // param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[512]{0} reduce(mul, const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      const.2 = f32[] parameter(0)
      ROOT reduce.2 = f32[512]{0} reduce(p1.2, const.2), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      fusion.1 = f32[512] fusion(p0, p1), kind=kInput, calls=fused_computation_1
      fusion.2 = f32[512] fusion(p0, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[512]{0}, f32[512]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce())));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionNoSiblingFusionForCommonScalar) {
  // Two sibling fusions with bitcast roots sharing the same scalar input param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      param_0.87 = bf16[32,4096,16384]{2,1,0} parameter(0)
      param_1.4620 = s32[] parameter(1)
      constant_3949 = s32[] constant(0)
      compare.1026 = pred[] compare(param_1.4620, constant_3949), direction=LT
      constant_5437 = s32[] constant(32)
      add.6859 = s32[] add(param_1.4620, constant_5437)
      select.1599 = s32[] select(compare.1026, add.6859, param_1.4620)
      dynamic-slice.59 = bf16[1,4096,16384]{2,1,0} dynamic-slice(param_0.87, select.1599, constant_3949, constant_3949), dynamic_slice_sizes={1,4096,16384}
      ROOT bitcast.41089 = bf16[4096,16384]{1,0} bitcast(dynamic-slice.59)
    }

    fused_computation_2 {
      param_0 = bf16[32,4096,16384]{2,1,0} parameter(0)
      param_1 = s32[] parameter(1)
      constant = s32[] constant(0)
      compare = pred[] compare(param_1, constant), direction=LT
      constant.32 = s32[] constant(32)
      add = s32[] add(param_1, constant.32)
      select = s32[] select(compare, add, param_1)
      dynamic-slice = bf16[1,4096,16384]{2,1,0} dynamic-slice(param_0, select, constant, constant), dynamic_slice_sizes={1,4096,16384}
      ROOT bitcast.41087 = bf16[4096,16384]{1,0} bitcast(dynamic-slice)
    }

    ENTRY entry {
      p0 = s32[] parameter(0)
      p1 = bf16[32,4096,16384]{2,1,0} parameter(1)
      p2 = bf16[32,4096,16384]{2,1,0} parameter(2)
      fusion.1 = bf16[4096,16384]{1,0} fusion(p1, p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = bf16[4096,16384]{1,0} fusion(p2, p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (bf16[4096,16384]{1,0}, bf16[4096,16384]{1,0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingReduceAndReduceMultiOutputFusion) {
  // Multi-output fusion with two reduce instructions root and a sibling reduce
  // instruction sharing the same input param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation (p0: f32[128,512,28,28]) -> (f32[512], f32[512]) {
      const.1 = f32[] constant(1)
      p0.1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(f32[128,512,28,28]{3,2,1,0} p0.1, f32[128,512,28,28]{3,2,1,0} p0.1)
      reduce.1 = f32[512]{0} reduce(f32[128,512,28,28]{3,2,1,0} mul, f32[] const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
      reduce.2 = f32[512]{0} reduce(f32[128,512,28,28]{3,2,1,0} p0.1, f32[] const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT tuple = (f32[512]{0}, f32[512]{0}) tuple(f32[512]{0} reduce.1, f32[512]{0} reduce.2)
    }

    ENTRY entry (p0: f32[128,512,28,28]) -> (f32[512], f32[512], f32[512]) {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      const = f32[] constant(1)
      fusion = (f32[512]{0}, f32[512]{0}) fusion(f32[128,512,28,28]{3,2,1,0} p0), kind=kInput, calls=fused_computation
      get-tuple-element = f32[512]{0} get-tuple-element((f32[512]{0}, f32[512]{0}) fusion), index=0
      get-tuple-element.1 = f32[512]{0} get-tuple-element((f32[512]{0}, f32[512]{0}) fusion), index=1
      reduce.3 = f32[512]{0} reduce(p0, const), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT root = (f32[512]{0}, f32[512]{0}, f32[512]{0}) tuple(f32[512]{0} get-tuple-element, f32[512]{0} get-tuple-element.1, f32[512]{0} reduce.3)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce(), m::Reduce())));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingFusionCheckAgainstReduceOperand) {
  // Verify that if we already have a multi-output fusion that we prefer to pick
  // a reduce op from its operands for checking shape compatibility.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[10,10]{1,0} parameter(1)
      mul = f32[10,10]{1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      reduce.1 = f32[] reduce(p1.1, const.1), dimensions={0,1}, to_apply=scalar_add_computation
      ROOT tuple = (f32[10,10], f32[]) tuple(mul, reduce.1)
    }

    fused_computation_2 {
      p1.2 = f32[10,10]{1,0} parameter(1)
      const.2 = f32[] parameter(0)
      ROOT reduce.2 = f32[10] reduce(p1.2, const.2), dimensions={0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[10,10]{1,0} parameter(1)
      p2 = f32[] parameter(2)
      fusion.1 = (f32[10,10], f32[]) fusion(p0, p1), kind=kInput, calls=fused_computation_1
      get-tuple-element.1 = f32[10,10] get-tuple-element((f32[10,10], f32[]) fusion.1), index=0
      get-tuple-element.2 = f32[] get-tuple-element((f32[10,10], f32[]) fusion.1), index=1
      fusion.2 = f32[10] fusion(p2, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[10,10], f32[], f32[10]) tuple(get-tuple-element.1, get-tuple-element.2, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, LoopVariadicReductionFusions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation.94 {
      tmp_0 = f32[] parameter(0)
      tmp_1 = f32[] parameter(1)
      tmp_2 = pred[] compare(tmp_0, tmp_1), direction=GE
      tmp_3 = f32[] select(tmp_2, tmp_0, tmp_1)
      tmp_4 = pred[] compare(tmp_0, tmp_1), direction=EQ
      tmp_5 = s32[] parameter(2)
      tmp_6 = s32[] parameter(3)
      tmp_7 = s32[] minimum(tmp_5, tmp_6)
      tmp_8 = s32[] select(tmp_2, tmp_5, tmp_6)
      tmp_9 = s32[] select(tmp_4, tmp_7, tmp_8)
      ROOT tmp_10 = (f32[], s32[]) tuple(tmp_3, tmp_9)
    }

    minmax_func.1536 {
      tmp_0 = f32[] parameter(0)
      tmp_1 = f32[] parameter(2)
      tmp_2 = s32[] parameter(1)
      tmp_3 = s32[] parameter(3)
      ROOT tmp_4 = (f32[], s32[]) fusion(tmp_0, tmp_1, tmp_2, tmp_3), kind=kLoop, calls=fused_computation.94
    }

    fused_computation {
      tmp_0 = f32[554112,10]{1,0} parameter(0)
      tmp_1 = s32[554112,10]{1,0} iota(), iota_dimension=1
      tmp_2 = f32[] constant(-inf)
      tmp_3 = s32[] constant(0)
      ROOT tmp_4 = (f32[554112]{0}, s32[554112]{0}) reduce(tmp_0, tmp_1, tmp_2, tmp_3), dimensions={1}, to_apply=minmax_func.1536
    }

    fused_computation2 {
      tmp_0 = f32[554112,10]{1,0} parameter(0)
      tmp_1 = s32[554112,10]{1,0} iota(), iota_dimension=1
      tmp_2 = f32[] constant(inf)
      tmp_3 = s32[] constant(1)
      ROOT tmp_4 = (f32[554112]{0}, s32[554112]{0}) reduce(tmp_0, tmp_1, tmp_2, tmp_3), dimensions={1}, to_apply=minmax_func.1536
    }

    ENTRY e {
      tmp_0 = f32[554112,10]{1,0} parameter(0)
      tmp_1 = (f32[554112]{0}, s32[554112]{0}) fusion(tmp_0), kind=kLoop, calls=fused_computation
      tmp_2 = s32[554112]{0} get-tuple-element(tmp_1), index=1
      tmp_4 = (f32[554112]{0}, s32[554112]{0}) fusion(tmp_0), kind=kLoop, calls=fused_computation2
      tmp_5 = s32[554112]{0} get-tuple-element(tmp_4), index=1
      ROOT tmp_6 = s32[554112]{0} add(tmp_2, tmp_5)
    })"))
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, InputVariadicReductionFusions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation.1117 {
      param_0.2433 = f32[] parameter(0)
      param_1.2571 = f32[] parameter(1)
      compare.1770 = pred[] compare(param_0.2433, param_1.2571), direction=LE
      select.682 = f32[] select(compare.1770, param_0.2433, param_1.2571)
      compare.1303.clone.1 = pred[] compare(param_0.2433, param_1.2571), direction=EQ
      param_2.6460 = s32[] parameter(2)
      param_3.6755 = s32[] parameter(3)
      minimum.633.clone.1 = s32[] minimum(param_2.6460, param_3.6755)
      select.398.clone.1 = s32[] select(compare.1770, param_2.6460, param_3.6755)
      select.397.clone.1 = s32[] select(compare.1303.clone.1, minimum.633.clone.1, select.398.clone.1)
      ROOT tuple.151 = (f32[], s32[]) tuple(select.682, select.397.clone.1)
    }

    minmax_func.223 {
      lhs_value.224 = f32[] parameter(0)
      rhs_value.226 = f32[] parameter(2)
      lhs_index.225 = s32[] parameter(1)
      rhs_index.227 = s32[] parameter(3)
      ROOT fusion.1117 = (f32[], s32[]) fusion(lhs_value.224, rhs_value.226, lhs_index.225, rhs_index.227), kind=kLoop, calls=fused_computation.1117
    }

    fused_computation.73 {
      bitcast.86661 = f32[3,1024,300]{2,1,0} parameter(0)
      iota.734 = s32[3,1,1024,300]{3,2,1,0} iota(), iota_dimension=3
      bitcast.97555 = s32[3,1024,300]{2,1,0} bitcast(iota.734)
      constant_3917 = f32[] constant(inf)
      constant_3918 = s32[] constant(0)
      ROOT reduce.1069 = (f32[3,1024]{1,0}, s32[3,1024]{1,0}) reduce(bitcast.86661, bitcast.97555, constant_3917, constant_3918), dimensions={2}, to_apply=minmax_func.223
    }

    fused_computation.84 {
      bitcast.86676 = f32[3,1024,300]{2,1,0} parameter(0)
      iota.732 = s32[3,1,1024,300]{3,2,1,0} iota(), iota_dimension=3
      bitcast.97553 = s32[3,1024,300]{2,1,0} bitcast(iota.732)
      constant_3915 = f32[] constant(inf)
      constant_3916 = s32[] constant(0)
      ROOT reduce.1070 = (f32[3,1024]{1,0}, s32[3,1024]{1,0}) reduce(bitcast.86676, bitcast.97553, constant_3915, constant_3916), dimensions={2}, to_apply=minmax_func.223
    }

    ENTRY e {
      p0 = f32[3,1024,300]{2,1,0} parameter(0)
      fusion.84 = (f32[3,1024]{1,0}, s32[3,1024]{1,0}) fusion(p0), kind=kInput, calls=fused_computation.84
      gte.391 = s32[3,1024]{1,0} get-tuple-element(fusion.84), index=1
      fusion.73 = (f32[3,1024]{1,0}, s32[3,1024]{1,0}) fusion(p0), kind=kInput, calls=fused_computation.73
      gte.393 = s32[3,1024]{1,0} get-tuple-element(fusion.73), index=1
      ROOT r = s32[3,1024]{1,0} add(gte.391, gte.393)
    })"))
                    .value();
  EXPECT_TRUE(mof_.Run(module.get()).value());
  EXPECT_EQ(module->entry_computation()->parameter_instruction(0)->user_count(),
            1);
  const HloInstruction* fusion =
      module->entry_computation()->parameter_instruction(0)->users()[0];
  EXPECT_THAT(fusion, GmockMatch(m::Fusion()));
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce())));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionTwoLoops) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    fused_computation_2 {
      p0.2 = f32[6400]{0} parameter(0)
      const.2 = f32[] constant(1)
      broadcast = f32[6400]{0} broadcast(const.2), dimensions={}
      ROOT div = f32[6400]{0} divide(p0.2, broadcast)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (f32[6400]{0}, f32[6400]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Multiply(), m::Divide())));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionLoopElementwise) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      const.2 = f32[] constant(1)
      broadcast = f32[6400]{0} broadcast(const.2), dimensions={}
      div = f32[6400]{0} divide(p0, broadcast)
      ROOT root = (f32[6400]{0}, f32[6400]{0}) tuple(fusion.1, div)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Multiply(), m::Divide())));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingLoopsDifferentShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      ROOT mul = f32[8,1,5,16,1,2]{5,4,3,2,1,0} multiply(p0.1, p0.1)
    }

    fused_computation_2 {
      p0.2 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      const.2 = f32[] constant(0)
      ROOT reduce = f32[1,5,1,2]{3,2,1,0} reduce(p0.2, const.2), dimensions={0,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      fusion.1 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[1,5,1,2]{3,2,1,0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (f32[8,1,5,16,1,2]{5,4,3,2,1,0}, f32[1,5,1,2]{3,2,1,0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingLoopAndMultiOutputLoop) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      mul = f32[8,1,5,16,1,1]{5,4,3,2,1,0} multiply(p0.1, p0.1)
      exp = f32[8,1,5,16,1,1]{5,4,3,2,1,0} exponential(p0.1)
      ROOT tuple = (f32[8,1,5,16,1,1]{5,4,3,2,1,0},
        f32[8,1,5,16,1,1]{5,4,3,2,1,0}) tuple(mul, exp)
    }

    fused_computation_2 {
      p0.2 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      const.2 = f32[] constant(0)
      broadcast = f32[8,1,5,16,1,1]{5,4,3,2,1,0} broadcast(const.2),
        dimensions={}
      ROOT add = f32[8,1,5,16,1,1]{5,4,3,2,1,0} add(p0.2, broadcast)
    }

    ENTRY entry {
      p0 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      fusion.1 = (f32[8,1,5,16,1,1]{5,4,3,2,1,0},
        f32[8,1,5,16,1,1]{5,4,3,2,1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      fusion.2 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} fusion(p0), kind=kLoop,
        calls=fused_computation_2
      gte0 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=0
      gte1 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=1
      ROOT root = (f32[8,1,5,16,1,1]{5,4,3,2,1,0},
        f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0})
        tuple(gte0, gte1, fusion.2)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Multiply(), m::Exp(), m::Add())));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingMultiOutputLoopAndMultiOutputLoop) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[8,16]{1,0} parameter(0)
      mul = f32[8,16]{1,0} multiply(p0.1, p0.1)
      exp = f32[8,16]{1,0} exponential(p0.1)
      ROOT tuple = (f32[8,16]{1,0}, f32[8,16]{1,0}) tuple(mul, exp)
    }

    fused_computation_2 {
      p0.2 = f32[8,16]{1,0} parameter(0)
      const.2 = f32[] constant(0)
      broadcast = f32[8,16]{1,0} broadcast(const.2),
        dimensions={}
      add = f32[8,16]{1,0} add(p0.2, broadcast)
      ROOT tuple.1 = (f32[8,16]{1,0}, f32[8,16]{1,0}) tuple(add, broadcast)
    }

    ENTRY entry {
      p0 = f32[8,16]{1,0} parameter(0)
      fusion.1 = (f32[8,16]{1,0}, f32[8,16]{1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      fusion.2 = (f32[8,16]{1,0}, f32[8,16]{1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_2
      gte0 = f32[8,16]{1,0} get-tuple-element(fusion.1), index=0
      gte1 = f32[8,16]{1,0} get-tuple-element(fusion.1), index=1
      gte2 = f32[8,16]{1,0} get-tuple-element(fusion.2), index=0
      gte3 = f32[8,16]{1,0} get-tuple-element(fusion.2), index=1
      ROOT root = (f32[8,16]{1,0}, f32[8,16]{1,0}, f32[8,16]{1,0},
        f32[8,16]{1,0})
        tuple(gte0, gte1, gte2, gte3)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(
      fusion->fused_expression_root(),
      GmockMatch(m::Tuple(m::Multiply(), m::Exp(), m::Add(), m::Broadcast())));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingLoopAndMultiOutputLoopDifferentShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      mul = f32[8,1,5,16,1,2]{5,4,3,2,1,0} multiply(p0.1, p0.1)
      exp = f32[8,1,5,16,1,2]{5,4,3,2,1,0} exponential(p0.1)
      ROOT tuple = (f32[8,1,5,16,1,2]{5,4,3,2,1,0},
        f32[8,1,5,16,1,2]{5,4,3,2,1,0}) tuple(mul, exp)
    }

    fused_computation_2 {
      p0.2 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      const.2 = f32[] constant(0)
      ROOT reduce = f32[1,5,1,2]{3,2,1,0} reduce(p0.2, const.2),
        dimensions={0,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      fusion.1 = (f32[8,1,5,16,1,2]{5,4,3,2,1,0},
        f32[8,1,5,16,1,2]{5,4,3,2,1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      fusion.2 = f32[1,5,1,2]{3,2,1,0} fusion(p0), kind=kLoop,
        calls=fused_computation_2
      gte0 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=0
      gte1 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=1
      ROOT root = (f32[8,1,5,16,1,2]{5,4,3,2,1,0},
        f32[8,1,5,16,1,2]{5,4,3,2,1,0}, f32[1,5,1,2]{3,2,1,0})
        tuple(gte0, gte1, fusion.2)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, SiblingFusionBitcastAndLoopFusionNotFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test

fused_computation_1 {
  p0.1 = f32[2048,16000]{1,0} parameter(0)
  bitcast = f32[2048,1,16000]{2,1,0} bitcast(p0.1)
  ROOT exp = f32[2048,1,16000]{2,1,0} exponential(bitcast)
}

ENTRY main {
  param_0 = f32[2048,16000]{1,0} parameter(0)
  fusion = f32[2048,1,16000]{2,1,0} fusion(param_0), kind=kLoop, calls=fused_computation_1
  bitcast = f32[16000,1,2048]{2,1,0} bitcast(param_0)
  ROOT tuple.143 = (f32[16000,1,2048]{2,1,0}, f32[2048,1,16000]{2,1,0}) tuple(bitcast, fusion)
})")
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest,
       ProducerConsumerFusionBitcastAndElementwiseNotFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test

ENTRY main {
  param_0 = f32[2048,16000]{1,0} parameter(0)
  convert = bf16[2048,16000]{1,0} convert(param_0)
  bitcast = bf16[16000,1,2048]{2,1,0} bitcast(convert)
  ROOT tuple.143 = (bf16[16000,1,2048]{2,1,0}, bf16[2048,16000]{1,0}) tuple(bitcast, convert)
})")
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionElementwiseAndReduce) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      exp = f32[32,32,32]{2,1,0} exponential(p0)
      reduce = f32[32,32]{1,0} reduce(exp, c0), dimensions={2},
        to_apply=scalar_add_computation
      ROOT root = (f32[32,32]{1,0}, f32[32,32,32]{2,1,0}) tuple(reduce, exp)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::GetTupleElement())));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Exp())));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionLoopFusionAndReduce) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0.1 = f32[32,32,32]{2,1,0} parameter(0)
      p1.1 = f32[32,32,32]{2,1,0} parameter(1)
      ROOT add = f32[32,32,32]{2,1,0} add(p0.1, p1.1)
    }

    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      add = f32[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_add
      reduce = f32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
      ROOT root = (f32[32,32]{1,0}, f32[32,32,32]{2,1,0}) tuple(reduce, add)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::GetTupleElement())));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Add())));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionLoopFusionAndReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_select {
      p1.1 = f32[32,32,32]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      broadcast = f32[32,32,32]{2,1,0} broadcast(f32[] c0), dimensions={}
      greater-than = pred[32,32,32]{2,1,0} compare(f32[32,32,32]{2,1,0} p1.1,
        f32[32,32,32]{2,1,0} broadcast), direction=GT
      p0.1 = f32[32,32,32]{2,1,0} parameter(0)
      ROOT select = f32[32,32,32]{2,1,0} select(pred[32,32,32]{2,1,0}
        greater-than, f32[32,32,32]{2,1,0} p0.1, f32[32,32,32]{2,1,0} broadcast)
    }

    fused_reduce {
      p0.2 = f32[32,32,32]{2,1,0} parameter(0)
      c1 = f32[] constant(0)
      r1 = f32[32,32]{1,0} reduce(p0.2, c1), dimensions={2},
        to_apply=scalar_add_computation
      mul = f32[32,32,32]{2,1,0} multiply(p0.2, p0.2)
      r2 = f32[32,32]{1,0} reduce(mul, c1), dimensions={2},
        to_apply=scalar_add_computation
      ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      select = f32[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_select
      fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(select), kind=kInput,
        calls=fused_reduce
      gte0 = f32[32,32]{1,0} get-tuple-element(fusion), index=0
      gte1 = f32[32,32]{1,0} get-tuple-element(fusion), index=1
      ROOT root = (f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32,32]{2,1,0})
        tuple(gte1, gte1, select)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root,
              GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                  m::GetTupleElement(), m::GetTupleElement())));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce(), m::Select())));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionDoNotFuseLoopReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_element_wise {
      p0.1 = f32[2,2,2]{2,1,0} parameter(0)
      p1.1 = f32[2,2,2]{2,1,0} parameter(1)
      ROOT root = f32[2,2,2]{2,1,0} add(p0.1, p1.1)
    }

    fused_reduce {
      p0.2 = f32[2,2,2]{2,1,0} parameter(0)
      mul = f32[2,2,2]{2,1,0} multiply(f32[2,2,2]{2,1,0} p0.2,
        f32[2,2,2]{2,1,0} p0.2)
      broadcast = f32[2,2,2,2]{3,2,1,0} broadcast(mul), dimensions={3,2,1}
      c1 = f32[] constant(0)
      ROOT reduce = f32[2,2]{1,0} reduce(f32[2,2,2,2]{3,2,1,0} broadcast,
        f32[] c1), dimensions={1,3}, to_apply=scalar_add_computation
    }

    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{2,1,0} parameter(1)
      element_wise = f32[2,2,2]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_element_wise
      fusion = f32[2,2]{1,0} fusion(element_wise), kind=kLoop, calls=fused_reduce
      ROOT root = (f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(fusion, element_wise)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest,
       ProducerConsumerFusionFp16LoopFusionAndReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_select {
      p1.1 = f16[32,32,32]{2,1,0} parameter(1)
      c0 = f16[] constant(0)
      broadcast = f16[32,32,32]{2,1,0} broadcast(f16[] c0), dimensions={}
      greater-than = pred[32,32,32]{2,1,0} compare(f16[32,32,32]{2,1,0} p1.1,
        f16[32,32,32]{2,1,0} broadcast), direction=GT
      p0.1 = f16[32,32,32]{2,1,0} parameter(0)
      ROOT select = f16[32,32,32]{2,1,0} select(pred[32,32,32]{2,1,0}
        greater-than, f16[32,32,32]{2,1,0} p0.1, f16[32,32,32]{2,1,0} broadcast)
    }
    fused_reduce {
      p0.2 = f16[32,32,32]{2,1,0} parameter(0)
      convert = f32[32,32,32]{2,1,0} convert(p0.2)
      c1 = f32[] constant(0)
      r1 = f32[32,32]{1,0} reduce(convert, c1), dimensions={2},
        to_apply=scalar_add_computation
      mul = f32[32,32,32]{2,1,0} multiply(convert, convert)
      r2 = f32[32,32]{1,0} reduce(mul, c1), dimensions={2},
        to_apply=scalar_add_computation
      ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(r1, r2)
    }
    ENTRY reduce {
      p0 = f16[32,32,32]{2,1,0} parameter(0)
      p1 = f16[32,32,32]{2,1,0} parameter(1)
      select = f16[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_select
      fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(select), kind=kInput,
        calls=fused_reduce
      gte0 = f32[32,32]{1,0} get-tuple-element(fusion), index=0
      gte1 = f32[32,32]{1,0} get-tuple-element(fusion), index=1
      ROOT root = (f32[32,32]{1,0}, f32[32,32]{1,0}, f16[32,32,32]{2,1,0})
        tuple(gte1, gte1, select)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root,
              GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                  m::GetTupleElement(), m::GetTupleElement())));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce(), m::Select())));
}

TEST_F(MultiOutputFusionTest,
       ProducerConsumerFusionReduceUnfriendlyLoopFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    mixed_input_layouts_computation {
      p0.1 = f16[128,32,32,1024]{3,2,1,0} parameter(0)
      p1.1 = f16[128,1024,32,32]{3,2,1,0} parameter(1)
      transpose = f16[128,32,32,1024]{3,2,1,0} transpose(p1.1), dimensions={0,2,3,1}
      c0 = f16[] constant(0)
      broadcast = f16[128,32,32,1024]{3,2,1,0} broadcast(c0), dimensions={}
      greater-than = pred[128,32,32,1024]{3,2,1,0} compare(transpose, broadcast), direction=GT
      ROOT root = f16[128,32,32,1024]{3,2,1,0} select(greater-than, p0.1, broadcast)
    }
    fused_reduce {
      p0.2 = f16[128,32,32,1024]{3,2,1,0} parameter(0)
      convert = f32[128,32,32,1024]{3,2,1,0} convert(p0.2)
      c0.2 = f32[] constant(0)
      ROOT reduce = f32[1024]{0} reduce(convert, c0.2), dimensions={0,1,2}, to_apply=scalar_add_computation
    }
    ENTRY reduce {
      p0 = f16[128,32,32,1024]{3,2,1,0} parameter(0)
      p1 = f16[128,1024,32,32]{3,2,1,0} parameter(1)
      loop_fusion = f16[128,32,32,1024]{3,2,1,0} fusion(p0, p1), kind=kLoop, calls=mixed_input_layouts_computation
      reduce_fusion = f32[1024]{0} fusion(loop_fusion), kind=kInput, calls=fused_reduce
      ROOT root = (f32[1024]{0}, f16[128,32,32,1024]{3,2,1,0}) tuple(reduce_fusion, loop_fusion)
    })"))
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionAvoidsCycles) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      ROOT add = f32[32,32,32]{2,1,0} add(p0, p1)
    }

    fused_mul {
      p2 = f32[64,64,64]{2,1,0} parameter(0)
      p3 = f32[64,64,64]{2,1,0} parameter(1)
      ROOT multiply = f32[64,64,64]{2,1,0} multiply(p2, p3)
    }

    fused_reduce_1 {
      p4 = f32[32,32,32]{2,1,0} parameter(0)
      p5 = f32[64,64,64]{2,1,0} parameter(1)
      slice = f32[32,32,32]{2,1,0} slice(p5), slice={[0:32], [0:32], [0:32]}
      add = f32[32,32,32]{2,1,0} add(p4, slice)
      c0 = f32[] constant(0)
      ROOT r1 = f32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
    }

    fused_reduce_2 {
      p6 = f32[32,32,32]{2,1,0} parameter(0)
      p7 = f32[64,64,64]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      pad = f32[64,64,64]{2,1,0} pad(p6, c0), padding=16_16x16_16x16_16
      mul = f32[64,64,64]{2,1,0} multiply(pad, p7)
      ROOT r1 = f32[64,64]{1,0} reduce(mul, c0), dimensions={2},
        to_apply=scalar_add_computation
    }

    ENTRY reduce {
      p8 = f32[32,32,32]{2,1,0} parameter(0)
      p9 = f32[64,64,64]{2,1,0} parameter(1)
      // `add` and `mul` can be multi-output fused with `reduce1` and `reduce2`,
      // respectively. However, both isn't possible, because multi-output fusion
      // will introduce an extra dependency from `neg` to `abs` or vice versa.
      // Hence, the second multi-output fusion would introduce a cycle.
      add = f32[32,32,32]{2,1,0} fusion(p8, p8), kind=kLoop, calls=fused_add
      mul = f32[64,64,64]{2,1,0} fusion(p9, p9), kind=kLoop, calls=fused_mul

      reduce1 = f32[32,32]{1,0} fusion(add, mul), kind=kInput,
          calls=fused_reduce_1
      reduce2 = f32[64,64]{1,0} fusion(add, mul), kind=kInput,
          calls=fused_reduce_2
      ROOT root = (f32[32,32,32]{2,1,0}, f32[32,32]{1,0}, f32[64,64]{1,0},
                   f32[64,64,64]{2,1,0}) tuple(add, reduce1, reduce2, mul)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  EXPECT_EQ(1, CountMultiOutputFusions(module.get()));
}

TEST_F(MultiOutputFusionTest, PreferFuseProducerIntoFusionConsumer) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      ROOT add = f32[32,32,32]{2,1,0} add(p0, p1)
    }
    fused_reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[64,64,64]{2,1,0} parameter(1)
      slice = f32[32,32,32]{2,1,0} slice(p1), slice={[0:32], [0:32], [0:32]}
      add = f32[32,32,32]{2,1,0} add(p0, slice)
      c0 = f32[] constant(0)
      ROOT r1 = f32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
    }
    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[64,64,64]{2,1,0} parameter(1)
      add = f32[32,32,32]{2,1,0} fusion(p0, p0), kind=kLoop, calls=fused_add
      c0 = f32[] constant(0)
      reduce2 = f32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
      reduce = f32[32,32]{1,0} fusion(add, p1), kind=kInput, calls=fused_reduce
      ROOT root = (f32[32,32,32]{2,1,0}, f32[32,32]{1,0}, f32[32,32]{1,0})
                  tuple(add, reduce, reduce2)
    })"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  int multi_output_fusion_count = 0;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instr : computation->instructions()) {
      if (instr->IsMultiOutputFusion()) {
        multi_output_fusion_count++;
      }
    }
  }
  EXPECT_EQ(1, multi_output_fusion_count);
}

// Check that we limit the number of operands to fusions we create.
TEST_F(MultiOutputFusionTest, AvoidsLargeFusion) {
  constexpr int64_t kNumParams = 200;
  ASSERT_GT(kNumParams, MaxOperandsAndOutputsPerFusion());

  // Compute
  //   p0 * p1,
  //   p0 * p1 + p1 * p2
  //   p0 * p1 + p1 * p2 + p2 * p3
  //   ...
  // where each of the (pi * pj)'s is represented as a fusion node so that
  // multi-output fusion will pay attention to it.
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {10, 100});

  std::vector<HloInstruction*> params;
  for (int64_t i = 0; i < kNumParams; ++i) {
    params.push_back(
        b.AddInstruction(HloInstruction::CreateParameter(i, shape, "p")));
  }

  // Creates a fusion node that calculates x*y.
  auto make_fusion = [&](HloInstruction* x, HloInstruction* y) {
    HloComputation::Builder sub_builder("subcomp");
    auto* p0 = sub_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "p"));
    auto* p1 = sub_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "p"));
    sub_builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, p0, p1));
    HloComputation* subcomp =
        module->AddEmbeddedComputation(sub_builder.Build());
    return HloInstruction::CreateFusion(
        shape, HloInstruction::FusionKind::kLoop, {x, y}, subcomp);
  };

  auto* sum = b.AddInstruction(make_fusion(params[0], params[1]));
  for (int64_t i = 2; i < kNumParams; ++i) {
    sum = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, sum,
        b.AddInstruction(make_fusion(params[i - 1], params[i]))));
  }
  auto computation = module->AddEntryComputation(b.Build());
  EXPECT_TRUE(mof_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  for (const HloInstruction* instr : computation->instructions()) {
    EXPECT_LE(instr->operand_count() + ShapeUtil::SubshapeCount(instr->shape()),
              MaxOperandsAndOutputsPerFusion())
        << instr->ToString();
  }
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDUS) {
  auto module = ParseAndReturnVerifiedModule(R"(HloModule dus_mof
    fusion.1 {
      p.0 = f16[50,96,1024]{2,1,0} parameter(0)
      p.1 = f16[1,96,1024]{2,1,0} parameter(1)
      c.0 = s32[3]{0} constant({0, 0, 0})
      ROOT %dynamic-update-slice = f16[50,96,1024]{2,1,0} dynamic-update-slice(p.0, p.1, c.0)
    }

    fusion.2 {
      p.0 = f16[50,96,1024]{2,1,0} parameter(0)
      p.1 = f16[1,96,1024]{2,1,0} parameter(1)
      c.0 = s32[3]{0} constant({0, 0, 0})
      ROOT %dynamic-update-slice = f16[50,96,1024]{2,1,0} dynamic-update-slice(p.0, p.1, c.0)
    }

    ENTRY entry {
      p.00 = f16[50,96,1024]{2,1,0} parameter(0)
      p.01 = f16[50,96,1024]{2,1,0} parameter(1)
      p.1 = f16[1,96,1024]{2,1,0} parameter(2)

      f1 = f16[50,96,1024] fusion(p.00, p.1), kind=kLoop, calls=fusion.1
      f2 = f16[50,96,1024] fusion(p.01, p.1), kind=kLoop, calls=fusion.2
      ROOT tuple = (f16[50,96,1024],f16[50,96,1024]) tuple(f1, f2)
    })")
                    .value();
  ASSERT_FALSE(mof_.Run(module.get()).value());
}

// Check that we don't fuse too many reductions together.
TEST_F(MultiOutputFusionTest, SharedMemoryBudget) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation0 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation1 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation2 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation3 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation4 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation5 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation6 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation7 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation8 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    fused_computation9 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={0},
        to_apply=scalar_add_computation
    }
    ENTRY computation {
      zero = f32[] constant(0)
      param0 = f32[64,64] parameter(0)
      param1 = f32[64,64] parameter(1)
      param2 = f32[64,64] parameter(2)
      param3 = f32[64,64] parameter(3)
      param4 = f32[64,64] parameter(4)
      param5 = f32[64,64] parameter(5)
      param6 = f32[64,64] parameter(6)
      param7 = f32[64,64] parameter(7)
      param8 = f32[64,64] parameter(8)
      param9 = f32[64,64] parameter(9)
      out0 = f32[64] fusion(param0, param1, zero), kind=kInput, calls=fused_computation0
      out1 = f32[64] fusion(param1, param2, zero), kind=kInput, calls=fused_computation1
      out2 = f32[64] fusion(param2, param3, zero), kind=kInput, calls=fused_computation2
      out3 = f32[64] fusion(param3, param4, zero), kind=kInput, calls=fused_computation3
      out4 = f32[64] fusion(param4, param5, zero), kind=kInput, calls=fused_computation4
      out5 = f32[64] fusion(param5, param6, zero), kind=kInput, calls=fused_computation5
      out6 = f32[64] fusion(param6, param7, zero), kind=kInput, calls=fused_computation6
      out7 = f32[64] fusion(param7, param8, zero), kind=kInput, calls=fused_computation7
      out8 = f32[64] fusion(param8, param9, zero), kind=kInput, calls=fused_computation8
      out9 = f32[64] fusion(param9, param0, zero), kind=kInput, calls=fused_computation9
      ROOT out = (f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64]) tuple(f32[64] out0, f32[64] out1, f32[64] out2, f32[64] out3, f32[64] out4, f32[64] out5, f32[64] out6, f32[64] out7, f32[64] out8, f32[64] out9)
    }
  )"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());

  EXPECT_EQ(5, CountMultiOutputFusions(module.get()));
}

TEST_F(MultiOutputFusionTest, DoNotGroupTooManyReductions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation0 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation1 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation2 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation3 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation4 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation5 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation6 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation7 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation8 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    fused_computation9 {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      p2 = f32[] parameter(2)
      add = f32[64,64] add(p0, p1)
      ROOT reduce = f32[64] reduce(f32[64,64] add, f32[] p2), dimensions={1},
        to_apply=scalar_add_computation
    }
    ENTRY computation {
      zero = f32[] constant(0)
      param0 = f32[64,64] parameter(0)
      param1 = f32[64,64] parameter(1)
      param2 = f32[64,64] parameter(2)
      param3 = f32[64,64] parameter(3)
      param4 = f32[64,64] parameter(4)
      param5 = f32[64,64] parameter(5)
      param6 = f32[64,64] parameter(6)
      param7 = f32[64,64] parameter(7)
      param8 = f32[64,64] parameter(8)
      param9 = f32[64,64] parameter(9)
      out0 = f32[64] fusion(param0, param1, zero), kind=kInput, calls=fused_computation0
      out1 = f32[64] fusion(param1, param2, zero), kind=kInput, calls=fused_computation1
      out2 = f32[64] fusion(param2, param3, zero), kind=kInput, calls=fused_computation2
      out3 = f32[64] fusion(param3, param4, zero), kind=kInput, calls=fused_computation3
      out4 = f32[64] fusion(param4, param5, zero), kind=kInput, calls=fused_computation4
      out5 = f32[64] fusion(param5, param6, zero), kind=kInput, calls=fused_computation5
      out6 = f32[64] fusion(param6, param7, zero), kind=kInput, calls=fused_computation6
      out7 = f32[64] fusion(param7, param8, zero), kind=kInput, calls=fused_computation7
      out8 = f32[64] fusion(param8, param9, zero), kind=kInput, calls=fused_computation8
      out9 = f32[64] fusion(param9, param0, zero), kind=kInput, calls=fused_computation9
      ROOT out = (f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64]) tuple(f32[64] out0, f32[64] out1, f32[64] out2, f32[64] out3, f32[64] out4, f32[64] out5, f32[64] out6, f32[64] out7, f32[64] out8, f32[64] out9)
    }
  )"))
                    .value();
  ASSERT_TRUE(mof_.Run(module.get()).value());

  EXPECT_EQ(2, CountMultiOutputFusions(module.get()));
}

TEST_F(MultiOutputFusionTest, NoFusionToAvoidUsingTooMuchSharedMemory) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule xla_computation_update_step.10931

%scalar_add_computation.1 (scalar_lhs.1: f64[], scalar_rhs.1: f64[]) -> f64[] {
  %scalar_lhs.1 = f64[] parameter(0)
  %scalar_rhs.1 = f64[] parameter(1)
  ROOT %add.1257 = f64[] add(f64[] %scalar_lhs.1, f64[] %scalar_rhs.1)
}

%fused_computation.1 (param_0.8: f64[64,64], param_1.11: f64[64,64], param_2.9: f64[64,64]) -> (f64[64], f64[64]) {
  %param_0.8 = f64[64,64]{1,0} parameter(0)
  %param_1.11 = f64[64,64]{1,0} parameter(1)
  %multiply.2 = f64[64,64]{1,0} multiply(f64[64,64]{1,0} %param_0.8, f64[64,64]{1,0} %param_1.11)
  %constant_5217.3 = f64[] constant(0)
  %broadcast.1 = f64[64,64]{1,0} broadcast(f64[] %constant_5217.3), dimensions={}
  %multiply.0 = f64[64,64]{1,0} multiply(f64[64,64]{1,0} %multiply.2, f64[64,64]{1,0} %broadcast.1)
  %reduce.0 = f64[64]{0} reduce(f64[64,64]{1,0} %multiply.0, f64[] %constant_5217.3), dimensions={0}, to_apply=%scalar_add_computation.1
  %param_2.9 = f64[64,64]{1,0} parameter(2)
  %multiply.1514.clone.0.clone.1 = f64[64,64]{1,0} multiply(f64[64,64]{1,0} %param_2.9, f64[64,64]{1,0} %param_1.11)
  %constant_5217.1.clone.1 = f64[] constant(0)
  %broadcast.0.clone.1 = f64[64,64]{1,0} broadcast(f64[] %constant_5217.1.clone.1), dimensions={}
  %multiply.1341.clone.0.clone.1 = f64[64,64]{1,0} multiply(f64[64,64]{1,0} %multiply.1514.clone.0.clone.1, f64[64,64]{1,0} %broadcast.0.clone.1)
  %reduce.630.clone.0.clone.1 = f64[64]{0} reduce(f64[64,64]{1,0} %multiply.1341.clone.0.clone.1, f64[] %constant_5217.1.clone.1), dimensions={0}, to_apply=%scalar_add_computation.1
  ROOT %tuple = (f64[64]{0}, f64[64]{0}) tuple(f64[64]{0} %reduce.0, f64[64]{0} %reduce.630.clone.0.clone.1)
}

%primitive_computation_add__1.6426 (parameter.6427: f64[], parameter.6428: f64[]) -> f64[] {
  %parameter.6427 = f64[] parameter(0)
  %parameter.6428 = f64[] parameter(1)
  ROOT %add.6429 = f64[] add(f64[] %parameter.6427, f64[] %parameter.6428)
}

%fused_computation.2 (param_0.7: f64[64,64], param_1.9: f64[64,64]) -> f64[64] {
  %param_0.7 = f64[64,64]{1,0} parameter(0)
  %param_1.9 = f64[64,64]{1,0} parameter(1)
  %multiply.1 = f64[64,64]{1,0} multiply(f64[64,64]{1,0} %param_0.7, f64[64,64]{1,0} %param_1.9)
  %constant_5217.2 = f64[] constant(0)
  ROOT %reduce.740.clone.0 = f64[64]{0} reduce(f64[64,64]{1,0} %multiply.1, f64[] %constant_5217.2), dimensions={0}, to_apply=%primitive_computation_add__1.6426
}

ENTRY %reproducer (param_0.1090: f64[64,64], param_1.1377: f64[64,64], param_2.1948: f64[64,64]) -> (f64[64], f64[64], f64[64]) {
  %param_0.1090 = f64[64,64]{1,0} parameter(0)
  %param_1.1377 = f64[64,64]{1,0} parameter(1)
  %param_2.1948 = f64[64,64]{1,0} parameter(2)
  %fusion.1 = (f64[64]{0}, f64[64]{0}) fusion(f64[64,64]{1,0} %param_0.1090, f64[64,64]{1,0} %param_1.1377, f64[64,64]{1,0} %param_2.1948), kind=kInput, calls=%fused_computation.1
  %get-tuple-element = f64[64]{0} get-tuple-element((f64[64]{0}, f64[64]{0}) %fusion.1), index=0
  %fusion.2 = f64[64]{0} fusion(f64[64,64]{1,0} %param_0.1090, f64[64,64]{1,0} %param_1.1377), kind=kInput, calls=%fused_computation.2
  %get-tuple-element.1 = f64[64]{0} get-tuple-element((f64[64]{0}, f64[64]{0}) %fusion.1), index=1
  ROOT %tuple.428 = (f64[64]{0}, f64[64]{0}, f64[64]{0}) tuple(f64[64]{0} %get-tuple-element, f64[64]{0} %fusion.2, f64[64]{0} %get-tuple-element.1)
}
  )")
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, NoProblemWithCodeDuplication) {
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
  fusion.2 = f32[2,20,256]{2,0,1} fusion(param_0.0, param_1.0, param_2.0, param_3.0, fusion.1, param_5.0, param_10, param_7.0, param_8.0, param_9.0), kind=kLoop, calls=fused_computation.2
  ROOT root = (f32[2,20,256]{2,0,1}, f32[2,20,256]{2,0,1}) tuple(fusion.1, fusion.2)
}
  )")
                    .value();
  EXPECT_TRUE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, DoNotFuseRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule module

no_op {
  arg_empty_tuple = () parameter(0)
  ROOT tuple = () tuple()
}

fused_computation {
  param_0 = f32[] parameter(0)
  ROOT convert = s32[] convert(param_0)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  fusion = s32[] fusion(param_0), kind=kLoop, calls=fused_computation
  tuple = () tuple()
  conditional = () conditional(fusion, tuple, tuple), branch_computations={no_op, no_op}
  constant = f32[] constant(1)
  ROOT root = f32[] add(param_0, constant)
}
  )")
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, CostBasedNoMerge) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

region_3.63 {
  Arg_0.64 = f32[] parameter(0)
  Arg_1.65 = f32[] parameter(1)
  ROOT add.66 = f32[] add(Arg_0.64, Arg_1.65)
}

fused_computation.29 {
  param_0.161 = f32[5,32,32,1]{3,2,1,0} parameter(0)
  multiply.208 = f32[5,32,32,1]{3,2,1,0} multiply(param_0.161, param_0.161)
  bitcast.67 = f32[5,32,32]{2,1,0} bitcast(multiply.208)
  constant.265 = f32[] constant(0)
  reduce-window.81 = f32[5,30,31]{2,1,0} reduce-window(bitcast.67, constant.265), window={size=1x3x2}, to_apply=region_3.63
  constant.264 = f32[] constant(0.166666672)
  broadcast.204 = f32[5,30,31]{2,1,0} broadcast(constant.264), dimensions={}
  multiply.205 = f32[5,30,31]{2,1,0} multiply(reduce-window.81, broadcast.204)
  constant.263 = f32[] constant(0)
  reduce-window.80 = f32[5,30,31]{2,1,0} reduce-window(multiply.205, constant.263), window={size=1x2x3 pad=0_0x0_1x1_1}, to_apply=region_3.63
  constant.262 = f32[] constant(0.0138888899)
  broadcast.201 = f32[5,30,31]{2,1,0} broadcast(constant.262), dimensions={}
  multiply.204 = f32[5,30,31]{2,1,0} multiply(reduce-window.80, broadcast.201)
  constant.261 = f32[] constant(0)
  reduce-window.78 = f32[5,30,31]{2,1,0} reduce-window(multiply.204, constant.261), window={size=1x1x2 pad=0_0x0_0x0_1}, to_apply=region_3.63
  constant.113 = f32[] constant(0.5)
  broadcast.137 = f32[5,30,31]{2,1,0} broadcast(constant.113), dimensions={}
  multiply.125 = f32[5,30,31]{2,1,0} multiply(reduce-window.78, broadcast.137)
  constant.114 = f32[] constant(0)
  ROOT reduce-window.17 = f32[5,30,31]{2,1,0} reduce-window(multiply.125, constant.114), window={size=1x2x1 pad=0_0x0_1x0_0}, to_apply=region_3.63
}

fused_computation.15 {
  constant.108 = f32[] constant(0.5)
  broadcast.105 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.108), dimensions={}
  param_3.126 = f32[5,30,31]{2,1,0} parameter(3)
  constant.295 = f32[] constant(0.25)
  broadcast.234 = f32[5,30,31]{2,1,0} broadcast(constant.295), dimensions={}
  multiply.242 = f32[5,30,31]{2,1,0} multiply(param_3.126, broadcast.234)
  broadcast.233 = f32[5,5,30,31]{3,2,1,0} broadcast(multiply.242), dimensions={0,2,3}
  param_2.154 = f32[5,30,31]{2,1,0} parameter(2)
  multiply.241 = f32[5,30,31]{2,1,0} multiply(param_2.154, broadcast.234)
  broadcast.232 = f32[5,5,30,31]{3,2,1,0} broadcast(multiply.241), dimensions={1,2,3}
  multiply.240 = f32[5,5,30,31]{3,2,1,0} multiply(broadcast.233, broadcast.232)
  param_1.188 = f32[5,5,30,31]{3,2,1,0} parameter(1)
  constant.294 = f32[] constant(0.159154937)
  broadcast.231 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.294), dimensions={}
  multiply.239 = f32[5,5,30,31]{3,2,1,0} multiply(param_1.188, broadcast.231)
  param_0.164 = f32[5,5,30,31]{3,2,1,0} parameter(0)
  add.19 = f32[5,5,30,31]{3,2,1,0} add(multiply.239, param_0.164)
  constant.293 = f32[] constant(0)
  reduce-window.90 = f32[5,5,30,31]{3,2,1,0} reduce-window(add.19, constant.293), window={size=1x1x1x2 pad=0_0x0_0x0_0x0_1}, to_apply=region_3.63
  constant.292 = f32[] constant(0.5)
  broadcast.230 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.292), dimensions={}
  multiply.238 = f32[5,5,30,31]{3,2,1,0} multiply(reduce-window.90, broadcast.230)
  constant.291 = f32[] constant(0)
  reduce-window.89 = f32[5,5,30,31]{3,2,1,0} reduce-window(multiply.238, constant.291), window={size=1x1x2x1 pad=0_0x0_0x0_1x0_0}, to_apply=region_3.63
  constant.290 = f32[] constant(0.25)
  broadcast.229 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.290), dimensions={}
  multiply.237 = f32[5,5,30,31]{3,2,1,0} multiply(reduce-window.89, broadcast.229)
  multiply.236 = f32[5,5,30,31]{3,2,1,0} multiply(multiply.237, multiply.237)
  subtract.10 = f32[5,5,30,31]{3,2,1,0} subtract(multiply.240, multiply.236)
  constant.289 = f32[] constant(0)
  broadcast.228 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.289), dimensions={}
  maximum.6 = f32[5,5,30,31]{3,2,1,0} maximum(subtract.10, broadcast.228)
  sqrt.6 = f32[5,5,30,31]{3,2,1,0} sqrt(maximum.6)
  constant.110 = f32[] constant(0)
  broadcast.107 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.110), dimensions={}
  compare.4 = pred[5,5,30,31]{3,2,1,0} compare(sqrt.6, broadcast.107), direction=EQ
  constant.243 = f32[] constant(0.159154937)
  broadcast.193 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.243), dimensions={}
  multiply.194 = f32[5,5,30,31]{3,2,1,0} multiply(param_1.188, broadcast.193)
  add.15 = f32[5,5,30,31]{3,2,1,0} add(multiply.194, param_0.164)
  constant.242 = f32[] constant(0)
  reduce-window.66 = f32[5,5,30,31]{3,2,1,0} reduce-window(add.15, constant.242), window={size=1x1x1x2 pad=0_0x0_0x0_0x0_1}, to_apply=region_3.63
  constant.241 = f32[] constant(0.5)
  broadcast.192 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.241), dimensions={}
  multiply.193 = f32[5,5,30,31]{3,2,1,0} multiply(reduce-window.66, broadcast.192)
  constant.240 = f32[] constant(0)
  reduce-window.65 = f32[5,5,30,31]{3,2,1,0} reduce-window(multiply.193, constant.240), window={size=1x1x2x1 pad=0_0x0_0x0_1x0_0}, to_apply=region_3.63
  constant.239 = f32[] constant(0.25)
  broadcast.191 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.239), dimensions={}
  multiply.192 = f32[5,5,30,31]{3,2,1,0} multiply(reduce-window.65, broadcast.191)
  compare.3 = pred[5,5,30,31]{3,2,1,0} compare(multiply.192, broadcast.107), direction=EQ
  and.1 = pred[5,5,30,31]{3,2,1,0} and(compare.4, compare.3)
  constant.109 = f32[] constant(1.57079637)
  broadcast.104 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.109), dimensions={}
  atan2.1 = f32[5,5,30,31]{3,2,1,0} atan2(sqrt.6, multiply.192)
  select.4 = f32[5,5,30,31]{3,2,1,0} select(and.1, broadcast.104, atan2.1)
  constant.107 = f32[] constant(0.159154937)
  broadcast.106 = f32[5,5,30,31]{3,2,1,0} broadcast(constant.107), dimensions={}
  multiply.100 = f32[5,5,30,31]{3,2,1,0} multiply(select.4, broadcast.106)
  ROOT subtract.3 = f32[5,5,30,31]{3,2,1,0} subtract(broadcast.105, multiply.100)
}

fused_computation.4 {
  param_0.172 = f32[5,30,31]{2,1,0} parameter(0)
  constant.315 = f32[] constant(0.125)
  broadcast.242 = f32[5,30,31]{2,1,0} broadcast(constant.315), dimensions={}
  multiply.250 = f32[5,30,31]{2,1,0} multiply(param_0.172, broadcast.242)
  constant.314 = f32[] constant(0)
  reduce-window.100 = f32[5,30,31]{2,1,0} reduce-window(multiply.250, constant.314), window={size=1x3x3 pad=0_0x1_1x1_1}, to_apply=region_3.63
  constant.79 = f32[] constant(0.055555556)
  broadcast.85 = f32[5,30,31]{2,1,0} broadcast(constant.79), dimensions={}
  multiply.80 = f32[5,30,31]{2,1,0} multiply(reduce-window.100, broadcast.85)
  constant.81 = f32[] constant(0)
  reduce-window.1 = f32[5,30,31]{2,1,0} reduce-window(multiply.80, constant.81), window={size=1x3x3 pad=0_0x1_1x1_1}, to_apply=region_3.63
  constant.80 = f32[] constant(0.111111112)
  broadcast.86 = f32[5,30,31]{2,1,0} broadcast(constant.80), dimensions={}
  multiply.79 = f32[5,30,31]{2,1,0} multiply(reduce-window.1, broadcast.86)
  bitcast.26 = f32[5,930]{1,0} bitcast(multiply.79)
  ROOT reduce.8 = f32[5]{0} reduce(bitcast.26, constant.81), dimensions={1}, to_apply=region_3.63
}

ENTRY e {
  Arg_0.1 = f32[5,32,32,1]{3,2,1,0} parameter(0)
  p1 = f32[5,5,30,31]{3,2,1,0} parameter(1)
  p2 = f32[5,5,30,31]{3,2,1,0} parameter(2)
  p3 = f32[5,30,31]{2,1,0} parameter(3)
  fusion.29 = f32[5,30,31]{2,1,0} fusion(Arg_0.1), kind=kLoop, calls=fused_computation.29
  fusion.15 = f32[5,5,30,31]{3,2,1,0} fusion(p2, p1, p3, fusion.29), kind=kLoop, calls=fused_computation.15
  ROOT fusion.4 = f32[5]{0} fusion(fusion.29), kind=kInput, calls=fused_computation.4
})")
                    .value();
  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, NoOverlappingRead) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation_1 {
      p0.1 = f32[100,200]{1,0} parameter(0)
      slice.0 = f32[50,100]{1,0} slice(p0.1), slice={[0:50],[0:100]}
      mul = f32[50,100]{1,0} multiply(slice.0, slice.0)
      exp = f32[50,100]{1,0} exponential(slice.0)
      ROOT tuple = (f32[50,100]{1,0}, f32[50,100]{1,0}) tuple(mul, exp)
    }

    fused_computation_2 {
      p0.2 = f32[100,200]{1,0} parameter(0)
      slice.1 = f32[50,100]{1,0} slice(p0.2), slice={[0:50],[100:200]}
      const.2 = f32[] constant(0)
      broadcast = f32[50,100]{1,0} broadcast(const.2), dimensions={}
      ROOT add = f32[50,100]{1,0} add(slice.1, broadcast)
    }

    ENTRY entry {
      p0 = f32[100,200]{1,0} parameter(0)
      fusion.1 = (f32[50,100]{1,0}, f32[50,100]{1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      gte0 = f32[50,100]{1,0} get-tuple-element(fusion.1), index=0
      gte1 = f32[50,100]{1,0} get-tuple-element(fusion.1), index=1
      fusion.2 = f32[50,100]{1,0} fusion(p0), kind=kLoop,
        calls=fused_computation_2
      ROOT root = (f32[50,100]{1,0}, f32[50,100]{1,0}, f32[50,100]{1,0})
        tuple(gte0, gte1, fusion.2)
    })")
                    .value();

  EXPECT_FALSE(mof_.Run(module.get()).value());
}

TEST_F(MultiOutputFusionTest, OverlappingRead) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation_1 {
      p0.1 = f32[100,200]{1,0} parameter(0)
      slice.0 = f32[50,100]{1,0} slice(p0.1), slice={[0:50],[50:150]}
      mul = f32[50,100]{1,0} multiply(slice.0, slice.0)
      exp = f32[50,100]{1,0} exponential(slice.0)
      ROOT tuple = (f32[50,100]{1,0}, f32[50,100]{1,0}) tuple(mul, exp)
    }

    fused_computation_2 {
      p0.2 = f32[100,200]{1,0} parameter(0)
      slice.1 = f32[50,100]{1,0} slice(p0.2), slice={[30:80],[20:120]}
      const.2 = f32[] constant(0)
      broadcast = f32[50,100]{1,0} broadcast(const.2), dimensions={}
      ROOT add = f32[50,100]{1,0} add(slice.1, broadcast)
    }

    ENTRY entry {
      p0 = f32[100,200]{1,0} parameter(0)
      fusion.1 = (f32[50,100]{1,0}, f32[50,100]{1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      gte0 = f32[50,100]{1,0} get-tuple-element(fusion.1), index=0
      gte1 = f32[50,100]{1,0} get-tuple-element(fusion.1), index=1
      fusion.2 = f32[50,100]{1,0} fusion(p0), kind=kLoop,
        calls=fused_computation_2
      ROOT root = (f32[50,100]{1,0}, f32[50,100]{1,0}, f32[50,100]{1,0})
        tuple(gte0, gte1, fusion.2)
    })")
                    .value();

  EXPECT_TRUE(mof_.Run(module.get()).value());
}

class TransposeMultiOutputFusionTest : public MultiOutputFusionTest {
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        MultiOutputFusionTest::GetDebugOptionsForTest();
    return debug_options;
  }
};

TEST_F(TransposeMultiOutputFusionTest, MultipleTransposes) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = f32[16,32]{1,0} parameter(0)
  s.1 = f32[16,32]{1,0} sqrt(param_0.1)
  ROOT t.1 = f32[32,16]{1,0} transpose(s.1), dimensions={1,0}
}

ENTRY main {
  p = f32[16,32]{1,0} parameter(0)
  fusion = f32[32,16]{1,0} fusion(p), kind=kInput, calls=fused_computation
  t1 = f32[32,16]{1,0} transpose(p), dimensions={1,0}
  ROOT t = (f32[32,16]{1,0}, f32[32,16]{1,0}) tuple(fusion, t1)
}
  )";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation (param_0.1: f32[16,32]) -> (f32[32,16], f32[32,16]) {
// CHECK-NEXT:   [[param_0_1_0:%[^ ]+]] = f32[16,32]{1,0} parameter(0)
// CHECK-NEXT:   [[s_1_1:%[^ ]+]] = f32[16,32]{1,0} sqrt([[param_0_1_0]])
// CHECK-NEXT:   [[c_1_2:%[^ ]+]] = f32[32,16]{1,0} transpose([[s_1_1]]), dimensions={1,0}
// CHECK-NEXT:   [[c1_1_3:%[^ ]+]] = f32[32,16]{1,0} transpose([[param_0_1_0]]), dimensions={1,0}
// CHECK-NEXT:   ROOT [[tuple_4:%[^ ]+]] = (f32[32,16]{1,0}, f32[32,16]{1,0}) tuple([[c_1_2]], [[c1_1_3]])
// CHECK-NEXT: }

// CHECK: [[fusion_0:%[^ ]+]] = (f32[32,16]{1,0}, f32[32,16]{1,0}) fusion([[p_1:%[^ ]+]]), kind=kInput, calls=[[fused_computation_2:%[^ ]+]]
)");
}

TEST_F(TransposeMultiOutputFusionTest, MultipleTransposesDifferentTypes) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = f16[16,32]{1,0} parameter(0)
  s.1 = s16[16,32]{1,0} convert(param_0.1)
  ROOT t.1 = s16[32,16]{1,0} transpose(s.1), dimensions={1,0}
}

ENTRY main {
  p = f16[16,32]{1,0} parameter(0)
  fusion = s16[32,16]{1,0} fusion(p), kind=kInput, calls=fused_computation
  t1 = f16[32,16]{1,0} transpose(p), dimensions={1,0}
  ROOT t = (s16[32,16]{1,0}, f16[32,16]{1,0}) tuple(fusion, t1)
}
  )";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation (param_0.1: f16[16,32]) -> (s16[32,16], f16[32,16]) {
// CHECK-NEXT:   [[param_0_1_0:%[^ ]+]] = f16[16,32]{1,0} parameter(0)
// CHECK-NEXT:   [[s_1_1:%[^ ]+]] = s16[16,32]{1,0} convert([[param_0_1_0]])
// CHECK-NEXT:   [[c_1_2:%[^ ]+]] = s16[32,16]{1,0} transpose([[s_1_1]]), dimensions={1,0}
// CHECK-NEXT:   [[c1_1_3:%[^ ]+]] = f16[32,16]{1,0} transpose([[param_0_1_0]]), dimensions={1,0}
// CHECK-NEXT:   ROOT [[tuple_4:%[^ ]+]] = (s16[32,16]{1,0}, f16[32,16]{1,0}) tuple([[c_1_2]], [[c1_1_3]])
// CHECK:   [[fusion_5:%[^ ]+]] = (s16[32,16]{1,0}, f16[32,16]{1,0}) fusion([[p_6:%[^ ]+]]), kind=kInput, calls=[[fused_computation_7:%[^ ]+]]
)");
}

// Do not group transpose and reduction.
TEST_F(TransposeMultiOutputFusionTest, TiledReduceTranspose) {
  const char* hlo = R"(
HloModule module

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = add(lhs, rhs)
}

fused_computation {
  param_0.1 = f32[16,32]{1,0} parameter(0)
  s.1 = f32[16,32]{1,0} sqrt(param_0.1)
  ROOT t.1 = f32[32,16]{1,0} transpose(s.1), dimensions={1,0}
}

ENTRY main {
  p = f32[16,32]{1,0} parameter(0)
  fusion = f32[32,16]{1,0} fusion(p), kind=kInput, calls=fused_computation
  z = f32[] constant(0)
  r1 = f32[32]{0} reduce(p, z), dimensions={0}, to_apply=add
  ROOT t = (f32[32,16]{1,0}, f32[32]{0}) tuple(fusion, r1)
}
  )";

  CheckMultiOutputFusion(hlo, std::nullopt);
}

// Do not group incompatible transposes.
TEST_F(TransposeMultiOutputFusionTest, IncompatibleTransposes) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = f32[18,16,32]{2,1,0} parameter(0)
  param_1.1 = f32[32,16,18]{2,1,0} parameter(1)
  s.1 = f32[18,16,32]{2,1,0} sqrt(param_0.1)
  t.1 = f32[32,16,18]{2,1,0} transpose(s.1), dimensions={2,1,0}
  sub.1 = f32[32,16,18]{2,1,0} subtract(t.1, param_1.1)
  exp.1 = f32[32,16,18]{2,1,0} exponential(sub.1)
  ROOT add.1 = f32[32,16,18]{2,1,0} add(exp.1, exp.1)
}

fused_computation.2 {
  param_0.2 = f32[18,16,32]{2,1,0} parameter(0)
  s.2 = f32[18,16,32]{2,1,0} sqrt(param_0.2)
  ROOT t.2 = f32[18,32,16]{2,1,0} transpose(s.2), dimensions={0,2,1}
}

ENTRY main {
  p = f32[18,16,32]{2,1,0} parameter(0)
  p2 = f32[32,16,18]{2,1,0} parameter(1)
  fusion = f32[32,16,18]{2,1,0} fusion(p, p2), kind=kLoop, calls=fused_computation
  fusion2 = f32[18,32,16]{2,1,0} fusion(p), kind=kInput, calls=fused_computation.2
  ROOT t = (f32[32,16,18]{2,1,0}, f32[18,32,16]{2,1,0}) tuple(fusion, fusion2)
}
  )";

  CheckMultiOutputFusion(hlo, std::nullopt);
}

// A variation of the test above, where no CSE was run.
TEST_F(TransposeMultiOutputFusionTest, TransposesNoCSE) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = f32[18,16,32]{2,1,0} parameter(0)
  param_1.1 = f32[32,16,18]{2,1,0} parameter(1)
  s.1 = f32[18,16,32]{2,1,0} sqrt(param_0.1)
  t.1 = f32[32,16,18]{2,1,0} transpose(s.1), dimensions={2,1,0}
  sub.1 = f32[32,16,18]{2,1,0} subtract(t.1, param_1.1)
  exp.1 = f32[32,16,18]{2,1,0} exponential(sub.1)
  exp.2 = f32[32,16,18]{2,1,0} exponential(sub.1)
  ROOT add.1 = f32[32,16,18]{2,1,0} add(exp.1, exp.2)
}

fused_computation.2 {
  param_0.2 = f32[18,16,32]{2,1,0} parameter(0)
  s.2 = f32[18,16,32]{2,1,0} sqrt(param_0.2)
  ROOT t.2 = f32[18,32,16]{2,1,0} transpose(s.2), dimensions={0,2,1}
}

ENTRY main {
  p = f32[18,16,32]{2,1,0} parameter(0)
  p2 = f32[32,16,18]{2,1,0} parameter(1)
  fusion = f32[32,16,18]{2,1,0} fusion(p, p2), kind=kLoop, calls=fused_computation
  fusion2 = f32[18,32,16]{2,1,0} fusion(p), kind=kInput, calls=fused_computation.2
  ROOT t = (f32[32,16,18]{2,1,0}, f32[18,32,16]{2,1,0}) tuple(fusion, fusion2)
}
  )";

  CheckMultiOutputFusion(hlo, std::nullopt);
}

TEST_F(TransposeMultiOutputFusionTest, TransposeAndInput) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = f32[16,32]{1,0} parameter(0)
  s.1 = f32[16,32]{1,0} sqrt(param_0.1)
  ROOT t.1 = f32[32,16]{1,0} transpose(s.1), dimensions={1,0}
}

ENTRY main {
  p = f32[16,32]{1,0} parameter(0)
  fusion = f32[32,16]{1,0} fusion(p), kind=kInput, calls=fused_computation
  c1 = f32[16,32]{1,0} exponential(p)
  ROOT t = (f32[32,16]{1,0}, f32[16,32]{1,0}) tuple(fusion, c1)
}
  )";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation (param_0.1: f32[16,32]) -> (f32[32,16], f32[16,32]) {
// CHECK-NEXT:   [[param_0_1_0:%[^ ]+]] = f32[16,32]{1,0} parameter(0)
// CHECK-NEXT:   [[s_1_1:%[^ ]+]] = f32[16,32]{1,0} sqrt([[param_0_1_0]])
// CHECK-NEXT:   [[c_1_2:%[^ ]+]] = f32[32,16]{1,0} transpose([[s_1_1]]), dimensions={1,0}
// CHECK-NEXT:   [[c1_1_3:%[^ ]+]] = f32[16,32]{1,0} exponential([[param_0_1_0]])
// CHECK-NEXT:   ROOT [[tuple_4:%[^ ]+]] = (f32[32,16]{1,0}, f32[16,32]{1,0}) tuple([[c_1_2]], [[c1_1_3]])
// CHECK-NEXT: }
// CHECK:   [[fusion_0:%[^ ]+]] = (f32[32,16]{1,0}, f32[16,32]{1,0}) fusion([[p_1:%[^ ]+]]), kind=kInput, calls=[[fused_computation_2:%[^ ]+]]
)");
}

TEST_F(TransposeMultiOutputFusionTest, TransposeAndInputEpilogueFusion) {
  const char* hlo = R"(
HloModule module

fused_computation {
  param_0.1 = f32[1,16,32]{2,1,0} parameter(0)
  s.1 = f32[1,16,32]{2,1,0} sqrt(param_0.1)
  t.1 = f32[1,32,16]{2,1,0} transpose(s.1), dimensions={0,2,1}
  ROOT out = f32[32,16,1]{2,1,0} bitcast(t.1)
}

ENTRY main {
  p = f32[1,16,32]{2,1,0} parameter(0)
  fusion = f32[32,16,1]{2,1,0} fusion(p), kind=kInput, calls=fused_computation
  c1 = f32[1,16,32]{2,1,0} exponential(p)
  ROOT t = (f32[32,16,1]{2,1,0}, f32[1,16,32]{2,1,0}) tuple(fusion, c1)
}
  )";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation
// CHECK-NEXT:   [[param_0_1_0:%[^ ]+]] = f32[1,16,32]{2,1,0} parameter(0)
// CHECK-NEXT:   [[s_1_1:%[^ ]+]] = f32[1,16,32]{2,1,0} sqrt([[param_0_1_0]])
// CHECK-NEXT:   [[c_1_2:%[^ ]+]] = f32[1,32,16]{2,1,0} transpose([[s_1_1]])
// CHECK-NEXT:   [[out_3:%[^ ]+]] = f32[32,16,1]{2,1,0} bitcast([[c_1_2]])
// CHECK-NEXT:   [[c1_1_4:%[^ ]+]] = f32[1,16,32]{2,1,0} exponential([[param_0_1_0]])
// CHECK-NEXT:   ROOT [[tuple_5:%[^ ]+]] = (f32[32,16,1]{2,1,0}, f32[1,16,32]{2,1,0}) tuple([[out_3]], [[c1_1_4]])
// CHECK-NEXT: }
// CHECK: [[fusion_0:%[^ ]+]] = (f32[32,16,1]{2,1,0}, f32[1,16,32]{2,1,0}) fusion([[p_1:%[^ ]+]]), kind=kInput, calls=[[fused_computation_2:%[^ ]+]]
)");
}

class ReduceMultiOutputFusionTest : public MultiOutputFusionTest {};

TEST_F(ReduceMultiOutputFusionTest, ReduceAndLoop) {
  const char* hlo = R"(
HloModule module

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = f32[] add(a, b)
}

fused_reduction {
  p = f32[200] parameter(0)
  z = f32[] constant(0)
  e = f32[200] exponential(p)
  ROOT r = f32[] reduce(e, z), dimensions={0}, to_apply=add
}

fused_elementwise {
  p = f32[200] parameter(0)
  ROOT r = f32[200] sqrt(p)
}

ENTRY computation {
  p = f32[200] parameter(0)
  o1 = f32[200] fusion(p), kind=kLoop, calls=fused_elementwise
  o2 = f32[] fusion(p), kind=kInput, calls=fused_reduction
  ROOT out = (f32[200], f32[]) tuple(o1, o2)
}

)";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_elementwise
// CHECK-NEXT:  [[p_1_0:%[^ ]+]] = f32[200]{0} parameter(0)
// CHECK-NEXT:  [[r_1_1:%[^ ]+]] = f32[200]{0} sqrt([[p_1_0]])
// CHECK-NEXT:  [[e_2:%[^ ]+]].clone.1 = f32[200]{0} exponential([[p_1_0]])
// CHECK-NEXT:  [[z_3:%[^ ]+]].clone.1 = f32[] constant(0)
// CHECK-NEXT:  [[r_4:%[^ ]+]].clone.1 = f32[] reduce([[e_2]].clone.1, [[z_3]].clone.1), dimensions={0}, to_apply=[[add_5:%[^ ]+]]
// CHECK-NEXT:  ROOT [[tuple_6:%[^ ]+]] = (f32[200]{0}, f32[]) tuple([[r_1_1]], [[r_4]].clone.1)
// CHECK-NEXT:}
// CHECK: [[o1_0:%[^ ]+]] = (f32[200]{0}, f32[]) fusion([[p_2_1:%[^ ]+]]), kind=kInput, calls=[[fused_elementwise_2:%[^ ]+]]
  )");
}

TEST_F(ReduceMultiOutputFusionTest, ReduceAndLoopDifferentShape) {
  const char* hlo = R"(
HloModule module

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = f32[] add(a, b)
}

fused_reduction {
  p = f32[10,20] parameter(0)
  z = f32[] constant(0)
  e = f32[10,20] exponential(p)
  b = f32[200] bitcast(e)
  ROOT r = f32[] reduce(b, z), dimensions={0}, to_apply=add
}

fused_elementwise {
  p = f32[10,20] parameter(0)
  ROOT r = f32[10,20] sqrt(p)
}

ENTRY computation {
  p = f32[10,20] parameter(0)
  o1 = f32[10,20] fusion(p), kind=kLoop, calls=fused_elementwise
  o2 = f32[] fusion(p), kind=kInput, calls=fused_reduction
  ROOT out = (f32[10,20], f32[]) tuple(o1, o2)
}
)";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_elementwise (p.1: f32[10,20]) -> (f32[10,20], f32[]) {
// CHECK-NEXT:   [[p_1_0:%[^ ]+]] = f32[10,20]{1,0} parameter(0)
// CHECK-NEXT:   [[r_1_1:%[^ ]+]] = f32[10,20]{1,0} sqrt([[p_1_0]])
// CHECK-NEXT:   [[e_2:%[^ ]+]].clone.1 = f32[10,20]{1,0} exponential([[p_1_0]])
// CHECK-NEXT:   [[b_1_3:%[^ ]+]].clone.1 = f32[200]{0} bitcast([[e_2]].clone.1)
// CHECK-NEXT:   [[z_4:%[^ ]+]].clone.1 = f32[] constant(0)
// CHECK-NEXT:   [[r_5:%[^ ]+]].clone.1 = f32[] reduce([[b_1_3]].clone.1, [[z_4]].clone.1), dimensions={0}, to_apply=[[add_6:%[^ ]+]]
// CHECK-NEXT:   ROOT [[tuple_7:%[^ ]+]] = (f32[10,20]{1,0}, f32[]) tuple([[r_1_1]], [[r_5]].clone.1)
// CHECK-NEXT: }
  )");
}

TEST_F(ReduceMultiOutputFusionTest, ReduceAndLoopDifferentShapeDifferentType) {
  const char* hlo = R"(
HloModule module, entry_computation_layout={(f16[100,200]{1,0},f32[],f32[])->(f16[100,200]{1,0}, f32[])}

max {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = f32[] maximum(a, b)
}

fused_computation {
  one_5 = f32[] constant(1)
  one_b.5 = f32[100,200]{1,0} broadcast(one_5), dimensions={}
  param_1.15 = f16[100,200]{1,0} parameter(1)
  c.6 = f32[100,200]{1,0} convert(param_1.15)
  param_0.11 = f32[] parameter(0)
  b.6 = f32[100,200]{1,0} broadcast(param_0.11), dimensions={}
  d.5 = f32[100,200]{1,0} divide(c.6, b.6)
  a.6 = f32[100,200]{1,0} add(one_b.5, d.5)
  bitcast.1 = f32[20000]{0} bitcast(a.6)
  z_1 = f32[] constant(0)
  ROOT r.1 = f32[] reduce(bitcast.1, z_1), dimensions={0}, to_apply=max
}

fused_computation.1 {
  one_3 = f32[] constant(1)
  one_b.3 = f32[100,200]{1,0} broadcast(one_3), dimensions={}
  param_2.7 = f16[100,200]{1,0} parameter(2)
  c.4 = f32[100,200]{1,0} convert(param_2.7)
  param_1.10 = f32[] parameter(1)
  b.4 = f32[100,200]{1,0} broadcast(param_1.10), dimensions={}
  d.3 = f32[100,200]{1,0} divide(c.4, b.4)
  a.4 = f32[100,200]{1,0} add(one_b.3, d.3)
  param_0.8 = f32[] parameter(0)
  output_scale_broadcast.1 = f32[100,200]{1,0} broadcast(param_0.8), dimensions={}
  a_scaled.1 = f32[100,200]{1,0} multiply(a.4, output_scale_broadcast.1)
  ROOT a_scaled_converted.1 = f16[100,200]{1,0} convert(a_scaled.1)
}

ENTRY computation {
  output_scale = f32[] parameter(2)
  input_scale = f32[] parameter(1)
  p = f16[100,200]{1,0} parameter(0)
  fusion.1 = f16[100,200]{1,0} fusion(output_scale, input_scale, p), kind=kLoop, calls=fused_computation.1
  fusion = f32[] fusion(input_scale, p), kind=kInput, calls=fused_computation
  ROOT out = (f16[100,200]{1,0}, f32[]) tuple(fusion.1, fusion)
}
)";

  CheckMultiOutputFusion(hlo, R"(
// CHECK: %fused_computation.1 (param_0.8: f32[], param_1.10: f32[], param_2.7: f16[100,200]) -> (f16[100,200], f32[]) {
// CHECK-NEXT:   [[one_3_0:%[^ ]+]] = f32[] constant(1)
// CHECK-NEXT:   [[one_b_3_1:%[^ ]+]] = f32[100,200]{1,0} broadcast([[one_3_0]]), dimensions={}
// CHECK-NEXT:   [[param_2_7_2:%[^ ]+]] = f16[100,200]{1,0} parameter(2)
// CHECK-NEXT:   [[c_4_3:%[^ ]+]] = f32[100,200]{1,0} convert([[param_2_7_2]])
// CHECK-NEXT:   [[param_1_10_4:%[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:   [[b_4_5:%[^ ]+]] = f32[100,200]{1,0} broadcast([[param_1_10_4]]), dimensions={}
// CHECK-NEXT:   [[d_3_6:%[^ ]+]] = f32[100,200]{1,0} divide([[c_4_3]], [[b_4_5]])
// CHECK-NEXT:   [[a_4_7:%[^ ]+]] = f32[100,200]{1,0} add([[one_b_3_1]], [[d_3_6]])
// CHECK-NEXT:   [[param_0_8_8:%[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:   [[output_scale_broadcast_1_9:%[^ ]+]] = f32[100,200]{1,0} broadcast([[param_0_8_8]]), dimensions={}
// CHECK-NEXT:   [[a_scaled_1_10:%[^ ]+]] = f32[100,200]{1,0} multiply([[a_4_7]], [[output_scale_broadcast_1_9]])
// CHECK-NEXT:   [[a_scaled_converted_1_11:%[^ ]+]] = f16[100,200]{1,0} convert([[a_scaled_1_10]])
// CHECK-NEXT:   [[one_5_12:%[^ ]+]].clone.1 = f32[] constant(1)
// CHECK-NEXT:   [[one_b_5_13:%[^ ]+]].clone.1 = f32[100,200]{1,0} broadcast([[one_5_12]].clone.1), dimensions={}
// CHECK-NEXT:   [[c_6_14:%[^ ]+]].clone.1 = f32[100,200]{1,0} convert([[param_2_7_2]])
// CHECK-NEXT:   [[b_6_15:%[^ ]+]].clone.1 = f32[100,200]{1,0} broadcast([[param_1_10_4]]), dimensions={}
// CHECK-NEXT:   [[d_5_16:%[^ ]+]].clone.1 = f32[100,200]{1,0} divide([[c_6_14]].clone.1, [[b_6_15]].clone.1)
// CHECK-NEXT:   [[a_6_17:%[^ ]+]].clone.1 = f32[100,200]{1,0} add([[one_b_5_13]].clone.1, [[d_5_16]].clone.1)
// CHECK-NEXT:   [[bitcast_1_18:%[^ ]+]].clone.1 = f32[20000]{0} bitcast([[a_6_17]].clone.1)
// CHECK-NEXT:   [[z_1_19:%[^ ]+]].clone.1 = f32[] constant(0)
// CHECK-NEXT:   [[r_1_20:%[^ ]+]].clone.1 = f32[] reduce([[bitcast_1_18]].clone.1, [[z_1_19]].clone.1), dimensions={0}, to_apply=[[max_21:%[^ ]+]]
// CHECK-NEXT:   ROOT [[tuple_22:%[^ ]+]] = (f16[100,200]{1,0}, f32[]) tuple([[a_scaled_converted_1_11]], [[r_1_20]].clone.1)
// CHECK-NEXT: }
  )");
}

TEST_F(ReduceMultiOutputFusionTest, GetTupleElementMakeTupleSequence) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    fusion {
      p0 = s32[] parameter(0)
      p1 = s32[32] parameter(1)
      custom-call = (bf16[], s32[], u32[]) custom-call(p1), custom_call_target="my_custom_call"
      get-tuple-element.0 = bf16[] get-tuple-element(custom-call), index=0
      get-tuple-element.1 = s32[] get-tuple-element(custom-call), index=1
      bitcast = s32[1] bitcast(get-tuple-element.1)
      dynamic-update-slice = s32[32] dynamic-update-slice(p1, bitcast, p0)
      get-tuple-element.2 = u32[] get-tuple-element(custom-call), index=2
      ROOT tuple.30 = (bf16[], s32[32], u32[]) tuple(get-tuple-element.0, dynamic-update-slice, get-tuple-element.2)
    }

    ENTRY entry{
      p0 = s32[] parameter(0)
      bitcast = s32[32] bitcast(p0)
      ROOT address_computation.7.0 = (bf16[], s32[32], u32[]) fusion(p0, bitcast), kind=kCustom, calls=fusion
    }
  )")
                    .value();

  ASSERT_FALSE(mof_.Run(module.get()).value());
}

}  // namespace gpu
}  // namespace xla
