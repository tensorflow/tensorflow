/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class GpuHloCostAnalysisTest : public HloTestBase {
 public:
  HloCostAnalysis::Options options_{.count_multiple_input_accesses = true};
  GpuHloCostAnalysis analysis_{options_};
  GpuHloCostAnalysisTest() : HloTestBase() {}
};

TEST_F(GpuHloCostAnalysisTest, ConvCustomCall) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = s8[128,12,24,24,4]{4,3,2,1,0} parameter(0)
  p1 = s8[16,12,5,5,4]{4,3,2,1,0} parameter(1)
  p2 = f32[16]{0} parameter(2)
  conv1 = (s8[128,4,24,24,4]{4,3,2,1,0}, u8[0]{0}) custom-call(p0, p1, p2),
              window={size=5x5 pad=2_2x2_2},
              dim_labels=bf01_oi01->bf01,
              custom_call_target="__cudnn$convBiasActivationForward"
  ROOT tuple = tuple(conv1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));
  HloComputation* comp = module->entry_computation();
  const HloInstruction* conv1 = comp->GetInstructionWithName("conv1");
  int op0_size = sizeof(int8_t) * 128 * 12 * 24 * 24 * 4;
  int op1_size = sizeof(int8_t) * 16 * 12 * 5 * 5 * 4;
  int op2_size = sizeof(float) * 16;
  int out_size = sizeof(int8_t) * 128 * 4 * 24 * 24 * 4;
  EXPECT_EQ(analysis_.operand_bytes_accessed(*conv1, 0), op0_size);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*conv1, 1), op1_size);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*conv1, 2), op2_size);
  EXPECT_EQ(analysis_.output_bytes_accessed(*conv1, {0}), out_size);
  EXPECT_EQ(analysis_.bytes_accessed(*conv1),
            op0_size + op1_size + op2_size + out_size);
  EXPECT_EQ(analysis_.flop_count(*conv1), 159694848);
}

TEST_F(GpuHloCostAnalysisTest, ReduceWindowWithOverlapsRepeatedReads) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

add {
  a0 = f32[] parameter(0)
  a1 = f32[] parameter(1)
  ROOT _ = f32[] add(a0, a1)
}

ENTRY entry {
  p0 = f32[8,8] parameter(0)
  c0 = f32[] constant(0)
  ROOT _ = f32[3,4] reduce-window(p0, c0), window={size=4x5 stride=2x1}, to_apply=add
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* root = module->entry_computation()->root_instruction();
  int n_output_elements = 3 * 4;

  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  // Each of the output elements are generated from reducing [4x5] elements;
  // each elementwise operation is counted as 3 flops.
  EXPECT_EQ(analysis_.flop_count(), 3 * n_output_elements * (4 * 5 - 1));

  EXPECT_EQ(analysis_.bytes_accessed(),
            sizeof(float) * (8 * 8 + 1 + n_output_elements));

  // For every output element (window size) elements are read from operand 0
  // independently.
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0),
            sizeof(float) * n_output_elements * 4 * 5);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 1), sizeof(float) * 1);
  EXPECT_EQ(analysis_.output_bytes_accessed(*root),
            sizeof(float) * n_output_elements);
}

TEST_F(GpuHloCostAnalysisTest, BroadcastWithRepeats) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p1 = s8[] parameter(0)
  c1 = s8[] constant(0)
  a1 = s8[] add(p1, c1)
  b1 = s8[10000] broadcast(a1), dimensions={}
  b2 = s8[10000] broadcast(c1), dimensions={}
  ROOT r1 = s8[10000] add(b1, b2)
}

ENTRY e {
  p0 = s8[] parameter(0)
  ROOT r0 = s8[10000] fusion(p0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 10000);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 10000);
  // Operand + output.
  EXPECT_EQ(analysis_.bytes_accessed(*root), 2 * 10000);
  EXPECT_EQ(analysis_.bytes_accessed(), 2 * 10000);
}

TEST_F(GpuHloCostAnalysisTest, WithoutRepeats) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p1 = s8[] parameter(0)
  a1 = s8[] add(p1, p1)
  b1 = s8[10000] broadcast(a1), dimensions={}
  a2 = s8[10000] add(b1, b1)
  slice1 = s8[8000] slice(a2), slice={[0:8000]}
  slice2 = s8[8000] slice(a2), slice={[2000:10000]}
  c = s8[10000] constant({...})
  slicec1 = s8[8000] slice(c), slice={[0:8000]}
  slicec2 = s8[8000] slice(c), slice={[2000:10000]}
  a3 = s8[8000] add(slice1, slice2)
  a4 = s8[8000] add(slicec1, slicec2)
  ROOT a5 = s8[8000] add(a3, a4)
}

ENTRY e {
  p0 = s8[] parameter(0)
  ROOT r0 = s8[8000] fusion(p0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  options_.count_multiple_input_accesses = false;
  GpuHloCostAnalysis analysis{options_};
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis));

  EXPECT_EQ(analysis.output_bytes_accessed(*root), 8000);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), 1);
  // Operand + output + constant.
  EXPECT_EQ(analysis.bytes_accessed(*root), 1 + 8000 + 10000);
  EXPECT_EQ(analysis.bytes_accessed(), 1 + 8000 + 10000);
}

TEST_F(GpuHloCostAnalysisTest, BroadcastFlops) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  i0 = f32[1024] iota(), iota_dimension=0
  m0 = f32[1024] add(i0, i0)
  s0 = f32[1024] multiply(m0, m0)
  b0 = f32[1024,1024] broadcast(s0), dimensions={0}
  ROOT r0 = f32[1024,1024] negate(b0)
}

ENTRY e {
  ROOT r = f32[1024,1024] fusion(), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto n_elements = 1024 * 1024;
  EXPECT_EQ(analysis_.output_bytes_accessed(*root), n_elements * 4);
  EXPECT_EQ(analysis_.bytes_accessed(*root), n_elements * 4);
  EXPECT_EQ(analysis_.bytes_accessed(), n_elements * 4);
  EXPECT_EQ(analysis_.flop_count(), n_elements * 3 * 3);
  EXPECT_EQ(analysis_.IrSize(*root), 5);
}

TEST_F(GpuHloCostAnalysisTest, Slice) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p1 = s8[100000000] parameter(0)
  i1 = s8[100000000] iota(), iota_dimension=0
  a1 = s8[100000000] add(p1, i1)
  ROOT r1 = s8[1] slice(a1), slice={[0:1]}
}

ENTRY e {
  p0 = s8[100000000] parameter(0)
  ROOT r0 = s8[1] fusion(p0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 1);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 1);
  EXPECT_EQ(analysis_.bytes_accessed(*root), 2);
  EXPECT_EQ(analysis_.bytes_accessed(), 2);
  EXPECT_EQ(analysis_.IrSize(*root), 4);
}

TEST_F(GpuHloCostAnalysisTest, TwoSlices) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p1 = s8[100] parameter(0)
  i1 = s8[100] iota(), iota_dimension=0
  a1 = s8[100] add(p1, i1)
  slice1 = s8[1] slice(a1), slice={[0:1]}
  slice2 = s8[1] slice(a1), slice={[3:4]}
  ROOT r = s8[1] add(slice1, slice2)
}

ENTRY e {
  p0 = s8[100] parameter(0)
  ROOT r0 = s8[1] fusion(p0), kind=kInput, calls=f
}

)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 1);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 2);
  EXPECT_EQ(analysis_.bytes_accessed(*root), 3);
  EXPECT_EQ(analysis_.bytes_accessed(), 3);
  EXPECT_EQ(analysis_.IrSize(*root), 9);
}

TEST_F(GpuHloCostAnalysisTest, MultipleTrivialUsers) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = s8[] parameter(0)
  m0 = s8[] multiply(p0, p0)
  n0 = s8[] negate(p0)
  ROOT a0 = s8[] add(m0, n0)
}

ENTRY e {
  param0 = s8[] parameter(0)
  ROOT r0 = s8[] fusion(param0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  // Expect that uses of p0 by different trivial users (m0, n0) can be
  // combined into a single memory access.
  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 1);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 1);
  EXPECT_EQ(analysis_.bytes_accessed(*root), 1 + 1);
  EXPECT_EQ(analysis_.bytes_accessed(), 1 + 1);
  EXPECT_EQ(analysis_.IrSize(*root), 4);
}

TEST_F(GpuHloCostAnalysisTest, MixedUsers) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = s8[10] parameter(0)
  n0 = s8[10] negate(p0)
  m0 = s8[10] multiply(n0, n0)
  a0 = s8[10] add(n0, n0)
  s0 = s8[5] slice(a0), slice={[0:5]}
  svar1 = s8[2] slice(n0), slice={[4:6]}
  n1 = s8[2] negate(svar1)
  ROOT c0 = s8[17] concatenate(s0, m0, n1), dimensions={0}
}

ENTRY e {
  param0 = s8[10] parameter(0)
  ROOT r0 = s8[17] fusion(param0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  // Expect that uses of n0 by different trivial users (m0, a0) can be
  // combined into a single memory access, but slices have to be counted
  // separately.
  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 17);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 17);
  EXPECT_EQ(analysis_.bytes_accessed(*root), 17 + 17);
  EXPECT_EQ(analysis_.bytes_accessed(), 17 + 17);
  // There are 2 slice accesses + 1 element-wise from the root.
  EXPECT_EQ(analysis_.IrSize(*root->fused_parameter(0)), 3);
  // Because p0 is only directly used by elementwise n0 their code sizes
  // have to be equal.
  EXPECT_EQ(analysis_.IrSize(*root->fused_parameter(0)),
            analysis_.IrSize(*root->fused_parameter(0)->users()[0]));
  EXPECT_EQ(analysis_.IrSize(*root), 12);
}

TEST_F(GpuHloCostAnalysisTest, FractionalUseRoundingUp) {
  absl::string_view hlo_string = R"(
HloModule m

add_s8 {
  lhs = s8[] parameter(0)
  rhs = s8[] parameter(1)
  ROOT add = s8[] add(lhs, rhs)
}

f {
  p0 = s8[] parameter(0)
  b0 = s8[10] broadcast(p0), dimensions={}
  c0 = s8[] constant(0)
  r0 = s8[] reduce(b0, c0), dimensions={0}, to_apply=add_s8
  bitcast0 = s8[1] bitcast(r0)
  i0 = s8[5] iota(), iota_dimension=0
  cat0 = s8[6] concatenate(bitcast0, i0), dimensions={0}
  p1 = s32[] parameter(1)
  ROOT s0 = s8[2] dynamic-slice(cat0, p1), dynamic_slice_sizes={2}
}

ENTRY e {
  p0 = s8[] parameter(0)
  p1 = s32[] parameter(1)
  ROOT r = s8[2] fusion(p0, p1), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 2);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 10);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 1), 4);
  EXPECT_EQ(analysis_.bytes_accessed(*root), 2 + 10 + 4);
  EXPECT_EQ(analysis_.bytes_accessed(), 2 + 10 + 4);
}

TEST_F(GpuHloCostAnalysisTest, LargeConstant) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = s8[1000] parameter(0)
  c0 = s8[1000] constant({...})
  ROOT a0 = s8[1000] add(p0, c0)
}

ENTRY e {
  p0 = s8[1000] parameter(0)
  ROOT r = s8[1000] fusion(p0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  EXPECT_EQ(analysis_.output_bytes_accessed(*root), 1000);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*root, 0), 1000);
  // Parameter + output + constant.
  EXPECT_EQ(analysis_.bytes_accessed(*root), 3000);
  EXPECT_EQ(analysis_.bytes_accessed(), 3000);
  EXPECT_EQ(analysis_.IrSize(*root), 3);
}

TEST_F(GpuHloCostAnalysisTest, DynUpdateSliceUsingOperandData) {
  const char* hlo_fusion_module_str = R"(
  HloModule m

  f {
    to_update = s8[3,1,1,1] parameter(0)
    update = s8[1,1,1,1] constant(0)
    a = s32[] constant(0)
    dus = s8[3,1,1,1] dynamic-update-slice(to_update, update, a, a, a, a)
    ROOT _ = s8[3,1,1,1] negate(dus)
  }

  ENTRY _ {
    to_update = s8[3,1,1,1] parameter(0)
    ROOT _ = s8[3,1,1,1] fusion(to_update), kind=kLoop, calls=f
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_EQ(fusion->opcode(), HloOpcode::kFusion);

  // Input size minus update size.
  EXPECT_EQ(analysis_.operand_bytes_accessed(*fusion, 0), 3 - 1);
  EXPECT_EQ(analysis_.output_bytes_accessed(*fusion), 3);
}

TEST_F(GpuHloCostAnalysisTest, DynUpdateSliceNotUsingOperandData) {
  const char* hlo_fusion_module_str = R"(
  HloModule m

  f {
    to_update = s8[3,1,1,1] parameter(0)
    update = s8[1,1,1,1] constant(0)
    a = s32[] constant(0)
    ROOT dus = s8[3,1,1,1] dynamic-update-slice(to_update, update, a, a, a, a)
  }

  ENTRY _ {
    to_update = s8[3,1,1,1] parameter(0)
    ROOT _ = s8[3,1,1,1] fusion(to_update), kind=kLoop, calls=f
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_EQ(fusion->opcode(), HloOpcode::kFusion);

  EXPECT_EQ(analysis_.operand_bytes_accessed(*fusion, 0), 0);
  EXPECT_EQ(analysis_.output_bytes_accessed(*fusion), 1);
}

TEST_F(GpuHloCostAnalysisTest, CommonElementwiseUseTwoParameters) {
  const char* hlo_fusion_module_str = R"(
  HloModule m

  add {
    p0 = s8[] parameter(0)
    p1 = s8[] parameter(1)
    ROOT _ = s8[] add(p0, p1)
  }

  f {
    p0 = s8[10] parameter(0)
    p1 = s8[10] parameter(1)
    a = s8[10] add(p0, p1)
    c0 = s8[] constant(0)
    r0 = s8[] reduce(a, c0), dimensions={0}, to_apply=add
    c1 = s8[] constant(100)
    r1 = s8[] reduce(a, c1), dimensions={0}, to_apply=add
    ROOT _ = s8[] add(r0, r1)
  }

  ENTRY _ {
    p0 = s8[10] parameter(0)
    p1 = s8[10] parameter(1)
    ROOT _ = s8[] fusion(p0, p1), kind=kLoop, calls=f
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  EXPECT_EQ(analysis_.CommonElementwiseUtilization(fusion->fused_parameter(0),
                                                   fusion->fused_parameter(1)),
            2.f);
}

TEST_F(GpuHloCostAnalysisTest, CommonElementwiseUseParameterAndRoot) {
  const char* hlo_fusion_module_str = R"(
  HloModule m

  f {
    p0 = s8[10] parameter(0)
    p1 = s8[] parameter(1)
    p1b = s8[10] broadcast(p1)
    a = s8[10] add(p0, p1b)
    ROOT _ = s8[10] negate(a)
  }

  ENTRY _ {
    p0 = s8[10] parameter(0)
    p1 = s8[] parameter(1)
    ROOT _ = s8[10] fusion(p0, p1), kind=kLoop, calls=f
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  EXPECT_EQ(analysis_.CommonElementwiseUtilization(
                fusion->fused_parameter(0), fusion->fused_expression_root()),
            1.f);
  EXPECT_EQ(analysis_.CommonElementwiseUtilization(
                fusion->fused_parameter(1), fusion->fused_expression_root()),
            0.f);
}

TEST_F(GpuHloCostAnalysisTest,
       CommonElementwiseUseParameterAndRootMultiOutputFusion) {
  const char* hlo_fusion_module_str = R"(
  HloModule m

  f {
    p0 = s8[10] parameter(0)
    p1 = s8[] parameter(1)
    p1b = s8[10] broadcast(p1)
    a = s8[10] add(p0, p1b)
    neg = s8[10] negate(a)
    ROOT _ = (s8[10], s8[10]) tuple(a, neg)
  }

  ENTRY _ {
    p0 = s8[10] parameter(0)
    p1 = s8[] parameter(1)
    ROOT _ = (s8[10], s8[10]) fusion(p0, p1), kind=kLoop, calls=f
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  EXPECT_EQ(analysis_.CommonElementwiseUtilization(
                fusion->fused_parameter(0), fusion->fused_expression_root()),
            1.f);
  EXPECT_EQ(analysis_.CommonElementwiseUtilization(
                fusion->fused_parameter(1), fusion->fused_expression_root()),
            0.f);
}

TEST_F(GpuHloCostAnalysisTest, Reduce) {
  absl::string_view hlo_string = R"(
HloModule m

add {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add.0 = f32[] add(param_0, param_1)
}

ENTRY entry_computation {
  param_0.3 = f32[32,40]{1,0} parameter(0)
  constant = f32[] constant(0)
  ROOT reduce = f32[32]{0} reduce(param_0.3, constant), dimensions={1}, to_apply=add
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();

  int64_t input_bytes_accessed = 4 * 32 * 40;
  int64_t init_bytes_accessed = 4 * 32;
  int64_t output_bytes_accessed = 4 * 32;

  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 0), input_bytes_accessed);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 1), init_bytes_accessed);
  EXPECT_EQ(analysis_.output_bytes_accessed(*reduce), output_bytes_accessed);
  EXPECT_EQ(analysis_.bytes_accessed(*reduce),
            input_bytes_accessed + init_bytes_accessed + output_bytes_accessed);
  EXPECT_EQ(analysis_.flop_count(*reduce), 32 * 39 * 3);
}

TEST_F(GpuHloCostAnalysisTest, VariadicReduce) {
  absl::string_view hlo_string = R"(
HloModule m

add {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  param_2 = f32[] parameter(2)
  param_3 = f32[] parameter(3)
  add.0 = f32[] add(param_0, param_2)
  add.1 = f32[] add(param_1, param_3)
  ROOT t = (f32[], f32[]) tuple(add.0, add.1)
}

ENTRY entry_computation {
  param_0.3 = f32[32,40]{1,0} parameter(0)
  param_1.3 = f32[32,40]{1,0} parameter(1)
  param_2.2 = f32[] parameter(2)
  constant = f32[] constant(0)
  ROOT reduce = (f32[32]{0}, f32[32]{0}) reduce(param_0.3, param_1.3, param_2.2, constant), dimensions={1}, to_apply=add
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();

  int64_t input_bytes_accessed = 4 * 32 * 40;
  int64_t init_bytes_accessed = 4 * 32;
  int64_t output_bytes_accessed = 2 * 4 * 32;

  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 0), input_bytes_accessed);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 1), input_bytes_accessed);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 2), init_bytes_accessed);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*reduce, 3), init_bytes_accessed);
  EXPECT_EQ(analysis_.output_bytes_accessed(*reduce), output_bytes_accessed);
  EXPECT_EQ(analysis_.bytes_accessed(*reduce), 2 * input_bytes_accessed +
                                                   2 * init_bytes_accessed +
                                                   output_bytes_accessed);
  EXPECT_EQ(analysis_.flop_count(*reduce), 32 * 39 * 6);
}

TEST_F(GpuHloCostAnalysisTest, AsyncAllReduce) {
  absl::string_view hlo_string = R"(
HloModule m

add {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT t = f32[] add(param_0, param_1)
}

ENTRY entry_computation {
  p = f32[4096] parameter(0)
  ar-start = f32[4096] all-reduce-start(p), to_apply=add
  ROOT _ = f32[4096] all-reduce-done(ar-start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  const HloInstruction* all_reduce =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(analysis_.BytesTransferred(*all_reduce), 4096 * 4);
  EXPECT_EQ(analysis_.bytes_accessed(*all_reduce), 4096 * 4);
}

TEST_F(GpuHloCostAnalysisTest, AllGather) {
  absl::string_view hlo_string = R"(
HloModule m, num_partitions=4

ENTRY entry_computation {
  p = f32[1024] parameter(0)
  ROOT _ = f32[4096] all-gather(p), dimensions={0}, use_global_device_ids=true,
    replica_groups={{0,1,2,3}}, channel_id=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis_.BytesTransferred(*all_gather), 4096 * 4);
  // write = (3 + 4) * 1024 * 4 bytes
  // read = 4 * 1024 * 4 bytes
  EXPECT_EQ(analysis_.bytes_accessed(*all_gather), 45056);
}

TEST_F(GpuHloCostAnalysisTest, AsyncAllGather) {
  absl::string_view hlo_string = R"(
HloModule m, num_partitions=4

ENTRY entry_computation {
  p.0 = f32[1024] parameter(0)
  p.1 = f32[512] parameter(1)
  ag-start = ((f32[1024],f32[512]), (f32[4096],f32[2048])) all-gather-start(p.0,p.1),
    dimensions={0}, use_global_device_ids=true, replica_groups={{0,1,2,3}},
    channel_id=1
  ROOT _ = (f32[4096],f32[2048]) all-gather-done(ag-start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction()->operand(0);
  // Output is (f32[4096], f32[2048]).
  EXPECT_EQ(analysis_.BytesTransferred(*all_gather), 4096 * 4 + 2048 * 4);
  // write = (3 + 4) * (1024 + 512) * 4 bytes
  // read = 4 * (1024 + 512) * 4 bytes
  EXPECT_EQ(analysis_.bytes_accessed(*all_gather), 67584);
}

TEST_F(GpuHloCostAnalysisTest, ReduceScatter) {
  absl::string_view hlo_string = R"(
HloModule m, num_partitions=4

add {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT t = f32[] add(param_0, param_1)
}

ENTRY entry_computation {
  p = f32[4096] parameter(0)
  ROOT _ = f32[1024] reduce-scatter(p), dimensions={0}, to_apply=add,
      use_global_device_ids=true, replica_groups={{0,1,2,3}}, channel_id=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  const HloInstruction* reduce_scatter =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis_.BytesTransferred(*reduce_scatter), 4096 * 4);
  // read = (3 + 4) * 1024 * 4 bytes
  // write = 4 * 1024 * 4 bytes
  EXPECT_EQ(analysis_.bytes_accessed(*reduce_scatter), 45056);
}

TEST_F(GpuHloCostAnalysisTest, AsyncReduceScatter) {
  absl::string_view hlo_string = R"(
HloModule m, num_partitions=4

add {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT t = f32[] add(param_0, param_1)
}

async_computation {
  param_3 = f32[4096] parameter(0)
  param_4 = f32[2048] parameter(1)
  ROOT r = (f32[1024],f32[512]) reduce-scatter(param_3,param_4),
    dimensions={0}, to_apply=add, use_global_device_ids=true,
    replica_groups={{0,1,2,3}}, channel_id=1
}

ENTRY entry_computation {
  p.0 = f32[4096] parameter(0)
  p.1 = f32[2048] parameter(1)
  rs-start = ((f32[4096],f32[2048]),(f32[1024],f32[512])) async-start(p.0,p.1),
    calls=async_computation
  ROOT _ = (f32[1024],f32[512]) async-done(rs-start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  const HloInstruction* reduce_scatter =
      module->entry_computation()->root_instruction()->operand(0);
  // Output is (f32[1024],f32[512]).
  EXPECT_EQ(analysis_.BytesTransferred(*reduce_scatter), 4096 * 4 + 2048 * 4);
  // read = (3 + 4) * (1024 + 512) * 4 bytes
  // write = 4 * (1024 + 512) * 4 bytes
  EXPECT_EQ(analysis_.bytes_accessed(*reduce_scatter), 67584);
}

TEST_F(GpuHloCostAnalysisTest, CustomOpProfileIsUsed) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY entry_computation {
  param_0 = f32[10] parameter(0)
  param_1 = f32[10] parameter(1)
  param_2 = f32[10] parameter(2)
  param_3 = f32[10] parameter(3)
  tanh = f32[10] tanh(param_0)
  mul = f32[10] multiply(tanh, param_1)
  ROOT clamp = f32[10] clamp(mul, param_2, param_3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloOpProfiles::HloOpProfile hlo_op_profile;

  const int kF32ClampFlopsPerElement = 7;
  const int kF32MultiplyFlopsPerElement = 11;
  const int kF32TanhFlopsPerElement = 13;

  const int kNumElements = 10;

  hlo_op_profile[{HloOpcode::kClamp, PrimitiveType::F32}] =
      kF32ClampFlopsPerElement;
  hlo_op_profile[{HloOpcode::kMultiply, PrimitiveType::F32}] =
      kF32MultiplyFlopsPerElement;
  hlo_op_profile[{HloOpcode::kTanh, PrimitiveType::F32}] =
      kF32TanhFlopsPerElement;

  GpuHloCostAnalysis analysis(options_, hlo_op_profile);
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis));

  const HloInstruction* clamp = module->entry_computation()->root_instruction();
  const HloInstruction* mul = clamp->operand(0);
  const HloInstruction* tanh = mul->operand(0);

  EXPECT_EQ(analysis.flop_count(*clamp),
            kF32ClampFlopsPerElement * kNumElements);
  EXPECT_EQ(analysis.flop_count(*mul),
            kF32MultiplyFlopsPerElement * kNumElements);
  EXPECT_EQ(analysis.flop_count(*tanh), kF32TanhFlopsPerElement * kNumElements);
};

}  // namespace gpu
}  // namespace xla
