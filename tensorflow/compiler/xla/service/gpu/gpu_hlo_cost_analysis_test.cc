/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.h"

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

constexpr int64_t kPointerSize = 8;

int64_t ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

using GpuHloCostAnalysisTest = HloTestBase;

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
  GpuHloCostAnalysis analysis({ShapeSize});
  ASSERT_IS_OK(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  HloComputation* comp = module->entry_computation();
  const HloInstruction* conv1 = comp->GetInstructionWithName("conv1");
  EXPECT_EQ(analysis.operand_bytes_accessed(*conv1, 0),
            sizeof(int8_t) * 128 * 12 * 24 * 24 * 4);
  EXPECT_EQ(analysis.operand_bytes_accessed(*conv1, 1),
            sizeof(int8_t) * 16 * 12 * 5 * 5 * 4);
  EXPECT_EQ(analysis.output_bytes_accessed(*conv1),
            sizeof(int8_t) * 128 * 4 * 24 * 24 * 4);
  EXPECT_EQ(analysis.flop_count(*conv1), 159694848);
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
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  int n_output_elements = 3 * 4;

  GpuHloCostAnalysis analysis({ShapeSize});
  ASSERT_IS_OK(root->Accept(&analysis));

  // Each of the output elements are generated from reducing [4x5] elements.
  EXPECT_EQ(analysis.flop_count(), n_output_elements * (4 * 5 - 1));

  EXPECT_EQ(analysis.bytes_accessed(),
            sizeof(float) * (8 * 8 + 1 + n_output_elements));

  // For every output element (window size) elements are read from operand 0
  // independently.
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0),
            sizeof(float) * n_output_elements * 4 * 5);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * 1);
  EXPECT_EQ(analysis.output_bytes_accessed(*root),
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

  GpuHloCostAnalysis analysis({ShapeSize});
  ASSERT_IS_OK(root->Accept(&analysis));

  EXPECT_EQ(analysis.output_bytes_accessed(*root), 10000);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), 10000);
  // operand + output
  EXPECT_EQ(analysis.bytes_accessed(*root), 2 * 10000);
  EXPECT_EQ(analysis.bytes_accessed(), 2 * 10000);
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
  GpuHloCostAnalysis analysis({ShapeSize});
  ASSERT_IS_OK(root->Accept(&analysis));

  auto n_elements = 1024 * 1024;
  EXPECT_EQ(analysis.output_bytes_accessed(*root), n_elements * 4);
  EXPECT_EQ(analysis.bytes_accessed(*root), n_elements * 4);
  EXPECT_EQ(analysis.bytes_accessed(), n_elements * 4);
  EXPECT_EQ(analysis.flop_count(), n_elements * 3);
}

}  // namespace gpu
}  // namespace xla
