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
  HloCostAnalysis::Options options{ShapeSize};
  GpuHloCostAnalysis analysis(options);
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

}  // namespace gpu
}  // namespace xla
