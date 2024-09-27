/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/xprof_gpu_cost_analysis.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace profiler {

class XprofGpuHloCostAnalysisTest : public xla::HloTestBase {
  xla::HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const xla::Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return xla::ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

 public:
  xla::HloCostAnalysis::Options options_{
      ShapeSizeBytesFunction(),
      /*per_second_rates=*/{},
      /*min_latencies_seconds=*/{},
      /*count_multiple_input_accesses=*/true};
  XProfGpuCostAnalysis analysis_{options_};
  XprofGpuHloCostAnalysisTest() : xla::HloTestBase() {}
};

TEST_F(XprofGpuHloCostAnalysisTest, Fp16GemmNoAdjustment) {
  absl::string_view hlo_string = R"(
HloModule r

ENTRY e {
  arg0 = f16[65536,32800] parameter(0)
  arg1 = f16[32800,32] parameter(1)
  gemm = (f16[65536,32], s8[0]) custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config="{
        \"gemm_backend_config\": {
            \"alpha_real\":1,
            \"beta\":0,
            \"dot_dimension_numbers\":{
                \"lhs_contracting_dimensions\":[\"1\"],
                \"rhs_contracting_dimensions\":[\"0\"],
                \"lhs_batch_dimensions\":[],
                \"rhs_batch_dimensions\":[]
            },
            \"alpha_imag\":0,
            \"precision_config\":{
                \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
            },
            \"epilogue\":\"DEFAULT\"
        }
    }"
  ROOT get-tuple-element = f16[65536,32]
    get-tuple-element((f16[65536,32], s8[0]) gemm), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));
  xla::HloComputation* comp = module->entry_computation();
  const xla::HloInstruction* fp16gemm = comp->GetInstructionWithName("gemm");
  // flops of gemm A * B = rows(A) * cols(B) * cols(A) * 2
  // where 2 is for the add and multiply
  int64_t gold_flops = 65536LL * 32800 * 32 * 2;
  EXPECT_EQ(analysis_.flop_count(*fp16gemm), gold_flops);
  EXPECT_EQ(analysis_.GetDeviceFlopsAdjustment(*fp16gemm), 0);
}

TEST_F(XprofGpuHloCostAnalysisTest, S8GemmAdjustment) {
  absl::string_view hlo_string = R"(
HloModule r

ENTRY e {
  arg0 = s8[65536,32800] parameter(0)
  arg1 = s8[32800,32] parameter(1)
  gemm = (s32[65536,32], s8[0]) custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config="{
        \"gemm_backend_config\": {
            \"alpha_real\":1,
            \"beta\":0,
            \"dot_dimension_numbers\":{
                \"lhs_contracting_dimensions\":[\"1\"],
                \"rhs_contracting_dimensions\":[\"0\"],
                \"lhs_batch_dimensions\":[],
                \"rhs_batch_dimensions\":[]
            },
            \"alpha_imag\":0,
            \"precision_config\":{
                \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
            },
            \"epilogue\":\"DEFAULT\"
        }
    }"
  ROOT get-tuple-element = s32[65536,32]
    get-tuple-element((s32[65536,32], s8[0]) gemm), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));
  xla::HloComputation* comp = module->entry_computation();
  const xla::HloInstruction* s8gemm = comp->GetInstructionWithName("gemm");
  int64_t gold_flops = 65536LL * 32800 * 32 * 2;
  EXPECT_EQ(analysis_.flop_count(*s8gemm), gold_flops);
  // Matmul of int8 * int8 -> int32, normalized it to equivalent fp16 flops by
  // dividing by 2 as all inputs are 8 bits
  EXPECT_EQ(analysis_.GetDeviceFlopsAdjustment(*s8gemm), gold_flops / 2);
}

}  // namespace profiler
}  // namespace tensorflow
