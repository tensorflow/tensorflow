/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {

namespace {

class GemmBroadcastFoldingRewriteTest : public GpuCodegenTest {};

TEST_F(GemmBroadcastFoldingRewriteTest, BroadcastedStridedRewriteRhs) {
  const char* hlo_text = R"(
HloModule BroadcastedInput

ENTRY AddDotsFunc {
  x = f32[3,2,2]{2,1,0} parameter(0)
  y = f32[2,2]{1,0} parameter(1)
  y_broadcast = f32[3,2,2]{2,1,0} broadcast(y), dimensions={1,2}
  ROOT dot_a = f32[3,2,2]{2,1,0} dot(x, y_broadcast), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[3,2,2], y: f32[2,2]) -> f32[3,2,2] {
; CHECK-NEXT:    %x = f32[3,2,2]{2,1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    ROOT %cublas-batch-gemm.1 = f32[3,2,2]{2,1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[\"0\"],\"rhs_batch_dimensions\":[]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}

TEST_F(GemmBroadcastFoldingRewriteTest, BroadcastedStridedRewriteLhs) {
  const char* hlo_text = R"(
HloModule BroadcastedInput

ENTRY AddDotsFunc {
  x = f32[2,2]{1,0} parameter(0)
  y = f32[3,2,2]{2,1,0} parameter(1)
  x_broadcast = f32[3,2,2]{2,1,0} broadcast(x), dimensions={1,2}
  ROOT dot_a = f32[3,2,2]{2,1,0} dot(x_broadcast, y), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[3,2,2]) -> f32[3,2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[3,2,2]{2,1,0} parameter(1)
; CHECK-NEXT:    ROOT %cublas-batch-gemm.1 = f32[3,2,2]{2,1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[\"0\"]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}
}  // namespace
}  // namespace gpu
}  // namespace xla
