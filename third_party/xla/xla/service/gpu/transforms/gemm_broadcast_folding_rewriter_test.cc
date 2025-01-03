/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/gemm_broadcast_folding_rewriter.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {

namespace {

class GemmBroadcastFoldingRewriteTest : public GpuCodegenTest {
 protected:
  const auto& GpuComputeComp() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // These tests test the cuBLAS rewriter so we have to make sure that we use
    // cuBLAS for them.
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
    return debug_options;
  }
};

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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[3,2,2], {{.*}}: f32[2,2]) -> f32[3,2,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[3,2,2]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas${{(lt\$matmul|gemm)}}",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["2"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":["0"]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,2], {{.*}}: f32[3,2,2]) -> f32[3,2,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,2,2]{2,1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK    :       custom_call_target="__cublas${{(lt\$matmul|gemm)}}",
; CHECK    :       backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":["0"]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(GemmBroadcastFoldingRewriteTest,
       BroadcastedStridedRewriteRhsPassChanged) {
  const char* hlo_text = R"(
HloModule BroadcastedInput

ENTRY AddDotsFunc {
  x = f32[3,2,2]{2,1,0} parameter(0)
  y = f32[2,2]{1,0} parameter(1)
  y_broadcast = f32[3,2,2]{2,1,0} broadcast(y), dimensions={1,2}
  ROOT dot_a = f32[3,2,2]{2,1,0} dot(x, y_broadcast), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  // Use GemmRewriter to generate cublasGemm call.
  GemmRewriter gemm_rewriter(
      GpuComputeComp(),
      /*toolkit_version=*/stream_executor::SemanticVersion{12, 4, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          this->RunHloPass(&gemm_rewriter, module.get()));
  EXPECT_TRUE(changed);
  GemmBroadcastFoldingRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(GemmBroadcastFoldingRewriteTest,
       BroadcastedStridedRewriteLhsPassChanged) {
  const char* hlo_text = R"(
HloModule BroadcastedInput

ENTRY AddDotsFunc {
  x = f32[2,2]{1,0} parameter(0)
  y = f32[3,2,2]{2,1,0} parameter(1)
  x_broadcast = f32[3,2,2]{2,1,0} broadcast(x), dimensions={1,2}
  ROOT dot_a = f32[3,2,2]{2,1,0} dot(x_broadcast, y), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  // Use GemmRewriter to generate cublasGemm call.
  GemmRewriter gemm_rewriter(
      GpuComputeComp(),
      /*toolkit_version=*/stream_executor::SemanticVersion{12, 4, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          this->RunHloPass(&gemm_rewriter, module.get()));
  EXPECT_TRUE(changed);
  GemmBroadcastFoldingRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(GemmBroadcastFoldingRewriteTest, LHSBatchDimNonZero) {
  const char* hlo_text = R"(
HloModule LHSBatchDimNonZero

ENTRY %LHSBatchDimNonZero (Arg_1: f32[4,3], Arg_2: f32[4,7,3]) -> f32[4,7,7] {
  %Arg_1 = f32[4,3]{1,0} parameter(0)
  %Arg_2 = f32[4,7,3]{2,1,0} parameter(1)
  %broadcast.22 = f32[7,4,3]{2,1,0} broadcast(f32[4,3]{1,0} %Arg_1), dimensions={1,2}
  ROOT %dot.24 = f32[4,7,7]{2,1,0} dot(f32[7,4,3]{2,1,0} %broadcast.22, f32[4,7,3]{2,1,0} %Arg_2), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={2}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  // Use GemmRewriter to generate cublasGemm call.
  GemmRewriter gemm_rewriter(
      GpuComputeComp(),
      /*toolkit_version=*/stream_executor::SemanticVersion{12, 4, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          this->RunHloPass(&gemm_rewriter, module.get()));
  EXPECT_TRUE(changed);
  GemmBroadcastFoldingRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(GemmBroadcastFoldingRewriteTest, RHSBatchDimNonZero) {
  const char* hlo_text = R"(
HloModule RHSBatchDimNonZero

ENTRY %RHSBatchDimNonZero (Arg_1: f32[4,3], Arg_2: f32[4,7,3]) -> f32[4,7,7] {
  %Arg_1 = f32[4,3]{1,0} parameter(0)
  %Arg_2 = f32[4,7,3]{2,1,0} parameter(1)
  %broadcast.22 = f32[7,4,3]{2,1,0} broadcast(f32[4,3]{1,0} %Arg_1), dimensions={1,2}
  ROOT %dot.24 = f32[4,7,7]{2,1,0} dot(f32[4,7,3]{2,1,0} %Arg_2, f32[7,4,3]{2,1,0} %broadcast.22), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter gemm_rewriter(
      GpuComputeComp(),
      /*toolkit_version=*/stream_executor::SemanticVersion{12, 4, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          this->RunHloPass(&gemm_rewriter, module.get()));
  EXPECT_TRUE(changed);
  GemmBroadcastFoldingRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
