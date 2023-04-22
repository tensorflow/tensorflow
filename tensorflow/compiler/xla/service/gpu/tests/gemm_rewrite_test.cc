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

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

namespace {

class GemmRewriteTest : public GpuCodegenTest {
 public:
  void CheckNumberOfAllocations(const std::string& hlo,
                                int expected_number_of_allocations) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        backend().compiler()->RunBackend(
            std::move(optimized_module), backend().default_stream_executor(),
            backend().default_stream_executor()->GetAllocator()));
    GpuExecutable* gpu_executable =
        static_cast<GpuExecutable*>(executable.get());
    absl::Span<const BufferAllocation> allocations =
        gpu_executable->GetAllocations();
    CHECK_EQ(allocations.size(), expected_number_of_allocations);
  }

  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

TEST_F(GemmRewriteTest, SimpleRewrite) {
  const char* hlo_text = R"(
HloModule SimpleGemm

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  ROOT dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    ROOT %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"batch_size\":\"1\",\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}

TEST_F(GemmRewriteTest, TestBatchedAutotuning) {
  const char* hlo_text = R"(
HloModule ComplexDotMultipleNonContracting

ENTRY %test {
  %lhs = f32[7,17,10,13]{3,2,1,0} parameter(0)
  %rhs = f32[7,9,10,13,6]{4,3,2,1,0} parameter(1)
  ROOT %dot = f32[10,7,17,9,6]{4,3,2,1,0} dot(%lhs, %rhs), lhs_batch_dims={2,0}, rhs_batch_dims={2,0}, lhs_contracting_dims={3}, rhs_contracting_dims={3}
}

)";

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: \"batch_size\":\"70\"
; CHECK: selected_algorithm
      )");
}

TEST_F(GemmRewriteTest, SimpleRewriteDeterministic) {
  const char* hlo_text = R"(
HloModule SimpleGemm

ENTRY AddDotsFunc {
  x = f32[128,128] parameter(0)
  y = f32[128,128] parameter(1)
  ROOT dot_a = f32[128,128] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto get_module = [&]() -> StatusOr<std::unique_ptr<HloModule>> {
    HloModuleConfig config;
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_deterministic_ops(true);
    config.set_debug_options(debug_options);
    return ParseAndReturnVerifiedModule(hlo_text, config);
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> optimized_module,
      backend().compiler()->RunHloPasses(
          *get_module(), backend().default_stream_executor(),
          backend().default_stream_executor()->GetAllocator()));
  StatusOr<bool> filecheck_result = RunFileCheck(optimized_module->ToString(),
                                                 R"(
; CHECK:    \"selected_algorithm\":\"-1\"
      )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
  EXPECT_TRUE(RunAndCompare(*get_module(), ErrorSpec{1e-5, 1e-5}));
}

TEST_F(GemmRewriteTest, ArgTransposeFoldCheck) {
  const char* hlo_text = R"(
HloModule ArgTransposeFoldGemm

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  x_transposed = f32[2,2] transpose(x), dimensions={1, 0}
  ROOT dot_a = f32[2,2] dot(x_transposed, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    ROOT %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"0\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"batch_size\":\"1\",\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}

TEST_F(GemmRewriteTest, InstrTransposeFoldCheck) {
  const char* hlo_text = R"(
HloModule InstrTransposeFoldGemm

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,2] transpose(dot_a), dimensions={1, 0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    ROOT %custom-call = f32[2,2]{1,0} custom-call(%y, %x), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"0\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"batch_size\":\"1\",\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}

TEST_F(GemmRewriteTest, AlphaSimpleRewrite) {
  const char* hlo_text = R"(
HloModule AlphaSimpleRewrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  k = f32[] constant(3.0)
  k_broadcast = f32[2, 2] broadcast(k), dimensions={}
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    ROOT %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":3,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"batch_size\":\"1\",\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}

TEST_F(GemmRewriteTest, ComplexAlphaSimpleRewrite) {
  const char* hlo_text = R"(
HloModule ComplexAlphaSimpleRewrite

ENTRY AddDotsFunc {
  x = c64[2,2] parameter(0)
  y = c64[2,2] parameter(1)
  k = c64[] constant((3.0, 3.0))
  k_broadcast = c64[2, 2] broadcast(k), dimensions={}
  dot_a = c64[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_a_multiplied = c64[2, 2] multiply(dot_a, k_broadcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: c64[2,2], y: c64[2,2]) -> c64[2,2] {
; CHECK-NEXT:    %x = c64[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = c64[2,2]{1,0} parameter(1)
; CHECK-NEXT:    ROOT %custom-call = c64[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":3,\"alpha_imag\":3,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"batch_size\":\"1\",\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}

TEST_F(GemmRewriteTest, AlphaMultipleUsersNoRewrite) {
  const char* hlo_text = R"(
HloModule AlphaMultipleUsersNoRewrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  k = f32[] constant(3.0)
  k_broadcast = f32[2, 2] broadcast(k), dimensions={}
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  ROOT out = f32[2,2] add(dot_a_multiplied, dot_a)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK:    %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"batch_size\":\"1\",\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}

TEST_F(GemmRewriteTest, AlphaVectorNoRewrite) {
  const char* hlo_text = R"(
HloModule AlphaVectorNoRewrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  alpha = f32[2] constant({1, 2})
  alpha_broadcast = f32[2,2] broadcast(alpha), dimensions={1}
  dot = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_a_multiplied = f32[2, 2] multiply(dot, alpha_broadcast)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"batch_size\":\"1\",\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}

TEST_F(GemmRewriteTest, AlphaBetaRewrite) {
  const char* hlo_text = R"(
HloModule NonZeroAlphaBeta

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] parameter(2)
  k = f32[] constant(3.0)
  k_broadcast = f32[2, 2] broadcast(k), dimensions={}
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  ROOT out = f32[2,2] add(dot_a_multiplied, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], bias: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    %bias = f32[2,2]{1,0} parameter(2)
; CHECK-NEXT:    ROOT %custom-call.1 = f32[2,2]{1,0} custom-call(%x, %y, %bias), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":3,\"alpha_imag\":0,\"beta\":1,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"batch_size\":\"1\",\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}

TEST_F(GemmRewriteTest, BiasMultipleUsersNoRewrite) {
  const char* hlo_text = R"(
HloModule BiasMultipleUsersNoRewrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] parameter(2)
  k = f32[] constant(3.0)
  k_broadcast = f32[2, 2] broadcast(k), dimensions={}
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  biased_out = f32[2,2] add(dot_a_multiplied, bias)
  ROOT out = f32[2,2] add(biased_out, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], bias: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %bias = f32[2,2]{1,0} parameter(2)
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":3,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"batch_size\":\"1\",\"selected_algorithm\":\"{{-?[0-9]+}}\"}"
      )");
}

TEST_F(GemmRewriteTest, SharedBufferAssignment) {
  const char* hlo_text = R"(
HloModule SharedBufferAssignment

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] add(x, y)
  dot = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,2] add(dot, bias)
}

)";

  // Bias should be fused into the multiplication.
  CheckNumberOfAllocations(hlo_text, 3);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(GemmRewriteTest, BF16Gemm) {
  const char* hlo_text = R"(
HloModule bf16gemm

ENTRY bf16gemm {
  %parameter.1 = bf16[12,4]{1,0} parameter(0)
  %parameter.2 = bf16[4,8]{1,0} parameter(1)
  ROOT %dot.8 = bf16[12,8] dot(bf16[12,4] %parameter.1, bf16[4,8] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: bf16[16,8]{1,0} custom-call(bf16[16,8]{1,0} {{.*}}, bf16[8,8]{1,0} {{.*}}), custom_call_target="__cublas$gemm"
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: bf16[12,8]{1,0} custom-call(bf16[12,4]{1,0} %parameter.1, bf16[4,8]{1,0} %parameter.2), custom_call_target="__cublas$gemm"

  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_F(GemmRewriteTest, BF16GemmStrided) {
  const char* hlo_text = R"(
HloModule bf16gemm

ENTRY bf16gemm {
  %parameter.1 = bf16[3,3,4] parameter(0)
  %parameter.2 = bf16[3,3,2] parameter(1)
  ROOT %dot.3 = bf16[3,4,2]{2,1,0} dot(bf16[3,3,4]{2,1,0} %parameter.1, bf16[3,3,2]{2,1,0} %parameter.2), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}, operand_precision={highest,highest}
}

  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
    ; CHECK: bf16[3,8,8]{2,1,0} custom-call(bf16[3,8,8]{2,1,0} {{.*}}, bf16[3,8,8]{2,1,0} {{.*}}), custom_call_target="__cublas$gemm"
    )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
    ; CHECK: ROOT %custom-call = bf16[3,4,2]{2,1,0} custom-call(bf16[3,3,4]{2,1,0} %parameter.1, bf16[3,3,2]{2,1,0} %parameter.2), custom_call_target="__cublas$gemm"
    )",
                      /*print_operand_shape=*/true);
  }
}

TEST_F(GemmRewriteTest, BF16GemmCodeGen) {
  const char* hlo_text = R"(
HloModule bf16codegendgemm

ENTRY bf16gemm {
  %parameter.1 = bf16[2]{0} parameter(0)
  %parameter.2 = bf16[2]{0} parameter(1)
  ROOT %dot.3 = bf16[] dot(bf16[2]{0} %parameter.1, bf16[2]{0} %parameter.2), lhs_contracting_dims={0}, rhs_contracting_dims={0}, operand_precision={highest,highest}
}
  )";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK:  %param_1.3 = bf16[2]{0} parameter(1)
; CHECK:  %convert.5 = f32[2]{0} convert(%param_1.3)
; CHECK:  %param_0.3 = bf16[2]{0} parameter(0)
; CHECK:  %convert.4 = f32[2]{0} convert(%param_0.3)
; CHECK:  %multiply.1 = f32[2]{0} multiply(%convert.5, %convert.4)
; CHECK:  %constant_1 = f32[] constant(0)
; CHECK:  %reduce.1 = f32[] reduce(%multiply.1, %constant_1), dimensions={0}, to_apply=%scalar_add_computation
; CHECK:  ROOT %convert.3 = bf16[] convert(%reduce.1)
  )");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(GemmRewriteTest, BF16Transpose) {
  const char* hlo_text = R"(
HloModule broadcast

ENTRY broadcast {
  p = bf16[9] parameter(0)
  ROOT out = bf16[1,9] broadcast(p), dimensions={1}
}
)";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: bf16[1,9]{1,0} bitcast
; CHECK: bf16[1,9]{1,0} copy
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(GemmRewriteTest, Int8Gemm) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[12,4]{1,0} parameter(0)
  %parameter.2 = s8[4,8]{1,0} parameter(1)
  ROOT %dot.8 = s32[12,8] dot(s8[12,4] %parameter.1, s8[4,8] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} custom-call(s8[12,4]{1,0} %parameter.1, s8[4,8]{1,0} %parameter.2), custom_call_target="__cublas$gemm"
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} dot(s32[12,4]{1,0} %bitcast, s32[4,8]{1,0} %bitcast.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_F(GemmRewriteTest, Int8GemmNoAlphaRewrite) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[12,4]{1,0} parameter(0)
  %parameter.2 = s8[4,8]{1,0} parameter(1)
  k = s32[] constant(2)
  k_broadcast = s32[12,8] broadcast(k), dimensions={}
  %dot.8 = s32[12,8] dot(s8[12,4] %parameter.1, s8[4,8] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_multiplied = s32[12,8] multiply(%dot.8, k_broadcast)
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} custom-call(s8[12,4]{1,0} %parameter.1, s8[4,8]{1,0} %parameter.2), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} dot(s32[12,4]{1,0} %bitcast, s32[4,8]{1,0} %bitcast.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_F(GemmRewriteTest, Int8GemmNoBetaRewrite) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[12,4]{1,0} parameter(0)
  %parameter.2 = s8[4,8]{1,0} parameter(1)
  bias = s32[12,8] parameter(2)
  %dot.8 = s32[12,8] dot(s8[12,4] %parameter.1, s8[4,8] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = s32[12,8] add(%dot.8, bias)
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} custom-call(s8[12,4]{1,0} %parameter.1, s8[4,8]{1,0} %parameter.2), custom_call_target="__cublas$gemm", backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} dot(s32[12,4]{1,0} %bitcast, s32[4,8]{1,0} %bitcast.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_F(GemmRewriteTest, Int8GemmNotMultipleOfFour) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[13,4]{1,0} parameter(0)
  %parameter.2 = s8[4,9]{1,0} parameter(1)
  ROOT %dot.9 = s32[13,9] dot(s8[13,4] %parameter.1, s8[4,9] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[16,12]{1,0} custom-call(s8[16,4]{1,0} %pad, s8[4,12]{1,0} %pad.1)
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[13,9]{1,0} dot(s32[13,4]{1,0} %bitcast, s32[4,9]{1,0} %bitcast.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                      /*print_operand_shape=*/true);
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
