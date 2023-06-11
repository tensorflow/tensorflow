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

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/tensor_float_32_utils.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {

namespace {

namespace m = ::xla::match;

class GemmRewriteTest : public GpuCodegenTest {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
  void SetUp() override {
    tf32_state_ = tsl::tensor_float_32_execution_enabled();
    tsl::enable_tensor_float_32_execution(false);
  }
  void TearDown() override {
    tsl::enable_tensor_float_32_execution(tf32_state_);
  }

 private:
  bool tf32_state_;
};

// A test fixture class for tests which should have similar results with legacy
// cublas and cublasLt
class ParameterizedGemmRewriteTest
    : public GemmRewriteTest,
      public ::testing::WithParamInterface<bool> {
 public:
  ParameterizedGemmRewriteTest() {
    const bool kUsingCublasLt = GetParam();
    replacements_[kCustomCallTargetPlaceholder] =
        kUsingCublasLt ? "__cublas$lt$matmul" : "__cublas$gemm";
  }
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GemmRewriteTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(GetParam());
    return debug_options;
  }
  void MatchOptimizedHlo(absl::string_view hlo, const absl::string_view pattern,
                         bool print_operand_shape = false) {
    GemmRewriteTest::MatchOptimizedHlo(
        hlo, absl::StrReplaceAll(pattern, replacements_), print_operand_shape);
  }
  absl::string_view CustomCallTarget() {
    return replacements_[kCustomCallTargetPlaceholder];
  }

 protected:
  absl::flat_hash_map<absl::string_view, absl::string_view> replacements_;

 private:
  static constexpr const char* kCustomCallTargetPlaceholder{
      "<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"};
};

class ParameterizedFp8GemmRewriteTest : public ParameterizedGemmRewriteTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options =
        ParameterizedGemmRewriteTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_xla_runtime_executable(false);
    return debug_options;
  }

 protected:
  void CheckFp8IfOnHopper(absl::string_view hlo_text,
                          ErrorSpec error_spec = ErrorSpec{1e-2, 1e-2}) {
    if (!GetCudaComputeCapability().IsAtLeast(
            se::CudaComputeCapability::HOPPER)) {
      return;
    }
    EXPECT_TRUE(RunAndCompare(hlo_text, error_spec));
    // Most FP8 tests directly create a GemmRewriter and check the output.
    // Here, also run the entire HLO pass pipeline to ensure no other passes
    // interfere with GemmRewriter's pattern matching.
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo_text));
    const HloInstruction* call =
        FindInstruction(optimized_module.get(), HloOpcode::kCustomCall);
    ASSERT_NE(call, nullptr);
    EXPECT_EQ(call->custom_call_target(), "__cublas$lt$matmul$f8");
  }
};

TEST_P(ParameterizedFp8GemmRewriteTest, DoNotRewriteToF8OnPreHopper) {
  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP() << "Test requires a pre-Hopper GPU.";
  }
  const char* hlo_text = R"(
    HloModule test

    ENTRY PreHopperTest {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      ROOT out = f8e4m3fn[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %PreHopperTest (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16]) -> f8e4m3fn[16,16] {
; CHECK:    {{.*}} = f16[16,16]{1,0} custom-call({{.*}}, {{.*}})
; CHECK-DAG:  custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"
          )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, UnsupportedTypesF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  // Test with types unsupported by cuBLAS LT when FP8 is used. cuBLAS LT with
  // FP8 requires one of the operands to be F8E4M3FN.
  const char* hlo_text = R"(
    HloModule test

    ENTRY unsupported_types {
      x = f8e5m2[16,16] parameter(0)
      y = f8e5m2[16,16] parameter(1)
      ROOT out = f8e5m2[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
  RunAndFilecheckHloRewrite(hlo_text, GemmRewriter(GetCudaComputeCapability()),
                            absl::StrReplaceAll(R"(
; CHECK-LABEL: ENTRY %unsupported_types (x: f8e5m2[16,16], y: f8e5m2[16,16]) -> f8e5m2[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e5m2[16,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P0_CONVERT:%[^ ]+]] = f16[16,16]{1,0} convert([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e5m2[16,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_CONVERT:%[^ ]+]] = f16[16,16]{1,0} convert([[P1]])
; CHECK-NEXT:    [[DOT:%[^ ]+]] = f16[16,16]{1,0} custom-call([[P0_CONVERT]], [[P1_CONVERT]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f8e5m2[16,16]{1,0} convert([[DOT]])
      )",
                                                replacements_));
}

TEST_P(ParameterizedFp8GemmRewriteTest, UnscaledABUnscaledDF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      ROOT out = f8e4m3fn[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16]) -> f8e4m3fn[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f8e4m3fn[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C1]], [[C1]], [[C1]], /*index=5*/[[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[C1]], /*index=5*/[[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDPaddedF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[13,17] parameter(0)
      y = f8e4m3fn[17,31] parameter(1)
      x_f32 = f32[13,17] convert(x)
      y_f32 = f32[17,31] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[13,17] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[17,31] broadcast(y_scale), dimensions={}
      x_unscaled = f32[13,17] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[17,31] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[13,31] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[13,17], y: f8e4m3fn[17,31], x_scale: f32[], y_scale: f32[]) -> f32[13,31] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[13,17]{1,0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f8e4m3fn[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = f8e4m3fn[16,32]{1,0} pad([[P0]], [[C0]]), padding=0_3x0_15
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[17,31]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[31,17]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:%[^ ]+]] = f8e4m3fn[] constant(0)
; CHECK-NEXT:    [[P1_TRANSPOSE_PADDED:%[^ ]+]] = f8e4m3fn[32,32]{1,0} pad([[P1_TRANSPOSE]], [[C1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C4:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[DOT:%[^ ]+]] = f32[16,32]{1,0} custom-call([[P0_PADDED]], [[P1_TRANSPOSE_PADDED]], [[P2]], [[P3]], [[C4]], /*index=5*/[[C4]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT: ROOT [[OUT:%[^ ]+]] = f32[13,31]{1,0} slice([[DOT]]), slice={[0:13], [0:31]}
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDBitcastF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[2,8,16] parameter(0)
      y = f8e4m3fn[16,16] parameter(1)
      x_f32 = f32[2,8,16] convert(x)
      y_f32 = f32[16,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[2,8,16] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[16,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[2,8,16] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[16,16] multiply(y_f32, y_scale_bcast)
      x_bitcast = f32[16,16] bitcast(x_unscaled)
      ROOT out = f32[16,16] dot(x_bitcast, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(
      se::CudaComputeCapability{se::CudaComputeCapability::HOPPER, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::CustomCall({"__cublas$lt$matmul$f8"}).WithShape(F32, {16, 16})));
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDUnaryOpsF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[3] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[3] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[3] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[3] multiply(x_f32, x_scale_bcast)
      zero = f32[] constant(0)
      x_unscaled_padded = f32[30] pad(x_unscaled, zero), padding=0_27
      x_unscaled_padded_bcast = f32[30,8,5] broadcast(x_unscaled_padded), dimensions={0}
      x_unscaled_padded_bcast_sliced = f32[16,8,4] slice(x_unscaled_padded_bcast), slice={[2:18], [0:8], [0:4]}
      x_unscaled_padded_bcast_sliced_reshaped = f32[16,32] reshape(x_unscaled_padded_bcast_sliced)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[16,16] dot(x_unscaled_padded_bcast_sliced_reshaped, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";
  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[3], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[3]{0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C0_CONVERT:%[^ ]+]] = f8e4m3fn[] convert([[C0]])
; CHECK-NEXT:    [[P0_U0:%[^ ]+]] = f8e4m3fn[30]{0} pad([[P0]], [[C0_CONVERT]]), padding=0_27
; CHECK-NEXT:    [[P0_U1:%[^ ]+]] = f8e4m3fn[30,8,5]{2,1,0} broadcast([[P0_U0]]), dimensions={0}
; CHECK-NEXT:    [[P0_U2:%[^ ]+]] = f8e4m3fn[16,8,4]{2,1,0} slice([[P0_U1]]), slice={[2:18], [0:8], [0:4]}
; CHECK-NEXT:    [[P0_U3:%[^ ]+]] = f8e4m3fn[16,32]{1,0} reshape([[P0_U2]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0_U3]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[C2]], /*index=5*/[[C2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, BatchedScaledABUnscaledDF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[10,16,32] parameter(0)
      y = f8e4m3fn[10,32,16] parameter(1)
      x_f32 = f32[10,16,32] convert(x)
      y_f32 = f32[10,32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[10,16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[10,32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[10,16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[10,32,16] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[10,16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[10,16,32], y: f8e4m3fn[10,32,16], x_scale: f32[], y_scale: f32[]) -> f32[10,16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[10,16,32]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[10,32,16]{2,1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[10,16,32]{2,1,0} transpose([[P1]]), dimensions={0,2,1}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[10,16,16]{2,1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[C1]], /*index=5*/[[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"2\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"2\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:           \"rhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABAlphaDF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      k = f32[] constant(3.0)
      k_bcast = f32[16,16] broadcast(k), dimensions={}
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT out = f32[16,16] multiply(dot_a, k_bcast)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[C1]], /*index=5*/[[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":3
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDReluActivationF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      c = f32[] constant(0)
      c_bcast = f32[16,16] broadcast(c), dimensions={}
      ROOT out = f32[16,16] maximum(dot_a, c_bcast)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[C1]], /*index=5*/[[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, InvScaledABUnscaledDF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] divide(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] divide(y_f32, y_scale_bcast)
      ROOT out = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDMatrixBiasF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      b = f32[16,16] parameter(2)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT out = add(dot_a, b)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f32[16,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[16,16]{1,0} parameter(2)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0]], [[P2]], [[P3]], /*index=5*/[[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDMatrixBiasPaddedF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[14,31] parameter(0)
      y = f8e4m3fn[31,14] parameter(1)
      b = f32[14,14] parameter(2)
      x_f32 = f32[14,31] convert(x)
      y_f32 = f32[31,14] convert(y)
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[14,31] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[31,14] broadcast(y_scale), dimensions={}
      x_unscaled = f32[14,31] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[31,14] multiply(y_f32, y_scale_bcast)
      dot_a = f32[14,14] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT out = add(dot_a, b)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[14,31], y: f8e4m3fn[31,14], b: f32[14,14], x_scale: f32[], y_scale: f32[]) -> f32[14,14] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[14,31]{1,0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f8e4m3fn[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = f8e4m3fn[16,32]{1,0} pad([[P0]], [[C0]]), padding=0_2x0_1
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[31,14]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[14,31]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:%[^ ]+]] = f8e4m3fn[] constant(0)
; CHECK-NEXT:    [[P1_TRANSPOSE_PADDED:%[^ ]+]] = f8e4m3fn[16,32]{1,0} pad([[P1_TRANSPOSE]], [[C1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[14,14]{1,0} parameter(2)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[P2_PADDED:%[^ ]+]] = f32[16,16]{1,0} pad([[P2]], [[C2]]), padding=0_2x0_2
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[C3:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[DOT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0_PADDED]], [[P1_TRANSPOSE_PADDED]], [[P2_PADDED]], [[P3]], [[P4]], /*index=5*/[[C3]], [[C3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT: ROOT [[OUT:%[^ ]+]] = f32[14,14]{1,0} slice([[DOT]]), slice={[0:14], [0:14]}
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      z_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_scaled = f32[16,16] divide(dot_a, z_scale_bcast)
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[], z_scale: f32[]) -> f8e4m3fn[16,16] {
; CHECK:         [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f8e4m3fn[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[C1]], /*index=5*/[[P4_INV]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABInvScaledDF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      z_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_scaled = f32[16,16] multiply(dot_a, z_scale_bcast)
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-NOT:     divide

; CHECK:           custom_call_target="__cublas$lt$matmul$f8",

      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDReluActivationF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      z_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      c = f32[] constant(0)
      c_bcast = f32[16,16] broadcast(c), dimensions={}
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      relu_a = f32[16,16] maximum(dot_a, c_bcast)
      relu_a_scaled = f32[16,16] divide(relu_a, z_scale_bcast)
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      relu_a_clamped = f32[16,16] clamp(c1_bcast, relu_a_scaled, c2_bcast)
      ROOT out = f8e4m3fn[16,16] convert(relu_a_clamped)
          }
)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[], z_scale: f32[]) -> f8e4m3fn[16,16] {
; CHECK:         [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f8e4m3fn[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[C1]], /*index=5*/[[P4_INV]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDMatrixBiasF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f16 = f16[16,32] convert(x)
      y_f16 = f16[32,16] convert(y)
      b = f16[16,16] parameter(2)
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      z_scale = f16[] parameter(5)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f16[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f16, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f16, y_scale_bcast)
      dot_a = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bias = f16[16,16] add(dot_a, b)
      dot_a_scaled = f16[16,16] divide(dot_a_bias, z_scale_bcast)
      c1 = f16[] constant(-448.)
      c1_bcast = f16[16,16] broadcast(c1), dimensions={}
      c2 = f16[] constant(448.)
      c2_bcast = f16[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f16[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfOnHopper(hlo_text, ErrorSpec{0.1, 0.1});
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f16[16,16], x_scale: f16[], y_scale: f16[], z_scale: f16[]) -> f8e4m3fn[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f16[16,16]{1,0} parameter(2)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK:         [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK:         [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK:         [[P4:%[^ ]+]] = f16[] parameter(5)
; CHECK:       ROOT [[OUT:%[^ ]+]] = f8e4m3fn[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0]], [[DUMMY0:%[^ ]+]], [[DUMMY1:%[^ ]+]], /*index=5*/[[C1]], [[DUMMY2:%[^ ]+]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDVectorBiasF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f16 = f16[16,32] convert(x)
      y_f16 = f16[32,16] convert(y)
      b = f16[16] parameter(2)
      b_bcast = f16[16,16] broadcast(b), dimensions={1}
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      z_scale = f16[] parameter(5)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f16[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f16, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f16, y_scale_bcast)
      dot_a = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bias = f16[16,16] add(dot_a, b_bcast)
      dot_a_scaled = f16[16,16] divide(dot_a_bias, z_scale_bcast)
      c1 = f16[] constant(-448.)
      c1_bcast = f16[16,16] broadcast(c1), dimensions={}
      c2 = f16[] constant(448.)
      c2_bcast = f16[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f16[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfOnHopper(hlo_text, ErrorSpec{0.1, 0.1});
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f16[16], x_scale: f16[], y_scale: f16[], z_scale: f16[]) -> f8e4m3fn[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[CV:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[CV1:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[C:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f16[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f16[] parameter(5)
; CHECK-NEXT:    [[DV:%[^ ]+]] = f16[] divide([[C2]], [[P4]])
; CHECK-NEXT:    [[CV2:%[^ ]+]] = f32[] convert([[DV]])
; CHECK-NEXT:    [[VB:%[^ ]+]] = f16[16]{0} parameter(2)
; CHECK:         ROOT [[OUT:%[^ ]+]] = f8e4m3fn[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[CV]], [[CV1]], [[C]], /*index=5*/[[CV2]], [[VB]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{ 
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[] 
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDF32VectorBiasF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      b = f32[16] parameter(2)
      b_bf16 = bf16[16] convert(b)
      b_f32 = f32[16] convert(b_bf16)
      b_bcast = f32[16,16] broadcast(b_f32), dimensions={1}
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT out = f32[16,16] add(dot_a, b_bcast)
           }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f32[16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK:         [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[C:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[VB:%[^ ]+]] = f32[16]{0} parameter(2)
; CHECK-NEXT:    [[VBC:%[^ ]+]] = bf16[16]{0} convert([[VB]])
; CHECK:         ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[C]], /*index=5*/[[C]], [[VBC]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{ 
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[] 
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, Rank3ScaledABUnscaledDVectorBiasF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "A matrix bias on a matmul is only supported in CUDA 12";
#endif
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = f8e4m3fn[4,16,16] parameter(0)
      y = f8e4m3fn[16,32] parameter(1)
      b = f32[32] parameter(2)
      b_f16 = f16[32] convert(b)
      b_bcast = f16[4,16,32] broadcast(b_f16), dimensions={2}
      x_f16 = f16[4,16,16] convert(x)
      y_f16 = f16[16,32] convert(y)
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      x_scale_bcast = f16[4,16,16] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[16,32] broadcast(y_scale), dimensions={}
      x_unscaled = f16[4,16,16] multiply(x_f16, x_scale_bcast)
      x_unscaled_bitcast = f16[64,16] bitcast(x_unscaled)
      y_unscaled = f16[16,32] multiply(y_f16, y_scale_bcast)
      dot_a = f16[64,32] dot(x_unscaled_bitcast, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bitcast = f16[4,16,32]{2,1,0} bitcast(dot_a)
      ROOT out = f16[4,16,32] add(dot_a_bitcast, b_bcast)
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(
      se::CudaComputeCapability{se::CudaComputeCapability::HOPPER, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Bitcast(m::CustomCall({"__cublas$lt$matmul$f8"})
                                        .WithShape(F16, {64, 32}))
                             .WithShape(F16, {4, 16, 32})));

  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[4,16,16], y: f8e4m3fn[16,32], b: f32[32], x_scale: f16[], y_scale: f16[]) -> f16[4,16,32] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[4,16,16]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f8e4m3fn[64,16]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[32,16]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[P2_CV:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[P3_CV:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[C:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[B:%[^ ]+]] = f32[32]{0} parameter(2)
; CHECK-NEXT:    [[B_F16:%[^ ]+]] = f16[32]{0} convert([[B]])
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f16[64,32]{1,0} custom-call([[P0_BITCAST]], [[P1_TRANSPOSE]], [[P2_CV]], [[P3_CV]], [[C]], /*index=5*/[[C]], [[B_F16]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
; CHECK:         ROOT [[OUT:%[^ ]+]] = f16[4,16,32]{2,1,0} bitcast([[GEMM]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       Rank3ScaledABUnscaledDVectorBiasPaddedF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "A matrix bias on a matmul is only supported in CUDA 12";
#endif
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = f8e4m3fn[4,15,15] parameter(0)
      y = f8e4m3fn[15,31] parameter(1)
      b = f32[31] parameter(2)
      b_f16 = f16[31] convert(b)
      b_bcast = f16[4,15,31] broadcast(b_f16), dimensions={2}
      x_f16 = f16[4,15,15] convert(x)
      y_f16 = f16[15,31] convert(y)
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      x_scale_bcast = f16[4,15,15] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[15,31] broadcast(y_scale), dimensions={}
      x_unscaled = f16[4,15,15] multiply(x_f16, x_scale_bcast)
      x_unscaled_bitcast = f16[60,15] bitcast(x_unscaled)
      y_unscaled = f16[15,31] multiply(y_f16, y_scale_bcast)
      dot_a = f16[60,31] dot(x_unscaled_bitcast, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bitcast = f16[4,15,31]{2,1,0} bitcast(dot_a)
      ROOT out = f16[4,15,31] add(dot_a_bitcast, b_bcast)
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(
      se::CudaComputeCapability{se::CudaComputeCapability::HOPPER, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Bitcast(m::Slice(m::CustomCall({"__cublas$lt$matmul$f8"})
                                         .WithShape(F16, {64, 32}))
                                .WithShape(F16, {60, 31}))
                     .WithShape(F16, {4, 15, 31})));

  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[4,15,15], y: f8e4m3fn[15,31], b: f32[31], x_scale: f16[], y_scale: f16[]) -> f16[4,15,31] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[4,15,15]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f8e4m3fn[60,15]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[C1:%[^ ]+]] = f8e4m3fn[] constant(0)
; CHECK-NEXT:    [[P0_PAD:%[^ ]+]] = f8e4m3fn[64,16]{1,0} pad([[P0_BITCAST]], [[C1]]), padding=0_4x0_1
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[15,31]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[31,15]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C2:%[^ ]+]] = f8e4m3fn[] constant(0)
; CHECK-NEXT:    [[P1_PAD:%[^ ]+]] = f8e4m3fn[32,16]{1,0} pad([[P1_TRANSPOSE]], [[C2]]), padding=0_1x0_1
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[P2_CV:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[P3_CV:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[C:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[B:%[^ ]+]] = f32[31]{0} parameter(2)
; CHECK-NEXT:    [[B_F16:%[^ ]+]] = f16[31]{0} convert([[B]])
; CHECK-NEXT:    [[C3:%[^ ]+]] = f16[] constant(0)
; CHECK-NEXT:    [[P2_PAD:%[^ ]+]] = f16[32]{0} pad([[B_F16]], [[C3]]), padding=0_1
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f16[64,32]{1,0} custom-call([[P0_PAD]], [[P1_PAD]], [[P2_CV]], [[P3_CV]], [[C]], /*index=5*/[[C]], [[P2_PAD]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
; CHECK-NEXT:     [[SLICE:%[^ ]+]] = f16[60,31]{1,0} slice([[GEMM]]), slice={[0:60], [0:31]}
; CHECK-NEXT:     ROOT [[OUT:%[^ ]+]] = f16[4,15,31]{2,1,0} bitcast([[SLICE]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, Rank3ScaledABUnscaledDMatrixBiasF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "A matrix bias on a matmul is only supported in CUDA 12";
#endif
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = f8e4m3fn[4,16,16] parameter(0)
      y = f8e4m3fn[16,32] parameter(1)
      b = f32[4,16,32] parameter(2)
      x_f32 = f32[4,16,16] convert(x)
      y_f32 = f32[16,32] convert(y)
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[4,16,16] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[16,32] broadcast(y_scale), dimensions={}
      x_unscaled = f32[4,16,16] multiply(x_f32, x_scale_bcast)
      x_unscaled_bitcast = f32[64,16] bitcast(x_unscaled)
      y_unscaled = f32[16,32] multiply(y_f32, y_scale_bcast)
      dot_a = f32[64,32] dot(x_unscaled_bitcast, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bitcast = f32[4,16,32]{2,1,0} bitcast(dot_a)
      ROOT out = f32[4,16,32] add(dot_a_bitcast, b)
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(
      se::CudaComputeCapability{se::CudaComputeCapability::HOPPER, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Bitcast(m::CustomCall({"__cublas$lt$matmul$f8"})
                                        .WithShape(F32, {64, 32}))
                             .WithShape(F32, {4, 16, 32})));

  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[4,16,16], y: f8e4m3fn[16,32], b: f32[4,16,32], x_scale: f32[], y_scale: f32[]) -> f32[4,16,32] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[4,16,16]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f8e4m3fn[64,16]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[32,16]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[B:%[^ ]+]] = f32[4,16,32]{2,1,0} parameter(2)
; CHECK-NEXT:    [[B_BITCAST:%[^ ]+]] = f32[64,32]{1,0} bitcast([[B]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[C:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f32[64,32]{1,0} custom-call([[P0_BITCAST]], [[P1_TRANSPOSE]], [[B_BITCAST]], [[P2]], [[P3]], /*index=5*/[[C]], [[C]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK:         ROOT [[OUT:%[^ ]+]] = f32[4,16,32]{2,1,0} bitcast([[GEMM]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       Rank3ScaledABUnscaledDMatrixBiasPaddedF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "A matrix bias on a matmul is only supported in CUDA 12";
#endif
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = f8e4m3fn[3,15,15] parameter(0)
      y = f8e4m3fn[15,31] parameter(1)
      b = f32[3,15,31] parameter(2)
      x_f32 = f32[3,15,15] convert(x)
      y_f32 = f32[15,31] convert(y)
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[3,15,15] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[15,31] broadcast(y_scale), dimensions={}
      x_unscaled = f32[3,15,15] multiply(x_f32, x_scale_bcast)
      x_unscaled_bitcast = f32[45,15] bitcast(x_unscaled)
      y_unscaled = f32[15,31] multiply(y_f32, y_scale_bcast)
      dot_a = f32[45,31] dot(x_unscaled_bitcast, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bitcast = f32[3,15,31]{2,1,0} bitcast(dot_a)
      ROOT out = f32[3,15,31] add(dot_a_bitcast, b)
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(
      se::CudaComputeCapability{se::CudaComputeCapability::HOPPER, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Bitcast(m::Slice(m::CustomCall({"__cublas$lt$matmul$f8"})
                                         .WithShape(F32, {48, 32}))
                                .WithShape(F32, {45, 31}))
                     .WithShape(F32, {3, 15, 31})));

  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[3,15,15], y: f8e4m3fn[15,31], b: f32[3,15,31], x_scale: f32[], y_scale: f32[]) -> f32[3,15,31] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[3,15,15]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f8e4m3fn[45,15]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[C1:%[^ ]+]] = f8e4m3fn[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = f8e4m3fn[48,16]{1,0} pad([[P0_BITCAST]], [[C1]]), padding=0_3x0_1
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[15,31]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[31,15]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C2:%[^ ]+]] = f8e4m3fn[] constant(0)
; CHECK-NEXT:    [[P1_PADDED:%[^ ]+]] = f8e4m3fn[32,16]{1,0} pad([[P1_TRANSPOSE]], [[C2]]), padding=0_1x0_1
; CHECK-NEXT:    [[B:%[^ ]+]] = f32[3,15,31]{2,1,0} parameter(2)
; CHECK-NEXT:    [[B_BITCAST:%[^ ]+]] = f32[45,31]{1,0} bitcast([[B]])
; CHECK-NEXT:    [[C3:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[P2_PADDED:%[^ ]+]] = f32[48,32]{1,0} pad([[B_BITCAST]], [[C3]]), padding=0_3x0_1
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[C:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f32[48,32]{1,0} custom-call([[P0_PADDED]], [[P1_PADDED]], [[P2_PADDED]], [[P2]], [[P3]], /*index=5*/[[C]], [[C]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:      [[SLICE:%[^ ]+]] = f32[45,31]{1,0} slice([[GEMM]]), slice={[0:45], [0:31]}
; CHECK-NEXT:      ROOT [[OUT:%[^ ]+]] = f32[3,15,31]{2,1,0} bitcast([[SLICE]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDVectorBiasThenReluActivationF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      b = f16[16] parameter(2)
      b_bcast = f16[16,16] broadcast(b), dimensions={1}
      x_f32 = f16[16,32] convert(x)
      y_f32 = f16[32,16] convert(y)
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f32, y_scale_bcast)
      c = f16[] constant(0)
      c_bcast = f16[16,16] broadcast(c), dimensions={}
      dot_a0 = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a = f16[16,16] add(dot_a0, b_bcast)
      ROOT out = f16[16,16] maximum(dot_a, c_bcast)
          }
)";

  CheckFp8IfOnHopper(hlo_text, ErrorSpec{2e-3, 0.});
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f16[16], x_scale: f16[], y_scale: f16[]) -> f16[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[CV:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[CV1:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[C:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[VB:%[^ ]+]] = f16[16]{0} parameter(2)
; CHECK     :    ROOT [[OUT:%[^ ]+]] = f16[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[CV]], [[CV1]], [[C]], /*index=5*/[[C]], [[VB]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDMatrixBiasThenVectorBiasF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f16 = f16[16,32] convert(x)
      y_f16 = f16[32,16] convert(y)
      b = f16[16] parameter(2)
      b_bcast = f16[16,16] broadcast(b), dimensions={1}
      b2 = f16[16,16] parameter(3)
      x_scale = f16[] parameter(4)
      y_scale = f16[] parameter(5)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f16, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f16, y_scale_bcast)
      dot_a = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bias1 = f16[16,16] add(dot_a, b2)
      ROOT dot_a_bias = f16[16,16] add(dot_a_bias1, b_bcast)
          }

)";
  CheckFp8IfOnHopper(hlo_text, ErrorSpec{2e-3, 0.});
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL:   ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f16[16], b2: f16[16,16], x_scale: f16[], y_scale: f16[]) -> f16[16,16] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[MB:%[^ ]+]] = f16[16,16]{1,0} parameter(3)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[CV0:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(5)
; CHECK-NEXT:    [[CV1:%[^ ]+]] = f32[] convert([[P3]])
; CHECK:         [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK:         [[GEMMOUT:%[^ ]+]] = f16[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[MB]], [[CV0]], [[CV1]], /*index=5*/[[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{ 
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[] 
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK:         [[VB:%[^ ]+]] = f16[16]{0} parameter(2)
; CHECK:         [[VBC:%[^ ]+]] = f16[16,16]{1,0} broadcast([[VB]]), dimensions={1}
; CHECK:         ROOT [[OUT:%[^ ]+]] = f16[16,16]{1,0} add([[GEMMOUT]], [[VBC]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDWithDAmaxF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] maximum(a, b)
    }

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      z_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      abs_dot_a = f32[16,16] abs(dot_a)
      c0 = f32[] constant(-inf)
      amax = f32[] reduce(abs_dot_a, c0), dimensions={0,1}, to_apply=apply
      dot_a_scaled = f32[16,16] divide(dot_a, z_scale_bcast)
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
      ROOT out = (f8e4m3fn[16,16], f32[]) tuple(dot_a_f8, amax)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[], z_scale: f32[]) -> (f8e4m3fn[16,16], f32[]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f8e4m3fn[16,16]{1,0}, f32[]) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[C1]], /*index=5*/[[P4_INV]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABScaledDWithDAmaxF8WithF16Intermediates) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  // This is the same as ScaledABScaledDWithDAmaxF8, but uses F16 intermediate
  // values instead of F32 intermediate values.
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f16[] parameter(0)
      b = f16[] parameter(1)
      ROOT c = f16[] maximum(a, b)
    }

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f16 = f16[16,32] convert(x)
      y_f16 = f16[32,16] convert(y)
      x_scale = f16[] parameter(2)
      y_scale = f16[] parameter(3)
      z_scale = f16[] parameter(4)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f16[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f16, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f16, y_scale_bcast)
      dot_a = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      abs_dot_a = f16[16,16] abs(dot_a)
      c0 = f16[] constant(-inf)
      amax = f16[] reduce(abs_dot_a, c0), dimensions={0,1}, to_apply=apply
      dot_a_scaled = f16[16,16] divide(dot_a, z_scale_bcast)
      c1 = f16[] constant(-448.)
      c1_bcast = f16[16,16] broadcast(c1), dimensions={}
      c2 = f16[] constant(448.)
      c2_bcast = f16[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f16[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
      ROOT out = (f8e4m3fn[16,16], f16[]) tuple(dot_a_f8, amax)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f16[], y_scale: f16[], z_scale: f16[]) -> (f8e4m3fn[16,16], f16[]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(2)
; CHECK-NEXT:    [[P2_CONVERT:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[P3_CONVERT:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f16[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[P4_INV:%[^ ]+]] = f16[] divide([[C2]], [[P4]])
; CHECK-NEXT:    [[P4_INV_CONVERT:%[^ ]+]] = f32[] convert([[P4_INV]])
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f8e4m3fn[16,16]{1,0}, f32[]) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2_CONVERT]], [[P3_CONVERT]], [[C1]], /*index=5*/[[P4_INV_CONVERT]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABScaledDReluActivationWithDAmaxF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] maximum(a, b)
    }

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      z_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      czero = f32[] constant(0)
      czero_bcast = f32[16,16] broadcast(czero), dimensions={}
      dot_a_relu = f32[16,16] maximum(dot_a, czero_bcast)
      c0 = f32[] constant(-inf)
      amax = f32[] reduce(dot_a_relu, c0), dimensions={0,1}, to_apply=apply
      dot_a_scaled = f32[16,16] divide(dot_a_relu, z_scale_bcast)
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
      ROOT out = (f8e4m3fn[16,16], f32[]) tuple(dot_a_f8, amax)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[], z_scale: f32[]) -> (f8e4m3fn[16,16], f32[]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f8e4m3fn[16,16]{1,0}, f32[]) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[C1]], /*index=5*/[[P4_INV]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDF8Parameterized) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  std::array<std::array<absl::string_view, 7>, 32> combinations;
  int i = 0;

  for (bool d_is_col : {false, true}) {
    for (bool a_is_col : {false, true}) {
      for (bool b_is_col : {false, true}) {
        for (int lhs_contracting_dim : {0, 1}) {
          for (int rhs_contracting_dim : {0, 1}) {
            const absl::string_view lcd =
                lhs_contracting_dim == 1 ? "{1}" : "{0}";
            const absl::string_view rcd =
                rhs_contracting_dim == 1 ? "{1}" : "{0}";
            const absl::string_view a_shape =
                lhs_contracting_dim == 1 ? "[64,32]" : "[32,64]";
            const absl::string_view b_shape =
                rhs_contracting_dim == 0 ? "[32,16]" : "[16,32]";
            const absl::string_view a_layout = a_is_col ? "{0,1}" : "{1,0}";
            const absl::string_view b_layout = b_is_col ? "{0,1}" : "{1,0}";
            const absl::string_view output_layout =
                d_is_col ? "{0,1}" : "{1,0}";
            combinations[i++] = std::array{
                lcd, rcd, a_shape, b_shape, a_layout, b_layout, output_layout};
          }
        }
      }
    }
  }

  const char* hlo_template = R"(
      HloModule test
    ENTRY test {
      x = f8e4m3fn<<Ashape>><<Alayout>> parameter(0)
      x_f32 = f32<<Ashape>><<Alayout>> convert(x)
      x_scale = f32[] parameter(2)
      x_scale_bcast = f32<<Ashape>> broadcast(x_scale), dimensions={}
      x_unscaled = f32<<Ashape>> multiply(x_f32, x_scale_bcast)
      y = f8e4m3fn<<Bshape>><<Blayout>> parameter(1)
      y_f32 = f32<<Bshape>><<Blayout>> convert(y)
      y_scale = f32[] parameter(3)
      y_scale_bcast = f32<<Bshape>> broadcast(y_scale), dimensions={}
      y_unscaled = f32<<Bshape>> multiply(y_f32, y_scale_bcast)
      ROOT out = f32[64,16]<<Olayout>> dot(x_unscaled, y_unscaled), lhs_contracting_dims=<<Lcd>>, rhs_contracting_dims=<<Rcd>>
    }
      )";
  for (const auto& combination : combinations) {
    absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
    replacements["<<Lcd>>"] = std::get<0>(combination);
    replacements["<<Rcd>>"] = std::get<1>(combination);
    replacements["<<Ashape>>"] = std::get<2>(combination);
    replacements["<<Bshape>>"] = std::get<3>(combination);
    replacements["<<Alayout>>"] = std::get<4>(combination);
    replacements["<<Blayout>>"] = std::get<5>(combination);
    replacements["<<Olayout>>"] = std::get<6>(combination);
    const auto hlo_text = absl::StrReplaceAll(hlo_template, replacements);
    CheckFp8IfOnHopper(hlo_text);

    RunAndFilecheckHloRewrite(hlo_text,
                              GemmRewriter(se::CudaComputeCapability{
                                  se::CudaComputeCapability::HOPPER, 0}),
                              R"(
    ; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
          )");
  }
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDF8ParameterizedBatched) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  // TODO(wenscarl): For batched matmaul, not all combinations of A, B and
  // output layouts get pattern matched successfully to FP8 custom call. Only
  // a handful of cases are tested here.
  std::array<std::array<std::string, 7>, 32> combinations;
  std::string lcd, rcd, a_shape, b_shape, a_layout, b_layout, o_layout;
  int i = 0;
  for (bool o_is_col : {false, true}) {
    for (int lhs_contracting_dim : {2, 1}) {
      for (int rhs_contracting_dim : {2, 1}) {
        lcd = lhs_contracting_dim == 2 ? "{2}" : "{1}";
        rcd = rhs_contracting_dim == 2 ? "{2}" : "{1}";
        a_shape = lhs_contracting_dim == 2 ? "[2,64,32]" : "[2,32,64]";
        b_shape = rhs_contracting_dim == 1 ? "[2,32,16]" : "[2,16,32]";
        o_layout = o_is_col ? "{2, 0, 1}" : "{2, 1, 0}";
        for (std::string a_layout : {"{2,1,0}", "{1,2,0}"}) {
          for (std::string b_layout : {"{2,1,0}", "{1,2,0}"}) {
            combinations[i++] = std::array{lcd,      rcd,      a_shape, b_shape,
                                           a_layout, b_layout, o_layout};
          }
        }
      }
    }
  }

  const char* hlo_template = R"(
      HloModule m
ENTRY f {
  x_q = f8e4m3fn<<Ashape>><<Alayout>> parameter(0)
  x_scale = f32[] parameter(2)
  x_scale_broadcast = f32<<Ashape>><<Alayout>> broadcast(x_scale), dimensions={}
  x_q_convert = f32<<Ashape>><<Alayout>> convert(x_q)
  x_qdq = f32<<Ashape>><<Alayout>> multiply(x_q_convert, x_scale_broadcast)

  y_q = f8e4m3fn<<Bshape>><<Blayout>> parameter(1)
  y_scale = f32[] parameter(3)
  y_scale_broadcast = f32<<Bshape>><<Blayout>> broadcast(y_scale), dimensions={}
  y_q_convert = f32<<Bshape>><<Blayout>> convert(y_q)
  y_qdq = f32<<Bshape>><<Blayout>> multiply(y_q_convert, y_scale_broadcast)

  ROOT out = f32[2,64,16]<<Olayout>> dot(x_qdq, y_qdq), lhs_batch_dims={0}, lhs_contracting_dims=<<Lcd>>, rhs_batch_dims={0}, rhs_contracting_dims=<<Rcd>>
}
     )";
  for (const auto& combination : combinations) {
    absl::flat_hash_map<std::string, std::string> replacements;
    replacements["<<Lcd>>"] = std::get<0>(combination);
    replacements["<<Rcd>>"] = std::get<1>(combination);
    replacements["<<Ashape>>"] = std::get<2>(combination);
    replacements["<<Bshape>>"] = std::get<3>(combination);
    replacements["<<Alayout>>"] = std::get<4>(combination);
    replacements["<<Blayout>>"] = std::get<5>(combination);
    replacements["<<Olayout>>"] = std::get<6>(combination);

    const auto hlo_text = absl::StrReplaceAll(hlo_template, replacements);
    CheckFp8IfOnHopper(hlo_text);

    RunAndFilecheckHloRewrite(hlo_text,
                              GemmRewriter(se::CudaComputeCapability{
                                  se::CudaComputeCapability::HOPPER, 0}),
                              R"(
    ; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
          )");
  }
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDF8TF32E5M2) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
#endif  // CUDA_VERSION < 12000
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e5m2[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";
  bool tf32_state_ = tsl::tensor_float_32_execution_enabled();
  tsl::enable_tensor_float_32_execution(true);

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
    ; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
          )");
  tsl::enable_tensor_float_32_execution(tf32_state_);
}

INSTANTIATE_TEST_SUITE_P(Fp8CublasTestsBothLegacyAndLt,
                         ParameterizedFp8GemmRewriteTest, ::testing::Bool());

}  // namespace
}  // namespace gpu
}  // namespace xla
