/* Copyright 2025 The OpenXLA Authors.

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

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class MxScaledDotExecutionTest : public HloPjRtGpuTestBase {
 protected:
  void SetUp() override {
    const auto& gpu_cc = device_description().gpu_compute_capability();
    const auto* rocm_cc = gpu_cc.rocm_compute_capability();
    if (rocm_cc == nullptr || !rocm_cc->has_mx_type_support()) {
      GTEST_SKIP() << "MX execution test requires MX type support (gfx950+).";
    }
  }

  // Runs numerical correctness and verifies that the optimized HLO uses the
  // expected custom call target.
  void RunMxCorrectnessTest(absl::string_view hlo_string,
                            const ErrorSpec& error_spec) {
    HloModuleConfig ref_config = GetModuleConfigForTest();
    ref_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(false);
    ref_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
    TF_ASSERT_OK_AND_ASSIGN(auto ref_optimized,
                            GetOptimizedModule(hlo_string, ref_config));
    EXPECT_THAT(
        RunFileCheck(ref_optimized->ToString(),
                     R"(CHECK: {{__cublas\$lt\$matmul|__cublas\$gemm}})"),
        absl_testing::IsOkAndHolds(true));

    HloModuleConfig test_config = GetModuleConfigForTest();
    test_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(true);
    test_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(true);
    TF_ASSERT_OK_AND_ASSIGN(auto test_optimized,
                            GetOptimizedModule(hlo_string, test_config));
    // The autotuner may pick any of the MX-aware ROCm backends:
    //   __triton_nested_gemm_fusion -> Triton
    //   __cublas$lt$matmul$mx       -> hipBLASLt
    //   __cublas$lt$matmul          -> HIPBLASLT_FISSION (should never win)
    EXPECT_THAT(
        RunFileCheck(
            test_optimized->ToString(),
            R"(CHECK: {{__triton_nested_gemm_fusion|__cublas\$lt\$matmul\$mx}})"),
        absl_testing::IsOkAndHolds(true));

    EXPECT_TRUE(RunAndCompareTwoModules(std::move(test_optimized),
                                        std::move(ref_optimized), error_spec,
                                        /*run_hlo_passes=*/false));
  }
};

constexpr absl::string_view kMxFp8Hlo = R"(
HloModule mx_test
ENTRY main {
  %lhs = f8e4m3fn[32,256] parameter(0)
  %rhs = f8e4m3fn[16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[16,8] parameter(3)
  ROOT %result = f32[32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

TEST_F(MxScaledDotExecutionTest, MxFp8Correctness) {
  RunMxCorrectnessTest(kMxFp8Hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp8BatchedHlo = R"(
HloModule mx_batched_test
ENTRY main {
  %lhs = f8e4m3fn[1,32,256] parameter(0)
  %rhs = f8e4m3fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_F(MxScaledDotExecutionTest, MxFp8BatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp8BatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp8MixedTypesBatchedHlo = R"(
HloModule mx_fp8_mixed_types_batched_test
ENTRY main {
  %lhs = f8e4m3fn[1,32,256] parameter(0)
  %rhs = f8e5m2[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_F(MxScaledDotExecutionTest, MxFp8MixedTypesBatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp8MixedTypesBatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp4Hlo = R"(
HloModule mx_fp4_test
ENTRY main {
  %lhs = f4e2m1fn[32,256] parameter(0)
  %rhs = f4e2m1fn[16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[16,8] parameter(3)
  ROOT %result = f32[32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

TEST_F(MxScaledDotExecutionTest, MxFp4Correctness) {
  RunMxCorrectnessTest(kMxFp4Hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp4BatchedHlo = R"(
HloModule mx_fp4_batched_test
ENTRY main {
  %lhs = f4e2m1fn[1,32,256] parameter(0)
  %rhs = f4e2m1fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_F(MxScaledDotExecutionTest, MxFp4BatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp4BatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp4Fp8MixedBatchedHlo = R"(
HloModule mx_fp4_fp8_mixed_batched_test
ENTRY main {
  %lhs = f4e2m1fn[1,32,256] parameter(0)
  %rhs = f8e4m3fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_F(MxScaledDotExecutionTest, MxFp4Fp8MixedBatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp4Fp8MixedBatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

}  // namespace
}  // namespace xla::gpu
