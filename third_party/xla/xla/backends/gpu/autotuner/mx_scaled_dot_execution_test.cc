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
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// ROCm backends that can lower a scaled-dot fusion. HIPBLASLT_FISSION uses no
// MX: it converts the operands and scales to bf16 and runs a regular GEMM.
constexpr autotuner::Backend kTriton = autotuner::Backend::TRITON;
constexpr autotuner::Backend kHipblasLt = autotuner::Backend::HIPBLASLT;
constexpr autotuner::Backend kHipblasLtFission =
    autotuner::Backend::HIPBLASLT_FISSION;
constexpr autotuner::Backend kAllMxBackends[] = {kTriton, kHipblasLt,
                                                 kHipblasLtFission};

struct MxBackendParam {
  autotuner::Backend backend;
  // FileCheck regex, so the '$' in the custom call targets is escaped.
  absl::string_view expected_target_regex;
  absl::string_view name;
};

class MxScaledDotExecutionTest
    : public HloPjRtGpuTestBase,
      public ::testing::WithParamInterface<MxBackendParam> {
 protected:
  void SetUp() override {
    const auto& gpu_cc = device_description().gpu_compute_capability();
    const auto* rocm_cc = gpu_cc.rocm_compute_capability();
    if (rocm_cc == nullptr || !rocm_cc->has_mx_type_support()) {
      GTEST_SKIP() << "MX execution test requires MX type support (gfx950+).";
    }
  }

  // If the backend under test is in `accepted_by`, checks its lowering against
  // the non-MX reference; otherwise checks that it rejects the scaled dot.
  void RunMxTest(absl::string_view hlo_string, const ErrorSpec& error_spec,
                 absl::Span<const autotuner::Backend> accepted_by) {
    HloModuleConfig test_config = GetModuleConfigForTest();
    test_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(true);
    test_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(true);
    // All three backends work on the Triton GEMM fusion wrapping the scaled
    // dot, so the two flags above must stay on. Enable only the backend under
    // test, plus NATIVE_EMITTER for the other, non-GEMM fusions.
    test_config.mutable_debug_options().clear_xla_config_assigner_backends();
    test_config.mutable_debug_options().add_xla_config_assigner_backends(
        GetParam().backend);
    test_config.mutable_debug_options().add_xla_config_assigner_backends(
        autotuner::Backend::NATIVE_EMITTER);

    if (!absl::c_linear_search(accepted_by, GetParam().backend)) {
      // No other enabled backend takes GEMM fusions, so compilation must fail.
      EXPECT_THAT(GetOptimizedModule(hlo_string, test_config),
                  StatusIs(absl::StatusCode::kNotFound,
                           HasSubstr("No supported configs found")));
      return;
    }

    HloModuleConfig ref_config = GetModuleConfigForTest();
    ref_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(false);
    ref_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
    ASSERT_OK_AND_ASSIGN(auto ref_optimized,
                         GetOptimizedModule(hlo_string, ref_config));
    EXPECT_THAT(
        RunFileCheck(ref_optimized->ToString(),
                     R"(CHECK: {{__cublas\$lt\$matmul|__cublas\$gemm}})"),
        IsOkAndHolds(true));

    ASSERT_OK_AND_ASSIGN(auto test_optimized,
                         GetOptimizedModule(hlo_string, test_config));
    EXPECT_THAT(
        RunFileCheck(
            test_optimized->ToString(),
            absl::StrCat("CHECK: {{", GetParam().expected_target_regex, "}}")),
        IsOkAndHolds(true));

    EXPECT_TRUE(RunAndCompareTwoModules(std::move(test_optimized),
                                        std::move(ref_optimized), error_spec,
                                        /*run_hlo_passes=*/false));
  }
};

// *****************************************************************************
// *                               All backends                                *
// *****************************************************************************

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

TEST_P(MxScaledDotExecutionTest, MxFp8) {
  RunMxTest(kMxFp8Hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5), kAllMxBackends);
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

TEST_P(MxScaledDotExecutionTest, MxFp8Batched) {
  RunMxTest(kMxFp8BatchedHlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
            kAllMxBackends);
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

TEST_P(MxScaledDotExecutionTest, MxFp4) {
  RunMxTest(kMxFp4Hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5), kAllMxBackends);
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

TEST_P(MxScaledDotExecutionTest, MxFp4Batched) {
  RunMxTest(kMxFp4BatchedHlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
            kAllMxBackends);
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

TEST_P(MxScaledDotExecutionTest, MxFp8MixedTypesBatched) {
  RunMxTest(kMxFp8MixedTypesBatchedHlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
            kAllMxBackends);
}

// The scaled-dot operands are reshapes of higher-rank parameters, mimicking the
// in-graph block-scaled quantization case where the fp8 operand is produced in
// [batch, M, K/block, block] form and only flattened to [batch, M, K] by a
// reshape/bitcast that gets absorbed into the gemm fusion. This exercises the
// path where the fusion's external operands have a different rank than the
// scaled-dot operands; the MX rewrite must re-bitcast them back so the custom
// call operands stay consistent with the dot dimension numbers.
constexpr absl::string_view kMxFp8ReshapedOperandsHlo = R"(
HloModule mx_fp8_reshaped_operands_test
ENTRY main {
  %lhs_blocked = f8e4m3fn[1,32,8,32] parameter(0)
  %rhs_blocked = f8e4m3fn[1,16,8,32] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  %lhs = f8e4m3fn[1,32,256] reshape(%lhs_blocked)
  %rhs = f8e4m3fn[1,16,256] reshape(%rhs_blocked)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_P(MxScaledDotExecutionTest, MxFp8ReshapedOperands) {
  RunMxTest(kMxFp8ReshapedOperandsHlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
            kAllMxBackends);
}

// *****************************************************************************
// *                            hipBLASLt-specific                             *
// *****************************************************************************

// hipBLASLt MX requires a batch size of 1.
constexpr absl::string_view kMxFp8Batch2Hlo = R"(
HloModule mx_fp8_batch2_test
ENTRY main {
  %lhs = f8e4m3fn[2,32,256] parameter(0)
  %rhs = f8e4m3fn[2,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[2,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[2,16,8] parameter(3)
  ROOT %result = f32[2,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_P(MxScaledDotExecutionTest, MxFp8Batch2) {
  RunMxTest(kMxFp8Batch2Hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
            {kTriton, kHipblasLtFission});
}

// hipBLASLt MX requires M and N to be multiples of 16; here N is 8.
constexpr absl::string_view kMxFp8N8Hlo = R"(
HloModule mx_fp8_n8_test
ENTRY main {
  %lhs = f8e4m3fn[32,256] parameter(0)
  %rhs = f8e4m3fn[8,256] parameter(1)
  %lhs_scale = f8e8m0fnu[32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[8,8] parameter(3)
  ROOT %result = f32[32,8] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

TEST_P(MxScaledDotExecutionTest, MxFp8N8) {
  RunMxTest(kMxFp8N8Hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
            {kTriton, kHipblasLtFission});
}

// *****************************************************************************
// *                              Triton-specific                              *
// *****************************************************************************

// The Triton backend has no configs for mixed fp4 x fp8 operands on ROCm.
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

TEST_P(MxScaledDotExecutionTest, MxFp4Fp8MixedBatched) {
  RunMxTest(kMxFp4Fp8MixedBatchedHlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
            {kHipblasLt, kHipblasLtFission});
}

INSTANTIATE_TEST_SUITE_P(
    MxBackends, MxScaledDotExecutionTest,
    ::testing::Values(
        MxBackendParam{kTriton, R"(__triton_nested_gemm_fusion)", "Triton"},
        MxBackendParam{kHipblasLt,
                       R"(custom_call_target="__cublas\$lt\$matmul\$mx")",
                       "HipblasLt"},
        MxBackendParam{kHipblasLtFission,
                       R"(custom_call_target="__cublas\$lt\$matmul")",
                       "HipblasLtFission"}),
    [](const ::testing::TestParamInfo<MxBackendParam>& info) {
      return std::string(info.param.name);
    });

}  // namespace
}  // namespace xla::gpu
