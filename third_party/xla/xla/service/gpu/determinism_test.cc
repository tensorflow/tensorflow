/* Copyright 2023 The OpenXLA Authors.

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
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

class DeterminismTest : public GpuCodegenTest {
 public:
  DeterminismTest() : debug_options_(HloTestBase::GetDebugOptionsForTest()) {
    debug_options_.set_xla_gpu_exclude_nondeterministic_ops(true);
  }

  // Runs the HLO several times with the same random inputs, and asserts the
  // outputs are bitwise identical.
  void AssertDeterminism(absl::string_view hlo_string, int num_runs = 10) {
    // Set during the first iteration.
    std::vector<Literal> fake_arguments;
    std::vector<Literal*> fake_arguments_ptrs;
    std::optional<Literal> canonical_output;

    for (int i = 0; i < num_runs; ++i) {
      // Clear the autotune cache every iteration to ensure autotuning, if run,
      // is deterministic.
      AutotunerUtil::ClearAutotuneResults();

      TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                              ParseAndReturnVerifiedModule(hlo_string));
      if (i == 0) {
        fake_arguments = MakeFakeArguments(module.get()).value();
        for (Literal& literal : fake_arguments) {
          fake_arguments_ptrs.push_back(&literal);
        }
      }

      TF_ASSERT_OK_AND_ASSIGN(Literal output,
                              Execute(std::move(module), fake_arguments_ptrs));
      if (!canonical_output.has_value()) {
        canonical_output = std::move(output);
      } else {
        ASSERT_TRUE(LiteralTestUtil::Equal(*canonical_output, output));
      }
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    return debug_options_;
  }

  DebugOptions debug_options_;

  enum class TimerCreation { kAllowed, kForbidden };

  // Runs the HLO passes with the given HLO string and matches the
  // resulting HLO against the given expect_hlo_regex using FileCheck.
  //
  // Calls to GpuExecutor::CreateEventBasedTimer can be forbidden by setting
  // timer_creation to kForbidden. (The test fails when the function is called
  // in this case.)
  void MatchOptimizedHlo(absl::string_view hlo_string,
                         absl::string_view expected_hlo_regex,
                         TimerCreation timer_creation) {
    if (timer_creation == TimerCreation::kAllowed) {
      HloTestBase::MatchOptimizedHlo(hlo_string, expected_hlo_regex);
      return;
    }

    // If timer creation is forbidden we inject a mock GPU executor that
    // prevents timer creation.
    TF_ASSERT_OK_AND_ASSIGN(stream_executor::Platform * default_platform,
                            PlatformUtil::GetDefaultPlatform());
    stream_executor::MockStreamExecutor executor;
    EXPECT_CALL(executor, GetPlatform).WillRepeatedly([&] {
      return default_platform;
    });
    EXPECT_CALL(executor, CreateEventBasedTimer).Times(0);
    EXPECT_CALL(executor, GetDeviceDescription)
        .WillRepeatedly([this]() -> const se::DeviceDescription& {
          return backend().default_stream_executor()->GetDeviceDescription();
        });
    EXPECT_CALL(executor, GetPlatform).WillRepeatedly([&]() {
      return default_platform;
    });
    EXPECT_CALL(executor, AsDnn).WillRepeatedly([&]() {
      return backend().default_stream_executor()->AsDnn();
    });
    EXPECT_CALL(executor, device_ordinal).WillRepeatedly([]() { return 0; });

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_string));
    TF_ASSERT_OK_AND_ASSIGN(auto optimized_module,
                            backend().compiler()->RunHloPasses(
                                std::move(module), &executor, GetAllocator()));
    absl::StatusOr<bool> filecheck_result =
        RunFileCheck(optimized_module->ToString(), expected_hlo_regex);
    TF_ASSERT_OK(filecheck_result.status());
    EXPECT_TRUE(filecheck_result.value());
  }

  bool IsVoltaOrLater() const {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability()
        .IsAtLeastVolta();
  }

  bool IsRocm() const {
    return std::holds_alternative<stream_executor::RocmComputeCapability>(
        backend()
            .default_stream_executor()
            ->GetDeviceDescription()
            .gpu_compute_capability());
  }

  bool HasHipblasLt() const {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .rocm_compute_capability()
        .has_hipblaslt();
  }
};

TEST_F(DeterminismTest, CublasDot) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = f32[128,128] parameter(0)
  p1 = f32[128,128] parameter(1)
  ROOT d = f32[128,128] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  if (IsRocm()) {
    if (!HasHipblasLt()) {
      GTEST_SKIP() << "No hipblas-lt support on this architecture!";
    }
    debug_options_.set_xla_gpu_enable_triton_gemm(false);
  }

  debug_options_.set_xla_gpu_triton_fusion_level(0);
  MatchOptimizedHlo(kHloText, R"(; CHECK: custom_call_target="__cublas$gemm")",
                    TimerCreation::kForbidden);
  AssertDeterminism(kHloText);

  debug_options_.set_xla_gpu_enable_cublaslt(true);
  MatchOptimizedHlo(kHloText,
                    R"(; CHECK: custom_call_target="__cublas$lt$matmul")",
                    TimerCreation::kForbidden);
  AssertDeterminism(kHloText);
}

TEST_F(DeterminismTest, DeterministicTritonGemmUsesDefaultConfig) {
  if (!IsVoltaOrLater()) {
    GTEST_SKIP() << "Triton is not supported on non-NVIDIA and "
                    "pre-Volta NVIDIA GPUs.";
  }

  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = bf16[128,128] parameter(0)
  p0_convert = f32[128,128] convert(p0)
  p1 = f32[128,128] parameter(1)
  ROOT d = f32[128,128] dot(p0_convert, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  // Disable autotuning.
  debug_options_.set_xla_gpu_deterministic_ops(true);
  // Check that triton is used but without autotuning (default config).
  AutotunerUtil::ClearAutotuneResults();
  MatchOptimizedHlo(kHloText, R"(
    CHECK: __triton_gemm
    CHECK: {"block_m":"32","block_n":"32","block_k":"32","split_k":"1","num_stages":"1","num_warps":"4","num_ctas":"1"}
  )",
                    TimerCreation::kForbidden);
  AssertDeterminism(kHloText, /*num_runs=*/3);
}

TEST_F(DeterminismTest, ExcludingNonDeterministicOpsDoesNotDisableAutotuning) {
  if (!IsVoltaOrLater()) {
    GTEST_SKIP() << "Triton is not supported on non-NVIDIA and "
                    "pre-Volta NVIDIA GPUs.";
  }

  debug_options_.set_xla_gpu_cublas_fallback(false);
  ASSERT_TRUE(debug_options_.xla_gpu_exclude_nondeterministic_ops());
  ASSERT_FALSE(debug_options_.xla_gpu_deterministic_ops());
  AutotunerUtil::ClearAutotuneResults();
  // The default config is not used when autotuning is on.
  MatchOptimizedHlo(R"(
ENTRY e {
  p0 = bf16[128,128] parameter(0)
  p0_convert = f32[128,128] convert(p0)
  p1 = f32[128,128] parameter(1)
  ROOT d = f32[128,128] dot(p0_convert, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})",
                    R"(
    CHECK: __triton_gemm
    CHECK-NOT: {"block_m":"32","block_n":"32","block_k":"32","split_k":"1","num_stages":"1","num_warps":"4","num_ctas":"1"}
  )",
                    TimerCreation::kAllowed);
}

TEST_F(DeterminismTest, Conv) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  input = f32[16,3,64,64] parameter(0)
  filter = f32[3,3,3,64] parameter(1)
  conv = f32[16,64,64,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
})";

  AssertDeterminism(kHloText);
}

}  // namespace gpu
}  // namespace xla
