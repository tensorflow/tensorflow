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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/literal.h"
#include "xla/service/gpu/autotuning/autotuner_cache.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

class DeterminismTest : public HloPjRtGpuTestBase {
 public:
  DeterminismTest()
      : debug_options_(HloPjRtGpuTestBase::GetDebugOptionsForTest()) {
    debug_options_.set_xla_gpu_exclude_nondeterministic_ops(true);
  }

  se::StreamExecutor* stream_executor() const {
    auto platform =
        se::PlatformManager::PlatformWithId(stream_executor_platform_id());
    CHECK_OK(platform);
    auto executor = (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor);
    return *executor;
  }

  se::CudaComputeCapability get_cuda_cc() const {
    return device_description().cuda_compute_capability();
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
      AutotunerCache::ClearAutotuneResults();

      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<HloModule> module,
          ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest()));
      if (i == 0) {
        ASSERT_OK_AND_ASSIGN(fake_arguments, MakeFakeArguments(module.get()));
        for (Literal& literal : fake_arguments) {
          fake_arguments_ptrs.push_back(&literal);
        }
      }

      ASSERT_OK_AND_ASSIGN(Literal output,
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
  absl::StatusOr<bool> MatchOptimizedHlo(absl::string_view hlo_string,
                                         absl::string_view expected_hlo_regex,
                                         TimerCreation timer_creation) {
    if (timer_creation == TimerCreation::kAllowed) {
      HloPjRtGpuTestBase::MatchOptimizedHlo(hlo_string, expected_hlo_regex);
      return true;
    }

    // If timer creation is forbidden we inject a mock GPU executor that
    // prevents timer creation.
    ASSIGN_OR_RETURN(
        stream_executor::Platform * default_platform,
        se::PlatformManager::PlatformWithId(stream_executor_platform_id()));
    stream_executor::MockStreamExecutor executor;
    EXPECT_CALL(executor, GetPlatform).WillRepeatedly([&] {
      return default_platform;
    });
    EXPECT_CALL(executor, CreateEventBasedTimer).Times(0);
    EXPECT_CALL(executor, GetDeviceDescription)
        .WillRepeatedly([this]() -> const se::DeviceDescription& {
          return device_description();
        });
    EXPECT_CALL(executor, GetPlatform).WillRepeatedly([&]() {
      return default_platform;
    });
    EXPECT_CALL(executor, AsDnn).WillRepeatedly([&]() {
      return stream_executor()->AsDnn();
    });
    EXPECT_CALL(executor, device_ordinal).WillRepeatedly([]() { return 0; });
    EXPECT_CALL(executor, SynchronizeAllActivity).WillRepeatedly([&]() -> bool {
      return true;
    });
    EXPECT_CALL(executor, CreateStream).WillRepeatedly([&] {
      return stream_executor()->CreateStream();
    });
    EXPECT_CALL(executor, AsBlas).WillRepeatedly([&] {
      return stream_executor()->AsBlas();
    });

    ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo_string));
    ASSIGN_OR_RETURN(
        auto optimized_module,
        compiler()->RunHloPasses(std::move(module), &executor, nullptr));
    absl::StatusOr<bool> filecheck_result =
        RunFileCheck(optimized_module->ToString(), expected_hlo_regex);
    CHECK_OK(filecheck_result.status());
    return *filecheck_result;
  }

  bool IsAmpereOrLater() const { return get_cuda_cc().IsAtLeastAmpere(); }

  bool IsRocm() const {
    return device_description().gpu_compute_capability().IsRocm();
  }

  bool HasHipblasLt() const {
    return device_description().rocm_compute_capability().has_hipblaslt();
  }
};

TEST_F(DeterminismTest, CublasLtDot) {
  debug_options_.clear_xla_gpu_experimental_autotune_backends();
  if (IsRocm()) {
    if (!HasHipblasLt()) {
      GTEST_SKIP() << "No hipblas-lt support on this architecture!";
    }
  }
  auto backend =
      IsRocm() ? autotuner::Backend::HIPBLASLT : autotuner::Backend::CUBLASLT;
  debug_options_.add_xla_gpu_experimental_autotune_backends(backend);

  constexpr absl::string_view kHloText = R"hlo(
ENTRY e {
  p0 = f32[128,128] parameter(0)
  p1 = f32[128,128] parameter(1)
  ROOT d = f32[128,128] dot(p0, p1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0}
})hlo";

  debug_options_.set_xla_gpu_enable_triton_gemm(false);

  EXPECT_THAT(
      MatchOptimizedHlo(kHloText,
                        R"(; CHECK: custom_call_target="__cublas$lt$matmul")",
                        TimerCreation::kForbidden),
      absl_testing::IsOkAndHolds(true));
  AssertDeterminism(kHloText);
}

TEST_F(DeterminismTest, DeterministicOpsUsesFirstConfig) {
  if (!IsAmpereOrLater()) {
    GTEST_SKIP() << "Triton is not supported on non-NVIDIA and "
                    "pre-Ampere NVIDIA GPUs.";
  }
  if (get_cuda_cc().IsAtLeastBlackwell()) {
    // TODO(b/445172709): Re-enable once fixed.
    GTEST_SKIP();
  }

  constexpr absl::string_view kHloText = R"hlo(
ENTRY e {
  p0 = bf16[128,128] parameter(0)
  p0_convert = f32[128,128] convert(p0)
  p1 = f32[128,128] parameter(1)
  ROOT d = f32[128,128] dot(p0_convert, p1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0}
})hlo";

  // Disable autotuning.
  debug_options_.set_xla_gpu_exclude_nondeterministic_ops(true);
  AutotunerCache::ClearAutotuneResults();
  // Deterministic GEMM should use the first config from the list and with the
  // current backend order this should be cuBLAS LT.
  EXPECT_THAT(MatchOptimizedHlo(kHloText, R"(
    CHECK: ENTRY
    CHECK: __cublas$lt$matmul
  )",
                                TimerCreation::kForbidden),
              absl_testing::IsOkAndHolds(true));
  AssertDeterminism(kHloText, /*num_runs=*/3);
}

TEST_F(DeterminismTest, ExcludingNonDeterministicOpsUsesFirstConfig) {
  if (!IsAmpereOrLater()) {
    GTEST_SKIP() << "Triton is not supported on non-NVIDIA and "
                    "pre-Ampere NVIDIA GPUs.";
  }

  ASSERT_TRUE(debug_options_.xla_gpu_exclude_nondeterministic_ops());
  AutotunerCache::ClearAutotuneResults();
  // We select the first config from the list and with the current backend order
  // this should be cuBLAS LT.
  EXPECT_THAT(MatchOptimizedHlo(R"hlo(
ENTRY e {
  p0 = bf16[128,128] parameter(0)
  p0_convert = f32[128,128] convert(p0)
  p1 = f32[128,128] parameter(1)
  ROOT d = f32[128,128] dot(p0_convert, p1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0}
})hlo",
                                R"(
    CHECK: ENTRY
    CHECK: __cublas$lt$matmul
  )",
                                TimerCreation::kAllowed),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(DeterminismTest, Conv) {
  constexpr absl::string_view kHloText = R"hlo(
ENTRY e {
  input = f32[16,3,64,64] parameter(0)
  filter = f32[3,3,3,64] parameter(1)
  conv = f32[16,64,64,64] convolution(input, filter),
    window={size=3x3 pad=1_1x1_1},
    dim_labels=bf01_01io->bf01,
    feature_group_count=1
})hlo";

  AssertDeterminism(kHloText);
}

}  // namespace xla::gpu
