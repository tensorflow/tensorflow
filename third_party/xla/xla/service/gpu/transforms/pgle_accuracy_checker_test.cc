/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/pgle_accuracy_checker.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/service/profile_guided_latency_estimator.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using PGLEAccuracyCheckerTest = HloHardwareIndependentTestBase;
using ::tensorflow::profiler::ProfiledInstructionsProto;
using ::tsl::protobuf::TextFormat;
using ::tsl::testing::StatusIs;

// Constructs PGLE estimator for a given `profile`.
std::unique_ptr<ProfileGuidedLatencyEstimator> GetProfileGuidedLatencyEstimator(
    ProfiledInstructionsProto& profile) {
  auto gpu_latency_estimator =
      std::make_unique<GpuLatencyEstimator>(/*pointer_size=*/8);
  SchedulerConfig config;
  auto aggregator = std::make_unique<GPUProfileStatisticsAggregator>();
  return std::make_unique<ProfileGuidedLatencyEstimator>(
      config, std::move(gpu_latency_estimator), profile, std::move(aggregator));
}

TEST_F(PGLEAccuracyCheckerTest,
       ReturnsOkAndNoIRChangeIfAllInstructionsAreFoundInTheProfile) {
  const absl::string_view kHloString = R"(
  HloModule m

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[32] parameter(0)
    p1 = f32[32,32] parameter(1)
    p2 = f32[32,32] parameter(2)
    p3 = f32[32] parameter(3)

    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    dot1 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    add0 = f32[32,32] add(dot0, dot1)

    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    ar-done = f32[32] all-reduce-done(ar-start)

    ar-start1 = f32[32] all-reduce-start(p3), to_apply=apply_op
    ar-done1 = f32[32] all-reduce-done(ar-start1)

    ROOT _ = (f32[32],f32[32],f32[32,32]) tuple(ar-done, ar-done1, add0)
  })";

  // Profile string, cost does not matter.
  const std::string kProfileString = R"pb(
    costs { name: "dot0" cost_us: 1.0 }
    costs { name: "dot1" cost_us: 1.0 }
    costs { name: "add0" cost_us: 1.0 }
    costs { name: "ar-start" cost_us: 1.0 }
    costs { name: "ar-start1" cost_us: 1.0 }
  )pb";

  ProfiledInstructionsProto profile;
  ASSERT_TRUE(TextFormat::ParseFromString(kProfileString, &profile));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  module->mutable_config().set_fdo_profile(kProfileString);

  auto pgle_estimator = GetProfileGuidedLatencyEstimator(profile);
  PGLEAccuracyChecker pgle_accuracy_checker(*pgle_estimator);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          pgle_accuracy_checker.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(PGLEAccuracyCheckerTest,
       ReturnsInvalidArgumentIfThereAreMissingInstructionsFromTheProfile) {
  const absl::string_view kHloString = R"(
  HloModule m

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[32] parameter(0)
    p1 = f32[32,32] parameter(1)
    p2 = f32[32,32] parameter(2)
    p3 = f32[32] parameter(3)

    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    dot1 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    add0 = f32[32,32] add(dot0, dot1)

    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    ar-done = f32[32] all-reduce-done(ar-start)

    ar-start1 = f32[32] all-reduce-start(p3), to_apply=apply_op
    ar-done1 = f32[32] all-reduce-done(ar-start1)

    ROOT _ = (f32[32],f32[32],f32[32,32]) tuple(ar-done, ar-done1, add0)
  })";

  // Profile string, cost does not matter.
  // We're missing `dot1` and `ar-start` from the profile.
  const std::string kProfileString = R"pb(
    costs { name: "dot0" cost_us: 1.0 }
    costs { name: "add0" cost_us: 1.0 }
    costs { name: "ar-start1" cost_us: 1.0 }
  )pb";

  ProfiledInstructionsProto profile;
  ASSERT_TRUE(TextFormat::ParseFromString(kProfileString, &profile));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  module->mutable_config().set_fdo_profile(kProfileString);
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_pgle_accuracy_checker(
          DebugOptions::PGLE_STRICTNESS_LEVEL_ERROR);

  auto pgle_estimator = GetProfileGuidedLatencyEstimator(profile);
  PGLEAccuracyChecker pgle_accuracy_checker(*pgle_estimator);
  EXPECT_THAT(pgle_accuracy_checker.Run(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::gpu
