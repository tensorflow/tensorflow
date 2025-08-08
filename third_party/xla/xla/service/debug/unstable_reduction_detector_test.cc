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

#include "xla/service/debug/unstable_reduction_detector.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "testing/base/public/mock-log.h"
#include "absl/base/log_severity.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

static constexpr absl::string_view kUnstableReductionHloModule = R"(
  red {
      p0 = bf16[] parameter(0)
      p1 = bf16[] parameter(1)
      ROOT red = bf16[] add(p0, p1)
  }

  ENTRY main {
      p0 = bf16[164] parameter(0)
      init = bf16[] constant(1.0)
      ROOT red = bf16[] reduce(p0, init), to_apply=red, dimensions={0}
  }
)";

using ::base_logging::WARNING;
using ::testing::_;

TEST(UnstableReductionDetectorTest, FailOnUnstableReductions) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(kUnstableReductionHloModule));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(
          DebugOptions::UNSTABLE_REDUCTION_DETECTION_MODE_FAIL);
  UnstableReductionDetector detector;
  ::testing::ScopedMockLog log(::testing::kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(WARNING, _,
                       "1 unstable reductions found in module 'module_main'"));
  EXPECT_CALL(log, Log(WARNING, _,
                       "Unstable reduction: %red.1 = bf16[] reduce(%p0.1, "
                       "%init), dimensions={0}, to_apply=%red"));
  log.StartCapturingLogs();
  EXPECT_THAT(detector.Run(module.get(), /*execution_threads=*/{}),
              ::testing::status::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  ::testing::HasSubstr(
                      "1 unstable reductions found in module 'module_main'")));
}

TEST(UnstableReductionDetectorTest, WarningOnUnstableReduction) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(kUnstableReductionHloModule));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(
          DebugOptions::UNSTABLE_REDUCTION_DETECTION_MODE_WARNING);
  UnstableReductionDetector detector;

  ::testing::ScopedMockLog log(::testing::kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(WARNING, _,
                       "1 unstable reductions found in module 'module_main'"));
  EXPECT_CALL(log, Log(WARNING, _,
                       "Unstable reduction: %red.1 = bf16[] reduce(%p0.1, "
                       "%init), dimensions={0}, to_apply=%red"));
  log.StartCapturingLogs();
  EXPECT_THAT(detector.Run(module.get(), /*execution_threads=*/{}),
              ::testing::status::IsOkAndHolds(false));
}

TEST(UnstableReductionDetectorTest, DoNothingOnUnstableReduction) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(kUnstableReductionHloModule));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(
          DebugOptions::UNSTABLE_REDUCTION_DETECTION_MODE_NONE);
  ::testing::ScopedMockLog log(::testing::kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(WARNING, _, _)).Times(0);
  UnstableReductionDetector detector;
  log.StartCapturingLogs();
  EXPECT_THAT(detector.Run(module.get(), /*execution_threads=*/{}),
              ::testing::status::IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
