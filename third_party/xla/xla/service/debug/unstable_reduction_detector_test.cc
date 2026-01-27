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
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

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
      ROOT red = bf16[] reduce(p0, init),
          to_apply=red,
          dimensions={0},
          metadata={op_name="op_name" source_file="source_file.py" source_line=42}
  }
)";

static constexpr absl::string_view kUnstableReductionNoMetadataHloModule = R"(
  red {
      p0 = bf16[] parameter(0)
      p1 = bf16[] parameter(1)
      ROOT red = bf16[] add(p0, p1)
  }

  ENTRY main {
      p0 = bf16[164] parameter(0)
      init = bf16[] constant(1.0)
      ROOT red = bf16[] reduce(p0, init),
          to_apply=red,
          dimensions={0}
  }
)";

static constexpr absl::string_view kNoOpUnstableReductionHloModule = R"(
  red {
      p0 = bf16[] parameter(0)
      p1 = bf16[] parameter(1)
      ROOT red = bf16[] add(p0, p1)
  }

  ENTRY main {
      p0 = bf16[1] parameter(0)
      init = bf16[] constant(1.0)
      ROOT red = bf16[] reduce(p0, init),
          to_apply=red,
          dimensions={0}
  }
)";

using ::absl::LogSeverity;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::HasSubstr;

TEST(UnstableReductionDetectorTest, FailOnUnstableReductions) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(kUnstableReductionHloModule));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(DebugOptions::DETECTION_MODE_FAIL);
  UnstableReductionDetector detector;
  ::absl::ScopedMockLog log;
  EXPECT_CALL(
      log,
      Log(LogSeverity::kWarning, _,
          HasSubstr("1 unstable reductions found in module 'module_main'")));
  EXPECT_CALL(log,
              Log(LogSeverity::kWarning, _,
                  "Unstable reduction: %red.1 = bf16[] reduce(%p0.1, %init), "
                  "dimensions={0}, to_apply=%red, "
                  "metadata={op_name=\"op_name\" "
                  "source_file=\"source_file.py\" source_line=42}"));
  log.StartCapturingLogs();
  EXPECT_THAT(
      detector.Run(module.get(), /*execution_threads=*/{}),
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr("1 unstable reductions found in module 'module_main'. List "
                    "of unique reduction ops:\nsource_file.py:42: op_name")));
}

TEST(UnstableReductionDetectorTest, WarningOnUnstableReduction) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(kUnstableReductionHloModule));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(DebugOptions::DETECTION_MODE_WARNING);
  UnstableReductionDetector detector;
  ::absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _,
                       "1 unstable reductions found in module 'module_main'"));
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _,
                       "Unstable reduction: %red.1 = bf16[] reduce(%p0.1, "
                       "%init), dimensions={0}, to_apply=%red, "
                       "metadata={op_name=\"op_name\" "
                       "source_file=\"source_file.py\" source_line=42}"));
  log.StartCapturingLogs();
  EXPECT_THAT(detector.Run(module.get(), /*execution_threads=*/{}),
              IsOkAndHolds(false));
}

TEST(UnstableReductionDetectorTest, FailOnUnstableReductionNoMetadata) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnUnverifiedModule(kUnstableReductionNoMetadataHloModule));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(DebugOptions::DETECTION_MODE_FAIL);
  UnstableReductionDetector detector;
  ::absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _,
                       "1 unstable reductions found in module 'module_main'"));
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _,
                       "Unstable reduction: %red.1 = bf16[] reduce(%p0.1, "
                       "%init), dimensions={0}, to_apply=%red"));
  log.StartCapturingLogs();
  EXPECT_THAT(detector.Run(module.get(), /*execution_threads=*/{}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("1 op names without metadata: red.1")));
}

TEST(UnstableReductionDetectorTest, DoNothingOnUnstableReduction) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(kUnstableReductionHloModule));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(DebugOptions::DETECTION_MODE_NONE);
  ::absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, _)).Times(0);
  UnstableReductionDetector detector;
  log.StartCapturingLogs();
  EXPECT_THAT(detector.Run(module.get(), /*execution_threads=*/{}),
              IsOkAndHolds(false));
}

TEST(UnstableReductionDetectorTest, NoOpUnstableReduction) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(
                                           kNoOpUnstableReductionHloModule));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_detect_unstable_reductions(DebugOptions::DETECTION_MODE_WARNING);
  UnstableReductionDetector detector;
  ::absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, _)).Times(0);
  EXPECT_CALL(log, Log(LogSeverity::kError, _, _)).Times(0);
  log.StartCapturingLogs();
  EXPECT_THAT(detector.Run(module.get(), /*execution_threads=*/{}),
              IsOkAndHolds(false));
  log.StopCapturingLogs();
}

}  // namespace
}  // namespace xla
