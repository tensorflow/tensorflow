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

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/gpu_codegen_test.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

using ::testing::_;

namespace {

TEST_F(GpuCodegenTest, OnNanShouldLogHloInstruction) {
  static constexpr absl::string_view kHloModule = R"hlo(
    HloModule test_module
    ENTRY main {
      zero = f32[] constant(0)
      zero_init = f32[1024] broadcast(zero), dimensions={}
      ROOT div = f32[1024] divide(zero_init, zero_init)
    }
  )hlo";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kHloModule));
  module->mutable_config().mutable_debug_options().set_xla_gpu_detect_nan(
      DebugOptions::DETECTION_MODE_WARNING);
  absl::ScopedMockLog log;
  EXPECT_CALL(
      log, Log(absl::LogSeverity::kError, _,
               ::testing::HasSubstr("Found entry with non zero nan count ")));
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       ::testing::HasSubstr("Found NaN in HLO instruction ")));
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kError, _,
                  ::testing::HasSubstr("HLO fusion instruction computation")));
  log.StartCapturingLogs();
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
  log.StopCapturingLogs();
}

TEST_F(GpuCodegenTest, OnInfShouldLogHloInstruction) {
  static constexpr absl::string_view kHloModule = R"hlo(
    HloModule test_module
    ENTRY main {
      p0 = f32[1024] parameter(0)
      zero = f32[] constant(0)
      zero_init = f32[1024] broadcast(zero), dimensions={}
      ROOT div = f32[1024] divide(p0, zero_init)
    }
  )hlo";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kHloModule));
  module->mutable_config().mutable_debug_options().set_xla_gpu_detect_inf(
      DebugOptions::DETECTION_MODE_WARNING);
  absl::ScopedMockLog log;
  EXPECT_CALL(
      log, Log(absl::LogSeverity::kError, _,
               ::testing::HasSubstr("Found entry with non zero inf count ")));
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       ::testing::HasSubstr("Found Inf in HLO instruction ")));
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kError, _,
                  ::testing::HasSubstr("HLO fusion instruction computation")));
  log.StartCapturingLogs();
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
  log.StopCapturingLogs();
}

}  // namespace
}  // namespace xla::gpu
