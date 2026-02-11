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
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
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
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kError, _,
                  ::testing::AllOf(::testing::HasSubstr("found NaN"),
                                   ::testing::HasSubstr("nan_count: 1024,"))));
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       ::testing::HasSubstr("In HLO instruction")));
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
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kError, _,
                  ::testing::AllOf(::testing::HasSubstr("found Inf"),
                                   ::testing::HasSubstr("inf_count: 1024,"))));
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       ::testing::HasSubstr("In HLO instruction ")));
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kError, _,
                  ::testing::HasSubstr("HLO fusion instruction computation")));
  log.StartCapturingLogs();
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
  log.StopCapturingLogs();
}

TEST_F(GpuCodegenTest, OnMinMaxShouldLogValuesAndHloInstruction) {
  static constexpr absl::string_view kHloModule = R"hlo(
    HloModule test_module
    ENTRY main {
      c0 = f32[] constant(-512)
      iota = f32[1024] iota(), iota_dimension=0
      offset = f32[1024] broadcast(c0), dimensions={}
      ROOT add = f32[1024] add(iota, offset)
    }
  )hlo";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kHloModule));
  module->mutable_config().mutable_debug_options().set_xla_gpu_log_minmax(true);
  absl::ScopedMockLog log;
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kError, _,
                  ::testing::AllOf(::testing::HasSubstr("min_value: -512."),
                                   ::testing::HasSubstr("max_value: 511."))));
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       ::testing::HasSubstr("In HLO instruction ")));
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kError, _,
                  ::testing::HasSubstr("HLO fusion instruction computation")));
  log.StartCapturingLogs();
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
  log.StopCapturingLogs();
}

}  // namespace
}  // namespace xla::gpu
