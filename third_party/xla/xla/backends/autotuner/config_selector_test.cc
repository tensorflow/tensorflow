/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/autotuner/config_selector.h"

#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/config_runner.h"
#include "xla/backends/autotuner/mock_codegen_backend.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using absl_testing::StatusIs;

ConfigRunner::ConfigProfile CreateProfile(
    CodegenBackend* backend, absl::string_view config_name,
    absl::Duration duration, int scratch_bytes = 0,
    std::optional<ConfigRunner::Failure> failure = std::nullopt) {
  return ConfigRunner::ConfigProfile{
      /*config=*/{backend, GetTestConfig(config_name)},
      /*failure=*/failure,
      /*duration=*/duration,
      /*scratch_bytes=*/scratch_bytes,
  };
}

TEST(ConfigSelectorTest, PicksFastestConfig) {
  MockCodegenBackend backend;
  std::vector<ConfigRunner::ConfigProfile> profiles;
  profiles.push_back(
      CreateProfile(&backend, "test_config_1", absl::Seconds(2)));
  profiles.push_back(
      CreateProfile(&backend, "test_config_2", absl::Seconds(1)));
  profiles.push_back(
      CreateProfile(&backend, "test_config_3", absl::Seconds(3)));

  ASSERT_OK_AND_ASSIGN(
      auto best, PickBestConfig(profiles, /*scratch_bytes_window_size_us=*/0));
  EXPECT_THAT(*best.config.backend_config, ConfigMatcher("test_config_2"));
  EXPECT_EQ(best.duration, absl::Seconds(1));
}

TEST(ConfigSelectorTest, SkipsFailedConfigs) {
  MockCodegenBackend backend;
  std::vector<ConfigRunner::ConfigProfile> profiles;
  profiles.push_back(CreateProfile(
      &backend, "test_config_1", absl::Seconds(1), /*scratch_bytes=*/0,
      ConfigRunner::Failure{ConfigRunner::FailureKind::kWrongResults,
                            "mismatch"}));
  profiles.push_back(
      CreateProfile(&backend, "test_config_2", absl::Seconds(2)));

  ASSERT_OK_AND_ASSIGN(
      auto best, PickBestConfig(profiles, /*scratch_bytes_window_size_us=*/0));
  EXPECT_THAT(*best.config.backend_config, ConfigMatcher("test_config_2"));
  EXPECT_EQ(best.duration, absl::Seconds(2));
}

TEST(ConfigSelectorTest, FailsWhenAllConfigsFail) {
  MockCodegenBackend backend;
  std::vector<ConfigRunner::ConfigProfile> profiles;
  profiles.push_back(CreateProfile(
      &backend, "test_config_1", absl::Seconds(1), /*scratch_bytes=*/0,
      ConfigRunner::Failure{ConfigRunner::FailureKind::kCompilationFailed,
                            "fail"}));
  profiles.push_back(CreateProfile(
      &backend, "test_config_2", absl::Seconds(2), /*scratch_bytes=*/0,
      ConfigRunner::Failure{ConfigRunner::FailureKind::kExecutionFailed,
                            "crash"}));

  EXPECT_THAT(PickBestConfig(profiles, /*scratch_bytes_window_size_us=*/0),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ConfigSelectorTest, OptimizesScratchBytesWithinWindow) {
  MockCodegenBackend backend;
  std::vector<ConfigRunner::ConfigProfile> profiles;
  profiles.push_back(CreateProfile(&backend, "config_most_time_less_scratch",
                                   absl::Microseconds(7), 100));
  profiles.push_back(CreateProfile(&backend, "config_less_time_less_scratch",
                                   absl::Microseconds(3), 100));
  profiles.push_back(CreateProfile(&backend, "config_least_time_most_scratch",
                                   absl::Microseconds(2), 200));
  profiles.push_back(CreateProfile(&backend, "config_more_time_less_scratch",
                                   absl::Microseconds(6), 100));

  // Window is 8us. The fastest config takes 2us, so any config <= 10us is
  // eligible. Among eligible configs with 100 scratch bytes,
  // config_less_time_less_scratch has duration 3us which is the minimum time
  // with min scratch bytes.
  ASSERT_OK_AND_ASSIGN(
      auto best, PickBestConfig(profiles, /*scratch_bytes_window_size_us=*/8));
  EXPECT_THAT(*best.config.backend_config,
              ConfigMatcher("config_less_time_less_scratch"));
  EXPECT_EQ(best.scratch_bytes, 100);
  EXPECT_EQ(best.duration, absl::Microseconds(3));
}

TEST(ConfigSelectorTest, IgnoresScratchBytesOutsideWindow) {
  MockCodegenBackend backend;
  std::vector<ConfigRunner::ConfigProfile> profiles;
  profiles.push_back(CreateProfile(&backend, "config_fast_more_scratch",
                                   absl::Microseconds(2), 200));
  profiles.push_back(CreateProfile(&backend, "config_slow_less_scratch",
                                   absl::Microseconds(10), 50));

  // Window is 2us. Duration limit is 4us. config_slow_less_scratch is not
  // eligible.
  ASSERT_OK_AND_ASSIGN(
      auto best, PickBestConfig(profiles, /*scratch_bytes_window_size_us=*/2));
  EXPECT_THAT(*best.config.backend_config,
              ConfigMatcher("config_fast_more_scratch"));
  EXPECT_EQ(best.scratch_bytes, 200);
}

}  // namespace
}  // namespace xla
