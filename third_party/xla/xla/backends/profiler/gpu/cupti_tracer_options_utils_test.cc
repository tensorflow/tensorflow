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
#include "xla/backends/profiler/gpu/cupti_tracer_options_utils.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace xla {
namespace profiler {
namespace {

using ::testing::ElementsAre;

TEST(CuptiTracerOptionsUtilsTest,
     SetPmSamplingCounterOptionsWithCountersAndInterval) {
  tensorflow::ProfileOptions profile_options;
  (*profile_options.mutable_advanced_configuration())["gpu_pm_sample_counters"]
      .set_string_value("metric3,metric4");
  (*profile_options
        .mutable_advanced_configuration())["gpu_pm_sample_interval_us"]
      .set_int64_value(500);
  absl::flat_hash_set<absl::string_view> input_keys = {
      "gpu_pm_sample_counters", "gpu_pm_sample_interval_us"};

  CuptiTracerOptions tracer_options;
  TF_ASSERT_OK(
      SetPmSamplingCounterOptions(profile_options, input_keys, tracer_options));
  EXPECT_TRUE(tracer_options.pm_sampler_options.enable);
  EXPECT_THAT(tracer_options.pm_sampler_options.metrics,
              ElementsAre("metric3", "metric4"));
  EXPECT_EQ(tracer_options.pm_sampler_options.sample_interval_ns, 500'000);
}

TEST(CuptiTracerOptionsUtilsTest, SetPmSamplingCounterOptionsWithConfigFile) {
  std::string config_content = R"pb(
    advanced_configuration {
      key: "gpu_pm_sample_counters"
      value { string_value: "metric5" }
    }
    advanced_configuration {
      key: "gpu_pm_sample_interval_us"
      value { int64_value: 2000 }
    }
  )pb";
  std::string config_path =
      tsl::io::JoinPath(::testing::TempDir(), "pm_config.textproto");
  TF_CHECK_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), config_path, config_content));

  tensorflow::ProfileOptions profile_options;
  (*profile_options
        .mutable_advanced_configuration())["gpu_pm_sample_config_path"]
      .set_string_value(config_path);
  absl::flat_hash_set<absl::string_view> input_keys = {
      "gpu_pm_sample_config_path"};

  CuptiTracerOptions tracer_options;
  TF_ASSERT_OK(
      SetPmSamplingCounterOptions(profile_options, input_keys, tracer_options));
  EXPECT_TRUE(tracer_options.pm_sampler_options.enable);
  EXPECT_THAT(tracer_options.pm_sampler_options.default_config_path,
              config_path);
}

}  // namespace
}  // namespace profiler
}  // namespace xla
