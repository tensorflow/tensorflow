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

#include "xla/backends/profiler/gpu/cupti_tracer_options_utils.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace xla {
namespace profiler {
namespace {

using ::tsl::testing::IsOk;

TEST(CuptiTracerOptionsUtilsTest, ReadOptionsFromEnvVars) {
  tsl::setenv("TF_GPU_CUPTI_ACTIVITY_BUFFER_SIZE", "65536", /*overwrite=*/1);
  tsl::setenv("TF_GPU_CUPTI_ACTIVITY_BUFFER_PREALLOCATION_COUNT", "16",
              /*overwrite=*/1);

  tensorflow::ProfileOptions profile_options;
  CuptiTracerOptions tracer_options;
  CuptiTracerCollectorOptions collector_options;

  EXPECT_THAT(UpdateCuptiTracerOptionsFromProfilerOptions(
                  profile_options, tracer_options, collector_options),
              IsOk());

  EXPECT_EQ(tracer_options.activity_buffer_size, 65536);
  EXPECT_EQ(tracer_options.activity_buffer_preallocation_count, 16);

  tsl::unsetenv("TF_GPU_CUPTI_ACTIVITY_BUFFER_SIZE");
  tsl::unsetenv("TF_GPU_CUPTI_ACTIVITY_BUFFER_PREALLOCATION_COUNT");
}

TEST(CuptiTracerOptionsUtilsTest, AdvancedConfigOverridesEnvVars) {
  tsl::setenv("TF_GPU_CUPTI_ACTIVITY_BUFFER_SIZE", "65536", /*overwrite=*/1);
  tsl::setenv("TF_GPU_CUPTI_ACTIVITY_BUFFER_PREALLOCATION_COUNT", "16",
              /*overwrite=*/1);

  tensorflow::ProfileOptions profile_options;
  (*profile_options
        .mutable_advanced_configuration())["gpu_activity_buffer_size"]
      .set_int64_value(131072);
  (*profile_options.mutable_advanced_configuration())
      ["gpu_activity_buffer_preallocation_count"]
          .set_int64_value(32);

  CuptiTracerOptions tracer_options;
  CuptiTracerCollectorOptions collector_options;

  EXPECT_THAT(UpdateCuptiTracerOptionsFromProfilerOptions(
                  profile_options, tracer_options, collector_options),
              IsOk());

  EXPECT_EQ(tracer_options.activity_buffer_size, 131072);
  EXPECT_EQ(tracer_options.activity_buffer_preallocation_count, 32);

  tsl::unsetenv("TF_GPU_CUPTI_ACTIVITY_BUFFER_SIZE");
  tsl::unsetenv("TF_GPU_CUPTI_ACTIVITY_BUFFER_PREALLOCATION_COUNT");
}

}  // namespace
}  // namespace profiler
}  // namespace xla
