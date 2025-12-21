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

#include "xla/backends/gpu/target_config/target_config.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

struct GpuTargetConfigTestCase {
  std::string test_name;
  std::string gpu_model;
  bool expect_ok;
};

using GetGpuTargetConfigTest =
    ::testing::TestWithParam<GpuTargetConfigTestCase>;

TEST_P(GetGpuTargetConfigTest, TestProtoRetrieval) {
  const GpuTargetConfigTestCase& test_case = GetParam();
  auto config = GetGpuTargetConfig(test_case.gpu_model);

  if (test_case.expect_ok) {
    ASSERT_THAT(config, absl_testing::IsOk());
    EXPECT_TRUE(config->has_gpu_device_info());
    EXPECT_GT(config->gpu_device_info().threads_per_block_limit(), 0);
  } else {
    EXPECT_THAT(config,
                absl_testing::StatusIs(absl::StatusCode::kNotFound,
                                       HasSubstr("Embedded file not found")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    GetGpuTargetConfigTests, GetGpuTargetConfigTest,
    ::testing::ValuesIn<GpuTargetConfigTestCase>({
        {"A100_PCIE_80", "a100_pcie_80", true},
        {"A100_SXM_40", "a100_sxm_40", true},
        {"A100_SXM_80", "a100_sxm_80", true},
        {"A6000", "a6000", true},
        {"B200", "b200", true},
        {"B300", "b300", true},
        {"H100_PCIE", "h100_pcie", true},
        {"H100_SXM", "h100_sxm", true},
        {"MI200", "mi200", true},
        {"P100", "p100", true},
        {"V100", "v100", true},
        {"UnknownModel", "unknown_gpu", false},
    }),
    [](const ::testing::TestParamInfo<GetGpuTargetConfigTest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
}  // namespace xla::gpu
