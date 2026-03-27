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
#include "google/protobuf/text_format.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

struct GpuTargetConfigTestCase {
  std::string test_name;
  GpuModel gpu_model;
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
        {"A100_PCIE_80", GpuModel::A100_PCIE_80, true},
        {"A100_SXM_40", GpuModel::A100_SXM_40, true},
        {"A100_SXM_80", GpuModel::A100_SXM_80, true},
        {"A6000", GpuModel::A6000, true},
        {"B200", GpuModel::B200, true},
        {"B300", GpuModel::B300, true},
        {"BMG_G21", GpuModel::BMG_G21, true},
        {"H100_PCIE", GpuModel::H100_PCIE, true},
        {"H100_SXM", GpuModel::H100_SXM, true},
        {"MI200", GpuModel::MI200, true},
        {"P100", GpuModel::P100, true},
        {"V100", GpuModel::V100, true},
        {"GB200", GpuModel::GB200, true},
        {"GB300", GpuModel::GB300, true},
        {"RTX6000PRO", GpuModel::RTX6000PRO, true},
    }),
    [](const ::testing::TestParamInfo<GetGpuTargetConfigTest::ParamType>&
           info) { return info.param.test_name; });

TEST(TargetConfigTest, CompareEqualFromSameProto) {
  stream_executor::GpuTargetConfigProto config_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        platform_name: "platform"
        dnn_version_info { major: 2 }
        runtime_version { major: 12 }
        gpu_device_info { threads_per_block_limit: 5 }
        device_description_str: "foo"
      )pb",
      &config_proto));

  ASSERT_OK_AND_ASSIGN(auto config1, GpuTargetConfig::FromProto(config_proto));
  ASSERT_OK_AND_ASSIGN(auto config2, GpuTargetConfig::FromProto(config_proto));
  EXPECT_THAT(config1, ::testing::Eq(config2));
}

TEST(TargetConfigTest, GetTargetConfigFromFile) {
  std::string filename =
      tsl::io::JoinPath(testing::TempDir(), "target_config.textproto");
  std::string proto_content = R"pb(
    platform_name: "platform"
    gpu_device_info { threads_per_block_limit: 5 }
  )pb";
  TF_ASSERT_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), filename, proto_content));

  ASSERT_OK_AND_ASSIGN(GpuTargetConfig config,
                       GetTargetConfigFromFile(filename));
  EXPECT_EQ(config.platform_name, "platform");
  EXPECT_EQ(config.device_description.threads_per_block_limit(), 5);
}

}  // namespace
}  // namespace xla::gpu
