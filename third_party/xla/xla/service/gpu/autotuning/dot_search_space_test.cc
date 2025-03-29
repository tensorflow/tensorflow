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

#include "xla/service/gpu/autotuning/dot_search_space.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"

namespace xla::gpu {
namespace {

class DotSearchSpaceTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_description_{
      se::DeviceDescription(se::GpuDeviceInfoProto::default_instance())};
  std::unique_ptr<VerifiedHloModule> default_module_{
      ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f16[1024,1024] parameter(0)
  p1 = f16[1024,1024] parameter(1)
  ROOT r = f16[1024,1024] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
          .value()};
};

TEST_F(DotSearchSpaceTest, ReturnsValidConfigList) {
  TritonDotFusionSearchSpace search_space(
      device_description_,
      Cast<HloDotInstruction>(
          default_module_->entry_computation()->root_instruction()));

  std::vector<TritonGemmConfig> configs = search_space.GenerateConfigs();

  ASSERT_FALSE(configs.empty());
  for (const auto& config : configs) {
    EXPECT_GE(config.block_m, 1);
    EXPECT_GE(config.block_n, 1);
    EXPECT_GE(config.block_k, 1);
    EXPECT_GE(config.split_k, 1);
    EXPECT_GE(config.num_stages, 1);
    EXPECT_GE(config.num_warps, 1);
    EXPECT_GE(config.num_ctas, 1);
  }
}

TEST_F(DotSearchSpaceTest, HonorsForcedContractingSplit) {
  TritonDotFusionSearchSpace search_space(
      device_description_,
      Cast<HloDotInstruction>(
          default_module_->entry_computation()->root_instruction()));

  std::vector<TritonGemmConfig> configs =
      search_space.GenerateConfigs(/*force_contracting_split=*/2);

  ASSERT_FALSE(configs.empty());
  for (const auto& config : configs) {
    EXPECT_EQ(config.split_k, 2);

    EXPECT_GE(config.block_m, 1);
    EXPECT_GE(config.block_n, 1);
    EXPECT_GE(config.block_k, 1);
    EXPECT_GE(config.num_stages, 1);
    EXPECT_GE(config.num_warps, 1);
    EXPECT_GE(config.num_ctas, 1);
  }
}

TEST_F(DotSearchSpaceTest, ConsidersContractingSplitForSmallOutputSize) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f16[16,1024] parameter(0)
  p1 = f16[1024,16] parameter(1)
  ROOT r = f16[16,16] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                                                  .value();
  TritonDotFusionSearchSpace search_space(
      device_description_,
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction()));

  std::vector<TritonGemmConfig> configs = search_space.GenerateConfigs();

  ASSERT_TRUE(
      std::any_of(configs.begin(), configs.end(),
                  [](const auto& config) { return config.split_k > 1; }));
}

TEST_F(DotSearchSpaceTest, LimitsContractingSplitForSmallerContractingSize) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f16[16,32] parameter(0)
  p1 = f16[32,16] parameter(1)
  ROOT r = f16[16,16] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                                                  .value();

  TritonDotFusionSearchSpace search_space(
      device_description_,
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction()));

  std::vector<TritonGemmConfig> configs = search_space.GenerateConfigs();

  ASSERT_FALSE(configs.empty());
  for (const auto& config : configs) {
    EXPECT_LE(config.split_k, 2);
  }
}

}  // namespace
}  // namespace xla::gpu
