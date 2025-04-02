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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::Field;
using ::testing::Ge;

auto IsValidConfig() {
  return AllOf(Field("block_m", &TritonGemmConfig::block_m, Ge(1)),
               Field("block_n", &TritonGemmConfig::block_n, Ge(1)),
               Field("block_k", &TritonGemmConfig::block_k, Ge(1)),
               Field("split_k", &TritonGemmConfig::split_k, Ge(1)),
               Field("num_stages", &TritonGemmConfig::num_stages, Ge(1)),
               Field("num_warps", &TritonGemmConfig::num_warps, Ge(1)),
               Field("num_ctas", &TritonGemmConfig::num_ctas, Ge(1)));
};

class DotSearchSpaceTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_description_{
      se::DeviceDescription(se::GpuDeviceInfoProto::default_instance())};
};

TEST_F(DotSearchSpaceTest, ReturnsValidConfigList) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  ROOT r = f32[1024,1024] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  TritonDotFusionSearchSpace search_space(
      device_description_,
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction()));

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(::testing::IsEmpty()), Each(IsValidConfig())));
}

}  // namespace
}  // namespace xla::gpu
