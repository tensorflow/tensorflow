/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/hlo_op_profiles.h"

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

constexpr char kDeviceHloOpProfiles[] = R"pb(
  entries {
    key: "sm_90"
    value {
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 32
      }
    }
  }

  entries {
    key: "sm_80"
    value {
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 64
      }
    }
  }
)pb";

using HloOpProfilesTest = ::testing::Test;

TEST_F(HloOpProfilesTest, GetProfile) {
  auto hlo_op_profiles = HloOpProfiles::Load(kDeviceHloOpProfiles,
                                             /*default_profile_name=*/"sm_80");
  auto device_info_sm_90 = TestGpuDeviceInfo::RTXA6000DeviceInfo(
      stream_executor::CudaComputeCapability(9, 0));

  const auto& op_profile = hlo_op_profiles->GetProfile(&device_info_sm_90);
  ASSERT_TRUE(op_profile.contains(
      std::make_pair(HloOpcode::kDivide, PrimitiveType::F32)));
  EXPECT_EQ(
      op_profile.at(std::make_pair(HloOpcode::kDivide, PrimitiveType::F32)),
      32);
}

TEST_F(HloOpProfilesTest, GetProfileDefault) {
  auto hlo_op_profiles = HloOpProfiles::Load(kDeviceHloOpProfiles,
                                             /*default_profile_name=*/"sm_80");
  auto device_info_sm_85 = TestGpuDeviceInfo::RTXA6000DeviceInfo(
      stream_executor::CudaComputeCapability(8, 5));

  // hlo_op_profiles only has sm_80 and sm_90, should return the default sm_80.
  const auto& op_profile = hlo_op_profiles->GetProfile(&device_info_sm_85);
  ASSERT_TRUE(op_profile.contains(
      std::make_pair(HloOpcode::kMultiply, PrimitiveType::F32)));
  EXPECT_EQ(
      op_profile.at(std::make_pair(HloOpcode::kMultiply, PrimitiveType::F32)),
      64);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
