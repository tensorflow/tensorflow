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

#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

using ::testing::Test;
using ::tsl::testing::IsOkAndHolds;

bool IsMultiHostTopology(se::CudaComputeCapability compute_capability,
                         int num_partitions, int replica_count) {
  HloModuleConfig config;
  config.set_num_partitions(num_partitions);
  config.set_replica_count(replica_count);
  se::DeviceDescription device_description;
  device_description.set_gpu_compute_capability(compute_capability);
  return xla::gpu::IsMultiHostTopology(config, device_description);
}

TEST(IsMultiHostTopologyTest, SingleHostSingleDevice) {
  EXPECT_FALSE(IsMultiHostTopology(se::CudaComputeCapability::Ampere(),
                                   /*num_partitions=*/1, /*replica_count=*/1));
  EXPECT_FALSE(IsMultiHostTopology(se::CudaComputeCapability::Hopper(),
                                   /*num_partitions=*/1, /*replica_count=*/1));
}

TEST(IsMultiHostTopologyTest, SingleHostMultiDevices) {
  EXPECT_FALSE(IsMultiHostTopology(se::CudaComputeCapability::Ampere(),
                                   /*num_partitions=*/16, /*replica_count=*/1));
  EXPECT_FALSE(IsMultiHostTopology(se::CudaComputeCapability::Ampere(),
                                   /*num_partitions=*/1, /*replica_count=*/16));
  EXPECT_FALSE(IsMultiHostTopology(se::CudaComputeCapability::Hopper(),
                                   /*num_partitions=*/8, /*replica_count=*/1));
  EXPECT_FALSE(IsMultiHostTopology(se::CudaComputeCapability::Hopper(),
                                   /*num_partitions=*/1, /*replica_count=*/8));
}

TEST(IsMultiHostTopologyTest, MultiHosts) {
  EXPECT_TRUE(IsMultiHostTopology(se::CudaComputeCapability::Ampere(),
                                  /*num_partitions=*/32, /*replica_count=*/1));
  EXPECT_TRUE(IsMultiHostTopology(se::CudaComputeCapability::Ampere(),
                                  /*num_partitions=*/1, /*replica_count=*/32));
  EXPECT_TRUE(IsMultiHostTopology(se::CudaComputeCapability::Hopper(),
                                  /*num_partitions=*/16, /*replica_count=*/1));
  EXPECT_TRUE(IsMultiHostTopology(se::CudaComputeCapability::Hopper(),
                                  /*num_partitions=*/1, /*replica_count=*/16));
}

class CommunicationTypeTest : public Test {
 protected:
  se::DeviceDescription& device_info() { return device_info_; }

 private:
  se::DeviceDescription device_info_ = TestGpuDeviceInfo::RTXA6000DeviceInfo(
      stream_executor::CudaComputeCapability(9, 0));
};

TEST_F(CommunicationTypeTest, DetectsSingleHost8Devices) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=8

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[1024] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups=[1,8]<=[8]
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(*instr, device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_HOST));
}

TEST_F(CommunicationTypeTest, DetectsSingleHost4Devices) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=8

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[512] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups=[1,4]<=[4]
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(*instr, device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_HOST));
}

TEST_F(CommunicationTypeTest, DetectsSingleHost16Devices) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[512] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups=[2,8]<=[16]
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(*instr, device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_HOST));
}

TEST_F(CommunicationTypeTest, DetectRailAlignedAllDevices) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[2048] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups=[1,16]<=[16]
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(*instr, device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::RAIL_ALIGNED));
}

TEST_F(CommunicationTypeTest, DetectRailAlignedHalfMesh) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=32

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[512] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups={
          {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
          {16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
        }
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(*instr, device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::RAIL_ALIGNED));
}

TEST_F(CommunicationTypeTest, DetectNonRailAligned) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[512] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups={{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(*instr, device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::NON_RAIL_ALIGNED));
}

}  // namespace
}  // namespace xla::gpu
