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

#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Test;

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
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_PARTITION));
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
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_PARTITION));
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
  EXPECT_THAT(CommunicationType(/*partition_size=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_PARTITION));
}

TEST_F(CommunicationTypeTest, DetectWorldLevelAllDevices) {
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
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_WORLD_LEVEL));
}

TEST_F(CommunicationTypeTest, DetectWorldLevelHalfMesh) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=32

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[2048] all-gather(p),
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
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_WORLD_LEVEL));
}

TEST_F(CommunicationTypeTest, DetectNonWorldLevel) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[256] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups={{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL));
}

TEST_F(CommunicationTypeTest, DetectsSingleHost16DevicesForEmptyReplicaGroups) {
  absl::string_view kHlo = R"(
    HloModule m, replica_count=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[512] all-gather(p),
        dimensions={0},
        replica_groups={}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/16, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_PARTITION));
}

TEST_F(CommunicationTypeTest, DetectWorldLevel8DevicesForEmptyReplicaGroups) {
  absl::string_view kHlo = R"(
    HloModule m, replica_count=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[2048] all-gather(p),
        dimensions={0},
        replica_groups={}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_WORLD_LEVEL));
}

TEST_F(CommunicationTypeTest, DetectNonWorldLevel16Devices) {
  absl::string_view kHlo = R"(
    HloModule m, replica_count=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[256] all-gather(p),
        dimensions={0},
        replica_groups={{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL));
}

TEST_F(CommunicationTypeTest, DetectsSingleHostCollectivePermute) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=8

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
        source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloChannelInstruction* instr = Cast<HloChannelInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_PARTITION));
}

TEST_F(CommunicationTypeTest, DetectsSingleHostCollectivePermuteSinglePair) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=8

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
        source_target_pairs={{0,7},{7,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloChannelInstruction* instr = Cast<HloChannelInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_PARTITION));
}

TEST_F(CommunicationTypeTest, DetectNonWorldLevelCollectivePermute) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
        source_target_pairs={{0,8},{8,0},{1,9},{9,1},{2,10},{10,2},{3,11},{11,3},
                             {4,12},{12,4},{5,13},{13,5},{6,14},{14,6},{7,15},{15,7}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloChannelInstruction* instr = Cast<HloChannelInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL));
}

TEST_F(CommunicationTypeTest, DetectWorldLevelCollectivePermute) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
       source_target_pairs={{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloChannelInstruction* instr = Cast<HloChannelInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL));
}

TEST_F(CommunicationTypeTest, DetectsCrossHostCollectivePermuteMixed) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
       source_target_pairs={{0,7},
                            {0,8},
                            {1,9},
                            {2,10},
                            {3,11},
                            {4,12},
                            {5,13},
                            {6,14},
                            {7,15}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloChannelInstruction* instr = Cast<HloChannelInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*num_devices_per_host=*/8, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL));
}

TEST_F(CommunicationTypeTest, DetectsSinglePartitionMultiHost) {
  // 16 devices across 2 hosts with partition_size=16 (single partition spanning
  // 2 hosts)
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
  EXPECT_THAT(CommunicationType(/*partition_size=*/16, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_PARTITION));
}

TEST_F(CommunicationTypeTest, DetectsMultiPartitionWith8DevicePartitions) {
  // 64 devices across 2 partitions with partition_size=32
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[2048] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups=[1, 64]<=[64]
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*partition_size=*/32, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_WORLD_LEVEL));
}

TEST_F(CommunicationTypeTest, DetectsMultiPartitionNonRailAligned) {
  // 64 devices with partition_size=36: partition 0 has 36 devices, partition 1
  // has 28 devices
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=12

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[1536] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups=[1, 64]<=[64]
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  // With partition_size=8, spans 2 partitions but not rail-aligned (8 and 4
  // devices)
  EXPECT_THAT(CommunicationType(/*partition_size=*/36, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL));
}

TEST_F(CommunicationTypeTest, DetectsSinglePartitionSubset) {
  // 6 devices within a single partition (partition_size=36)
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=4

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[512] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups={{0,1,2,3,4,5}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*partition_size=*/36, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::SINGLE_PARTITION));
}

TEST_F(CommunicationTypeTest, DetectsRailAlignedMultiPartition) {
  // 128 devices across 2 partitions with partition_size=8 (rail-aligned: 64
  // devices per partition)
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=32

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[4096] all-gather(p),
        dimensions={0},
        use_global_device_ids=true,
        channel_id=1,
        replica_groups=[1,128]<=[128]
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(CommunicationType(/*partition_size=*/64, *instr,
                                device_info().gpu_compute_capability()),
              IsOkAndHolds(GPUCommunicationType::MULTI_HOST_WORLD_LEVEL));
}

}  // namespace

TEST_F(CommunicationTypeTest, CollectivePermuteIntraPartitionOneWay) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=8

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
        source_target_pairs={{0,1},{2,3},{4,5},{6,7}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectivePermuteInstruction* instr =
      Cast<HloCollectivePermuteInstruction>(
          module->entry_computation()->root_instruction());
  EXPECT_EQ(GetCollectivePermuteCostModelType(*instr,
                                              /*num_devices_per_partition=*/8),
            CollectivePermuteCostModelType::kIntraPartitionOneWay);
}

TEST_F(CommunicationTypeTest, CollectivePermuteIntraPartitionTwoWayMutual) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=4

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
        source_target_pairs={{0,1},{1,0},{2,3},{3,2}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectivePermuteInstruction* instr =
      Cast<HloCollectivePermuteInstruction>(
          module->entry_computation()->root_instruction());
  EXPECT_EQ(GetCollectivePermuteCostModelType(*instr,
                                              /*num_devices_per_partition=*/8),
            CollectivePermuteCostModelType::kIntraPartitionTwoWayAllMutual);
}

TEST_F(CommunicationTypeTest, CollectivePermuteInterPartitionTwoWayMutual) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
        source_target_pairs={{0,8},{8,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectivePermuteInstruction* instr =
      Cast<HloCollectivePermuteInstruction>(
          module->entry_computation()->root_instruction());
  EXPECT_EQ(GetCollectivePermuteCostModelType(*instr,
                                              /*num_devices_per_partition=*/8),
            CollectivePermuteCostModelType::kInterPartitionTwoWayAllMutual);
}

TEST_F(CommunicationTypeTest, CollectivePermuteInterPartitionOneWay) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
        source_target_pairs={{0,8},{1,9}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectivePermuteInstruction* instr =
      Cast<HloCollectivePermuteInstruction>(
          module->entry_computation()->root_instruction());
  EXPECT_EQ(GetCollectivePermuteCostModelType(*instr,
                                              /*num_devices_per_partition=*/8),
            CollectivePermuteCostModelType::kInterPartitionOneWay);
}

TEST_F(CommunicationTypeTest,
       CollectivePermuteIntraPartitionTwoWayHasNonMutual) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=8

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
        source_target_pairs={{0,1},{1,2},{2,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectivePermuteInstruction* instr =
      Cast<HloCollectivePermuteInstruction>(
          module->entry_computation()->root_instruction());
  EXPECT_EQ(GetCollectivePermuteCostModelType(*instr,
                                              /*num_devices_per_partition=*/8),
            CollectivePermuteCostModelType::kIntraPartitionTwoWayHasNonMutual);
}

TEST_F(CommunicationTypeTest,
       CollectivePermuteInterPartitionTwoWayHasNonMutual) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=16
    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
        source_target_pairs={{0,8},{1,9},{8,2}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectivePermuteInstruction* instr =
      Cast<HloCollectivePermuteInstruction>(
          module->entry_computation()->root_instruction());
  EXPECT_EQ(GetCollectivePermuteCostModelType(*instr,
                                              /*num_devices_per_partition=*/8),
            CollectivePermuteCostModelType::kInterPartitionTwoWayHasNonMutual);
}

// TODO(b/460155942): remove once the collective-permute with empty pairs is
// disallowed by the HLO verifier.
TEST_F(CommunicationTypeTest, CollectivePermuteEmptyPairs) {
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=8

    ENTRY e {
      p = f32[128] parameter(0)
      ROOT _ = f32[128] collective-permute(p),
        source_target_pairs={}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));

  HloCollectivePermuteInstruction* instr =
      Cast<HloCollectivePermuteInstruction>(
          module->entry_computation()->root_instruction());
  EXPECT_EQ(GetCollectivePermuteCostModelType(*instr,
                                              /*num_devices_per_partition=*/8),
            CollectivePermuteCostModelType::kUnknown);
}

}  // namespace xla::gpu
