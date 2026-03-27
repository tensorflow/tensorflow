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

#include "xla/backends/gpu/target_config/gpu_topology_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(IsCompatibleWithTargetTopologyTest, SamePlatformsAreCompatible) {
  GpuTopology single_topology("platform_version", 1, 1, 1);
  EXPECT_THAT(IsCompatibleWithTargetTopology(single_topology, single_topology),
              IsOkAndHolds(true));
  EXPECT_THAT(IsCompatibleWithTargetTopology(single_topology.ToProto(),
                                             single_topology.ToProto()),
              IsOkAndHolds(true));
}

TEST(IsCompatibleWithTargetTopologyTest,
     DifferentPlatformVersionsButSameDeviceDescriptionsAreCompatible) {
  // Depending on where the topology is created, the platform version may not be
  // identical byte for byte. But the device description is authorative for
  // compatibility.
  ASSERT_OK_AND_ASSIGN(auto gpu_target_config_proto,
                       gpu::GetGpuTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(auto gpu_target_config, gpu::GpuTargetConfig::FromProto(
                                                   gpu_target_config_proto));
  GpuTopology first_h100_topology("platform_version", 1, 1, 1,
                                  gpu_target_config);
  GpuTopology second_h100_topology("other_platform_version", 1, 1, 1,
                                   gpu_target_config);
  EXPECT_THAT(
      IsCompatibleWithTargetTopology(first_h100_topology, second_h100_topology),
      IsOkAndHolds(true));
  EXPECT_THAT(IsCompatibleWithTargetTopology(first_h100_topology.ToProto(),
                                             second_h100_topology.ToProto()),
              IsOkAndHolds(true));
}

TEST(IsCompatibleWithTargetTopologyTest, B200andGB200AreCompatible) {
  ASSERT_OK_AND_ASSIGN(GpuTopology b200_topology,
                       GetGpuTopologyForPlatform("umbriel_b200", 1, 1, 1));
  ASSERT_OK_AND_ASSIGN(GpuTopology gb200_topology,
                       GetGpuTopologyForPlatform("oberon_b200", 1, 1, 1));
  EXPECT_THAT(IsCompatibleWithTargetTopology(b200_topology, gb200_topology),
              IsOkAndHolds(true));
  EXPECT_THAT(IsCompatibleWithTargetTopology(b200_topology.ToProto(),
                                             gb200_topology.ToProto()),
              IsOkAndHolds(true));
  // Compatibility is symmetric for B200 and GB200.
  EXPECT_THAT(IsCompatibleWithTargetTopology(gb200_topology, b200_topology),
              IsOkAndHolds(true));
  EXPECT_THAT(IsCompatibleWithTargetTopology(gb200_topology.ToProto(),
                                             b200_topology.ToProto()),
              IsOkAndHolds(true));
}

TEST(IsCompatibleWithTargetTopologyTest, B200andH100AreIncompatible) {
  ASSERT_OK_AND_ASSIGN(GpuTopology b200_topology,
                       GetGpuTopologyForPlatform("umbriel_b200", 1, 1, 1));
  ASSERT_OK_AND_ASSIGN(GpuTopology h100_topology,
                       GetGpuTopologyForPlatform("nvidia_h100", 1, 1, 1));
  EXPECT_THAT(IsCompatibleWithTargetTopology(b200_topology, h100_topology),
              IsOkAndHolds(false));
  EXPECT_THAT(IsCompatibleWithTargetTopology(b200_topology.ToProto(),
                                             h100_topology.ToProto()),
              IsOkAndHolds(false));
  EXPECT_THAT(IsCompatibleWithTargetTopology(h100_topology, b200_topology),
              IsOkAndHolds(false));
  EXPECT_THAT(IsCompatibleWithTargetTopology(h100_topology.ToProto(),
                                             b200_topology.ToProto()),
              IsOkAndHolds(false));
}

TEST(IsCompatibleWithTargetTopoologyTest, FailsWithInvalidProto) {
  GpuTopologyProto invalid_proto;
  // Setting cuda_compute_capability to an invalid feature extension to provoke
  // a failure in FromProto.
  invalid_proto.mutable_gpu_target_config()
      ->mutable_gpu_device_info()
      ->mutable_cuda_compute_capability()
      ->set_feature_extension(
          static_cast<
              stream_executor::CudaComputeCapabilityProto::FeatureExtension>(
              99));

  ASSERT_OK_AND_ASSIGN(GpuTopology single_topology,
                       GetGpuTopologyForPlatform("nvidia_h100", 1, 1, 1));
  EXPECT_THAT(
      IsCompatibleWithTargetTopology(invalid_proto, single_topology.ToProto()),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      IsCompatibleWithTargetTopology(single_topology.ToProto(), invalid_proto),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::gpu
