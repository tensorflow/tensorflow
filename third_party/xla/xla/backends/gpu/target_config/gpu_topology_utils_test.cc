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
#include "xla/service/gpu_topology.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.pb.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

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

TEST(IsCompatibleWithTargetTopologyTest,
     RequiredLocalityDomainCountMustMatchTarget) {
  ASSERT_OK_AND_ASSIGN(auto target_config_proto,
                       gpu::GetGpuTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(auto compiler_config,
                       gpu::GpuTargetConfig::FromProto(target_config_proto));
  auto target_config = compiler_config;
  compiler_config.device_description.set_locality_domain_count(2);
  target_config.device_description.set_locality_domain_count(1);
  GpuTopology compiler_topology("same_platform", 1, 1, 1, compiler_config);
  GpuTopology target_topology("same_platform", 1, 1, 1, target_config);
  EXPECT_THAT(
      IsCompatibleWithTargetTopology(compiler_topology, target_topology),
      IsOkAndHolds(false));
  EXPECT_THAT(IsCompatibleWithTargetTopology(compiler_topology.ToProto(),
                                             target_topology.ToProto()),
              IsOkAndHolds(false));
}

TEST(IsCompatibleWithTargetTopologyTest,
     UnspecifiedLocalityDomainCountDoesNotConstrainTarget) {
  ASSERT_OK_AND_ASSIGN(auto target_config_proto,
                       gpu::GetGpuTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(auto compiler_config,
                       gpu::GpuTargetConfig::FromProto(target_config_proto));
  auto target_config = compiler_config;
  compiler_config.device_description.set_locality_domain_count(0);
  target_config.device_description.set_locality_domain_count(2);
  GpuTopology compiler_topology("same_platform", 1, 1, 1, compiler_config);
  GpuTopology target_topology("same_platform", 1, 1, 1, target_config);
  EXPECT_THAT(
      IsCompatibleWithTargetTopology(compiler_topology, target_topology),
      IsOkAndHolds(true));
  EXPECT_THAT(IsCompatibleWithTargetTopology(compiler_topology.ToProto(),
                                             target_topology.ToProto()),
              IsOkAndHolds(true));
}

TEST(IsCompatibleWithTargetTopologyTest,
     MatchingLocalityDomainCountIsCompatible) {
  ASSERT_OK_AND_ASSIGN(auto target_config_proto,
                       gpu::GetGpuTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(auto compiler_config,
                       gpu::GpuTargetConfig::FromProto(target_config_proto));
  auto target_config = compiler_config;
  compiler_config.device_description.set_locality_domain_count(2);
  target_config.device_description.set_locality_domain_count(2);
  GpuTopology compiler_topology("same_platform", 1, 1, 1, compiler_config);
  GpuTopology target_topology("same_platform", 1, 1, 1, target_config);
  EXPECT_THAT(
      IsCompatibleWithTargetTopology(compiler_topology, target_topology),
      IsOkAndHolds(true));
  EXPECT_THAT(IsCompatibleWithTargetTopology(compiler_topology.ToProto(),
                                             target_topology.ToProto()),
              IsOkAndHolds(true));
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

TEST(IsCompatibleWithTargetTopologyTest,
     RegisteredPlatformSkuVariantsAreCompatible) {
  ASSERT_OK_AND_ASSIGN(GpuTopology gb300_topology,
                       GetGpuTopologyForPlatform("oberon_b300", 1, 1, 4));
  ASSERT_OK_AND_ASSIGN(GpuTopology b300_topology,
                       GetGpuTopologyForPlatform("umbriel_b300", 1, 1, 8));

  GpuTargetConfig maxq_config = gb300_topology.gpu_target_config();
  maxq_config.device_description_str = "NVIDIA GB300 Max-Q";
  maxq_config.device_description.set_name("NVIDIA GB300 Max-Q");
  GpuTopology gb300_maxq_topology("NVIDIA GB300 Max-Q", 1, 1, 4, maxq_config);

  EXPECT_THAT(
      IsCompatibleWithTargetTopology(gb300_topology, gb300_maxq_topology),
      IsOkAndHolds(true));
  EXPECT_THAT(
      IsCompatibleWithTargetTopology(b300_topology, gb300_maxq_topology),
      IsOkAndHolds(true));

  GpuTargetConfig b300_sxm_config = b300_topology.gpu_target_config();
  b300_sxm_config.device_description_str = "NVIDIA B300 SXM6 AC";
  b300_sxm_config.device_description.set_name("NVIDIA B300 SXM6 AC");
  GpuTopology b300_sxm_topology("NVIDIA B300 SXM6 AC", 1, 1, 8,
                                b300_sxm_config);
  EXPECT_THAT(IsCompatibleWithTargetTopology(b300_topology, b300_sxm_topology),
              IsOkAndHolds(true));

  ASSERT_OK_AND_ASSIGN(GpuTopology vr200_topology,
                       GetGpuTopologyForPlatform("velorum_r200", 1, 1, 4));
  GpuTargetConfig generic_vr_config = vr200_topology.gpu_target_config();
  generic_vr_config.device_description_str = "NVIDIA Graphics Device";
  generic_vr_config.device_description.set_name("NVIDIA Graphics Device");
  GpuTopology generic_vr_topology("NVIDIA Graphics Device", 1, 1, 4,
                                  generic_vr_config);
  EXPECT_THAT(
      IsCompatibleWithTargetTopology(vr200_topology, generic_vr_topology),
      IsOkAndHolds(true));

  // Distinct platforms that share a compute capability but are not in
  // `CompatibleGpuPair` remain incompatible.
  ASSERT_OK_AND_ASSIGN(GpuTopology h100_topology,
                       GetGpuTopologyForPlatform("nvidia_h100", 1, 1, 8));
  ASSERT_OK_AND_ASSIGN(GpuTopology h200_topology,
                       GetGpuTopologyForPlatform("tesla_h200", 1, 1, 8));
  EXPECT_THAT(IsCompatibleWithTargetTopology(h100_topology, h200_topology),
              IsOkAndHolds(false));

  ASSERT_OK_AND_ASSIGN(GpuTopology a100_40_topology,
                       GetGpuTopologyForPlatform("tesla_a100", 1, 1, 16));
  ASSERT_OK_AND_ASSIGN(GpuTopology a100_80_topology,
                       GetGpuTopologyForPlatform("tesla_a100_80gib", 1, 1, 8));
  EXPECT_THAT(
      IsCompatibleWithTargetTopology(a100_40_topology, a100_80_topology),
      IsOkAndHolds(false));
}

TEST(IsCompatibleWithTargetTopologyTest,
     FallsBackWhenTargetConfigDeviceDescriptionStrIsEmpty) {
  ASSERT_OK_AND_ASSIGN(GpuTopology target_topology,
                       GetGpuTopologyForPlatform("oberon_b300", 1, 1, 4));

  // Fallback to `device_description.name()` when `device_description_str` is
  // empty.
  GpuTargetConfig named_device_config = target_topology.gpu_target_config();
  named_device_config.device_description_str.clear();
  named_device_config.device_description.set_name("NVIDIA GB300 Max-Q");
  GpuTopology named_device_topology("custom_platform", 1, 1, 4,
                                    named_device_config);
  EXPECT_OK(ValidateSupportedGpuTopology(named_device_topology));
  EXPECT_THAT(
      IsCompatibleWithTargetTopology(named_device_topology, target_topology),
      IsOkAndHolds(true));

  // Fallback to `platform_version()` when both `device_description_str` and
  // `device_description.name()` are empty.
  GpuTargetConfig empty_desc_config = target_topology.gpu_target_config();
  empty_desc_config.device_description_str.clear();
  empty_desc_config.device_description.set_name("");
  GpuTopology platform_fallback_topology("NVIDIA GB300 Max-Q", 1, 1, 4,
                                         empty_desc_config);
  EXPECT_OK(ValidateSupportedGpuTopology(platform_fallback_topology));
  EXPECT_THAT(IsCompatibleWithTargetTopology(platform_fallback_topology,
                                             target_topology),
              IsOkAndHolds(true));

  // When all three identifiers are empty/unrecognized, compatibility fails fast
  // with `InvalidArgumentError`.
  GpuTopology unrecognized_a("unrecognized_p1", 1, 1, 4, empty_desc_config);
  GpuTopology unrecognized_b("unrecognized_p2", 1, 1, 4, empty_desc_config);
  EXPECT_THAT(
      ValidateSupportedGpuTopology(unrecognized_a),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unsupported GPU platform type: unrecognized_p1")));
  EXPECT_THAT(IsCompatibleWithTargetTopology(unrecognized_a, unrecognized_b),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ValidateSupportedGpuTopologyTest, AcceptsSupportedPlatformsAndSkus) {
  ASSERT_OK_AND_ASSIGN(GpuTopology gb300_topology,
                       GetGpuTopologyForPlatform("oberon_b300", 1, 1, 4));
  GpuTargetConfig maxq_config = gb300_topology.gpu_target_config();
  maxq_config.device_description_str = "NVIDIA GB300 Max-Q";
  GpuTopology gb300_maxq_topology("NVIDIA GB300 Max-Q", 1, 1, 4, maxq_config);
  EXPECT_OK(ValidateSupportedGpuTopology(gb300_maxq_topology));

  GpuTopology p100_pcie_topology("Tesla P100-PCIE-16GB", 1, 1, 1);
  EXPECT_OK(ValidateSupportedGpuTopology(p100_pcie_topology));

  GpuTopology named_only_topology("NVIDIA H100 80GB HBM3", 1, 1, 8);
  EXPECT_OK(ValidateSupportedGpuTopology(named_only_topology));
}

TEST(ValidateSupportedGpuTopologyTest, RejectsUnsupportedGpuTopology) {
  GpuTopology unknown_topology("Unsupported GPU SKU", 1, 1, 1);
  EXPECT_THAT(
      ValidateSupportedGpuTopology(unknown_topology),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Unsupported GPU platform type: Unsupported GPU SKU")));

  GpuTopology empty_topology("", 1, 1, 1);
  EXPECT_THAT(ValidateSupportedGpuTopology(empty_topology),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("GPU topology does not specify a device or "
                                 "platform identifier")));
}

}  // namespace
}  // namespace xla::gpu
