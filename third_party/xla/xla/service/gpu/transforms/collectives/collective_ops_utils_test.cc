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

#include <gtest/gtest.h>
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

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

}  // namespace
}  // namespace gpu
}  // namespace xla
