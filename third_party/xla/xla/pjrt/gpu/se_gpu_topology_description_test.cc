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

#include "xla/pjrt/gpu/se_gpu_topology_description.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_stream_executor_device_description.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

void CheckDeviceDescription(const PjRtDeviceDescription& device_desc,
                            int global_device_id, int local_device_id,
                            int process_index, int slice_index) {
  EXPECT_EQ(device_desc.id(), global_device_id);
  EXPECT_EQ(device_desc.process_index(), process_index);
  const auto& gpu_device_desc =
      dynamic_cast<const PjRtStreamExecutorDeviceDescription&>(device_desc);
  EXPECT_THAT(gpu_device_desc.coords(),
              ElementsAre(local_device_id, process_index, slice_index));
}

TEST(StreamExecutorGpuTopologyDescriptionTest, SymmetricTopology) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_slices=*/2,
          /*num_hosts_per_slice=*/2, /*num_devices_per_host=*/2);

  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  EXPECT_EQ(topology_desc.platform_id(), xla::CudaId());
  EXPECT_EQ(topology_desc.platform_name(), xla::CudaName());
  EXPECT_EQ(topology_desc.platform_version(), "12.3");

  const auto device_descs = topology_desc.DeviceDescriptions();
  EXPECT_EQ(device_descs.size(), 8);
  CheckDeviceDescription(*device_descs[0], 0, 0, 0, 0);
  CheckDeviceDescription(*device_descs[1], 1, 1, 0, 0);
  CheckDeviceDescription(*device_descs[2], 2, 0, 1, 0);
  CheckDeviceDescription(*device_descs[3], 3, 1, 1, 0);
  CheckDeviceDescription(*device_descs[4], 4, 0, 2, 1);
  CheckDeviceDescription(*device_descs[5], 5, 1, 2, 1);
  CheckDeviceDescription(*device_descs[6], 6, 0, 3, 1);
  CheckDeviceDescription(*device_descs[7], 7, 1, 3, 1);
}

TEST(StreamExecutorGpuTopologyDescriptionTest, AsymmetricTopology) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_slices=*/-1,
          /*num_hosts_per_slice=*/-1, /*num_devices_per_host=*/-1);

  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  EXPECT_EQ(topology_desc.platform_id(), xla::CudaId());
  EXPECT_EQ(topology_desc.platform_name(), xla::CudaName());
  EXPECT_EQ(topology_desc.platform_version(), "12.3");

  const auto device_descs = topology_desc.DeviceDescriptions();
  EXPECT_EQ(device_descs.size(), 0);
}

}  // namespace
}  // namespace xla
