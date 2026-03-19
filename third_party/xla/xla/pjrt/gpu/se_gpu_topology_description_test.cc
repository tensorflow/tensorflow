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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_stream_executor_device_description.h"
#include "xla/service/gpu_topology.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

void CheckDeviceDescription(const PjRtDeviceDescription& device_desc,
                            int global_device_id, int process_index,
                            const std::vector<int>& coords) {
  EXPECT_EQ(device_desc.id(), global_device_id);
  EXPECT_EQ(device_desc.process_index(), process_index);
  const auto& gpu_device_desc =
      dynamic_cast<const PjRtStreamExecutorDeviceDescription&>(device_desc);
  EXPECT_THAT(gpu_device_desc.coords(),
              ElementsAre(coords[0], coords[1], coords[2]));
}

TEST(StreamExecutorGpuTopologyDescriptionTest, SymmetricTopology) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_partitions=*/3,
          /*num_hosts_per_partition=*/2, /*num_devices_per_host=*/4);

  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  EXPECT_EQ(topology_desc.platform_id(), xla::CudaId());
  EXPECT_EQ(topology_desc.platform_name(), xla::CudaName());
  EXPECT_EQ(topology_desc.platform_version(), "12.3");

  const auto device_descs = topology_desc.DeviceDescriptions();
  EXPECT_EQ(device_descs.size(), 24);
  CheckDeviceDescription(*device_descs[0], 0, 0, {0, 0, 0});
  CheckDeviceDescription(*device_descs[1], 1, 0, {0, 0, 1});
  CheckDeviceDescription(*device_descs[2], 2, 0, {0, 0, 2});
  CheckDeviceDescription(*device_descs[3], 3, 0, {0, 0, 3});
  CheckDeviceDescription(*device_descs[4], 4, 1, {0, 1, 0});
  CheckDeviceDescription(*device_descs[5], 5, 1, {0, 1, 1});
  CheckDeviceDescription(*device_descs[6], 6, 1, {0, 1, 2});
  CheckDeviceDescription(*device_descs[7], 7, 1, {0, 1, 3});
  CheckDeviceDescription(*device_descs[8], 8, 2, {1, 0, 0});
  CheckDeviceDescription(*device_descs[9], 9, 2, {1, 0, 1});
  CheckDeviceDescription(*device_descs[10], 10, 2, {1, 0, 2});
  CheckDeviceDescription(*device_descs[11], 11, 2, {1, 0, 3});
  CheckDeviceDescription(*device_descs[12], 12, 3, {1, 1, 0});
  CheckDeviceDescription(*device_descs[13], 13, 3, {1, 1, 1});
  CheckDeviceDescription(*device_descs[14], 14, 3, {1, 1, 2});
  CheckDeviceDescription(*device_descs[15], 15, 3, {1, 1, 3});
  CheckDeviceDescription(*device_descs[16], 16, 4, {2, 0, 0});
  CheckDeviceDescription(*device_descs[17], 17, 4, {2, 0, 1});
  CheckDeviceDescription(*device_descs[18], 18, 4, {2, 0, 2});
  CheckDeviceDescription(*device_descs[19], 19, 4, {2, 0, 3});
  CheckDeviceDescription(*device_descs[20], 20, 5, {2, 1, 0});
  CheckDeviceDescription(*device_descs[21], 21, 5, {2, 1, 1});
  CheckDeviceDescription(*device_descs[22], 22, 5, {2, 1, 2});
  CheckDeviceDescription(*device_descs[23], 23, 5, {2, 1, 3});
}

TEST(StreamExecutorGpuTopologyDescriptionTest, AsymmetricTopology) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_partitions=*/-1,
          /*num_hosts_per_partition=*/-1, /*num_devices_per_host=*/-1);

  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  EXPECT_EQ(topology_desc.platform_id(), xla::CudaId());
  EXPECT_EQ(topology_desc.platform_name(), xla::CudaName());
  EXPECT_EQ(topology_desc.platform_version(), "12.3");

  const auto device_descs = topology_desc.DeviceDescriptions();
  EXPECT_EQ(device_descs.size(), 0);
}

TEST(PjRtTopologyUtilsGPUTest, GetDeviceCoords) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_partitions=*/1,
          /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/4);
  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(1)));
  auto [device_coords, core_id] = std::move(device_core);
  ASSERT_EQ(device_coords, (PjRtDeviceDimensions{0, 0, 1}));
  ASSERT_EQ(core_id, 0);
}

TEST(PjRtTopologyUtilsGPUTest, GetDeviceCoordsSingleHostScopedPartition) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_partitions=*/4,
          /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/4);
  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core1,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(1)));
  auto [device_coords1, core_id1] = std::move(device_core1);
  ASSERT_EQ(device_coords1, (PjRtDeviceDimensions{0, 0, 1}));
  ASSERT_EQ(core_id1, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core2,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(6)));
  auto [device_coords2, core_id2] = std::move(device_core2);
  ASSERT_EQ(device_coords2, (PjRtDeviceDimensions{1, 0, 2}));
  ASSERT_EQ(core_id2, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core3,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(10)));
  auto [device_coords3, core_id3] = std::move(device_core3);
  ASSERT_EQ(device_coords3, (PjRtDeviceDimensions{2, 0, 2}));
  ASSERT_EQ(core_id3, 0);
}

TEST(PjRtTopologyUtilsGPUTest, GetDeviceCoordsMultipleHostScopedPartition) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_partitions=*/1,
          /*num_hosts_per_partition=*/4, /*num_devices_per_host=*/4);
  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core1,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(1)));
  auto [device_coords1, core_id1] = std::move(device_core1);
  ASSERT_EQ(device_coords1, (PjRtDeviceDimensions{0, 0, 1}));
  ASSERT_EQ(core_id1, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core2,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(6)));
  auto [device_coords2, core_id2] = std::move(device_core2);
  ASSERT_EQ(device_coords2, (PjRtDeviceDimensions{0, 1, 2}));
  ASSERT_EQ(core_id2, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core3,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(10)));
  auto [device_coords3, core_id3] = std::move(device_core3);
  ASSERT_EQ(device_coords3, (PjRtDeviceDimensions{0, 2, 2}));
  ASSERT_EQ(core_id3, 0);
}

}  // namespace
}  // namespace xla
