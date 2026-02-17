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

#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.pb.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

TEST(CpuTopologyDescriptionTest, ToProto) {
  std::vector<CpuTopology::CpuDevice> cpu_devices = {{0, 0}, {0, 1}};
  std::vector<std::string> machine_attributes = {"attr1", "attr2"};
  CpuTopologyDescription topology(xla::CpuId(), "cpu", "1.0", cpu_devices,
                                  machine_attributes);

  TF_ASSERT_OK_AND_ASSIGN(PjRtTopologyDescriptionProto proto,
                          topology.ToProto());

  EXPECT_EQ(proto.platform_id(), xla::CpuId());
  EXPECT_EQ(proto.platform_name(), "cpu");
  EXPECT_EQ(proto.platform_version(), "1.0");

  CpuTopologyProto cpu_topology_proto;
  ASSERT_TRUE(proto.platform_specific_topology().UnpackTo(&cpu_topology_proto));

  for (int i = 0; i < cpu_devices.size(); ++i) {
    const auto& cpu_device = cpu_devices[i];
    EXPECT_EQ(cpu_device.process_id,
              cpu_topology_proto.cpu_devices(i).process_index());
    EXPECT_EQ(cpu_device.local_device_id,
              cpu_topology_proto.cpu_devices(i).local_hardware_id());
  }

  for (int i = 0; i < machine_attributes.size(); ++i) {
    EXPECT_EQ(machine_attributes[i], cpu_topology_proto.machine_attributes(i));
  }
}

TEST(CpuTopologyDescriptionTest, FromProto) {
  PjRtTopologyDescriptionProto proto;
  proto.set_platform_id(xla::CpuId());
  proto.set_platform_name("cpu");
  proto.set_platform_version("2.0");

  CpuTopologyProto cpu_topology_proto;
  auto* device1 = cpu_topology_proto.add_cpu_devices();
  device1->set_process_index(0);
  device1->set_local_hardware_id(0);
  auto* device2 = cpu_topology_proto.add_cpu_devices();
  device2->set_process_index(1);
  device2->set_local_hardware_id(1);
  cpu_topology_proto.add_machine_attributes("attrA");
  cpu_topology_proto.add_machine_attributes("attrB");

  proto.mutable_platform_specific_topology()->PackFrom(cpu_topology_proto);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtTopologyDescription> topology_desc,
      CpuTopologyDescription::FromProto(proto));

  CpuTopologyDescription* cpu_topology =
      dynamic_cast<CpuTopologyDescription*>(topology_desc.get());
  ASSERT_NE(cpu_topology, nullptr);

  EXPECT_EQ(cpu_topology->platform_id(), xla::CpuId());
  EXPECT_EQ(cpu_topology->platform_name(), "cpu");
  EXPECT_EQ(cpu_topology->platform_version(), "2.0");

  const auto& devices = cpu_topology->cpu_topology().devices();
  ASSERT_EQ(devices.size(), 2);
  EXPECT_EQ(devices[0].process_id, 0);
  EXPECT_EQ(devices[0].local_device_id, 0);
  EXPECT_EQ(devices[1].process_id, 1);
  EXPECT_EQ(devices[1].local_device_id, 1);

  EXPECT_THAT(cpu_topology->cpu_topology().machine_attributes(),
              ElementsAre("attrA", "attrB"));
}

TEST(CpuTopologyDescriptionTest,
     ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType) {
  std::vector<CpuTopology::CpuDevice> cpu_devices = {{0, 0}, {0, 1}};
  std::vector<std::string> machine_attributes = {"attr1", "attr2"};
  CpuTopologyDescription topology(xla::CpuId(), "cpu", "1.0", cpu_devices,
                                  machine_attributes);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core,
      topology.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          xla::PjRtGlobalDeviceId(1)));
  auto [device_coords, core_id] = std::move(device_core);
  ASSERT_EQ(device_coords, (PjRtDeviceDimensions{0, 0, 1}));
  ASSERT_EQ(core_id, 0);
}

}  // namespace
}  // namespace xla
