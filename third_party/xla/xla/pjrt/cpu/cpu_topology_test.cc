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

#include "xla/pjrt/cpu/cpu_topology.h"

#include <memory>

#include "tsl/platform/protobuf.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(CpuTopology, FromProto) {
  CpuTopologyProto msg;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        cpu_devices:
        [ { process_index: 2, local_hardware_id: 3 }]
        machine_attributes: [ "x86_64", "Intel" ]
      )pb",
      &msg));

  std::unique_ptr<const CpuTopology> cpu_topology = CpuTopology::FromProto(msg);
  EXPECT_EQ(cpu_topology->devices().size(), 1);
  EXPECT_EQ(cpu_topology->devices()[0].process_id, 2);
  EXPECT_EQ(cpu_topology->devices()[0].local_device_id, 3);
  EXPECT_EQ(cpu_topology->machine_attributes().size(), 2);
  EXPECT_EQ(cpu_topology->machine_attributes()[0], "x86_64");
  EXPECT_EQ(cpu_topology->machine_attributes()[1], "Intel");
}

TEST(CpuTopology, ToProto) {
  CpuTopology cpu_topology({{2, 3}}, {"ab", "cd"});
  CpuTopologyProto msg = cpu_topology.ToProto();
  EXPECT_EQ(msg.cpu_devices_size(), 1);
  EXPECT_EQ(msg.cpu_devices(0).process_index(), 2);
  EXPECT_EQ(msg.cpu_devices(0).local_hardware_id(), 3);
  EXPECT_EQ(msg.machine_attributes_size(), 2);
  EXPECT_EQ(msg.machine_attributes(0), "ab");
  EXPECT_EQ(msg.machine_attributes(1), "cd");
}

}  // namespace
}  // namespace xla
