/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/pjrt/distributed/topology_util.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/test_helpers.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

TEST(TopologyTest, BuildGlobalTopology) {
  std::vector<LocalTopologyProto> locals(2);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  TF_ASSERT_OK_AND_ASSIGN(
      GlobalTopologyProto global,
      BuildGlobalTopology(absl::Span<LocalTopologyProto>(locals),
                          /*assign_global_device_ids=*/true));
  EXPECT_EQ(global.nodes_size(), 2);
  EXPECT_EQ(global.nodes()[0].devices_size(), 2);
  EXPECT_EQ(global.nodes()[1].devices_size(), 2);
}

TEST(TopologyTest, BuildGlobalTopologyWithFabricUuid) {
  std::vector<LocalTopologyProto> locals(2);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  d0->set_fabric_uuid("00000000-0000-0000-0000-000000000001/0");
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(1);
  d1->set_fabric_uuid("00000000-0000-0000-0000-000000000001/0");
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  d2->set_fabric_uuid("00000000-0000-0000-0000-000000000001/0");
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);
  d3->set_fabric_uuid("00000000-0000-0000-0000-000000000001/0");

  TF_ASSERT_OK_AND_ASSIGN(
      GlobalTopologyProto global,
      BuildGlobalTopology(absl::Span<LocalTopologyProto>(locals),
                          /*assign_global_device_ids=*/true));
  EXPECT_EQ(global.nodes_size(), 2);
  EXPECT_EQ(global.nodes()[0].devices_size(), 2);
  EXPECT_EQ(global.nodes()[1].devices_size(), 2);
  EXPECT_EQ(global.nodes()[0].devices()[0].slice_index(), 0);
  EXPECT_EQ(global.nodes()[0].devices()[1].slice_index(), 0);
  EXPECT_EQ(global.nodes()[1].devices()[0].slice_index(), 0);
  EXPECT_EQ(global.nodes()[1].devices()[1].slice_index(), 0);
}

TEST(TopologyTest, BuildGlobalTopologyMultipleFabricUuid) {
  std::vector<LocalTopologyProto> locals(4);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  d0->set_fabric_uuid("00000000-0000-0000-0000-000000000001/0");
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(1);
  d1->set_fabric_uuid("00000000-0000-0000-0000-000000000001/0");
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  d2->set_fabric_uuid("00000000-0000-0000-0000-000000000001/0");
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);
  d3->set_fabric_uuid("00000000-0000-0000-0000-000000000001/0");
  DeviceProto* d4 = locals[2].add_devices();
  d4->set_local_device_ordinal(0);
  d4->set_fabric_uuid("00000000-0000-0000-0000-000000000002/0");
  DeviceProto* d5 = locals[2].add_devices();
  d5->set_local_device_ordinal(1);
  d5->set_fabric_uuid("00000000-0000-0000-0000-000000000002/0");
  DeviceProto* d6 = locals[3].add_devices();
  d6->set_local_device_ordinal(0);
  d6->set_fabric_uuid("00000000-0000-0000-0000-000000000002/0");
  DeviceProto* d7 = locals[3].add_devices();
  d7->set_local_device_ordinal(1);
  d7->set_fabric_uuid("00000000-0000-0000-0000-000000000002/0");

  TF_ASSERT_OK_AND_ASSIGN(
      GlobalTopologyProto global,
      BuildGlobalTopology(absl::Span<LocalTopologyProto>(locals),
                          /*assign_global_device_ids=*/true));
  EXPECT_EQ(global.nodes_size(), 4);
  EXPECT_EQ(global.nodes()[0].devices_size(), 2);
  EXPECT_EQ(global.nodes()[1].devices_size(), 2);
  EXPECT_EQ(global.nodes()[2].devices_size(), 2);
  EXPECT_EQ(global.nodes()[3].devices_size(), 2);
  EXPECT_EQ(global.nodes()[0].devices()[0].slice_index(), 0);
  EXPECT_EQ(global.nodes()[0].devices()[1].slice_index(), 0);
  EXPECT_EQ(global.nodes()[1].devices()[0].slice_index(), 0);
  EXPECT_EQ(global.nodes()[1].devices()[1].slice_index(), 0);
  EXPECT_EQ(global.nodes()[2].devices()[0].slice_index(), 1);
  EXPECT_EQ(global.nodes()[2].devices()[1].slice_index(), 1);
  EXPECT_EQ(global.nodes()[3].devices()[0].slice_index(), 1);
  EXPECT_EQ(global.nodes()[3].devices()[1].slice_index(), 1);
}

TEST(TopologyTest, ExchangeTopology) {
  int num_nodes = 2;
  std::vector<LocalTopologyProto> locals(num_nodes);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  InMemoryKeyValueStore kv_store;
  std::vector<GlobalTopologyProto> globals(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "TestPool",
                                        num_nodes);
    for (int i = 0; i < num_nodes; i++) {
      thread_pool.Schedule([&, i] {
        TF_ASSERT_OK(ExchangeTopologies(
            /*platform=*/"cuda", /*node_id=*/i, num_nodes,
            /*get_local_topology_timeout=*/
            absl::Seconds(10), /*get_global_topology_timeout=*/
            absl::Seconds(10), &kv_store, locals[i], &globals[i],
            /*assign_global_device_ids=*/true));
      });
    }
  }
  for (const GlobalTopologyProto& global : globals) {
    EXPECT_EQ(global.nodes_size(), 2);
    EXPECT_EQ(global.nodes()[0].devices_size(), 2);
    EXPECT_EQ(global.nodes()[1].devices_size(), 2);
  }
}

TEST(TopologyTest, ExchangeTopology_Twice_Succeeds) {
  int num_nodes = 2;
  std::vector<LocalTopologyProto> locals(num_nodes);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  InMemoryKeyValueStore kv_store(/*allow_overwrite=*/false);
  std::vector<GlobalTopologyProto> globals(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "TestPool",
                                        num_nodes);
    for (int i = 0; i < num_nodes; i++) {
      thread_pool.Schedule([&, i] {
        TF_ASSERT_OK(ExchangeTopologies(
            /*platform=*/"cuda", /*node_id=*/i, num_nodes,
            /*get_local_topology_timeout=*/
            absl::Seconds(10), /*get_global_topology_timeout=*/
            absl::Seconds(10), &kv_store, locals[i], &globals[i],
            /*assign_global_device_ids=*/true));
        // Simulate node 1 restarting and exchanging topologies again.
        if (i == 1) {
          TF_ASSERT_OK(ExchangeTopologies(
              /*platform=*/"cuda", /*node_id=*/i, num_nodes,
              /*get_local_topology_timeout=*/
              absl::Seconds(10), /*get_global_topology_timeout=*/
              absl::Seconds(10), &kv_store, locals[i], &globals[i],
              /*assign_global_device_ids=*/true));
        }
      });
    }
  }
  for (const GlobalTopologyProto& global : globals) {
    EXPECT_EQ(global.nodes_size(), 2);
    EXPECT_EQ(global.nodes()[0].devices_size(), 2);
    EXPECT_EQ(global.nodes()[1].devices_size(), 2);
  }
}

TEST(TopologyTest, ExchangeTopology_TwiceWithDifferentLocalTopology_Fails) {
  int num_nodes = 2;
  std::vector<LocalTopologyProto> locals(num_nodes);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  InMemoryKeyValueStore kv_store(/*allow_overwrite=*/false);
  std::vector<GlobalTopologyProto> globals(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "TestPool",
                                        num_nodes);
    for (int i = 0; i < num_nodes; i++) {
      thread_pool.Schedule([&, i] {
        TF_ASSERT_OK(ExchangeTopologies(
            /*platform=*/"cuda", /*node_id=*/i, num_nodes,
            /*get_local_topology_timeout=*/
            absl::Seconds(10), /*get_global_topology_timeout=*/
            absl::Seconds(10), &kv_store, locals[i], &globals[i],
            /*assign_global_device_ids=*/true));
        // Simulate node 1 restarting with different devices.
        if (i == 1) {
          DeviceProto* d4 = locals[1].add_devices();
          d4->set_local_device_ordinal(2);
          // This should fail because the local topology is unexpectedly
          // different.
          EXPECT_THAT(ExchangeTopologies(
                          /*platform=*/"cuda", /*node_id=*/i, num_nodes,
                          /*get_local_topology_timeout=*/
                          absl::Seconds(10), /*get_global_topology_timeout=*/
                          absl::Seconds(10), &kv_store, locals[i], &globals[i],
                          /*assign_global_device_ids=*/true),
                      tsl::testing::StatusIs(absl::StatusCode::kInternal));
        }
      });
    }
  }
}

TEST(TopologyTest, BuildGlobalTopologyWithExplicitSliceIndices) {
  // Set slice_index explicitly, and expect boot id to be ignored.
  std::string boot_id = "foo";
  std::vector<LocalTopologyProto> locals(2);
  locals[0].set_boot_id(boot_id);
  locals[1].set_boot_id(boot_id);
  locals[0].set_node_id(0);
  locals[1].set_node_id(1);
  locals[0].set_slice_index(1);
  locals[1].set_slice_index(0);
  // Adds 2 devices to each host.
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(1);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  TF_ASSERT_OK_AND_ASSIGN(
      GlobalTopologyProto global,
      BuildGlobalTopology(absl::Span<LocalTopologyProto>(locals),
                          /*assign_global_device_ids=*/true));

  EXPECT_EQ(global.nodes_size(), 2);
  EXPECT_EQ(global.nodes()[0].devices_size(), 2);
  EXPECT_EQ(global.nodes()[0].devices()[0].slice_index(), 1);
  EXPECT_EQ(global.nodes()[0].devices()[1].slice_index(), 1);
  EXPECT_EQ(global.nodes()[1].devices_size(), 2);
  EXPECT_EQ(global.nodes()[1].devices()[0].slice_index(), 0);
  EXPECT_EQ(global.nodes()[1].devices()[1].slice_index(), 0);
}

TEST(TopologyTest, BuildGpuTopology) {
  std::string slice_0_boot_id = "foo";
  std::string slice_1_boot_id = "bar";
  std::vector<LocalTopologyProto> locals(2);
  // Adds 1 host to slice 0 and 1 host to slice 1.
  locals[0].set_boot_id(slice_0_boot_id);
  locals[1].set_boot_id(slice_1_boot_id);
  locals[0].set_node_id(0);
  locals[1].set_node_id(1);
  // Adds 2 devices to host 0 and 2 devices to host 1.
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  d0->set_core_count(20);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(1);
  d1->set_core_count(20);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  d2->set_core_count(20);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);
  d3->set_core_count(20);

  TF_ASSERT_OK_AND_ASSIGN(
      GlobalTopologyProto global,
      BuildGlobalTopology(absl::Span<LocalTopologyProto>(locals),
                          /*assign_global_device_ids=*/true));

  TF_ASSERT_OK_AND_ASSIGN(auto gpu_topology, BuildGpuTopology(global));
  EXPECT_EQ(gpu_topology.device_ids_size(), 4);
  EXPECT_EQ(gpu_topology.num_slices(), 2);
  EXPECT_EQ(gpu_topology.num_hosts_per_slice(), 1);
  EXPECT_EQ(gpu_topology.num_devices_per_host(), 2);
}

TEST(TopologyTest, BuildGpuTopologyWithDifferentNumHostsPerSlice) {
  std::string slice_0_boot_id = "foo";
  std::string slice_1_boot_id = "bar";
  std::vector<LocalTopologyProto> locals(3);
  // Adds 2 hosts to slice 0 and 1 host to slice 1.
  locals[0].set_boot_id(slice_0_boot_id);
  locals[1].set_boot_id(slice_0_boot_id);
  locals[2].set_boot_id(slice_1_boot_id);
  locals[0].set_node_id(0);
  locals[1].set_node_id(1);
  locals[2].set_node_id(2);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[1].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[2].add_devices();
  d2->set_local_device_ordinal(0);

  TF_ASSERT_OK_AND_ASSIGN(
      GlobalTopologyProto global,
      BuildGlobalTopology(absl::Span<LocalTopologyProto>(locals),
                          /*assign_global_device_ids=*/true));

  TF_ASSERT_OK_AND_ASSIGN(auto gpu_topology, BuildGpuTopology(global));
  EXPECT_EQ(gpu_topology.device_ids_size(), 3);
  EXPECT_EQ(gpu_topology.num_slices(), -1);
  EXPECT_EQ(gpu_topology.num_hosts_per_slice(), -1);
  EXPECT_EQ(gpu_topology.num_devices_per_host(), -1);
}

TEST(TopologyTest, BuildGpuTopologyWithDifferentNumDevicesPerHost) {
  std::string slice_0_boot_id = "foo";
  std::string slice_1_boot_id = "bar";
  std::vector<LocalTopologyProto> locals(2);
  locals[0].set_boot_id(slice_0_boot_id);
  locals[1].set_boot_id(slice_1_boot_id);
  locals[0].set_node_id(0);
  locals[1].set_node_id(1);
  // Adds 2 devices to host 0 and 1 device to host 1.
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(1);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);

  TF_ASSERT_OK_AND_ASSIGN(
      GlobalTopologyProto global,
      BuildGlobalTopology(absl::Span<LocalTopologyProto>(locals),
                          /*assign_global_device_ids=*/true));

  TF_ASSERT_OK_AND_ASSIGN(auto gpu_topology, BuildGpuTopology(global));
  EXPECT_EQ(gpu_topology.device_ids_size(), 3);
  EXPECT_EQ(gpu_topology.num_slices(), -1);
  EXPECT_EQ(gpu_topology.num_hosts_per_slice(), -1);
  EXPECT_EQ(gpu_topology.num_devices_per_host(), -1);
}
}  // namespace
}  // namespace xla
