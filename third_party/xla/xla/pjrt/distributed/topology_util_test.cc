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
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

TEST(TopologyTest, GetNetworkNodes) {
  tsl::setenv("XLA_DISTRIBUTED_TOPOLOGY_NETWORK_NODES", "spine,pod,rack", true);
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> nodes, GetNetworkNodes());

  ASSERT_EQ(nodes.size(), 3);
  EXPECT_EQ(nodes[0], "spine");
  EXPECT_EQ(nodes[1], "pod");
  EXPECT_EQ(nodes[2], "rack");
}

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
  EXPECT_EQ(global.processes_size(), 2);
  EXPECT_EQ(global.processes()[0].devices_size(), 2);
  EXPECT_EQ(global.processes()[1].devices_size(), 2);
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
  EXPECT_EQ(global.processes_size(), 2);
  EXPECT_EQ(global.processes()[0].devices_size(), 2);
  EXPECT_EQ(global.processes()[1].devices_size(), 2);
  EXPECT_EQ(global.processes()[0].devices()[0].partition_index(), 0);
  EXPECT_EQ(global.processes()[0].devices()[1].partition_index(), 0);
  EXPECT_EQ(global.processes()[1].devices()[0].partition_index(), 0);
  EXPECT_EQ(global.processes()[1].devices()[1].partition_index(), 0);
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
  EXPECT_EQ(global.processes_size(), 4);
  EXPECT_EQ(global.processes()[0].devices_size(), 2);
  EXPECT_EQ(global.processes()[1].devices_size(), 2);
  EXPECT_EQ(global.processes()[2].devices_size(), 2);
  EXPECT_EQ(global.processes()[3].devices_size(), 2);
  EXPECT_EQ(global.processes()[0].devices()[0].partition_index(), 0);
  EXPECT_EQ(global.processes()[0].devices()[1].partition_index(), 0);
  EXPECT_EQ(global.processes()[1].devices()[0].partition_index(), 0);
  EXPECT_EQ(global.processes()[1].devices()[1].partition_index(), 0);
  EXPECT_EQ(global.processes()[2].devices()[0].partition_index(), 1);
  EXPECT_EQ(global.processes()[2].devices()[1].partition_index(), 1);
  EXPECT_EQ(global.processes()[3].devices()[0].partition_index(), 1);
  EXPECT_EQ(global.processes()[3].devices()[1].partition_index(), 1);
}

// Verifies that BuildGlobalTopology sorts hosts by network_nodes so that
// topologically-close hosts (sharing rack/pod switches) receive consecutive
// global device IDs. This minimizes spine switch traffic for collective
// operations where adjacent-ID devices communicate (e.g. ring all-reduce).
TEST(TopologyTest, BuildGlobalTopologyWithNetworkNodes) {
  // Network topology:
  //   spine1
  //   ├── pod1
  //   │   ├── rack1 → host_a, host_b
  //   │   └── rack2 → host_c
  //   └── pod2
  //       └── rack3 → host_d
  //
  // Input order is intentionally NOT sorted by network proximity.
  std::vector<LocalTopologyProto> locals(4);

  // host_d: pod2/rack3 (different pod from the others).
  locals[0].set_boot_id("host_d");
  locals[0].add_network_nodes("spine1");
  locals[0].add_network_nodes("pod2");
  locals[0].add_network_nodes("rack3");
  locals[0].add_devices()->set_local_device_ordinal(0);

  // host_b: pod1/rack1.
  locals[1].set_boot_id("host_b");
  locals[1].add_network_nodes("spine1");
  locals[1].add_network_nodes("pod1");
  locals[1].add_network_nodes("rack1");
  locals[1].add_devices()->set_local_device_ordinal(0);

  // host_c: pod1/rack2.
  locals[2].set_boot_id("host_c");
  locals[2].add_network_nodes("spine1");
  locals[2].add_network_nodes("pod1");
  locals[2].add_network_nodes("rack2");
  locals[2].add_devices()->set_local_device_ordinal(0);

  // host_a: pod1/rack1 (same rack as host_b).
  locals[3].set_boot_id("host_a");
  locals[3].add_network_nodes("spine1");
  locals[3].add_network_nodes("pod1");
  locals[3].add_network_nodes("rack1");
  locals[3].add_devices()->set_local_device_ordinal(0);

  TF_ASSERT_OK_AND_ASSIGN(
      GlobalTopologyProto global,
      BuildGlobalTopology(absl::Span<LocalTopologyProto>(locals),
                          /*assign_global_device_ids=*/true));

  ASSERT_EQ(global.processes_size(), 4);

  // Find each host's global device ID via its boot_id.
  auto find_global_id = [&](const std::string& boot_id) -> int {
    for (const auto& process : global.processes()) {
      if (process.boot_id() == boot_id) {
        return process.devices(0).global_device_id();
      }
    }
    return -1;
  };

  int host_a_id = find_global_id("host_a");
  int host_b_id = find_global_id("host_b");
  int host_c_id = find_global_id("host_c");
  int host_d_id = find_global_id("host_d");

  // Same-rack hosts (host_a, host_b under pod1/rack1) must have consecutive
  // global device IDs — their traffic stays within the rack switch.
  EXPECT_EQ(host_b_id + 1, host_a_id)
      << "Same-rack hosts should have consecutive IDs";

  // All pod1 hosts should have lower IDs than the pod2 host, so that
  // collective traffic between adjacent IDs stays within pod1's aggregation
  // switch and never traverses the spine.
  EXPECT_LT(host_a_id, host_d_id);
  EXPECT_LT(host_b_id, host_d_id);
  EXPECT_LT(host_c_id, host_d_id);

  // Verify the exact assignment:
  //   Lexicographic sort: host_b (pod1/rack1), host_a (pod1/rack1),
  //                       host_c (pod1/rack2), host_d (pod2/rack3)
  //   Partition indices assigned in encounter order: 0, 1, 2, 3
  //   Global device IDs follow partition order: 0, 1, 2, 3
  EXPECT_EQ(host_b_id, 0);
  EXPECT_EQ(host_a_id, 1);
  EXPECT_EQ(host_c_id, 2);
  EXPECT_EQ(host_d_id, 3);
}

TEST(TopologyTest, ExchangeTopology) {
  int num_processes = 2;
  std::vector<LocalTopologyProto> locals(num_processes);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  InMemoryKeyValueStore kv_store;
  std::vector<GlobalTopologyProto> globals(num_processes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "TestPool",
                                        num_processes);
    for (int i = 0; i < num_processes; i++) {
      thread_pool.Schedule([&, i] {
        TF_ASSERT_OK(ExchangeTopologies(
            /*platform=*/"cuda", /*node_id=*/i, num_processes,
            /*get_local_topology_timeout=*/
            absl::Seconds(10), /*get_global_topology_timeout=*/
            absl::Seconds(10), &kv_store, locals[i], &globals[i],
            /*assign_global_device_ids=*/true));
      });
    }
  }
  for (const GlobalTopologyProto& global : globals) {
    EXPECT_EQ(global.processes_size(), 2);
    EXPECT_EQ(global.processes()[0].devices_size(), 2);
    EXPECT_EQ(global.processes()[1].devices_size(), 2);
  }
}

TEST(TopologyTest, ExchangeTopology_Twice_Succeeds) {
  int num_processes = 2;
  std::vector<LocalTopologyProto> locals(num_processes);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  InMemoryKeyValueStore kv_store(/*allow_overwrite=*/false);
  std::vector<GlobalTopologyProto> globals(num_processes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "TestPool",
                                        num_processes);
    for (int i = 0; i < num_processes; i++) {
      thread_pool.Schedule([&, i] {
        TF_ASSERT_OK(ExchangeTopologies(
            /*platform=*/"cuda", /*node_id=*/i, num_processes,
            /*get_local_topology_timeout=*/
            absl::Seconds(10), /*get_global_topology_timeout=*/
            absl::Seconds(10), &kv_store, locals[i], &globals[i],
            /*assign_global_device_ids=*/true));
        // Simulate node 1 restarting and exchanging topologies again.
        if (i == 1) {
          TF_ASSERT_OK(ExchangeTopologies(
              /*platform=*/"cuda", /*node_id=*/i, num_processes,
              /*get_local_topology_timeout=*/
              absl::Seconds(10), /*get_global_topology_timeout=*/
              absl::Seconds(10), &kv_store, locals[i], &globals[i],
              /*assign_global_device_ids=*/true));
        }
      });
    }
  }
  for (const GlobalTopologyProto& global : globals) {
    EXPECT_EQ(global.processes_size(), 2);
    EXPECT_EQ(global.processes()[0].devices_size(), 2);
    EXPECT_EQ(global.processes()[1].devices_size(), 2);
  }
}

TEST(TopologyTest, ExchangeTopology_TwiceWithDifferentLocalTopology_Fails) {
  int num_processes = 2;
  std::vector<LocalTopologyProto> locals(num_processes);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  InMemoryKeyValueStore kv_store(/*allow_overwrite=*/false);
  std::vector<GlobalTopologyProto> globals(num_processes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "TestPool",
                                        num_processes);
    for (int i = 0; i < num_processes; i++) {
      thread_pool.Schedule([&, i] {
        TF_ASSERT_OK(ExchangeTopologies(
            /*platform=*/"cuda", /*node_id=*/i, num_processes,
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
                          /*platform=*/"cuda", /*node_id=*/i, num_processes,
                          /*get_local_topology_timeout=*/
                          absl::Seconds(10), /*get_global_topology_timeout=*/
                          absl::Seconds(10), &kv_store, locals[i], &globals[i],
                          /*assign_global_device_ids=*/true),
                      absl_testing::StatusIs(absl::StatusCode::kInternal));
        }
      });
    }
  }
}

TEST(TopologyTest, BuildGlobalTopologyWithExplicitSliceIndices) {
  // Set partition_index explicitly, and expect boot id to be ignored.
  std::string boot_id = "foo";
  std::vector<LocalTopologyProto> locals(2);
  locals[0].set_boot_id(boot_id);
  locals[1].set_boot_id(boot_id);
  locals[0].set_process_id(0);
  locals[1].set_process_id(1);
  locals[0].set_partition_index(1);
  locals[1].set_partition_index(0);
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

  EXPECT_EQ(global.processes_size(), 2);
  EXPECT_EQ(global.processes()[0].devices_size(), 2);
  EXPECT_EQ(global.processes()[0].devices()[0].partition_index(), 1);
  EXPECT_EQ(global.processes()[0].devices()[1].partition_index(), 1);
  EXPECT_EQ(global.processes()[1].devices_size(), 2);
  EXPECT_EQ(global.processes()[1].devices()[0].partition_index(), 0);
  EXPECT_EQ(global.processes()[1].devices()[1].partition_index(), 0);
}

TEST(TopologyTest, BuildGpuTopology) {
  std::string partition_0_boot_id = "foo";
  std::string partition_1_boot_id = "bar";
  std::vector<LocalTopologyProto> locals(2);
  // Adds 1 host to partition 0 and 1 host to partition 1.
  locals[0].set_boot_id(partition_0_boot_id);
  locals[1].set_boot_id(partition_1_boot_id);
  locals[0].set_process_id(0);
  locals[1].set_process_id(1);
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
  EXPECT_EQ(gpu_topology.num_partitions(), 2);
  EXPECT_EQ(gpu_topology.num_hosts_per_partition(), 1);
  EXPECT_EQ(gpu_topology.num_devices_per_host(), 2);
}

TEST(TopologyTest, BuildGpuTopologyWithDifferentNumHostsPerSlice) {
  std::string partition_0_boot_id = "foo";
  std::string partition_1_boot_id = "bar";
  std::vector<LocalTopologyProto> locals(3);
  // Adds 2 hosts to partition 0 and 1 host to partition 1.
  locals[0].set_boot_id(partition_0_boot_id);
  locals[1].set_boot_id(partition_0_boot_id);
  locals[2].set_boot_id(partition_1_boot_id);
  locals[0].set_process_id(0);
  locals[1].set_process_id(1);
  locals[2].set_process_id(2);
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
  EXPECT_EQ(gpu_topology.num_partitions(), -1);
  EXPECT_EQ(gpu_topology.num_hosts_per_partition(), -1);
  EXPECT_EQ(gpu_topology.num_devices_per_host(), -1);
}

TEST(TopologyTest, BuildGpuTopologyWithDifferentNumDevicesPerHost) {
  std::string partition_0_boot_id = "foo";
  std::string partition_1_boot_id = "bar";
  std::vector<LocalTopologyProto> locals(2);
  locals[0].set_boot_id(partition_0_boot_id);
  locals[1].set_boot_id(partition_1_boot_id);
  locals[0].set_process_id(0);
  locals[1].set_process_id(1);
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
  EXPECT_EQ(gpu_topology.num_partitions(), -1);
  EXPECT_EQ(gpu_topology.num_hosts_per_partition(), -1);
  EXPECT_EQ(gpu_topology.num_devices_per_host(), -1);
}
}  // namespace
}  // namespace xla
