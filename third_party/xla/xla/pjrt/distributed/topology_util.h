/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_DISTRIBUTED_TOPOLOGY_UTIL_H_
#define XLA_PJRT_DISTRIBUTED_TOPOLOGY_UTIL_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"

namespace xla {

// Retrieve content of /proc/sys/kernel/random/boot_id as a string.
// Empty on non-Linux platforms.
absl::StatusOr<std::string> GetBootIdString();

// Performs a distributed exchange of topologies using a KV store. Each process
// provides its local topology, and the local topologies are exchanged to
// form a global topology.
// If assign_global_device_ids is true, assigns global IDs to each node in the
// topology in the order they appear in the input. Otherwise leaves the global
// IDs as they were in the local topologies..
// TODO(phawkins): deprecate and remove assign_global_device_ids.
absl::Status ExchangeTopologies(absl::string_view platform, int node_id,
                                int num_nodes,
                                absl::Duration get_local_topology_timeout,
                                absl::Duration get_global_topology_timeout,
                                KeyValueStoreInterface* kv_store,
                                const LocalTopologyProto& local_topology,
                                GlobalTopologyProto* global_topology,
                                bool assign_global_device_ids);

// Functions below this point are public only for testing.

// Given a LocalTopologyProto object from each node, builds a
// GlobalTopologyProto that describes all nodes. Steals the contents of the
// LocalTopologyProtos.
GlobalTopologyProto BuildGlobalTopology(
    absl::Span<LocalTopologyProto> local_topologies,
    bool assign_global_device_ids);

// Builds a GpuTopologyProto representing the GPU configuration described in the
// given GlobalTopologyProto.
absl::StatusOr<GpuTopologyProto> BuildGpuTopology(
    const GlobalTopologyProto& global_topology);
}  // namespace xla

#endif  // XLA_PJRT_DISTRIBUTED_TOPOLOGY_UTIL_H_
