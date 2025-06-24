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

#include "xla/pjrt/distributed/topology_util.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/utils.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace xla {

namespace {
bool SameDevice(const DeviceProto& a, const DeviceProto& b) {
  return (a.name() == b.name() && a.vendor() == b.vendor() &&
          a.local_device_ordinal() == b.local_device_ordinal() &&
          a.core_count() == b.core_count() &&
          a.device_kind() == b.device_kind() &&
          a.slice_index() == b.slice_index() &&
          // Global device ID Might not be set for LocalTopologyProto, still
          // check it for default value.
          a.global_device_id() == b.global_device_id() &&
          a.compute_capability() == b.compute_capability());
}

bool SameLocalTopology(const LocalTopologyProto& a,
                       const LocalTopologyProto& b) {
  if (a.node_id() != b.node_id() || a.devices_size() != b.devices_size()) {
    return false;
  }
  for (int i = 0; i < a.devices_size(); ++i) {
    if (!SameDevice(a.devices(i), b.devices(i))) {
      return false;
    }
  }
  return true;
}

// Returns true if all devices have a valid fabric_uuid.
bool HasFabricUuid(absl::Span<LocalTopologyProto> local_topologies) {
  for (const LocalTopologyProto& local : local_topologies) {
    for (const DeviceProto& device : local.devices()) {
      if (device.fabric_uuid().empty() ||
          device.fabric_uuid() == "00000000-0000-0000-0000-000000000000/0") {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

// Exists on Linux systems. Unique per OS kernel restart.
static constexpr char kBootIdPath[] = "/proc/sys/kernel/random/boot_id";

// Retrieve content of /proc/sys/kernel/random/boot_id as a string.
// Note that procfs file may have file size 0 which throws off generic file
// readers such as tsl::ReadFileToString.
absl::StatusOr<std::string> GetBootIdString() {
  std::string boot_id_str;
#ifdef __linux__
  std::ifstream file(kBootIdPath);
  if (!file) {
    return NotFound("%s not found.", kBootIdPath);
  }
  std::string line;
  while (std::getline(file, line)) {
    absl::StripAsciiWhitespace(&line);
    absl::StrAppend(&boot_id_str, line);
  }
#endif
  return boot_id_str;
}

static std::string GetLocalTopologyKey(absl::string_view platform,
                                       int node_id) {
  return absl::StrCat("local_topology/", platform, "/", node_id);
}

static std::string GetGlobalTopologyKey(absl::string_view platform) {
  return absl::StrCat("global_topology/", platform);
}

static absl::StatusOr<std::vector<LocalTopologyProto>> GetAllLocalTopologies(
    absl::string_view platform, int num_nodes, KeyValueStoreInterface* kv_store,
    absl::Duration timeout) {
  std::vector<absl::StatusOr<std::string>> local_topology_strs(num_nodes);

  // TODO(ezhulenev): Should a thread pool become a function argument?
  tsl::thread::ThreadPool thread_pool(
      tsl::Env::Default(), "GetAllLocalTopologies", DefaultThreadPoolSize());

  absl::BlockingCounter blocking_counter(num_nodes);
  absl::Mutex mu;
  for (int i = 0; i < num_nodes; i++) {
    thread_pool.Schedule([&, i] {
      absl::StatusOr<std::string> local_topology_str =
          kv_store->Get(GetLocalTopologyKey(platform, i), timeout);
      {
        absl::MutexLock lock(&mu);
        local_topology_strs[i] = local_topology_str;
      }
      blocking_counter.DecrementCount();
    });
  }
  blocking_counter.Wait();

  std::vector<std::string> error_messages;
  std::vector<LocalTopologyProto> local_topologies;
  int max_num_failed_message = 10;
  int failed_count = 0;
  for (const absl::StatusOr<std::string>& str : local_topology_strs) {
    if (str.ok()) {
      LocalTopologyProto local;
      local.ParseFromString(*str);
      local_topologies.push_back(local);
    } else {
      error_messages.push_back(
          absl::StrCat("Error ", ++failed_count, ": ", str.status().message()));
      if (failed_count > max_num_failed_message) {
        break;
      }
    }
  }
  if (error_messages.empty()) {
    return local_topologies;
  }
  return absl::InternalError(
      absl::StrCat("Getting local topologies failed: ",
                   absl::StrJoin(error_messages, "\n\n")));
}

// Steals the contents of `local_topologies`.
absl::StatusOr<GlobalTopologyProto> BuildGlobalTopology(
    absl::Span<LocalTopologyProto> local_topologies,
    bool assign_global_device_ids) {
  CHECK(!local_topologies.empty());
  bool explicit_slice_indices = local_topologies[0].has_slice_index();
  if (explicit_slice_indices) {
    // Every local topology explicitly declares its slice_index.
    for (LocalTopologyProto& local : local_topologies) {
      if (!local.has_slice_index()) {
        return InvalidArgument(
            "Either all of or none of the local topologies "
            "should explicitly set slice_index");
      }
      int slice_index = local.slice_index();
      for (DeviceProto& device : *local.mutable_devices()) {
        device.set_slice_index(slice_index);
      }
    }
  } else {
    // Assign local devices of the same fabric_uuid/boot_id to the same
    // slice_index.
    const bool has_fabric_uuid = HasFabricUuid(local_topologies);
    absl::flat_hash_map<std::string, int> id_to_slice_index;
    for (LocalTopologyProto& local : local_topologies) {
      if (local.has_slice_index()) {
        return InvalidArgument(
            "Either all of or none of the local topologies "
            "should explicitly set slice_index");
      }
      for (DeviceProto& device : *local.mutable_devices()) {
        // Each new fabric_uuid/boot_id seen is treated as a new slice.
        auto [it, _] = id_to_slice_index.try_emplace(
            has_fabric_uuid ? device.fabric_uuid() : local.boot_id(),
            id_to_slice_index.size());
        device.set_slice_index(it->second);
      }
    }
    if (VLOG_IS_ON(10)) {
      for (auto it = id_to_slice_index.begin(); it != id_to_slice_index.end();
           ++it) {
        LOG(INFO) << "BuildGlobalTopology id_to_slice_index " << it->first
                  << "->" << it->second;
      }
    }
  }

  GlobalTopologyProto global_topology;
  int next_global_device_id = 0;
  for (LocalTopologyProto& local : local_topologies) {
    if (assign_global_device_ids) {
      for (DeviceProto& device : *local.mutable_devices()) {
        device.set_global_device_id(next_global_device_id++);
      }
    }
    global_topology.add_nodes()->Swap(&local);
  }
  return global_topology;
}

absl::Status ExchangeTopologies(absl::string_view platform, int node_id,
                                int num_nodes,
                                absl::Duration get_local_topology_timeout,
                                absl::Duration get_global_topology_timeout,
                                KeyValueStoreInterface* kv_store,
                                const LocalTopologyProto& local_topology,
                                GlobalTopologyProto* global_topology,
                                bool assign_global_device_ids) {
  VLOG(3) << "Local Topology for platform" << platform << ":\n"
          << local_topology.DebugString();
  if (num_nodes == 1) {
    LocalTopologyProto* topology = global_topology->add_nodes();
    *topology = local_topology;
    for (DeviceProto& device : *topology->mutable_devices()) {
      device.set_global_device_id(device.local_device_ordinal());
    }
    return absl::OkStatus();
  }
  CHECK(kv_store != nullptr);
  const std::string local_topology_key = GetLocalTopologyKey(platform, node_id);
  const std::string serialized_local_topology =
      local_topology.SerializeAsString();

  auto status = kv_store->Set(GetLocalTopologyKey(platform, node_id),
                              serialized_local_topology);
  if (absl::IsAlreadyExists(status)) {
    // Local topology has been set previously from the same node before
    // restart.
    absl::StatusOr<std::string> existing_local_topology =
        kv_store->TryGet(local_topology_key);
    LocalTopologyProto existing_local_topology_proto;
    existing_local_topology_proto.ParseFromString(*existing_local_topology);
    if (!SameLocalTopology(existing_local_topology_proto, local_topology)) {
      return absl::InternalError(absl::Substitute(
          "Different local topology for node $0 has been set previously, "
          "possibly before a restart.\nBefore: $1\nAfter: $2",
          node_id, existing_local_topology_proto.DebugString(),
          local_topology.DebugString()));
    }
  } else if (!status.ok()) {
    return status;
  }

  // The lead node gets all local topologies, builds the global topology and
  // puts it to the key-value store.
  std::string global_topology_key = GetGlobalTopologyKey(platform);
  if (node_id == 0) {
    TF_ASSIGN_OR_RETURN(std::vector<LocalTopologyProto> local_topologies,
                        GetAllLocalTopologies(platform, num_nodes, kv_store,
                                              get_local_topology_timeout));
    TF_ASSIGN_OR_RETURN(
        *global_topology,
        BuildGlobalTopology(absl::Span<LocalTopologyProto>(local_topologies),
                            assign_global_device_ids));
    TF_RETURN_IF_ERROR(kv_store->Set(global_topology_key,
                                     global_topology->SerializeAsString()));
  } else {
    TF_ASSIGN_OR_RETURN(
        std::string global_topology_str,
        kv_store->Get(global_topology_key, get_global_topology_timeout));
    global_topology->ParseFromString(global_topology_str);
  }
  VLOG(3) << "Global topology for platform " << platform << ":\n"
          << global_topology->DebugString();
  return absl::OkStatus();
}

bool IsGpuTopologySymmetric(
    const std::map<int, std::set<int>>& slice_id_to_node_ids,
    const std::map<int, int>& node_id_to_device_count) {
  CHECK(!slice_id_to_node_ids.empty());
  CHECK(!node_id_to_device_count.empty());

  int num_hosts_per_slice = slice_id_to_node_ids.begin()->second.size();
  int num_devices_per_host = node_id_to_device_count.begin()->second;
  for (const auto& [slice_id, node_ids] : slice_id_to_node_ids) {
    if (node_ids.size() != num_hosts_per_slice) {
      LOG(INFO) << "GpuTopology is asymmetric as it has different number "
                   "of hosts per slice.";
      return false;
    }
  }
  for (const auto& [node_id, device_count] : node_id_to_device_count) {
    if (device_count != num_devices_per_host) {
      LOG(INFO) << "GpuTopology is asymmetric as it has different number "
                   "of devices per host.";
      return false;
    }
  }
  return true;
}

absl::StatusOr<GpuTopologyProto> BuildGpuTopology(
    const GlobalTopologyProto& global_topology) {
  GpuTopologyProto gpu_topology;
  std::map<int, std::set<int>> slice_id_to_node_ids;
  std::map<int, int> node_id_to_device_count;
  std::vector<int> device_ids;
  for (int i = 0; i < global_topology.nodes_size(); ++i) {
    const LocalTopologyProto& local_topology = global_topology.nodes(i);

    node_id_to_device_count[local_topology.node_id()] =
        local_topology.devices_size();
    for (const DeviceProto& device : local_topology.devices()) {
      if (gpu_topology.platform_version().empty()) {
        gpu_topology.set_platform_version(device.name());
      }
      slice_id_to_node_ids[device.slice_index()].insert(
          local_topology.node_id());
      device_ids.push_back(device.global_device_id());
    }
  }

  if (IsGpuTopologySymmetric(slice_id_to_node_ids, node_id_to_device_count)) {
    gpu_topology.set_num_slices(slice_id_to_node_ids.size());
    gpu_topology.set_num_hosts_per_slice(
        slice_id_to_node_ids.begin()->second.size());
    gpu_topology.set_num_devices_per_host(
        node_id_to_device_count.begin()->second);
  } else {
    // If gpu topology is not symmetric, then we don't need to populate
    // the topology with the slice/host/device information.
    gpu_topology.set_num_slices(-1);
    gpu_topology.set_num_hosts_per_slice(-1);
    gpu_topology.set_num_devices_per_host(-1);
  }
  std::sort(device_ids.begin(), device_ids.end());
  gpu_topology.mutable_device_ids()->Add(device_ids.begin(), device_ids.end());
  return gpu_topology;
}

}  // namespace xla
