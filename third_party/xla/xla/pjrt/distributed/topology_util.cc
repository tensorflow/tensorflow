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

#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/utils.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace xla {

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

static std::string GetLocalTopologyKey(std::string_view platform, int node_id) {
  return absl::StrCat("local_topology/", platform, "/", node_id);
}

static std::string GetGlobalTopologyKey(std::string_view platform) {
  return absl::StrCat("global_topology/", platform);
}

static absl::StatusOr<std::vector<LocalTopologyProto>> GetAllLocalTopologies(
    std::string_view platform, int num_nodes, KeyValueStoreInterface* kv_store,
    absl::Duration timeout) {
  std::vector<StatusOr<std::string>> local_topology_strs(num_nodes);

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
GlobalTopologyProto BuildGlobalTopology(
    absl::Span<LocalTopologyProto> local_topologies) {
  GlobalTopologyProto global_topology;
  int next_global_device_id = 0;
  // Assign local devices of the same host to the same slice_index.
  int next_slice_index = 0;
  absl::flat_hash_map<std::string, int> boot_id_to_slice_index;
  for (LocalTopologyProto& local : local_topologies) {
    // Every new boot_id seen is treated as a new host/slice.
    std::string_view boot_id = local.boot_id();
    auto [it, inserted] =
        boot_id_to_slice_index.try_emplace(boot_id, next_slice_index);
    if (inserted) {
      ++next_slice_index;
    }
    for (DeviceProto& device : *local.mutable_devices()) {
      device.set_global_device_id(next_global_device_id++);
      device.set_slice_index(it->second);
    }
    global_topology.add_nodes()->Swap(&local);
  }
  if (VLOG_IS_ON(10)) {
    for (auto it = boot_id_to_slice_index.begin();
         it != boot_id_to_slice_index.end(); ++it) {
      LOG(INFO) << "BuildGlobalTopology boot_id_to_slice_index " << it->first
                << "->" << it->second;
    }
  }
  return global_topology;
}

Status ExchangeTopologies(std::string_view platform, int node_id, int num_nodes,
                          absl::Duration get_local_topology_timeout,
                          absl::Duration get_global_topology_timeout,
                          KeyValueStoreInterface* kv_store,
                          const LocalTopologyProto& local_topology,
                          GlobalTopologyProto* global_topology) {
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
  TF_RETURN_IF_ERROR(kv_store->Set(GetLocalTopologyKey(platform, node_id),
                                   local_topology.SerializeAsString()));

  // The lead node gets all local topologies, builds the global topology and
  // puts it to the key-value store.
  std::string global_topology_key = GetGlobalTopologyKey(platform);
  if (node_id == 0) {
    TF_ASSIGN_OR_RETURN(std::vector<LocalTopologyProto> local_topologies,
                        GetAllLocalTopologies(platform, num_nodes, kv_store,
                                              get_local_topology_timeout));
    *global_topology =
        BuildGlobalTopology(absl::Span<LocalTopologyProto>(local_topologies));
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

}  // namespace xla
