/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/container/flat_hash_map.h"
#include "tsl/platform/logging.h"

namespace xla {

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
    absl::string_view boot_id = local.boot_id();
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

}  // namespace xla
