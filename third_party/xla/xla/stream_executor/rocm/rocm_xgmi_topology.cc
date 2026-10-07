/* Copyright 2026 The OpenXLA Authors.
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

#include "xla/stream_executor/rocm/rocm_xgmi_topology.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/rocm/smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {

absl::StatusOr<XgmiTopologyInfo> GetRocmXgmiTopology(
    absl::string_view pci_bus_id) {
  absl::MutexLock lock(smi_mutex);

  ABSL_RETURN_IF_ERROR(InitSmi());

  ABSL_ASSIGN_OR_RETURN(BdfComponents bdf, ParseBdf(pci_bus_id));
  ABSL_ASSIGN_OR_RETURN(SmiDeviceHandle device, FindDevice(bdf));

  XgmiTopologyInfo info;

  // A device outside an xGMI hive is a normal configuration, and SMI reports
  // it the same way it reports a failed query, so this is not fatal.
  absl::StatusOr<uint64_t> hive_id = QueryHiveId(device);
  if (hive_id.ok()) {
    info.hive_id = *hive_id;
  } else {
    VLOG(2) << "xGMI hive ID query failed for " << pci_bus_id << ": "
            << hive_id.status().message()
            << "; device may not be in an xGMI hive.";
  }

  // Count peers reachable over xGMI by querying the link type to every other
  // device. This counts peer GPUs, not physical links.
  ABSL_ASSIGN_OR_RETURN(std::vector<SmiDeviceHandle> devices, EnumerateDevices());
  if (devices.size() <= 1) return info;

  int xgmi_links = 0;
  for (SmiDeviceHandle peer : devices) {
    if (peer == device) continue;
    absl::StatusOr<bool> is_peer = IsXgmiPeer(device, peer);
    if (!is_peer.ok()) {
      VLOG(2) << "xGMI link type query failed for " << pci_bus_id << ": "
              << is_peer.status().message();
      continue;
    }
    if (*is_peer) ++xgmi_links;
  }

  info.active_links = xgmi_links;

  VLOG(1) << "xGMI topology for " << pci_bus_id << ": " << xgmi_links
          << " active xGMI links"
          << " (hive_id=" << info.hive_id << ", num_devices=" << devices.size()
          << ")";

  return info;
}

}  // namespace stream_executor::gpu
