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
#include <optional>

#include "absl/strings/string_view.h"
#include "rocm/include/rocm_smi/rocm_smi.h"
#include "xla/stream_executor/rocm/rocm_smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {

XgmiTopologyInfo GetRocmXgmiTopology(absl::string_view pci_bus_id) {
  XgmiTopologyInfo info;

  if (!InitRocmSmi()) return info;

  std::optional<BdfComponents> bdf = ParseBdf(pci_bus_id);
  if (!bdf.has_value()) {
    LOG(WARNING) << "Failed to parse PCI bus ID for xGMI query: " << pci_bus_id;
    return info;
  }

  std::optional<uint32_t> dev_idx = FindDeviceIndex(*bdf);
  if (!dev_idx.has_value()) {
    LOG(WARNING) << "rocm_smi: could not find device for PCI bus ID "
                 << pci_bus_id << " (xGMI query)";
    return info;
  }

  // Query xGMI hive ID.
  uint64_t hive_id = 0;
  rsmi_status_t status = rsmi_dev_xgmi_hive_id_get(*dev_idx, &hive_id);
  if (status == RSMI_STATUS_SUCCESS) {
    info.hive_id = hive_id;
  } else {
    VLOG(1) << "rsmi_dev_xgmi_hive_id_get failed for " << pci_bus_id
            << "; device may not be in an xGMI hive.";
  }

  // Count xGMI links by querying link type to every other device.
  uint32_t num_devices = 0;
  status = rsmi_num_monitor_devices(&num_devices);
  if (status != RSMI_STATUS_SUCCESS || num_devices <= 1) {
    return info;
  }

  int xgmi_links = 0;
  for (uint32_t i = 0; i < num_devices; ++i) {
    if (i == *dev_idx) continue;

    uint64_t hops = 0;
    RSMI_IO_LINK_TYPE link_type = RSMI_IOLINK_TYPE_UNDEFINED;
    status = rsmi_topo_get_link_type(*dev_idx, i, &hops, &link_type);
    if (status != RSMI_STATUS_SUCCESS) continue;

    if (link_type == RSMI_IOLINK_TYPE_XGMI) {
      ++xgmi_links;
    }
  }

  info.active_links = xgmi_links;

  VLOG(1) << "xGMI topology for " << pci_bus_id << ": " << xgmi_links
          << " active xGMI links"
          << " (hive_id=" << hive_id << ", num_devices=" << num_devices << ")";

  return info;
}

}  // namespace stream_executor::gpu
