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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_XGMI_TOPOLOGY_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_XGMI_TOPOLOGY_H_

#include <cstdint>

#include "absl/strings/string_view.h"

namespace stream_executor::gpu {

struct XgmiTopologyInfo {
  int active_links = 0;  // Number of active xGMI P2P links to other GPUs
  uint64_t hive_id = 0;  // xGMI hive ID
};

// Queries xGMI topology information for a ROCm device.
// pci_bus_id is the PCI bus ID string from HIP (e.g. "0000:41:00.0").
// Returns topology info. On failure, returns default (0 links, no hive).
XgmiTopologyInfo GetRocmXgmiTopology(absl::string_view pci_bus_id);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_XGMI_TOPOLOGY_H_
