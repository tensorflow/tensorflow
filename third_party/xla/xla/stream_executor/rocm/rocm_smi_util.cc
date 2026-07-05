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

#include "xla/stream_executor/rocm/rocm_smi_util.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "rocm/include/rocm_smi/rocm_smi.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {

bool InitRocmSmi() {
  static bool initialized = []() {
    rsmi_status_t status = rsmi_init(0);
    if (status != RSMI_STATUS_SUCCESS) {
      const char* err_str = nullptr;
      rsmi_status_string(status, &err_str);
      LOG(WARNING) << "rsmi_init failed: "
                   << (err_str ? err_str : "unknown error");
      return false;
    }
    return true;
  }();
  return initialized;
}

// Parses a PCI bus/device/function ID string into its numeric components.
// Accepts two formats:
//   DDDD:BB:DD.F - domain:bus:device.function (e.g. "0000:41:00.0")
//   BB:DD.F      - bus:device.function, domain defaults to 0
// All fields are hex
std::optional<BdfComponents> ParseBdf(absl::string_view pci_bus_id) {
  BdfComponents bdf = {};

  // Determine which format we have by counting colons.
  // Two colons -> DDDD:BB:DD.F, one colon -> BB:DD.F.
  size_t first_colon = pci_bus_id.find(':');
  if (first_colon == absl::string_view::npos) return std::nullopt;

  size_t second_colon = pci_bus_id.find(':', first_colon + 1);
  size_t dot;

  if (second_colon != absl::string_view::npos) {
    // DDDD:BB:DD.F format
    dot = pci_bus_id.find('.', second_colon + 1);
    if (dot == absl::string_view::npos) return std::nullopt;

    if (!absl::SimpleHexAtoi(pci_bus_id.substr(0, first_colon), &bdf.domain))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(first_colon + 1, second_colon - first_colon - 1),
            &bdf.bus))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(second_colon + 1, dot - second_colon - 1),
            &bdf.device))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(dot + 1), &bdf.function))
      return std::nullopt;
  } else {
    // BB:DD.F format (domain = 0)
    dot = pci_bus_id.find('.', first_colon + 1);
    if (dot == absl::string_view::npos) return std::nullopt;

    bdf.domain = 0;
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(0, first_colon), &bdf.bus))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(first_colon + 1, dot - first_colon - 1),
            &bdf.device))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(dot + 1), &bdf.function))
      return std::nullopt;
  }

  return bdf;
}

std::optional<uint32_t> FindDeviceIndex(const BdfComponents& target_bdf) {
  uint32_t num_devices = 0;
  rsmi_status_t status = rsmi_num_monitor_devices(&num_devices);
  if (status != RSMI_STATUS_SUCCESS || num_devices == 0) {
    return std::nullopt;
  }

  for (uint32_t i = 0; i < num_devices; ++i) {
    uint64_t bdfid = 0;
    status = rsmi_dev_pci_id_get(i, &bdfid);
    if (status != RSMI_STATUS_SUCCESS) continue;

    // Unpack rocm_smi's 64-bit BDF format into individual fields.
    // See
    // rocm-systems/projects/rocm-smi-lib/src/rocm_smi.cc:rsmi_dev_pci_id_get
    // for details on the packing.
    uint32_t domain = (bdfid >> 32) & 0xFFFFFFFF;
    uint8_t bus = (bdfid >> 8) & 0xFF;
    uint8_t device = (bdfid >> 3) & 0x1F;
    uint8_t function = bdfid & 0x7;

    if (domain == target_bdf.domain && bus == target_bdf.bus &&
        device == target_bdf.device && function == target_bdf.function) {
      return i;
    }
  }

  return std::nullopt;
}

}  // namespace stream_executor::gpu
