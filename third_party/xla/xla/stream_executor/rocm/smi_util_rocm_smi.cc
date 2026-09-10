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

// rocm_smi backend for the SMI queries declared in smi_util.h. Compiled
// in below ROCm 7.13; smi_util_amd_smi.cc takes its place from 7.13 on.

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "rocm/include/rocm_smi/rocm_smi.h"
#include "xla/stream_executor/rocm/smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

uint32_t ToDeviceIndex(SmiDeviceHandle device) {
  return static_cast<uint32_t>(device.value);
}

SmiDeviceHandle ToDeviceHandle(uint32_t device_index) {
  return SmiDeviceHandle{device_index};
}

absl::Status SmiError(absl::string_view api, rsmi_status_t status) {
  const char* err_str = nullptr;
  rsmi_status_string(status, &err_str);
  return absl::InternalError(
      absl::StrCat(api, " failed: ", err_str ? err_str : "unknown error"));
}

rsmi_status_t InitLibrary() {
  rsmi_status_t status = rsmi_init(0);
  if (status == RSMI_STATUS_SUCCESS) {
    VLOG(1) << "SMI device queries go through rocm_smi.";
  }
  return status;
}

}  // namespace

absl::Status InitSmi() {
  // Caching the raw status, not an absl::Status, keeps this a plain static.
  static const rsmi_status_t status = InitLibrary();
  if (status != RSMI_STATUS_SUCCESS) return SmiError("rsmi_init", status);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<SmiDeviceHandle>> EnumerateDevices() {
  uint32_t num_devices = 0;
  if (rsmi_status_t status = rsmi_num_monitor_devices(&num_devices);
      status != RSMI_STATUS_SUCCESS) {
    return SmiError("rsmi_num_monitor_devices", status);
  }

  std::vector<SmiDeviceHandle> devices;
  devices.reserve(num_devices);
  for (uint32_t i = 0; i < num_devices; ++i) {
    devices.push_back(ToDeviceHandle(i));
  }
  return devices;
}

absl::StatusOr<SmiDeviceHandle> FindDevice(const BdfComponents& target_bdf) {
  ABSL_ASSIGN_OR_RETURN(std::vector<SmiDeviceHandle> devices, EnumerateDevices());

  for (SmiDeviceHandle device : devices) {
    uint64_t bdfid = 0;
    if (rsmi_status_t status =
            rsmi_dev_pci_id_get(ToDeviceIndex(device), &bdfid);
        status != RSMI_STATUS_SUCCESS) {
      VLOG(2) << "Skipping device " << ToDeviceIndex(device) << ": "
              << SmiError("rsmi_dev_pci_id_get", status).message();
      continue;
    }

    // Unpack rocm_smi's 64-bit BDF format into individual fields.
    // See
    // rocm-systems/projects/rocm-smi-lib/src/rocm_smi.cc:rsmi_dev_pci_id_get
    // for details on the packing.
    uint32_t domain = (bdfid >> 32) & 0xFFFFFFFF;
    uint8_t bus = (bdfid >> 8) & 0xFF;
    uint8_t device_number = (bdfid >> 3) & 0x1F;
    uint8_t function = bdfid & 0x7;

    if (domain == target_bdf.domain && bus == target_bdf.bus &&
        device_number == target_bdf.device && function == target_bdf.function) {
      return device;
    }
  }

  return absl::NotFoundError(
      absl::StrFormat("rocm_smi exposes no device with BDF %04x:%02x:%02x.%x",
                      target_bdf.domain, target_bdf.bus, target_bdf.device,
                      target_bdf.function));
}

absl::StatusOr<PcieLinkStatus> QueryPcieLinkStatus(SmiDeviceHandle device) {
  rsmi_gpu_metrics_t gpu_metrics = {};
  if (rsmi_status_t status =
          rsmi_dev_gpu_metrics_info_get(ToDeviceIndex(device), &gpu_metrics);
      status != RSMI_STATUS_SUCCESS) {
    return SmiError("rsmi_dev_gpu_metrics_info_get", status);
  }

  // rocm_smi reports pcie_link_speed in units of 0.1 GT/s, so scale to MT/s.
  return PcieLinkStatus{
      static_cast<uint32_t>(gpu_metrics.pcie_link_speed) * 100,
      gpu_metrics.pcie_link_width};
}

absl::StatusOr<uint64_t> QueryHiveId(SmiDeviceHandle device) {
  uint64_t hive_id = 0;
  if (rsmi_status_t status =
          rsmi_dev_xgmi_hive_id_get(ToDeviceIndex(device), &hive_id);
      status != RSMI_STATUS_SUCCESS) {
    return SmiError("rsmi_dev_xgmi_hive_id_get", status);
  }
  return hive_id;
}

absl::StatusOr<bool> IsXgmiPeer(SmiDeviceHandle src, SmiDeviceHandle dst) {
  // The API rejects a null hops pointer; only the link type is used.
  uint64_t hops = 0;
  RSMI_IO_LINK_TYPE link_type = RSMI_IOLINK_TYPE_UNDEFINED;
  if (rsmi_status_t status = rsmi_topo_get_link_type(
          ToDeviceIndex(src), ToDeviceIndex(dst), &hops, &link_type);
      status != RSMI_STATUS_SUCCESS) {
    return SmiError("rsmi_topo_get_link_type", status);
  }
  return link_type == RSMI_IOLINK_TYPE_XGMI;
}

}  // namespace stream_executor::gpu
