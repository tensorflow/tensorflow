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

#include "xla/stream_executor/rocm/rocm_pcie_bandwidth.h"

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"
#include "rocm/include/rocm_smi/rocm_smi.h"
#include "xla/stream_executor/rocm/rocm_smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

// PCIe encoding efficiencies by generation
constexpr double kPcieGen1Gen2Efficiency = 0.8;
constexpr double kPcieGen3To5Efficiency = 128.0 / 130.0;
constexpr double kPcieGen6Efficiency = 242.0 / 256.0;

// PCIe transfer rate thresholds in MT/s
constexpr uint32_t kPcieGen2MaxSpeedMTps = 5000;
constexpr uint32_t kPcieGen5MaxSpeedMTps = 32000;

constexpr double PcieEncodingEfficiency(uint32_t speed_mt_per_sec) {
  if (speed_mt_per_sec <= kPcieGen2MaxSpeedMTps) return kPcieGen1Gen2Efficiency;
  if (speed_mt_per_sec <= kPcieGen5MaxSpeedMTps) return kPcieGen3To5Efficiency;
  return kPcieGen6Efficiency;
}

constexpr int64_t ComputePcieBandwidthFromSpeedAndWidth(
    uint32_t speed_mt_per_sec, uint16_t width) {
  if (width == 0 || speed_mt_per_sec == 0) return 0;
  double efficiency = PcieEncodingEfficiency(speed_mt_per_sec);
  return static_cast<int64_t>(static_cast<double>(speed_mt_per_sec) * 1e6 *
                              width / 8.0 * efficiency);
}

}  // namespace

std::optional<int64_t> GetRocmPcieBandwidth(absl::string_view pci_bus_id) {
  if (!InitRocmSmi()) return std::nullopt;

  std::optional<BdfComponents> bdf = ParseBdf(pci_bus_id);
  if (!bdf.has_value()) {
    LOG(WARNING) << "Failed to parse PCI bus ID: " << pci_bus_id;
    return std::nullopt;
  }

  std::optional<uint32_t> dev_idx = FindDeviceIndex(*bdf);
  if (!dev_idx.has_value()) {
    LOG(WARNING) << "rocm_smi: could not find device for PCI bus ID "
                 << pci_bus_id;
    return std::nullopt;
  }

  rsmi_gpu_metrics_t gpu_metrics = {};
  rsmi_status_t status = rsmi_dev_gpu_metrics_info_get(*dev_idx, &gpu_metrics);
  if (status != RSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    rsmi_status_string(status, &err_str);
    LOG(WARNING) << "rsmi_dev_gpu_metrics_info_get failed for " << pci_bus_id
                 << ": " << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  uint32_t speed_mt_per_sec =
      static_cast<uint32_t>(gpu_metrics.pcie_link_speed) * 100;
  uint16_t width = gpu_metrics.pcie_link_width;

  if (speed_mt_per_sec == 0 || width == 0) {
    LOG(WARNING) << "rocm_smi gpu_metrics reported zero PCIe speed ("
                 << speed_mt_per_sec << " MT/s) or width (" << width
                 << " lanes) for " << pci_bus_id;
    return std::nullopt;
  }

  int64_t bandwidth =
      ComputePcieBandwidthFromSpeedAndWidth(speed_mt_per_sec, width);

  VLOG(1) << "PCIe bandwidth for " << pci_bus_id << ": " << speed_mt_per_sec
          << " MT/s x" << width << " = " << bandwidth / (1024 * 1024 * 1024)
          << " GB/s (" << bandwidth << " bytes/s)";

  return bandwidth;
}

}  // namespace stream_executor::gpu
