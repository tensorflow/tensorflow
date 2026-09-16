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

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/rocm/smi_util.h"
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

absl::StatusOr<int64_t> GetRocmPcieBandwidth(absl::string_view pci_bus_id) {
  absl::MutexLock lock(smi_mutex);

  ABSL_RETURN_IF_ERROR(InitSmi());

  ABSL_ASSIGN_OR_RETURN(BdfComponents bdf, ParseBdf(pci_bus_id));
  ABSL_ASSIGN_OR_RETURN(SmiDeviceHandle device, FindDevice(bdf));
  ABSL_ASSIGN_OR_RETURN(PcieLinkStatus link, QueryPcieLinkStatus(device));

  uint32_t speed_mt_per_sec = link.speed_mt_per_sec;
  uint16_t width = link.width;

  if (speed_mt_per_sec == 0 || width == 0) {
    return absl::InternalError(
        absl::StrCat("SMI reported zero PCIe speed (", speed_mt_per_sec,
                     " MT/s) or width (", width, " lanes)"));
  }

  int64_t bandwidth =
      ComputePcieBandwidthFromSpeedAndWidth(speed_mt_per_sec, width);

  VLOG(1) << "PCIe bandwidth for " << pci_bus_id << ": " << speed_mt_per_sec
          << " MT/s x" << width << " = " << bandwidth / (1024 * 1024 * 1024)
          << " GB/s (" << bandwidth << " bytes/s)";

  return bandwidth;
}

}  // namespace stream_executor::gpu
