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

#include "xla/stream_executor/rocm/rocm_memory_bandwidth.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/rocm/smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

constexpr int64_t kGbps = int64_t{1000} * 1000 * 1000;

// Tier 1, the firmware peak.
absl::StatusOr<int64_t> FirmwareBandwidth(absl::string_view pci_bus_id) {
  absl::MutexLock lock(smi_mutex);

  ABSL_RETURN_IF_ERROR(InitSmi());

  ABSL_ASSIGN_OR_RETURN(BdfComponents bdf, ParseBdf(pci_bus_id));
  ABSL_ASSIGN_OR_RETURN(SmiDeviceHandle device, FindDevice(bdf));
  ABSL_ASSIGN_OR_RETURN(uint64_t gbps, QueryPeakMemoryBandwidthGbps(device));

  return static_cast<int64_t>(gbps) * kGbps;
}

// Tiers 2 and 3, a per-gfx peak where the formula is known to be wrong, else
// the formula.
int64_t ArchOrFormulaBandwidth(const RocmComputeCapability& cc,
                               int64_t mem_bus_width_bits,
                               int64_t mem_clock_khz) {
  // On HBM2/HBM2e (gfx908 MI100, gfx90a MI210) the formula `2 * bus_width *
  // clock` lands at spec peak, so those arches fall through to it. On
  // HBM3/HBM3e (gfx942 MI300X, gfx950 MI350X) and GDDR6 (gfx1201) the formula
  // falls short of spec peak, so an explicit per-gfx value is used instead.
  if (cc.gfx9_mi300()) return 5300 * kGbps;     // MI300X, HBM3
  if (cc.gfx9_mi350()) return 7782 * kGbps;     // MI350X, HBM3e
  if (cc.gfx12_discrete()) return 640 * kGbps;  // RX 9070 XT, GDDR6

  // mem_bandwidth = 2 * mem_bus_width_in_bytes * mem_clock_rate_in_hz
  return 2 * (mem_bus_width_bits / 8) * (mem_clock_khz * 1000);
}

}  // namespace

int64_t GetRocmMemoryBandwidth(absl::string_view pci_bus_id,
                               const RocmComputeCapability& cc,
                               int64_t mem_bus_width_bits,
                               int64_t mem_clock_khz) {
  absl::StatusOr<int64_t> firmware = FirmwareBandwidth(pci_bus_id);
  if (firmware.ok()) {
    VLOG(1) << "Device memory bandwidth for " << pci_bus_id << ": " << *firmware
            << " bytes/s, from the SMI reported firmware peak.";
    return *firmware;
  }

  // rocm_smi cannot report the peak below ROCm 7.13 (Unimplemented) and some
  // firmware leaves the field unset (Unavailable), so only an unexpected SMI
  // failure is worth a warning.
  const absl::Status& status = firmware.status();
  std::string reason = absl::StrCat("No SMI firmware peak for ", pci_bus_id,
                                    " (", status.message(), ") ");
  if (absl::IsUnimplemented(status)) {
    VLOG(1) << reason << "is implemented for ROCm < 7.13.";
  } else if (absl::IsUnavailable(status)) {
    LOG(WARNING) << reason << "is available from amd-smi/firmware.";
  } else {
    LOG(WARNING) << reason;
  }

  return ArchOrFormulaBandwidth(cc, mem_bus_width_bits, mem_clock_khz);
}

}  // namespace stream_executor::gpu
