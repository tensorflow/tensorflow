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

#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor::gpu {
namespace {

constexpr int64_t kGbps = int64_t{1000} * 1000 * 1000;

}  // namespace

int64_t GetRocmMemoryBandwidth(const RocmComputeCapability& cc,
                               int64_t mem_bus_width_bits,
                               int64_t mem_clock_khz) {
  // On HBM2/HBM2e (gfx908 MI100, gfx90a MI210) the formula `2 * bus_width *
  // clock` lands at spec peak, so those arches fall through to it. On
  // HBM3/HBM3e (gfx942 MI300X, gfx950 MI350X) and GDDR6 (gfx1201) the formula
  // falls short of spec peak, so an explicit per-gfx value is used instead.
  if (cc.gfx9_mi300()) return 5300 * kGbps;  // MI300X, HBM3
  // TODO: MI355X shares gfx950 with MI350X but has a higher peak; the per-board
  // value can be read from amd_smi gpu_metrics. Postponed because amd_smi
  // embeds a copy of rocm_smi and exports the same symbols, which clash with
  // the rocm_smi that RCCL pulls into the process. Revisit once the ROCm floor
  // is >= 7.13 (where RCCL no longer depends on rocm_smi).
  if (cc.gfx9_mi350()) return 6810 * kGbps;     // MI350X, HBM3e
  if (cc.gfx12_discrete()) return 640 * kGbps;  // RX 9070 XT, GDDR6

  // mem_bandwidth = 2 * mem_bus_width_in_bytes * mem_clock_rate_in_hz
  return 2 * (mem_bus_width_bits / 8) * (mem_clock_khz * 1000);
}

}  // namespace stream_executor::gpu
