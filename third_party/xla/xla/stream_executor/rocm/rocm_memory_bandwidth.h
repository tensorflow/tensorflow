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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_BANDWIDTH_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_BANDWIDTH_H_

#include <cstdint>

#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor::gpu {

// Returns the device memory (HBM/GDDR) bandwidth in bytes/second.
//
// The legacy `2 * bus_width * clock` formula lands at spec peak on HBM2/HBM2e
// but falls short on HBM3/HBM3e and GDDR6. The value is resolved in two tiers,
// first hit wins:
//   1. a per-gfx peak for architectures the formula gets wrong;
//   2. the legacy formula otherwise (correct on HBM2/HBM2e).
//
// `mem_bus_width_bits` and `mem_clock_khz` come from hipDeviceProp_t
// (memoryBusWidth, memoryClockRate) and feed the formula.
int64_t GetRocmMemoryBandwidth(const RocmComputeCapability& cc,
                               int64_t mem_bus_width_bits,
                               int64_t mem_clock_khz);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_BANDWIDTH_H_
