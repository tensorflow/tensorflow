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

#ifndef XLA_TOOLS_COST_MODEL_GPU_BANDWIDTH_BENCHMARK_H_
#define XLA_TOOLS_COST_MODEL_GPU_BANDWIDTH_BENCHMARK_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace xla::gpu {

// Entry in a bandwidth table mapping DMA transfer size to fractional
// saturation of peak bandwidth in [0.0, 1.0].
struct BandwidthEntry {
  int64_t dma_size_bytes = 0;
  float bandwidth_fraction = 0.0f;
};

// Returns theoretical peak GPU memory bandwidth in bytes per second for
// `device_id`.
absl::StatusOr<double> GetPeakBandwidthBytesPerSec(int device_id);

// Formats bandwidth table entries into a human-readable table.
std::string FormatBandwidthTable(absl::Span<const BandwidthEntry> entries);

}  // namespace xla::gpu

#endif  // XLA_TOOLS_COST_MODEL_GPU_BANDWIDTH_BENCHMARK_H_
