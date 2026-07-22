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

#include "xla/tools/cost_model/gpu_bandwidth_benchmark.h"

#include <string>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"

namespace xla::gpu {

std::string FormatBandwidthTable(absl::Span<const BandwidthEntry> entries) {
  std::string result =
      "DMA Size (Bytes)    Bandwidth Fraction\n"
      "----------------    ------------------\n";
  for (const BandwidthEntry& entry : entries) {
    absl::StrAppendFormat(&result, "%16d    %18.8f\n", entry.dma_size_bytes,
                          entry.bandwidth_fraction);
  }
  return result;
}

}  // namespace xla::gpu
