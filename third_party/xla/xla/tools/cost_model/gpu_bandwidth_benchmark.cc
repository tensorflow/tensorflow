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

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

absl::StatusOr<double> GetPeakBandwidthBytesPerSec(int device_id) {
  absl::StatusOr<stream_executor::Platform*> platform =
      stream_executor::PlatformManager::PlatformWithName(
          stream_executor::GpuPlatformName());
  if (!platform.ok()) {
    return platform.status();
  }
  absl::StatusOr<stream_executor::StreamExecutor*> executor =
      (*platform)->ExecutorForDevice(device_id);
  if (!executor.ok()) {
    return executor.status();
  }
  const int64_t bandwidth =
      (*executor)->GetDeviceDescription().memory_bandwidth();
  if (bandwidth <= 0) {
    return absl::InternalError(absl::StrFormat(
        "Failed to determine peak memory bandwidth for device %d: "
        "memory_bandwidth is %v.",
        device_id, bandwidth));
  }
  return static_cast<double>(bandwidth);
}

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
