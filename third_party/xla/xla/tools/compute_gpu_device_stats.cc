/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tools/compute_gpu_device_stats.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "third_party/tensorflow/core/profiler/convert/xplane_to_memory_profile.h"
#include "third_party/tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "third_party/tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "third_party/tensorflow/core/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla::gpu {

namespace {

const char* const kMemcpyDetailsStatName = "memcpy_details";
const char* const kGpuDeviceName = "/device:GPU:0";

// Calculates the peak device memory usage by iterating across all allocators
// and skipping over any allocations made in host memory.
absl::StatusOr<int64_t> PeakDeviceMemUsageLifetime(
    const tensorflow::profiler::MemoryProfile& memory_profile) {
  int64_t peak_bytes = 0;
  // Finds the max memory usage among all memory allocators.
  for (const auto& [id, allocator] :
       memory_profile.memory_profile_per_allocator()) {
    // Skips the host memory.
    if (absl::StrContains(absl::AsciiStrToLower(id), "host")) continue;
    const auto& peak_stats = allocator.profile_summary().peak_stats();
    peak_bytes = std::max(
        {peak_bytes, allocator.profile_summary().peak_bytes_usage_lifetime(),
         peak_stats.peak_bytes_in_use()});
  }
  return peak_bytes;
}

// Returns the peak memory usage of the device.
absl::StatusOr<int64_t> GetPeakDeviceMemory(
    const tensorflow::profiler::XSpace& xspace) {
  const tensorflow::profiler::XPlane* host_plane =
      tensorflow::profiler::FindPlaneWithName(
          xspace, tsl::profiler::kHostThreadsPlaneName);
  if (!host_plane) {
    VLOG(1) << "No host plane found.";
    return 0;
  }
  tensorflow::profiler::MemoryProfile memory_profile =
      tensorflow::profiler::ConvertXPlaneToMemoryProfile(
          *host_plane, std::numeric_limits<int64_t>::max());
  return PeakDeviceMemUsageLifetime(memory_profile);
}
}  // namespace

bool IsMemcpy(const tensorflow::profiler::XEvent& event,
              int64_t memcpy_details_id) {
  for (const auto& stat : event.stats()) {
    if (stat.metadata_id() == memcpy_details_id) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<LineStats> ProcessLineEvents(
    const tensorflow::profiler::XLine& line, int64_t memcpy_details_id) {
  LineStats stats;
  for (const auto& event : line.events()) {
    stats.total_time_ps += event.duration_ps();
    if (IsMemcpy(event, memcpy_details_id)) {
      stats.memcpy_time_ps += event.duration_ps();
    }
  }
  return stats;
}

absl::StatusOr<GpuDeviceStats> ComputeGPUDeviceStats(
    const tensorflow::profiler::XSpace& xspace) {
  GpuDeviceStats result;
  int64_t total_time_ps = 0;
  int64_t memcpy_time_ps = 0;
  int64_t memcpy_details_id = -1;
  bool device_plane_found = false;

  for (const tensorflow::profiler::XPlane& plane : xspace.planes()) {
    if (plane.name() == kGpuDeviceName) {
      device_plane_found = true;

      absl::flat_hash_map<std::string, int64_t> stat_metadata_map;
      for (const auto& stat_metadata : plane.stat_metadata()) {
        stat_metadata_map[stat_metadata.second.name()] =
            stat_metadata.second.id();
      }

      if (auto it = stat_metadata_map.find(kMemcpyDetailsStatName);
          it != stat_metadata_map.end()) {
        memcpy_details_id = it->second;
      }

      for (const auto& line : plane.lines()) {
        TF_ASSIGN_OR_RETURN(LineStats line_stats,
                            ProcessLineEvents(line, memcpy_details_id));
        total_time_ps += line_stats.total_time_ps;
        memcpy_time_ps += line_stats.memcpy_time_ps;
      }
      break;
    }
  }

  if (!device_plane_found) {
    return absl::NotFoundError(absl::StrFormat(
        "Device plane '%s' not found in XSpace.", kGpuDeviceName));
  }

  result.device_time_us = static_cast<double>(total_time_ps) / 1e6;
  result.device_memcpy_time_us = static_cast<double>(memcpy_time_ps) / 1e6;

  TF_ASSIGN_OR_RETURN(result.peak_device_mem_bytes,
                      GetPeakDeviceMemory(xspace));
  return result;
}

absl::Status Run(absl::string_view input_file) {
  if (input_file.empty()) {
    return absl::InvalidArgumentError("Input file must be specified.");
  }
  LOG(INFO) << "Input file: " << input_file;

  // Read the XSpace protobuf
  tsl::Env* env = tsl::Env::Default();
  auto xspace_proto = std::make_unique<tensorflow::profiler::XSpace>();
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(env, std::string(input_file), xspace_proto.get()));

  LOG(INFO) << "Successfully parsed XSpace proto.";

  // Calculate the GPU device statistics: device time, memcpy time, and peak
  // device memory usage.
  absl::StatusOr<GpuDeviceStats> stats = ComputeGPUDeviceStats(*xspace_proto);
  if (!stats.ok()) {
    return stats.status();
  }

  // Print the results
  std::cout << absl::StrFormat("Device Time: %.2f us\n", stats->device_time_us)
            << absl::StrFormat("Device Memcpy Time: %.2f us\n",
                               stats->device_memcpy_time_us);
  std::cout << absl::StrFormat("Peak Device Memory Usage: %d bytes\n",
                               stats->peak_device_mem_bytes);

  return absl::OkStatus();
}

}  // namespace xla::gpu
