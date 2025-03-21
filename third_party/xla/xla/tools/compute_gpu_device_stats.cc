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

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla::gpu {

// Checks if an event is a memcpy operation.
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

absl::StatusOr<GpuDeviceStats> CalculateDeviceTimeAndMemcpy(
    const tensorflow::profiler::XSpace& xspace, absl::string_view device_name) {
  GpuDeviceStats result;
  int64_t total_time_ps = 0;
  int64_t memcpy_time_ps = 0;

  // Iterate over planes to find the device
  for (const tensorflow::profiler::XPlane& plane : xspace.planes()) {
    if (plane.name() != device_name) {
      continue;  // Skip planes that aren't the target device.
    }

    // Create a map for stat metadata
    absl::flat_hash_map<std::string, int64_t> stat_metadata_map;
    for (const auto& stat_metadata : plane.stat_metadata()) {
      stat_metadata_map[stat_metadata.second.name()] =
          stat_metadata.second.id();
    }

    // Determine the memcpy details ID.
    int64_t memcpy_details_id = -1;
    if (auto it = stat_metadata_map.find("memcpy_details");
        it != stat_metadata_map.end()) {
      memcpy_details_id = it->second;
    }

    // Process each line in the plane
    for (const auto& line : plane.lines()) {
      TF_ASSIGN_OR_RETURN(LineStats line_stats,
                          ProcessLineEvents(line, memcpy_details_id));
      total_time_ps += line_stats.total_time_ps;
      memcpy_time_ps += line_stats.memcpy_time_ps;
    }
    break;
  }

  // Calculate the time in microseconds
  result.device_time_us = static_cast<double>(total_time_ps) / 1e6;
  result.device_memcpy_time_us = static_cast<double>(memcpy_time_ps) / 1e6;
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

  // Calculate device and memcpy times
  const std::string device_name = "/device:GPU:0";
  absl::StatusOr<GpuDeviceStats> stats =
      CalculateDeviceTimeAndMemcpy(*xspace_proto, device_name);
  if (!stats.ok()) {
    return stats.status();
  }

  // Print the results
  std::cout << absl::StrFormat("Device Time: %.2f us\n", stats->device_time_us)
            << absl::StrFormat("Device Memcpy Time: %.2f us\n",
                               stats->device_memcpy_time_us);
  return absl::OkStatus();
}

}  // namespace xla::gpu
