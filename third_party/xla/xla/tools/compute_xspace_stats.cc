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

#include "xla/tools/compute_xspace_stats.h"

#include <cstdint>
#include <iostream>
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

absl::StatusOr<LineStats> ProcessLineEvents(
    const tensorflow::profiler::XLine& line) {
  LineStats line_stats;
  for (const auto& event : line.events()) {
    line_stats.total_time_ps += event.duration_ps();
  }
  return line_stats;
}

absl::StatusOr<int64_t> GetTotalTimePs(
    const tensorflow::profiler::XPlane& plane) {
  int64_t total_time_ps = 0;
  for (const auto& line : plane.lines()) {
    TF_ASSIGN_OR_RETURN(xla::gpu::LineStats line_stats,
                        xla::gpu::ProcessLineEvents(line));
    total_time_ps += line_stats.total_time_ps;
  }
  return total_time_ps;
}

absl::StatusOr<int64_t> GetWallTimePs(
    const tensorflow::profiler::XSpace& xspace) {
  int64_t wall_time_ps = 0;
  for (const tensorflow::profiler::XPlane& plane : xspace.planes()) {
    if (plane.name() != "Task Environment") {
      continue;
    }
    int64_t start_time_ns = 0;
    int64_t stop_time_ns = 0;
    absl::flat_hash_map<std::string, int64_t> stat_metadata_map;
    for (const auto& stat_metadata : plane.stat_metadata()) {
      stat_metadata_map[stat_metadata.second.name()] =
          stat_metadata.second.id();
    }

    for (const auto& stat : plane.stats()) {
      if (stat.metadata_id() == stat_metadata_map["profile_start_time"]) {
        start_time_ns = stat.uint64_value();
      } else if (stat.metadata_id() == stat_metadata_map["profile_stop_time"]) {
        stop_time_ns = stat.uint64_value();
      }
    }

    if (start_time_ns > 0 && stop_time_ns > 0) {
      wall_time_ps = (stop_time_ns - start_time_ns) * 1000;  // ns to ps
    }
    break;
  }
  return wall_time_ps;
}

absl::StatusOr<GpuDeviceStats> CalculateGpuDeviceStats(
    const tensorflow::profiler::XSpace& xspace) {
  GpuDeviceStats result;
  int64_t total_time_ps = 0;
  int64_t memcpy_time_ps = 0;
  absl::string_view device_name = "/device:GPU:0";

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
  // Calculate Wall Time from the "Task Environment" plane
  TF_ASSIGN_OR_RETURN(int64_t wall_time_ps, GetWallTimePs(xspace));
  result.wall_time_us = static_cast<double>(wall_time_ps) / 1e6;

  // Calculate the time in microseconds
  result.device_time_us = static_cast<double>(total_time_ps) / 1e6;
  result.device_memcpy_time_us = static_cast<double>(memcpy_time_ps) / 1e6;
  return result;
}

absl::StatusOr<xla::gpu::CpuStats> CalculateCpuStats(
    const tensorflow::profiler::XSpace& xspace) {
  xla::gpu::CpuStats result;

  // Iterate over planes to find the CPU plane
  for (const tensorflow::profiler::XPlane& plane : xspace.planes()) {
    if (plane.name() != "/host:CPU") {
      continue;  // Skip planes that aren't the target device.
    }
    TF_ASSIGN_OR_RETURN(int64_t total_time_ps, GetTotalTimePs(plane));
    result.cpu_time_us = static_cast<double>(total_time_ps) / 1e6;
    break;  // Assuming only one /host:CPU plane
  }

  // Calculate Wall Time from the "Task Environment" plane
  TF_ASSIGN_OR_RETURN(int64_t wall_time_ps, GetWallTimePs(xspace));
  result.wall_time_us = static_cast<double>(wall_time_ps) / 1e6;

  return result;
}

absl::Status Run(absl::string_view input_file, absl::string_view device_type) {
  if (input_file.empty()) {
    return absl::InvalidArgumentError("Input file must be specified.");
  }
  LOG(INFO) << "Input file: " << input_file;
  // Read the XSpace protobuf
  tsl::Env* env = tsl::Env::Default();
  tensorflow::profiler::XSpace xspace_proto;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(env, std::string(input_file), &xspace_proto));

  LOG(INFO) << "Successfully parsed XSpace proto.";

  if (device_type == "GPU") {
    absl::StatusOr<GpuDeviceStats> stats =
        CalculateGpuDeviceStats(xspace_proto);
    if (!stats.ok()) {
      return stats.status();
    }
    // Print the results
    std::cout << absl::StrFormat("Device Time: %.2f us\n",
                                 stats->device_time_us)
              << absl::StrFormat("Device Memcpy Time: %.2f us\n",
                                 stats->device_memcpy_time_us);
  } else if (device_type == "CPU") {
    absl::StatusOr<CpuStats> cpu_stats = CalculateCpuStats(xspace_proto);
    if (!cpu_stats.ok()) {
      return cpu_stats.status();
    }
    // Print the results
    std::cout << absl::StrFormat("CPU Time: %.2f us\n", cpu_stats->cpu_time_us)
              << absl::StrFormat("Wall Time: %.2f us\n",
                                 cpu_stats->wall_time_us);
  } else {
    return absl::InvalidArgumentError("Device type must be GPU or CPU.");
  }

  return absl::OkStatus();
}

}  // namespace xla::gpu
