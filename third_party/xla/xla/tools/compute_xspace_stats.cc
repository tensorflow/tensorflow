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
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla::gpu {

using ::tensorflow::profiler::XLine;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineVisitor;
using ::tsl::profiler::XPlaneVisitor;
using ::tsl::profiler::XStatVisitor;

namespace {

bool IsMemoryAllocation(int64_t event_type) {
  return event_type == tsl::profiler::HostEventType::kMemoryAllocation;
}

bool IsMemoryDeallocation(int64_t event_type) {
  return event_type == tsl::profiler::HostEventType::kMemoryDeallocation;
}

}  // namespace

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

absl::StatusOr<LineStats> ProcessLineEvents(const XLine& line,
                                            int64_t memcpy_details_id) {
  LineStats stats;
  for (const auto& event : line.events()) {
    stats.total_time_ps += event.duration_ps();
    if (IsMemcpy(event, memcpy_details_id)) {
      stats.memcpy_time_ps += event.duration_ps();
    }
  }
  return stats;
}

absl::StatusOr<LineStats> ProcessLineEvents(const XLine& line) {
  LineStats line_stats;
  for (const auto& event : line.events()) {
    line_stats.total_time_ps += event.duration_ps();
  }
  return line_stats;
}

absl::StatusOr<int64_t> GetTotalTimePs(const XPlane& plane) {
  int64_t total_time_ps = 0;
  for (const auto& line : plane.lines()) {
    TF_ASSIGN_OR_RETURN(xla::gpu::LineStats line_stats,
                        xla::gpu::ProcessLineEvents(line));
    total_time_ps += line_stats.total_time_ps;
  }
  return total_time_ps;
}

absl::StatusOr<int64_t> GetWallTimePs(const XSpace& xspace) {
  int64_t wall_time_ps = 0;
  if (const XPlane* env_plane = tsl::profiler::FindPlaneWithName(
          xspace, tsl::profiler::kTaskEnvPlaneName)) {
    std::optional<int64_t> start_time_ns;
    std::optional<int64_t> stop_time_ns;
    XPlaneVisitor visitor(env_plane, {}, {tsl::profiler::FindTaskEnvStatType});
    if (auto stat = visitor.GetStat(
            tsl::profiler::TaskEnvStatType::kEnvProfileStartTime)) {
      start_time_ns = stat->UintValue();
    }
    if (auto stat = visitor.GetStat(
            tsl::profiler::TaskEnvStatType::kEnvProfileStopTime)) {
      stop_time_ns = stat->UintValue();
    }

    if (start_time_ns.has_value() && stop_time_ns.has_value()) {
      wall_time_ps = (*stop_time_ns - *start_time_ns) * 1000;  // ns to ps
    }
  }
  return wall_time_ps;
}

absl::StatusOr<int64_t> GetGPUPeakMemory(const XPlane* plane) {
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  absl::flat_hash_map<std::string, int64_t> peak_memory_usage_per_allocator;
  plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      int64_t event_type = event.Type().value_or(
          tsl::profiler::HostEventType::kUnknownHostEventType);
      if (!(IsMemoryAllocation(event_type) ||
            IsMemoryDeallocation(event_type))) {
        return;
      }
      std::optional<std::string> memory_allocator_id;
      std::optional<int64_t> peak_bytes_in_use;
      event.ForEachStat([&](const XStatVisitor& stat) {
        if (!stat.Type().has_value()) {
          return;
        }
        switch (stat.Type().value()) {
          case StatType::kAllocatorName:
            memory_allocator_id = std::string(stat.StrOrRefValue());
            break;
          case StatType::kPeakBytesInUse:
            peak_bytes_in_use = stat.IntValue();
            break;
          default:
            break;
        }
      });
      if (memory_allocator_id.has_value() && peak_bytes_in_use.has_value()) {
        peak_memory_usage_per_allocator[*memory_allocator_id] =
            std::max(peak_memory_usage_per_allocator[*memory_allocator_id],
                     *peak_bytes_in_use);
      }
    });
  });

  // This logic relies on string matching on allocator names, which can be
  // brittle. If allocator naming conventions change, this function may need to
  // be updated. These are the current names being used:
  // Host allocators:
  //     - gpu_host_bfc
  //     - xla_gpu_host_bfc
  //     - bfc_cpu_allocator_for_gpu
  // GPU allocators:
  //     - GPU_ + device_ordinal + _bfc
  //     - GPU_collectivememory_ + device_ordinal + _bfc
  auto is_allocator_for_device = [&](absl::string_view allocator_name) {
    auto lower_case_name = absl::AsciiStrToLower(allocator_name);
    return absl::StrContains(lower_case_name, "gpu") &&
           !absl::StrContains(lower_case_name, "host") &&
           !absl::StrContains(lower_case_name, "cpu");
  };

  int64_t peak_bytes = -1;
  for (const auto& [id, bytes_used] : peak_memory_usage_per_allocator) {
    if (!is_allocator_for_device(id)) {
      continue;
    }
    peak_bytes = std::max(peak_bytes, bytes_used);
  }
  if (peak_bytes == -1) {
    return absl::InternalError("Could not find peak memory usage.");
  }
  return peak_bytes;
}

absl::StatusOr<GpuDeviceStats> CalculateGpuDeviceStats(const XSpace& xspace) {
  GpuDeviceStats result;
  int64_t total_time_ps = 0;
  int64_t memcpy_time_ps = 0;
  absl::string_view device_name = "/device:GPU:0";

  if (const XPlane* host_plane = tsl::profiler::FindPlaneWithName(
          xspace, tsl::profiler::kHostThreadsPlaneName)) {
    TF_ASSIGN_OR_RETURN(result.peak_memory_usage_bytes,
                        GetGPUPeakMemory(host_plane));
  }

  // Iterate over planes to find the device
  for (const XPlane& plane : xspace.planes()) {
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

absl::StatusOr<xla::gpu::CpuStats> CalculateCpuStats(const XSpace& xspace) {
  xla::gpu::CpuStats result;

  // Process the host CPU plane
  if (const XPlane* host_plane = tsl::profiler::FindPlaneWithName(
          xspace, tsl::profiler::kHostThreadsPlaneName)) {
    TF_ASSIGN_OR_RETURN(int64_t total_time_ps, GetTotalTimePs(*host_plane));
    result.cpu_time_us = static_cast<double>(total_time_ps) / 1e6;
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
  XSpace xspace_proto;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(env, std::string(input_file), &xspace_proto));

  LOG(INFO) << "Successfully parsed XSpace proto.";

  // Any change in the format of the output needs to reflect in
  // .github/workflows/benchmarks/run_benchmark.sh
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
                                 stats->device_memcpy_time_us)
              << absl::StrFormat("Peak Memory: %d bytes\n",
                                 stats->peak_memory_usage_bytes);
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
