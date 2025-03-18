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
// A library for computing GPU statistics from an XSpace protobuf.
#include <cstdint>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

#ifndef XLA_TOOLS_COMPUTE_XSPACE_STATS_H_
#define XLA_TOOLS_COMPUTE_XSPACE_STATS_H_

namespace xla::gpu {

// Structure to hold the calculated GPU device statistics.
struct GpuDeviceStats {
  double device_time_us = 0.0;
  double device_memcpy_time_us = 0.0;
  double wall_time_us = 0.0;
};

struct CpuStats {
  double cpu_time_us = 0.0;
  double wall_time_us = 0.0;
};

// Structure to hold the calculated statistics for XEvent.
struct LineStats {
  int64_t total_time_ps = 0;
  int64_t memcpy_time_ps = 0;
};

// Checks if an XEvent is a memcpy operation.
bool IsMemcpy(const tensorflow::profiler::XEvent& event,
              int64_t memcpy_details_id);

// Processes an XLine and calculates the total time and memcpy time.
absl::StatusOr<LineStats> ProcessLineEvents(
    const tensorflow::profiler::XLine& line, int64_t memcpy_details_id);

absl::StatusOr<LineStats> ProcessLineEvents(
    const tensorflow::profiler::XLine& line);

// Calculates GPU device and memcpy times from an XSpace.
absl::StatusOr<GpuDeviceStats> CalculateGpuDeviceStats(
    const tensorflow::profiler::XSpace& xspace);

absl::StatusOr<CpuStats> CalculateCpuStats(
    const tensorflow::profiler::XSpace& xspace);

// Reads an XSpace protobuf from a file and computes GPU statistics, and prints
// them to stdout.  Returns an error status if something goes wrong.
absl::Status Run(absl::string_view input_file, absl::string_view device_type);

}  // namespace xla::gpu

#endif  // XLA_TOOLS_COMPUTE_XSPACE_STATS_H_
