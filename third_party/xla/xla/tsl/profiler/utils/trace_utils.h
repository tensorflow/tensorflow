/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PROFILER_UTILS_TRACE_UTILS_H_
#define XLA_TSL_PROFILER_UTILS_TRACE_UTILS_H_

#include <optional>

#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace profiler {

// Constants used as trace_viewer PID (device_id in trace_events.proto).
// PID 0 is unused.
// Support up to 500 accelerator devices.
constexpr uint32 kFirstDeviceId = 1;
constexpr uint32 kLastDeviceId = 500;
// Support Upto 200 custom planes as fake devices (i.e., planes with a
// "/custom:" prefix). See `<project_name>::kCustomPlanePrefix` for more
// information
constexpr uint32 kFirstCustomPlaneDeviceId = kLastDeviceId + 1;
constexpr uint32 kMaxCustomPlaneDevicesPerHost = 200;
constexpr uint32 kLastCustomPlaneDeviceId =
    kFirstCustomPlaneDeviceId + kMaxCustomPlaneDevicesPerHost - 1;
// Host threads are shown as a single fake device.
constexpr uint32 kHostThreadsDeviceId = kLastCustomPlaneDeviceId + 1;

// Constants used as trace_viewer TID (resource_id in trace_events.proto).
constexpr int kThreadIdDerivedMin = 0xdeadbeef;
constexpr int kThreadIdStepInfo = kThreadIdDerivedMin;
constexpr int kThreadIdKernelLaunch = kThreadIdDerivedMin + 1;
constexpr int kThreadIdTfNameScope = kThreadIdDerivedMin + 2;
constexpr int kThreadIdTfOp = kThreadIdDerivedMin + 3;
constexpr int kThreadIdHloModule = kThreadIdDerivedMin + 4;
constexpr int kThreadIdHloOp = kThreadIdDerivedMin + 5;
constexpr int kThreadIdOverhead = kThreadIdDerivedMin + 6;
constexpr int kThreadIdSource = kThreadIdDerivedMin + 7;
constexpr int kThreadIdHostOffloadOpStart = kThreadIdDerivedMin + 8;
constexpr int kThreadIdHostOffloadOpEnd = kThreadIdDerivedMin + 48;
// Space for derived lines for host XLA Ops
constexpr int kThreadIdHostXlaRegionStart = kThreadIdDerivedMin + 49;
constexpr int kThreadIdHostXlaRegionEnd = kThreadIdHostXlaRegionStart + 240;
constexpr int kThreadIdDerivedMax = kThreadIdHostXlaRegionEnd;

static inline bool IsDerivedThreadId(int thread_id) {
  return thread_id >= kThreadIdDerivedMin && thread_id <= kThreadIdDerivedMax;
}

// Parses the device ordinal (N) from device names that use TensorFlow
// convention: "hostname /device:xPU:N".
static inline std::optional<uint32_t> ParseDeviceOrdinal(
    absl::string_view device_name) {
  if (auto pos = device_name.find_last_of(':');
      pos != absl::string_view::npos) {
    device_name.remove_prefix(pos + 1);
  }
  if (auto pos = device_name.find_first_of(' ');
      pos != absl::string_view::npos) {
    device_name.remove_suffix(device_name.size() - pos);
  }
  uint32_t device_id;
  if (absl::SimpleAtoi(device_name, &device_id)) return device_id;
  return std::nullopt;
}

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_TRACE_UTILS_H_
