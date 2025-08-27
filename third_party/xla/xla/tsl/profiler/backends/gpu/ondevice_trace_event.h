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

#ifndef XLA_TSL_PROFILER_BACKENDS_GPU_ONDEVICE_TRACE_EVENT_H_
#define XLA_TSL_PROFILER_BACKENDS_GPU_ONDEVICE_TRACE_EVENT_H_

#include <cstdint>

#include "absl/strings/string_view.h"

namespace tsl {
namespace profiler {

struct GpuOnDeviceTraceEvent {
  // One injection instance mapping to a single kernel execution.
  int32_t injection_instance_id = 0;
  absl::string_view tag_name = "";
  uint32_t tag_id = 0;
  uint32_t pid = 0;
  uint32_t tid = 0;
  int64_t start_time_ns = 0;
  int64_t duration_ps = 0;
};

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_BACKENDS_GPU_ONDEVICE_TRACE_EVENT_H_
