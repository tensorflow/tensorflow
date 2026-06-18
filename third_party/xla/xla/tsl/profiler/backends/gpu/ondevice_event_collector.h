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

#ifndef XLA_TSL_PROFILER_BACKENDS_GPU_ONDEVICE_EVENT_COLLECTOR_H_
#define XLA_TSL_PROFILER_BACKENDS_GPU_ONDEVICE_EVENT_COLLECTOR_H_

#include "absl/status/status.h"
#include "xla/tsl/profiler/backends/gpu/ondevice_trace_event.h"

namespace tsl {
namespace profiler {

// This is the interface for the the on-device event receiver to inject event.
class GpuOnDeviceTraceEventCollector {
 public:
  virtual ~GpuOnDeviceTraceEventCollector() = default;

  virtual absl::Status AddEvent(GpuOnDeviceTraceEvent&& event) = 0;
};

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_BACKENDS_GPU_ONDEVICE_EVENT_COLLECTOR_H_
