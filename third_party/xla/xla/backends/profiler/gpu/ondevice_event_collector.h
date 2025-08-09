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

#ifndef XLA_BACKENDS_PROFILER_GPU_ONDEVICE_EVENT_COLLECTOR_H_
#define XLA_BACKENDS_PROFILER_GPU_ONDEVICE_EVENT_COLLECTOR_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "xla/backends/profiler/gpu/ondevice_trace_events.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

struct GpuOnDeviceTraceEventCollectorOptions {
  uint32_t max_injection_instance = 0;  // 0 to disable injection
  uint32_t max_pid = 0xFFFFFFFF;        // 2^32 - 1
  uint32_t max_tid = 0xFFFFFFFF;        // 2^32 - 1
};

class GpuOnDeviceTraceEventCollector {
 public:
  virtual ~GpuOnDeviceTraceEventCollector() = default;

  virtual absl::Status Export(::tensorflow::profiler::XSpace* space,
                              uint64_t end_gpu_ns) = 0;

  virtual absl::Status AddEvent(GpuOnDeviceTraceEvent&& event) = 0;

  virtual absl::Status AddEvent(std::vector<GpuOnDeviceTraceEvent> events) = 0;

  virtual const GpuOnDeviceTraceEventCollectorOptions& options() const = 0;
};

std::unique_ptr<GpuOnDeviceTraceEventCollector>
CreateGpuOnDeviceTraceEventCollector(
    const GpuOnDeviceTraceEventCollectorOptions& options,
    uint64_t start_walltime_ns, uint64_t start_gputime_ns);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ONDEVICE_EVENT_COLLECTOR_H_
