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

#ifndef XLA_BACKENDS_PROFILER_GPU_ONDEVICE_EVENT_EXPORTER_H_
#define XLA_BACKENDS_PROFILER_GPU_ONDEVICE_EVENT_EXPORTER_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "xla/tsl/profiler/backends/gpu/ondevice_event_collector.h"
#include "xla/tsl/profiler/backends/gpu/ondevice_trace_event.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

struct GpuOnDeviceTraceEventCollectorOptions {
  int32_t max_injection_instance = 0;
  int32_t max_pid = 0;
  int32_t max_tid = 0;
};

// Add export() to GpuOnDeviceTraceEventCollector.
class GpuOnDeviceTraceEventExporter
    : public ::tsl::profiler::GpuOnDeviceTraceEventCollector {
 public:
  ~GpuOnDeviceTraceEventExporter() override = default;

  absl::Status AddEvent(
      ::tsl::profiler::GpuOnDeviceTraceEvent&& event) override = 0;

  virtual absl::Status Export(tensorflow::profiler::XSpace* space,
                              uint64_t end_gpu_ns) = 0;
};

std::unique_ptr<GpuOnDeviceTraceEventExporter>
CreateGpuOnDeviceTraceEventExporter(
    const GpuOnDeviceTraceEventCollectorOptions& options,
    uint64_t start_walltime_ns, uint64_t start_gputime_ns);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ONDEVICE_EVENT_EXPORTER_H_
