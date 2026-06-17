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

#ifndef XLA_TSL_PROFILER_BACKENDS_GPU_ONDEVICE_EVENT_RECEIVER_H_
#define XLA_TSL_PROFILER_BACKENDS_GPU_ONDEVICE_EVENT_RECEIVER_H_

#include <cstddef>
#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/profiler/backends/gpu/ondevice_event_collector.h"
#include "xla/tsl/profiler/backends/gpu/ondevice_trace_event.h"

namespace tsl {
namespace profiler {

//     GpuOnDeviceTraceEventReceiver is a singleton class that receives
// GpuOnDeviceTraceEvent(s). Parallel threads may postprocess hardware event
// buffer (may be dumped into host memory) and inject those software generated
// event(s) to this Receiver. It will be global available to users by its APIs,
// or Python bindings, to a) check if it is active (in some profiling session);
// b) inject events that user software generated.
//     Upon receiving the event(s), the receiver will forward the event(s) to
// the collector (if not nullptr) of type GpuOnDeviceTraceEventCollector.
// The Collector registers itself to this receiver by calling StartWith(), so
// that it could receive the event(s) from the receiver upon user injection.
//     Note that the collector could be wrapped by a profiler which follows
// profiler api, so that it will be created together with other profilers
// at the beginning of a session, and be destroyed at the end of the session.
class GpuOnDeviceTraceEventReceiver final {
 public:
  static GpuOnDeviceTraceEventReceiver* GetSingleton();

  // Returns non-zero version number if the there has a collector bound
  // to this receiver. Otherwise, returns zero. A version number is uniquely
  // assigned to each collector when it is bound to this receiver in the
  // StartWith() call.
  size_t ActiveVersion();

  // Return and increment the injection instance id if the version matches.
  int32_t StartInjectionInstance(size_t version);

  absl::Status Inject(size_t version, GpuOnDeviceTraceEvent&& event);

  // Return active version number if collector is successfully bound to this
  // receiver. Otherwise, error status. The max_injection_instance is the
  // maximum number of injection instances that the collector can handle.
  absl::StatusOr<size_t> StartWith(GpuOnDeviceTraceEventCollector* collector,
                                   int32_t max_injection_instance);

  absl::Status Stop();

 private:
  GpuOnDeviceTraceEventReceiver() = default;
  GpuOnDeviceTraceEventReceiver(const GpuOnDeviceTraceEventReceiver&) = delete;
  GpuOnDeviceTraceEventReceiver& operator=(
      const GpuOnDeviceTraceEventReceiver&) = delete;

  absl::Mutex mutex_;
  int32_t max_injection_instance_ ABSL_GUARDED_BY(mutex_) = 0;
  uint32_t current_injection_id_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t version_ ABSL_GUARDED_BY(mutex_) = 0;
  GpuOnDeviceTraceEventCollector* collector_ ABSL_GUARDED_BY(mutex_) = nullptr;
};

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_BACKENDS_GPU_ONDEVICE_EVENT_RECEIVER_H_
