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

#include "xla/tsl/profiler/backends/gpu/ondevice_event_receiver.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/profiler/backends/gpu/ondevice_event_collector.h"
#include "xla/tsl/profiler/backends/gpu/ondevice_trace_event.h"

namespace tsl {
namespace profiler {

GpuOnDeviceTraceEventReceiver* GpuOnDeviceTraceEventReceiver::GetSingleton() {
  static GpuOnDeviceTraceEventReceiver* receiver =
      new GpuOnDeviceTraceEventReceiver();
  return receiver;
}

size_t GpuOnDeviceTraceEventReceiver::ActiveVersion() {
  absl::MutexLock lock(mutex_);
  return (collector_ != nullptr) ? version_ : 0;
}

absl::Status GpuOnDeviceTraceEventReceiver::Inject(
    size_t version, GpuOnDeviceTraceEvent&& event) {
  absl::MutexLock lock(mutex_);
  if (collector_ != nullptr && version == version_) {
    if (event.injection_instance_id > current_injection_id_ ||
        event.injection_instance_id <= 0) {
      return absl::InternalError("Injection instance id is out of range.");
    }
    return collector_->AddEvent(std::move(event));
  }
  return collector_ == nullptr
             ? absl::InternalError("Can not inject to nullptr collector.")
             : absl::InternalError("Inject with mismatched version!");
}

absl::StatusOr<size_t> GpuOnDeviceTraceEventReceiver::StartWith(
    GpuOnDeviceTraceEventCollector* collector, int32_t max_injection_instance) {
  if (collector == nullptr) {
    return absl::InternalError("Can not bind nullptr collector.");
  }
  if (max_injection_instance <= 0) {
    return absl::InternalError("Max injection instance must be positive.");
  }
  absl::MutexLock lock(mutex_);
  if (collector_ != nullptr && collector_ != collector) {
    return absl::InternalError(
        "GpuOnDeviceTraceEventReceiver already bind with another collector.");
  }
  if (collector_ == nullptr) {
    version_++;
    current_injection_id_ = 0;
    collector_ = collector;
    max_injection_instance_ = max_injection_instance;
  }
  return version_;
}

// Return and increment the injection instance id if successful.
int32_t GpuOnDeviceTraceEventReceiver::StartInjectionInstance(size_t version) {
  absl::MutexLock lock(mutex_);
  if (collector_ == nullptr || version != version_ ||
      current_injection_id_ >= max_injection_instance_) {
    return 0;
  }
  return ++current_injection_id_;
}

absl::Status GpuOnDeviceTraceEventReceiver::Stop() {
  absl::MutexLock lock(mutex_);
  if (collector_ != nullptr) {
    collector_ = nullptr;
    version_++;
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tsl
