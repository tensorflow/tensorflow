/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_THREADPOOL_LISTENER_H_
#define TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_THREADPOOL_LISTENER_H_

#include "absl/status/status.h"
#include "tsl/platform/tracing.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/backends/cpu/threadpool_listener_state.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
namespace tsl {
namespace profiler {

class ThreadpoolEventCollector : public tsl::tracing::EventCollector {
 public:
  explicit ThreadpoolEventCollector() = default;

  void RecordEvent(uint64 arg) const override;
  void StartRegion(uint64 arg) const override;
  void StopRegion() const override;

  // Annotates the current thread with a name.
  void SetCurrentThreadName(const char* name) {}
  // Returns whether event collection is enabled.
  static bool IsEnabled() { return threadpool_listener::IsEnabled(); }
};

class ThreadpoolProfilerInterface : public ProfilerInterface {
 public:
  explicit ThreadpoolProfilerInterface() = default;

  absl::Status Start() override;
  absl::Status Stop() override;

  absl::Status CollectData(tensorflow::profiler::XSpace* space) override;

 private:
  absl::Status status_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_THREADPOOL_LISTENER_H_
