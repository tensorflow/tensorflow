/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Declares the "host" platform, which is a CPU-only implementation of the
// StreamExecutor. The host platform only supports memory operations and plugin
// routines, and is primarily used for testing.
#ifndef TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_PLATFORM_H_
#define TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_PLATFORM_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/trace_listener.h"

namespace perftools {
namespace gputools {
namespace host {

// Host (CPU) platform plugin, registered as a singleton value via module
// initializer.
class HostPlatform : public Platform {
 public:
  HostPlatform();
  ~HostPlatform() override;

  Platform::Id id() const override;

  // Device count is less clear-cut for CPUs than accelerators. This call
  // currently returns the number of thread units in the host, as reported by
  // base::NumCPUs().
  int VisibleDeviceCount() const override;

  const string& Name() const override;

  port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

  port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& config) override;

  port::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override;

  port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) override;

  void RegisterTraceListener(std::unique_ptr<TraceListener> listener) override;

  void UnregisterTraceListener(TraceListener* listener) override;

 private:
  // This platform's name.
  string name_;

  // mutex that guards the ordinal-to-executor map.
  mutable mutex executors_mutex_;

  // Cache of created StreamExecutors.
  ExecutorCache executor_cache_;

  SE_DISALLOW_COPY_AND_ASSIGN(HostPlatform);
};

}  // namespace host
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_PLATFORM_H_
