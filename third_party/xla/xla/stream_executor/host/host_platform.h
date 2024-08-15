/* Copyright 2016 The OpenXLA Authors.

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
#ifndef XLA_STREAM_EXECUTOR_HOST_HOST_PLATFORM_H_
#define XLA_STREAM_EXECUTOR_HOST_HOST_PLATFORM_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
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

  const std::string& Name() const override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  absl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

 private:
  // Returns a device constructed with ordinal without
  // looking in or storing to the Platform's executor cache.
  // Ownership IS transferred to the caller.
  absl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      int ordinal);

  // This platform's name.
  std::string name_;

  // Cache of created StreamExecutors.
  ExecutorCache executor_cache_;

  HostPlatform(const HostPlatform&) = delete;
  void operator=(const HostPlatform&) = delete;
};

}  // namespace host
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_PLATFORM_H_
