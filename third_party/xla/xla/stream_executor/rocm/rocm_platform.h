/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_PLATFORM_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_PLATFORM_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {

// Opaque and unique identifier for the ROCM platform plugin.
// This is needed so that plugins can refer to/identify this platform without
// instantiating a ROCmPlatform object.
extern const Platform::Id kROCmPlatformId;

// ROCm-specific platform plugin, registered as a singleton value via module
// initializer.
class ROCmPlatform : public Platform {
 public:
  ROCmPlatform();

  // Platform interface implementation:
  // Returns the same value as kROCmPlatform above.
  Platform::Id id() const override;

  // Returns -1 as a sentinel on internal failure (and logs the error).
  int VisibleDeviceCount() const override;

  const std::string& Name() const override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  absl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;
  absl::StatusOr<StreamExecutor*> FindExisting(int ordinal) override;

 private:
  // Returns a device constructed with ordinal without
  // looking in or storing to the Platform's executor cache.
  // Ownership IS transferred to the caller.
  absl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      int ordinal);

  // This platform's name.
  std::string name_;

  // mutex that guards internal state.
  mutable absl::Mutex mu_;

  // Cache of created executors.
  ExecutorCache executor_cache_;

  ROCmPlatform(const ROCmPlatform&) = delete;
  void operator=(const ROCmPlatform&) = delete;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_PLATFORM_H_
