/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_PLATFORM_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_PLATFORM_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::sycl {

// SYCL-specific platform plugin, registered as a singleton value via module
// initializer.
class SyclPlatform : public Platform {
 public:
  SyclPlatform();

  ~SyclPlatform() override;

  // Platform interface implementation:
  // Returns the same value as kSyclPlatformId above.
  Platform::Id id() const override;

  // Returns the number of visible SYCL devices.
  // Returns -1 as a sentinel on internal failure (and logs the error).
  int VisibleDeviceCount() const override;

  // Returns the name of this platform.
  const std::string& Name() const override;

  // Returns a populated DeviceDescription for the device at the given ordinal.
  absl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  // Returns a cached StreamExecutor for the given ordinal, if available.
  // Otherwise creates and caches a new one. Ownership is not transferred to
  // the caller.
  absl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

  // Returns an existing StreamExecutor for the given ordinal if one exists.
  absl::StatusOr<StreamExecutor*> FindExisting(int ordinal) override;

 private:
  // Returns a device constructed with ordinal without
  // looking in or storing to the Platform's executor cache.
  // Ownership IS transferred to the caller.
  absl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      int ordinal);

  // This platform's name.
  std::string name_;

  // Cache of created executors.
  ExecutorCache executor_cache_;

  // Move-only: allow move, disallow copy.
  SyclPlatform(SyclPlatform&& other) = default;
  SyclPlatform& operator=(SyclPlatform&& other) = default;

  SyclPlatform(const SyclPlatform&) = delete;
  SyclPlatform& operator=(const SyclPlatform&) = delete;
};

}  // namespace stream_executor::sycl

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_PLATFORM_H_
