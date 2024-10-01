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
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace sycl {
// Opaque and unique identifier for the SYCL platform plugin.
// This is needed so that plugins can refer to/identify this platform without
// instantiating a SyclPlatform object.
extern const Platform::Id kSyclPlatformId;
}  // namespace sycl

namespace gpu {
// Sycl-specific platform plugin, registered as a singleton value via module
// initializer.
class SyclPlatform : public Platform {
 public:
  SyclPlatform();
  ~SyclPlatform() override;

  // Platform interface implementation:
  // Returns the same value as kSyclPlatform above.
  Platform::Id id() const override;

  // Returns -1 as a sentinel on internal failure (and logs the error).
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

  // Cache of created executors.
  ExecutorCache executor_cache_;

  SyclPlatform(const SyclPlatform&) = delete;
  void operator=(const SyclPlatform&) = delete;
};

}  // namespace gpu

namespace sycl {

using SyclPlatform = gpu::SyclPlatform;

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_PLATFORM_H_
