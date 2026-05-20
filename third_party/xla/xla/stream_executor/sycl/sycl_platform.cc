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

#include "xla/stream_executor/sycl/sycl_platform.h"

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/sycl/sycl_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor {
namespace sycl {

SyclPlatform::SyclPlatform() : name_(kSyclPlatformId->ToName()) {}

SyclPlatform::~SyclPlatform() {}

Platform::Id SyclPlatform::id() const { return kSyclPlatformId; }

int SyclPlatform::VisibleDeviceCount() const {
  auto status = SyclDevicePool::GetDeviceCount();
  if (status.ok()) {
    return status.value();
  }
  LOG(ERROR) << "Failed to get device count: " << status;
  return -1;  // Return -1 as a sentinel on internal failure.
}

const std::string& SyclPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
SyclPlatform::DescriptionForDevice(int ordinal) const {
  return SyclExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> SyclPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<StreamExecutor*> SyclPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
SyclPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<SyclExecutor>(this, ordinal);
  TF_RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}

}  // namespace sycl

// Initializes and registers the SYCL platform if it is not already registered.
static void InitializeSyclPlatform() {
  auto status = PlatformManager::PlatformWithName("SYCL");
  if (!status.ok()) {
    TF_CHECK_OK(PlatformManager::RegisterPlatform(
        std::make_unique<sycl::SyclPlatform>()));
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    sycl_platform, stream_executor::InitializeSyclPlatform());
