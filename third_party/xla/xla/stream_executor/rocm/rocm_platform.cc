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

#include "xla/stream_executor/rocm/rocm_platform.h"

#include <memory>

#include "absl/base/call_once.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/rocm/rocm_executor.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "tsl/platform/errors.h"

namespace stream_executor {
namespace gpu {

ROCmPlatform::ROCmPlatform() : name_("ROCM") {}

Platform::Id ROCmPlatform::id() const { return rocm::kROCmPlatformId; }

int ROCmPlatform::VisibleDeviceCount() const {
  // Throw away the result - it logs internally, and this [containing] function
  // isn't in the path of user control. It's safe to call this > 1x.

  if (!gpu::GpuDriver::Init().ok()) {
    return -1;
  }

  return GpuDriver::GetDeviceCount();
}

const std::string& ROCmPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
ROCmPlatform::DescriptionForDevice(int ordinal) const {
  return RocmExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> ROCmPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<StreamExecutor*> ROCmPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
ROCmPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<RocmExecutor>(this, ordinal);
  TF_RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}

}  // namespace gpu

static void InitializeROCmPlatform() {
  auto status = PlatformManager::PlatformWithName("ROCM");
  if (!status.ok()) {
    TF_CHECK_OK(PlatformManager::RegisterPlatform(
        std::make_unique<gpu::ROCmPlatform>()));
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    rocm_platform, stream_executor::InitializeROCmPlatform());
