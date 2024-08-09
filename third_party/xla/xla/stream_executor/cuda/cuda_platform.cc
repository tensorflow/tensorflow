/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_platform.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace stream_executor {
namespace gpu {

CudaPlatform::CudaPlatform() : name_("CUDA") {}

CudaPlatform::~CudaPlatform() {}

Platform::Id CudaPlatform::id() const { return cuda::kCudaPlatformId; }

int CudaPlatform::VisibleDeviceCount() const {
  // Initialized in a thread-safe manner the first time this is run.
  static const int num_devices = [] {
    if (!GpuDriver::Init().ok()) return -1;
    return GpuDriver::GetDeviceCount();
  }();
  return num_devices;
}

const std::string& CudaPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
CudaPlatform::DescriptionForDevice(int ordinal) const {
  return GpuExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> CudaPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  return GetExecutor(config);
}

absl::StatusOr<StreamExecutor*> CudaPlatform::FindExisting(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  return executor_cache_.Get(config);
}

absl::StatusOr<StreamExecutor*> CudaPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  if (config.gpu_stream) {
    // If the GPU stream was provided, it's not possible to get-or-create a
    // stream with a required pointer: so we are looking for previously
    // allocated streams.
    return executor_cache_.Get(config);
  }
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
CudaPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = std::make_unique<GpuExecutor>(this, config.ordinal);
  TF_RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}

}  // namespace gpu

static void InitializeCudaPlatform() {
  TF_CHECK_OK(
      PlatformManager::RegisterPlatform(std::make_unique<gpu::CudaPlatform>()));
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    cuda_platform, stream_executor::InitializeCudaPlatform());
