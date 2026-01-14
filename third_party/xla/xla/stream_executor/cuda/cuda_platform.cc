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
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/nvml/include/nvml.h"
#include "xla/debug_options_flags.h"
#include "xla/stream_executor/cuda/cuda_diagnostics.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/cuda_memory_allocator.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla.pb.h"

namespace stream_executor {
namespace gpu {
namespace {

// Actually performs the work of CUDA initialization. Wrapped up in one-time
// execution guard.
static absl::Status InternalInit() {
  LOG(ERROR) << "InternalInit";
  absl::Status status =
      cuda::ToStatus(cuInit(0 /* = flags */), "Failed call to cuInit");
  if (!status.ok()) {
    LOG(ERROR) << "failed call to cuInit: " << status;
    cuda::Diagnostician::LogDiagnosticInformation();
    return status;
  }

  nvmlReturn_t init_result = nvmlInit();
  if (init_result != NVML_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("NVML init failed with ", init_result));
  }

  return absl::OkStatus();
}

static absl::Status PlatformInitialize() {
  LOG(ERROR) << "PlatformInitialize";
  // Cached return value from calling InternalInit(), as cuInit need only be
  // called once, but PlatformInitialize may be called many times.
  static absl::Status* initialization_status = [] {
    LOG(ERROR) << "PlatformInitialize: new absl::Status(InternalInit())";
    return new absl::Status(InternalInit());
  }();
  LOG(ERROR) << "PlatformInitialize: return *initialization_status"
             << *initialization_status;
  return *initialization_status;
}

}  // namespace

CudaPlatform::CudaPlatform() : name_("CUDA") {
  LOG(ERROR) << "CudaPlatform::CudaPlatform";
}

CudaPlatform::~CudaPlatform() {
  LOG(ERROR) << "CudaPlatform::~CudaPlatform";
  nvmlReturn_t shutdown_result = nvmlShutdown();
  if (shutdown_result != NVML_SUCCESS) {
    LOG(ERROR) << "NVML shutdown failed with " << shutdown_result;
  }
  LOG(ERROR) << "CudaPlatform::~CudaPlatform: done";
}

Platform::Id CudaPlatform::id() const { return cuda::kCudaPlatformId; }

int CudaPlatform::VisibleDeviceCount() const {
  LOG(ERROR) << "CudaPlatform::VisibleDeviceCount";
  // Initialized in a thread-safe manner the first time this is run.
  static const int num_devices = [] {
    LOG(ERROR) << "CudaPlatform::VisibleDeviceCount: new []";
    if (!PlatformInitialize().ok()) {
      LOG(ERROR)
          << "CudaPlatform::VisibleDeviceCount: -1!PlatformInitialize().o";
      return -1;
    }
    int device_count = 0;
    auto status = cuda::ToStatus(cuDeviceGetCount(&device_count));
    LOG(ERROR) << "CudaPlatform::VisibleDeviceCount: status: " << status
               << " device_count: " << device_count;
    if (!status.ok()) {
      LOG(ERROR) << "could not retrieve CUDA device count: " << status;
      return 0;
    }
    LOG(ERROR) << "CudaPlatform::VisibleDeviceCount: return device_count: "
               << device_count;

    return device_count;
  }();
  LOG(ERROR) << "CudaPlatform::VisibleDeviceCount: return num_devices: "
             << num_devices;
  return num_devices;
}

const std::string& CudaPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
CudaPlatform::DescriptionForDevice(int ordinal) const {
  TF_RETURN_IF_ERROR(PlatformInitialize());
  return CudaExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> CudaPlatform::ExecutorForDevice(int ordinal) {
  TF_RETURN_IF_ERROR(PlatformInitialize());
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<StreamExecutor*> CudaPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
CudaPlatform::GetUncachedExecutor(int ordinal) {
  // TODO(b/468297040): We should not be using DebugOptions here.
  xla::DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  auto executor = std::make_unique<CudaExecutor>(
      this, ordinal,
      debug_options.xla_gpu_experimental_enable_nvshmem()
          ? CollectiveAllocatorType::kNvshmem
          : CollectiveAllocatorType::kNccl);
  TF_RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}

}  // namespace gpu

static void InitializeCudaPlatform() {
  CHECK_OK(
      PlatformManager::RegisterPlatform(std::make_unique<gpu::CudaPlatform>()));
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    cuda_platform, stream_executor::InitializeCudaPlatform());
