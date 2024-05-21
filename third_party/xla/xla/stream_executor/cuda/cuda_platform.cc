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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "tsl/platform/status.h"

namespace stream_executor {
namespace gpu {

CudaPlatform::CudaPlatform()
    : name_("CUDA"), min_numa_node_(0), limit_numa_node_(0) {}

CudaPlatform::~CudaPlatform() {}

// Due to legacy issues in user code, we can't currently call InpectNumaNodes
// at module initialization time, because non-GPU programs still include this
// plugin via various methods, so instead, it has to be init-on-reference.
void CudaPlatform::InspectNumaNodes() {
  // To get NUMA node information, we need to create all executors, so we can
  // examine their device descriptions to see their bus assignments.
  static absl::once_flag once;
  absl::call_once(once, [&] {
    for (int i = 0; i < VisibleDeviceCount(); i++) {
      StreamExecutor* exec = *ExecutorForDevice(i);
      if (i == 0) {
        // NUMA nodes may not start at 0, so set the minimum node  based on the
        // first executor we see.
        min_numa_node_ = exec->GetDeviceDescription().numa_node();
        limit_numa_node_ = min_numa_node_ + 1;
      } else {
        min_numa_node_ =
            std::min(min_numa_node_, exec->GetDeviceDescription().numa_node());
        limit_numa_node_ = std::max(
            limit_numa_node_, exec->GetDeviceDescription().numa_node() + 1);
      }
    }
  });
}

int CudaPlatform::BusCount() {
  InspectNumaNodes();
  return limit_numa_node_ - min_numa_node_;
}

int CudaPlatform::DeviceToBus(int device_ordinal) {
  StreamExecutor* exec = *ExecutorForDevice(device_ordinal);
  return exec->GetDeviceDescription().numa_node() - min_numa_node_;
}

absl::StatusOr<StreamExecutor*> CudaPlatform::FirstExecutorForBus(
    int bus_ordinal) {
  InspectNumaNodes();
  CHECK_LT(bus_ordinal, BusCount()) << "bus ordinal out of available range";
  for (int i = 0; i < VisibleDeviceCount(); i++) {
    if (DeviceToBus(i) == bus_ordinal) {
      return *ExecutorForDevice(i);
    }
  }

  return absl::NotFoundError(
      absl::StrFormat("Executor for bus %d not found.", bus_ordinal));
}

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

absl::StatusOr<StreamExecutor*> CudaPlatform::ExecutorForDeviceAndStream(
    int ordinal, int stream_id) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.stream_id = stream_id;
  return GetExecutor(config);
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
  auto executor =
      std::make_unique<GpuExecutor>(this, config.ordinal, config.stream_id);
  auto init_status = executor->Init();
  if (!init_status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "failed initializing StreamExecutor for CUDA device "
        "ordinal %d stream group %d: %s",
        config.ordinal, config.stream_id, init_status.ToString()));
  }

  return std::move(executor);
}

}  // namespace gpu

static void InitializeCudaPlatform() {
  // Disabling leak checking, PlatformManager does not destroy its
  // registered platforms.

  std::unique_ptr<gpu::CudaPlatform> platform(new gpu::CudaPlatform);
  TF_CHECK_OK(PlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    cuda_platform, stream_executor::InitializeCudaPlatform());
