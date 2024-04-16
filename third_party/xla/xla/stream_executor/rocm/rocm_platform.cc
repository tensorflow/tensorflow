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
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "tsl/platform/errors.h"

namespace stream_executor {
namespace gpu {

ROCmPlatform::ROCmPlatform()
    : name_("ROCM"), min_numa_node_(0), limit_numa_node_(0) {}

ROCmPlatform::~ROCmPlatform() {}

// Due to legacy issues in user code, we can't currently call InpectNumaNodes
// at module initialization time, because non-GPU programs still include this
// plugin via various methods, so instead, it has to be init-on-reference.
void ROCmPlatform::InspectNumaNodes() {
  // To get NUMA node information, we need to create all executors, so we can
  // examine their device descriptions to see their bus assignments.
  absl::once_flag once;
  absl::call_once(once, [&] {
    StreamExecutorConfig config;
    for (int i = 0; i < VisibleDeviceCount(); i++) {
      config.ordinal = i;
      StreamExecutor* exec = GetExecutor(config).value();
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

int ROCmPlatform::BusCount() {
  InspectNumaNodes();
  return limit_numa_node_ - min_numa_node_;
}

int ROCmPlatform::DeviceToBus(int device_ordinal) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  StreamExecutor* exec = GetExecutor(config).value();
  return exec->GetDeviceDescription().numa_node() - min_numa_node_;
}

absl::StatusOr<StreamExecutor*> ROCmPlatform::FirstExecutorForBus(
    int bus_ordinal) {
  InspectNumaNodes();
  CHECK_LT(bus_ordinal, BusCount()) << "bus ordinal out of available range";
  for (int i = 0; i < VisibleDeviceCount(); i++) {
    if (DeviceToBus(i) == bus_ordinal) {
      StreamExecutorConfig config;
      config.ordinal = i;
      return GetExecutor(config).value();
    }
  }

  return absl::Status{
      absl::StatusCode::kNotFound,
      absl::StrFormat("Executor for bus %d not found.", bus_ordinal)};
}

Platform::Id ROCmPlatform::id() const { return rocm::kROCmPlatformId; }

int ROCmPlatform::VisibleDeviceCount() const {
  // Throw away the result - it logs internally, and this [containing] function
  // isn't in the path of user control. It's safe to call this > 1x.

  if (!gpu::GpuDriver::Init().ok()) {
    return -1;
  }

  return GpuDriver::GetDeviceCount();
}

const string& ROCmPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
ROCmPlatform::DescriptionForDevice(int ordinal) const {
  return GpuExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> ROCmPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  return GetExecutor(config);
}

absl::StatusOr<StreamExecutor*> ROCmPlatform::GetExecutor(
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
ROCmPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = std::make_unique<StreamExecutor>(
      this, std::make_unique<GpuExecutor>(config.ordinal));
  auto init_status = executor->Init();
  if (!init_status.ok()) {
    return absl::Status{
        absl::StatusCode::kInternal,
        absl::StrFormat(
            "failed initializing StreamExecutor for ROCM device ordinal %d: %s",
            config.ordinal, init_status.ToString().c_str())};
  }

  return std::move(executor);
}

}  // namespace gpu

static void InitializeROCmPlatform() {
  // Disabling leak checking, PlatformManager does not destroy its
  // registered platforms.
  auto status = PlatformManager::PlatformWithName("ROCM");
  if (!status.ok()) {
    std::unique_ptr<gpu::ROCmPlatform> platform(new gpu::ROCmPlatform);
    TF_CHECK_OK(PlatformManager::RegisterPlatform(std::move(platform)));
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    rocm_platform, stream_executor::InitializeROCmPlatform());
