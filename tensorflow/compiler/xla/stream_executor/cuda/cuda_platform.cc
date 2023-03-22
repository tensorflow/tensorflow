/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_platform.h"

#include "absl/base/call_once.h"
#include "absl/base/const_init.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/platform/initialize.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"

namespace stream_executor {
namespace gpu {
namespace {

// Synchronize with spinlocks.
const char kScheduleSpinString[] = "spin";
// Synchronize with spinlocks that also call CPU yield instructions.
const char kScheduleYieldString[] = "yield";
// Synchronize with a "synchronization primitive" (e.g. mutex).
const char kScheduleBlockingSyncString[] = "blocking_sync";

const DeviceOptions GetDeviceOptionsFromEnv() {
  const char* gpu_schedule_string =
      std::getenv("TF_CUDA_PLATFORM_GPU_DEVICE_SCHEDULE");

  if (gpu_schedule_string == nullptr) {
    return DeviceOptions::Default();
  }

  unsigned device_flags = 0;
  if (strcmp(kScheduleSpinString, gpu_schedule_string) == 0) {
    device_flags = DeviceOptions::kScheduleSpin;
  } else if (strcmp(kScheduleYieldString, gpu_schedule_string) == 0) {
    device_flags = DeviceOptions::kScheduleYield;
  } else if (strcmp(kScheduleBlockingSyncString, gpu_schedule_string) == 0) {
    device_flags = DeviceOptions::kScheduleBlockingSync;
  } else {
    LOG(QFATAL) << "Unknown option for environment variable "
                   "TF_CUDA_PLATFORM_GPU_DEVICE_SCHEDULE "
                << gpu_schedule_string << " should be one of {"
                << kScheduleBlockingSyncString << ", " << kScheduleSpinString
                << ", " << kScheduleYieldString << "}";
  }

  return DeviceOptions(device_flags);
}

}  // namespace

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

tsl::StatusOr<StreamExecutor*> CudaPlatform::FirstExecutorForBus(
    int bus_ordinal) {
  InspectNumaNodes();
  CHECK_LT(bus_ordinal, BusCount()) << "bus ordinal out of available range";
  for (int i = 0; i < VisibleDeviceCount(); i++) {
    if (DeviceToBus(i) == bus_ordinal) {
      return *ExecutorForDevice(i);
    }
  }

  return tsl::Status(
      absl::StatusCode::kNotFound,
      absl::StrFormat("Executor for bus %d not found.", bus_ordinal));
}

Platform::Id CudaPlatform::id() const { return cuda::kCudaPlatformId; }

int CudaPlatform::VisibleDeviceCount() const {
  // Throw away the result - it logs internally, and this [containing] function
  // isn't in the path of user control. It's safe to call this > 1x.
  if (!gpu::GpuDriver::Init().ok()) {
    return -1;
  }

  return GpuDriver::GetDeviceCount();
}

const std::string& CudaPlatform::Name() const { return name_; }

tsl::StatusOr<std::unique_ptr<DeviceDescription>>
CudaPlatform::DescriptionForDevice(int ordinal) const {
  return GpuExecutor::CreateDeviceDescription(ordinal);
}

tsl::StatusOr<StreamExecutor*> CudaPlatform::ExecutorForDevice(
    int ordinal, int32 stream_id) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.stream_id = stream_id;
  config.plugin_config = PluginConfig();
  config.device_options = GetDeviceOptionsFromEnv();
  return GetExecutor(config);
}

tsl::StatusOr<StreamExecutor*> CudaPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config, int32 stream_id) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.stream_id = stream_id;
  config.plugin_config = plugin_config;
  config.device_options = GetDeviceOptionsFromEnv();
  return GetExecutor(config);
}

tsl::StatusOr<StreamExecutor*> CudaPlatform::GetExecutor(
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

tsl::StatusOr<std::unique_ptr<StreamExecutor>>
CudaPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = std::make_unique<StreamExecutor>(
      this, std::make_unique<GpuExecutor>(config.plugin_config),
      config.stream_id << 16 + config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return tsl::Status(
        absl::StatusCode::kInternal,
        absl::StrFormat(
            "failed initializing StreamExecutor for CUDA device ordinal %d: %s",
            config.ordinal, init_status.ToString()));
  }

  return std::move(executor);
}

void CudaPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register CUDA trace listener";
}

void CudaPlatform::UnregisterTraceListener(TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister CUDA trace listener";
}

}  // namespace gpu

static void InitializeCudaPlatform() {
  // Disabling leak checking, MultiPlatformManager does not destroy its
  // registered platforms.

  std::unique_ptr<gpu::CudaPlatform> platform(new gpu::CudaPlatform);
  TF_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(cuda_platform,
                            stream_executor::InitializeCudaPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(cuda_platform, multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     cuda_platform);
