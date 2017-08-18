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

#include "tensorflow/stream_executor/cuda/cuda_platform.h"

#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace perftools {
namespace gputools {
namespace cuda {

CudaPlatform::CudaPlatform()
    : name_("CUDA"), min_numa_node_(0), limit_numa_node_(0) {}

CudaPlatform::~CudaPlatform() {}

// Due to legacy issues in user code, we can't currently call InpectNumaNodes
// at module initialization time, because non-GPU programs still include this
// plugin via various methods, so instead, it has to be init-on-reference.
void CudaPlatform::InspectNumaNodes() {
  // To get NUMA node information, we need to create all executors, so we can
  // examine their device descriptions to see their bus assignments.
  static bool initialized = false;
  static mutex numa_mutex(LINKER_INITIALIZED);
  mutex_lock lock(numa_mutex);
  if (initialized) {
    return;
  }

  StreamExecutorConfig config;
  for (int i = 0; i < VisibleDeviceCount(); i++) {
    config.ordinal = i;
    StreamExecutor* exec = GetExecutor(config).ValueOrDie();
    if (i == 0) {
      // NUMA nodes may not start at 0, so set the minimum node  based on the
      // first executor we see.
      min_numa_node_ = exec->GetDeviceDescription().numa_node();
      limit_numa_node_ = min_numa_node_ + 1;
    } else {
      min_numa_node_ =
          std::min(min_numa_node_, exec->GetDeviceDescription().numa_node());
      limit_numa_node_ = std::max(limit_numa_node_,
                                  exec->GetDeviceDescription().numa_node() + 1);
    }
  }
  initialized = true;
}

int CudaPlatform::BusCount() {
  InspectNumaNodes();
  return limit_numa_node_ - min_numa_node_;
}

int CudaPlatform::DeviceToBus(int device_ordinal) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  StreamExecutor* exec = GetExecutor(config).ValueOrDie();
  return exec->GetDeviceDescription().numa_node() - min_numa_node_;
}

port::StatusOr<StreamExecutor*> CudaPlatform::FirstExecutorForBus(
    int bus_ordinal) {
  InspectNumaNodes();
  CHECK_LT(bus_ordinal, BusCount()) << "bus ordinal out of available range";
  for (int i = 0; i < VisibleDeviceCount(); i++) {
    if (DeviceToBus(i) == bus_ordinal) {
      StreamExecutorConfig config;
      config.ordinal = i;
      return GetExecutor(config).ValueOrDie();
    }
  }

  return port::Status{
      port::error::NOT_FOUND,
      port::Printf("Executor for bus %d not found.", bus_ordinal)};
}

Platform::Id CudaPlatform::id() const { return kCudaPlatformId; }

int CudaPlatform::VisibleDeviceCount() const {
  // Throw away the result - it logs internally, and this [containing] function
  // isn't in the path of user control. It's safe to call this > 1x.
  if (!cuda::CUDADriver::Init().ok()) {
    return -1;
  }

  return CUDADriver::GetDeviceCount();
}

const string& CudaPlatform::Name() const { return name_; }

port::StatusOr<StreamExecutor*> CudaPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> CudaPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> CudaPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
CudaPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = port::MakeUnique<StreamExecutor>(
      this, port::MakeUnique<CUDAExecutor>(config.plugin_config));
  auto init_status = executor->Init(config.ordinal, config.device_options);
  if (!init_status.ok()) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf(
            "failed initializing StreamExecutor for CUDA device ordinal %d: %s",
            config.ordinal, init_status.ToString().c_str())};
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

}  // namespace cuda

static void InitializeCudaPlatform() {
  // Disabling leak checking, MultiPlatformManager does not destroy its
  // registered platforms.
  
  std::unique_ptr<cuda::CudaPlatform> platform(new cuda::CudaPlatform);
  SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(cuda_platform,
                            perftools::gputools::InitializeCudaPlatform());

DECLARE_MODULE_INITIALIZER(multi_platform_manager);
// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(cuda_platform, multi_platform_manager);
