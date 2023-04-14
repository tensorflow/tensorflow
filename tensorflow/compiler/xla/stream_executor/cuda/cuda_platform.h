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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_PLATFORM_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_PLATFORM_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "tensorflow/compiler/xla/stream_executor/executor_cache.h"
#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_internal.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/compiler/xla/stream_executor/trace_listener.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace stream_executor {
namespace cuda {
// Opaque and unique identifier for the CUDA platform plugin.
// This is needed so that plugins can refer to/identify this platform without
// instantiating a CudaPlatform object.
extern const Platform::Id kCudaPlatformId;
}  // namespace cuda

namespace gpu {
// Cuda-specific platform plugin, registered as a singleton value via module
// initializer.
class CudaPlatform : public Platform {
 public:
  CudaPlatform();
  ~CudaPlatform() override;

  // CudaPlatform-specific functionality
  // Returns the number of distinct buses / NUMA nodes on the machine.
  int BusCount();

  // Returns the bus/NUMA node for the specified device ordinal.
  int DeviceToBus(int device_ordinal);

  // Returns the lowest-ordinal-number StreamExecutor on the specified bus.
  tsl::StatusOr<StreamExecutor*> FirstExecutorForBus(int bus_ordinal);

  // Platform interface implementation:
  // Returns the same value as kCudaPlatform above.
  Platform::Id id() const override;

  // Returns -1 as a sentinel on internal failure (and logs the error).
  int VisibleDeviceCount() const override;

  const std::string& Name() const override;

  tsl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  tsl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override {
    return ExecutorForDevice(ordinal, 0);
  }
  tsl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal,
                                                   int stream_id) override;

  tsl::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& config) override {
    return ExecutorForDeviceWithPluginConfig(ordinal, config, 0);
  }
  tsl::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& config, int stream_id) override;

  tsl::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override;

  tsl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) override;

  void RegisterTraceListener(std::unique_ptr<TraceListener> listener) override;

  void UnregisterTraceListener(TraceListener* listener) override;

 private:
  // Determines the number of NUMA nodes and the assignment of executor to each.
  void InspectNumaNodes();

  // This platform's name.
  std::string name_;

  // Cache of created executors.
  ExecutorCache executor_cache_;

  // The smallest NUMA node value for any device managed by this machine
  // manager. Used, along with limit_numa_node_, to convert NUMA nodes into bus
  // ordinals. The NUMA node space occupied by GPUs is assumed to be dense./
  int min_numa_node_;

  // Larger than the NUMA node value for any device managed by this machine
  // manager.
  int limit_numa_node_;

  SE_DISALLOW_COPY_AND_ASSIGN(CudaPlatform);
};

}  // namespace gpu

namespace cuda {

using CudaPlatform = gpu::CudaPlatform;

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_PLATFORM_H_
