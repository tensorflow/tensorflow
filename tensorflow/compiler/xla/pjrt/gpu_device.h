/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_GPU_DEVICE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_GPU_DEVICE_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/base/macros.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"

namespace xla {

class GpuDevice : public PjRtStreamExecutorDevice {
 public:
  GpuDevice(int id, std::unique_ptr<LocalDeviceState> local_device_state,
            std::string device_kind, std::string device_vendor, int node_id);

  absl::string_view device_vendor();

  std::string ToString() const override;

 private:
  std::string device_vendor_;
};

struct GpuAllocatorConfig {
  enum class Kind {
    kDefault,   // Client picks the best option for the platform.
    kPlatform,  // The platform's default.
    kBFC,  // Allocator using a "Best-Fit with Coalescing" algorithm. Currently
           // only available for GPU.
    kCudaAsync,  // Use the CUDA async allocator.
  };
  Kind kind = Kind::kDefault;

  // Only used if kind == kBFC. The maximum fraction of available memory to
  // allocate.
  double memory_fraction = 0.9;

  // Only used if kind == kBFC. If true, the allocator will immediately allocate
  // the maximum amount allowed by `memory_fraction`. This reduces
  // fragmentation, allowing more of the total memory to be used. If false, the
  // allocator will allocate more memory as allocations are requested.
  bool preallocate = true;
};

// distributed_client may be nullptr in non-distributed settings.
// distributed_client should be in the connected state before calling this
// function.
StatusOr<std::unique_ptr<PjRtClient>> GetGpuClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config,
    std::shared_ptr<DistributedRuntimeClient> distributed_client, int node_id,
    const std::optional<std::set<int>>& allowed_devices = std::nullopt,
    std::optional<std::string> platform_name = std::nullopt);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_GPU_DEVICE_H_
