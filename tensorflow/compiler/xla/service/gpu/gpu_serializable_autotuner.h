/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_SERIALIZABLE_AUTOTUNER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_SERIALIZABLE_AUTOTUNER_H_

#include <string>
#include <variant>

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// An abstract class for specifying gpu autotuner for both device and
// deviceless configs.
class GpuSerializableAutotuner : public HloModulePass {
 public:
  // TODO(b/267776674): Move shared functions such as WriteAutotuneResults and
  // LoadAutotuneResults into this class.

  struct DeviceConfig {
    se::StreamExecutor* stream_exec;  // never null

    // If the `allocator` parameter is not null, we will use it to allocate temp
    // memory while timing the various convolution algorithms.  If it's null,
    // we'll use the default allocator on the StreamExecutor.
    se::DeviceMemoryAllocator* allocator;  // may be null
  };

  struct DevicelessConfig {
    // The human-readable description of the device.  It can be found by using
    // stream_exec->GetDeviceDescription().model_str() when the stream executor
    // is available.
    std::string model_str;

    // A field to determine the architecture of the device. We only pick an
    // algorithm for non-Ampere architectures.
    se::CudaComputeCapability cuda_compute_capability{0, 0};
  };

  explicit GpuSerializableAutotuner(DeviceConfig config) : config_(config) {}
  explicit GpuSerializableAutotuner(DevicelessConfig config)
      : config_(config) {}

 protected:
  std::variant<DeviceConfig, DevicelessConfig> config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_SERIALIZABLE_AUTOTUNER_H_
