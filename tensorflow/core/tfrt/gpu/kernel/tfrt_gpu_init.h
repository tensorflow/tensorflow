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
#ifndef TENSORFLOW_CORE_TFRT_GPU_KERNEL_TFRT_GPU_INIT_H_
#define TENSORFLOW_CORE_TFRT_GPU_KERNEL_TFRT_GPU_INIT_H_
#include "xla/tsl/framework/serving_device_selector_policies.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"

namespace tensorflow {
namespace gpu {

struct GpuRunnerOptions {
  int num_gpu_streams = 1;
  tsl::ServingDeviceSelectorPolicy serving_selector_policy =
      tsl::ServingDeviceSelectorPolicy::kRoundRobin;
};

Status InitTfrtGpu(const GpuRunnerOptions& options,
                   tensorflow::tfrt_stub::Runtime& runtime);

}  // namespace gpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GPU_KERNEL_TFRT_GPU_INIT_H_
