/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/platform/cpu_info.h"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace grappler {

int GetNumAvailableGPUs() {
  int num_eligible_gpus = 0;
#if GOOGLE_CUDA
  if (ValidateGPUMachineManager().ok()) {
    perftools::gputools::Platform* gpu_manager = GPUMachineManager();
    if (gpu_manager != nullptr) {
      int num_gpus = gpu_manager->VisibleDeviceCount();
      for (int i = 0; i < num_gpus; i++) {
        auto exec_status = gpu_manager->ExecutorForDevice(i);
        if (exec_status.ok()) {
          perftools::gputools::StreamExecutor* se = exec_status.ValueOrDie();
          const perftools::gputools::DeviceDescription& desc =
              se->GetDeviceDescription();
          int min_gpu_core_count = 8;
          if (desc.core_count() >= min_gpu_core_count) {
            num_eligible_gpus++;
          }
        }
      }
    }
  }
#endif  // GOOGLE_CUDA
  LOG(INFO) << "Number of eligible GPUs (core count >= 8): "
            << num_eligible_gpus;
  return num_eligible_gpus;
}

int GetNumAvailableLogicalCPUCores() { return port::NumSchedulableCPUs(); }

}  // end namespace grappler
}  // end namespace tensorflow
