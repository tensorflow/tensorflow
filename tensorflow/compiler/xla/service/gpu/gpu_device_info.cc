/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"

namespace xla {
namespace gpu {

GpuDeviceInfo GetGpuDeviceInfo(stream_executor::StreamExecutor* stream_exec) {
  GpuDeviceInfo gpu_device_info;
  gpu_device_info.threads_per_block_limit =
      stream_exec->GetDeviceDescription().threads_per_block_limit();
  gpu_device_info.threads_per_warp =
      stream_exec->GetDeviceDescription().threads_per_warp();
  gpu_device_info.shared_memory_per_block =
      stream_exec->GetDeviceDescription().shared_memory_per_block();
  gpu_device_info.threads_per_core_limit =
      stream_exec->GetDeviceDescription().threads_per_core_limit();
  gpu_device_info.core_count = stream_exec->GetDeviceDescription().core_count();
  gpu_device_info.block_dim_limit_x =
      stream_exec->GetDeviceDescription().block_dim_limit().x;
  gpu_device_info.block_dim_limit_y =
      stream_exec->GetDeviceDescription().block_dim_limit().y;
  gpu_device_info.block_dim_limit_z =
      stream_exec->GetDeviceDescription().block_dim_limit().z;
  return gpu_device_info;
}

}  // namespace gpu
}  // namespace xla
