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
  GpuDeviceInfo device_info;
  const stream_executor::DeviceDescription& d =
      stream_exec->GetDeviceDescription();
  device_info.threads_per_block_limit = d.threads_per_block_limit();
  device_info.threads_per_warp = d.threads_per_warp();
  device_info.shared_memory_per_block = d.shared_memory_per_block();
  device_info.shared_memory_per_core = d.shared_memory_per_core();
  device_info.threads_per_core_limit = d.threads_per_core_limit();
  device_info.core_count = d.core_count();
  device_info.fpus_per_core = d.fpus_per_core();
  device_info.block_dim_limit_x = d.block_dim_limit().x;
  device_info.block_dim_limit_y = d.block_dim_limit().y;
  device_info.block_dim_limit_z = d.block_dim_limit().z;
  device_info.memory_bandwidth = d.memory_bandwidth();
  device_info.l2_cache_size = d.l2_cache_size();
  device_info.clock_rate_ghz = d.clock_rate_ghz();
  return device_info;
}

}  // namespace gpu
}  // namespace xla
