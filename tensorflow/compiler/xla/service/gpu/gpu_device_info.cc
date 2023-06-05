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
namespace {
GpuDeviceInfo GetGpuDeviceInfo(
    const stream_executor::DeviceDescription& device) {
  GpuDeviceInfo device_info;
  device_info.name = device.name();
  device_info.threads_per_block_limit = device.threads_per_block_limit();
  device_info.threads_per_warp = device.threads_per_warp();
  device_info.shared_memory_per_block = device.shared_memory_per_block();
  device_info.shared_memory_per_block_optin =
      device.shared_memory_per_block_optin();
  device_info.shared_memory_per_core = device.shared_memory_per_core();
  device_info.threads_per_core_limit = device.threads_per_core_limit();
  device_info.core_count = device.core_count();
  device_info.fpus_per_core = device.fpus_per_core();
  device_info.block_dim_limit_x = device.block_dim_limit().x;
  device_info.block_dim_limit_y = device.block_dim_limit().y;
  device_info.block_dim_limit_z = device.block_dim_limit().z;
  device_info.memory_bandwidth = device.memory_bandwidth();
  device_info.l2_cache_size = device.l2_cache_size();
  device_info.clock_rate_ghz = device.clock_rate_ghz();
  device_info.device_memory_size = device.device_memory_size();
  return device_info;
}

}  // namespace

GpuDeviceInfo GetGpuDeviceInfo(
    const stream_executor::StreamExecutor* stream_exec) {
  return GetGpuDeviceInfo(stream_exec->GetDeviceDescription());
}

GpuDeviceInfo GetGpuDeviceInfo(const stream_executor::Platform* platform) {
  auto device = platform->DescriptionForDevice(0);
  return GetGpuDeviceInfo(**device);
}

}  // namespace gpu
}  // namespace xla
