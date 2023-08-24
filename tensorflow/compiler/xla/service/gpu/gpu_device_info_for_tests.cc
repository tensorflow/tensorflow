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

#include "tensorflow/compiler/xla/service/gpu/gpu_device_info_for_tests.h"

#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"

namespace xla {
namespace gpu {

GpuDeviceInfo TestGpuDeviceInfo::RTXA6000DeviceInfo() {
  GpuDeviceInfo info;
  info.name = "NVIDIA RTX A6000";
  info.threads_per_block_limit = 1024;
  info.threads_per_warp = 32;
  info.shared_memory_per_block = 48 * 1024;
  info.shared_memory_per_block_optin = 99 * 1024;
  info.shared_memory_per_core = 100 * 1024;
  info.threads_per_core_limit = 1536;
  info.core_count = 84;
  info.fpus_per_core = 128;
  info.block_dim_limit_x = 2'147'483'647;
  info.block_dim_limit_y = 65535;
  info.block_dim_limit_z = 65535;
  info.memory_bandwidth = 768'096'000'000;
  info.l2_cache_size = 6 * 1024 * 1024;
  info.clock_rate_ghz = 1.410;
  info.device_memory_size = 51'050'250'240;
  return info;
}

GpuDeviceInfo TestGpuDeviceInfo::AMDMI210DeviceInfo() {
  GpuDeviceInfo info;
  info.name = "AMD Instinct MI210";
  info.threads_per_block_limit = 1024;
  info.threads_per_warp = 64;
  info.shared_memory_per_block = 64 * 1024;
  info.shared_memory_per_block_optin = 0;
  info.shared_memory_per_core = 64 * 1024;
  info.threads_per_core_limit = 2048;
  info.core_count = 104;
  info.fpus_per_core = 0;
  info.block_dim_limit_x = 2'147'483'647;
  info.block_dim_limit_y = 2'147'483'647;
  info.block_dim_limit_z = 2'147'483'647;
  info.memory_bandwidth = 1'638'400'000'000;
  info.l2_cache_size = 8 * 1024 * 1024;
  info.clock_rate_ghz = 1.7;
  info.device_memory_size = 67'628'957'696;
  return info;
}

}  // namespace gpu
}  // namespace xla
