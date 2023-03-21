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

namespace xla {
namespace gpu {

/*static*/ GpuDeviceInfo TestGpuDeviceInfo::RTXA6000DeviceInfo() {
  GpuDeviceInfo d;
  d.name = "NVIDIA RTX A6000";
  d.threads_per_block_limit = 1024;
  d.threads_per_warp = 32;
  d.shared_memory_per_block = 49152;
  d.shared_memory_per_core = 100 * 1024;
  d.threads_per_core_limit = 1536;
  d.core_count = 84;
  d.fpus_per_core = 128;
  d.block_dim_limit_x = 2'147'483'647;
  d.block_dim_limit_y = 65535;
  d.block_dim_limit_z = 65535;
  d.memory_bandwidth = 768'096'000'000;
  d.l2_cache_size = 6 * 1024 * 1024;
  d.clock_rate_ghz = 1.410;
  return d;
}

}  // namespace gpu
}  // namespace xla
