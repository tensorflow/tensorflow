/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_device_info_for_tests.h"

#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla {
namespace gpu {

stream_executor::DeviceDescription TestGpuDeviceInfo::RTXA6000DeviceInfo(
    stream_executor::GpuComputeCapability cc) {
  stream_executor::DeviceDescription b;
  b.set_gpu_compute_capability(cc);
  b.set_threads_per_block_limit(1024);
  b.set_threads_per_warp(32);
  b.set_shared_memory_per_block(48 * 1024);
  b.set_shared_memory_per_block_optin(99 * 1024);
  b.set_shared_memory_per_core(100 * 1024);
  b.set_threads_per_core_limit(1536);
  b.set_core_count(84);
  b.set_fpus_per_core(128);
  b.set_block_dim_limit_x(2'147'483'647);
  b.set_block_dim_limit_y(65535);
  b.set_block_dim_limit_z(65535);
  b.set_memory_bandwidth(768'096'000'000);
  b.set_l2_cache_size(6 * 1024 * 1024);
  b.set_clock_rate_ghz(1.410);
  b.set_device_memory_size(51'050'250'240);
  b.set_registers_per_core_limit(65536);
  b.set_registers_per_block_limit(65536);
  b.set_runtime_version(stream_executor::SemanticVersion{12, 4, 0});
  b.set_driver_version(stream_executor::SemanticVersion{12, 4, 0});
  return b;
}

stream_executor::DeviceDescription TestGpuDeviceInfo::RTXH100SXMDeviceInfo(
    stream_executor::GpuComputeCapability cc) {
  stream_executor::DeviceDescription b;
  b.set_gpu_compute_capability(cc);
  b.set_threads_per_block_limit(1024);
  b.set_threads_per_warp(32);
  b.set_shared_memory_per_block(48 * 1024);
  b.set_shared_memory_per_block_optin(227 * 1024);
  b.set_shared_memory_per_core(228 * 1024);
  b.set_threads_per_core_limit(2048);
  b.set_core_count(132);
  b.set_fpus_per_core(128);
  b.set_block_dim_limit_x(2'147'483'647);
  b.set_block_dim_limit_y(65535);
  b.set_block_dim_limit_z(65535);
  b.set_memory_bandwidth(3'352'320'000'000);
  b.set_l2_cache_size(50 * 1024 * 1024);
  b.set_clock_rate_ghz(1.98);
  b.set_device_memory_size(84'978'434'048);
  b.set_registers_per_core_limit(65536);
  b.set_registers_per_block_limit(65536);
  b.set_runtime_version(stream_executor::SemanticVersion{12, 4, 0});
  b.set_driver_version(stream_executor::SemanticVersion{12, 4, 0});
  return b;
}

stream_executor::DeviceDescription TestGpuDeviceInfo::AMDMI210DeviceInfo() {
  stream_executor::DeviceDescription b;
  b.set_gpu_compute_capability(
      stream_executor::RocmComputeCapability("gfx90a"));
  b.set_threads_per_block_limit(1024);
  b.set_threads_per_warp(64);
  b.set_shared_memory_per_block(64 * 1024);
  b.set_shared_memory_per_block_optin(0);
  b.set_shared_memory_per_core(64 * 1024);
  b.set_threads_per_core_limit(2048);
  b.set_core_count(104);
  b.set_fpus_per_core(128);
  b.set_block_dim_limit_x(2'147'483'647);
  b.set_block_dim_limit_y(2'147'483'647);
  b.set_block_dim_limit_z(2'147'483'647);
  b.set_memory_bandwidth(1'638'400'000'000);
  b.set_l2_cache_size(8 * 1024 * 1024);
  b.set_clock_rate_ghz(1.7);
  b.set_device_memory_size(67'628'957'696);
  b.set_runtime_version(stream_executor::SemanticVersion{6, 0, 0});
  b.set_driver_version(stream_executor::SemanticVersion{6, 0, 0});
  return b;
}

stream_executor::DeviceDescription TestGpuDeviceInfo::AMDRX7900DeviceInfo() {
  stream_executor::DeviceDescription b;
  b.set_gpu_compute_capability(
      stream_executor::RocmComputeCapability("gfx1100"));
  return b;
}

stream_executor::DeviceDescription TestGpuDeviceInfo::CudaOrRocmDeviceInfo() {
#if defined(TENSORFLOW_USE_ROCM)
  return AMDMI210DeviceInfo();
#else
  return RTXA6000DeviceInfo();
#endif  // GOOGLE_CUDA
}

}  // namespace gpu
}  // namespace xla
