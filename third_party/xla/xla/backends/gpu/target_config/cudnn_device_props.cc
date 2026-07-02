/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/target_config/cudnn_device_props.h"

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#include "xla/stream_executor/cuda/cudnn_frontend_helpers.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

absl::StatusOr<std::shared_ptr<cudnn_frontend::DeviceProperties>>
BuildDeviceProperties(const stream_executor::DeviceDescription& desc) {
  const auto* cc = desc.gpu_compute_capability().cuda_compute_capability();
  int device_ver = cc ? (cc->major * 100 + cc->minor * 10) : 0;
  int driver_ver = desc.driver_version().major_version() * 1000 +
                   desc.driver_version().minor_version() * 10;
  std::string json = absl::StrFormat(
      R"json({
  "deviceVer":%d,
  "multiProcessorCount":%d,
  "warpSize":%d,
  "maxSharedMemoryPerBlock":%d,
  "maxSharedMemoryPerBlockOptin":%d,
  "reservedSharedMemoryPerBlock":%d,
  "maxRegistersPerSM":%d,
  "maxCtasPerSM":%d,
  "maxThreadsPerBlock":%d,
  "maxThreadsPerSM":%d,
  "regsPerBlock":%d,
  "totalGlobalMem":%d,
  "smClockRateKHz":%d,
  "l2CacheSize":%d,
  "maxBlockSize":[%d,%d,%d],
  "memClockRateKHz":%d,
  "maxGridSize":[%d,%d,%d],
  "supportCoopLaunch":%d,
  "pciDeviceId":0,
  "isTccDriver":0,
  "cudaDeviceId":0,
  "driverVer":%d,
  "deviceName":"%s"
})json",
      device_ver, desc.core_count(), desc.threads_per_warp(),
      desc.shared_memory_per_block(), desc.shared_memory_per_block_optin(),
      desc.reserved_shared_memory_per_block(), desc.registers_per_core_limit(),
      desc.max_blocks_per_multiprocessor(), desc.threads_per_block_limit(),
      desc.threads_per_core_limit(), desc.registers_per_block_limit(),
      desc.device_memory_size(),
      static_cast<int64_t>(desc.clock_rate_ghz() * 1e6), desc.l2_cache_size(),
      desc.thread_dim_limit().x, desc.thread_dim_limit().y,
      desc.thread_dim_limit().z,
      static_cast<int64_t>(desc.mem_clock_ghz() * 1e6),
      desc.block_dim_limit().x, desc.block_dim_limit().y,
      desc.block_dim_limit().z,
      /*desc.supports_coop_launch()*/ 1, driver_ver, desc.name());
  auto device_props = std::make_shared<cudnn_frontend::DeviceProperties>();
  RETURN_IF_CUDNN_FRONTEND_ERROR(device_props->deserialize(
      std::vector<uint8_t>(json.begin(), json.end())));
  return device_props;
}

}  // namespace xla::gpu
