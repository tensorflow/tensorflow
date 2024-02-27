/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/gpu_kernel.h"

namespace stream_executor {
namespace gpu {

hipFuncCache_t GpuKernel::GetGpuCacheConfig() const {
  switch (cache_config()) {
    case KernelCacheConfig::kNoPreference:
      return hipFuncCachePreferNone;
    case KernelCacheConfig::kPreferShared:
      return hipFuncCachePreferShared;
    case KernelCacheConfig::kPreferL1:
      return hipFuncCachePreferL1;
    case KernelCacheConfig::kPreferEqual:
      return hipFuncCachePreferEqual;
    default:
      LOG(FATAL) << "Unknown KernelCacheConfig"
                 << static_cast<int32>(cache_config());
  }
}

absl::StatusOr<int32_t> GpuKernel::GetMaxOccupiedBlocksPerCore(
    ThreadDim threads, size_t dynamic_shared_memory_bytes) const {
  int32_t threads_per_block = threads.x * threads.y * threads.z;
  VLOG(0) << "Get kernel block occupancy: " << name_
          << "; threads_per_block: " << threads_per_block
          << "; dynamic_shared_memory_bytes: " << dynamic_shared_memory_bytes;

  return GpuDriver::GetMaxOccupiedBlocksPerCore(gpu_context_, gpu_function_,
                                                threads_per_block,
                                                dynamic_shared_memory_bytes);
}

}  // namespace gpu
}  // namespace stream_executor
