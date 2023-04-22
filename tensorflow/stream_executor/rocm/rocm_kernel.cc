/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/gpu/gpu_kernel.h"

namespace stream_executor {
namespace gpu {

hipFuncCache_t GpuKernel::GetGpuCacheConfig() const {
  switch (preferred_cache_config_) {
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
                 << static_cast<int32>(preferred_cache_config_);
  }
}

}  // namespace gpu
}  // namespace stream_executor
