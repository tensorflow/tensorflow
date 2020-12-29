/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/cuda/cuda_kernel.h"

namespace stream_executor {
namespace gpu {

CUfunc_cache GpuKernel::GetGpuCacheConfig() const {
  switch (preferred_cache_config_) {
    case KernelCacheConfig::kNoPreference:
      return CU_FUNC_CACHE_PREFER_NONE;
    case KernelCacheConfig::kPreferShared:
      return CU_FUNC_CACHE_PREFER_SHARED;
    case KernelCacheConfig::kPreferL1:
      return CU_FUNC_CACHE_PREFER_L1;
    case KernelCacheConfig::kPreferEqual:
      return CU_FUNC_CACHE_PREFER_EQUAL;
    default:
      LOG(FATAL) << "Unknown KernelCacheConfig"
                 << static_cast<int32>(preferred_cache_config_);
  }
}

}  // namespace gpu
}  // namespace stream_executor
