/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

bool GPUBFCAllocator::GetAllowGrowthValue(const GPUOptions& gpu_options) {
  const char* force_allow_growth_string =
      std::getenv("TF_FORCE_GPU_ALLOW_GROWTH");
  if (force_allow_growth_string == nullptr) {
    return gpu_options.allow_growth();
  }

  if (strcmp("false", force_allow_growth_string) == 0) {
    if (gpu_options.allow_growth()) {
      LOG(WARNING)
          << "Overriding allow_growth setting because the"
          << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
          << " config value was " << gpu_options.allow_growth() << ".";
    }
    return false;
  } else if (strcmp("true", force_allow_growth_string) == 0) {
    if (!gpu_options.allow_growth()) {
      LOG(WARNING)
          << "Overriding allow_growth setting because the"
          << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
          << " config value was " << gpu_options.allow_growth() << ".";
    }
    return true;
  }

  LOG(ERROR)
      << "The TF_FORCE_GPU_ALLOW_GROWTH environment variable is set but could"
      << " not be parsed: \"" << force_allow_growth_string << "\". Valid"
      << " values are \"true\" or \"false\". Using original config value"
      << " of " << gpu_options.allow_growth() << ".";
  return gpu_options.allow_growth();
}

GPUBFCAllocator::GPUBFCAllocator(GPUMemAllocator* sub_allocator,
                                 size_t total_memory, const string& name)
    : GPUBFCAllocator(sub_allocator, total_memory, GPUOptions(), name) {}

GPUBFCAllocator::GPUBFCAllocator(GPUMemAllocator* sub_allocator,
                                 size_t total_memory,
                                 const GPUOptions& gpu_options,
                                 const string& name)
    : BFCAllocator(sub_allocator, total_memory,
                   GPUBFCAllocator::GetAllowGrowthValue(gpu_options), name) {}

}  // namespace tensorflow
