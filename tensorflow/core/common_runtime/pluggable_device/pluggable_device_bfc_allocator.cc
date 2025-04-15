/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.h"

#include <cstdlib>
#include <cstring>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "xla/tsl/framework/bfc_allocator.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

bool PluggableDeviceBFCAllocator::GetAllowGrowthValue(
    const GPUOptions& gpu_options, bool force_memory_growth_requested) {
  const char* force_allow_growth_string =
      std::getenv("TF_FORCE_GPU_ALLOW_GROWTH");
  if (force_allow_growth_string == nullptr) {
    if (force_memory_growth_requested && !gpu_options.allow_growth()) {
      LOG(WARNING) << "Overriding allow_growth setting because "
                      "force_memory_growth was requested by the device.";
      return true;
    }

    return gpu_options.allow_growth();
  }

  if (force_memory_growth_requested) {
    LOG(WARNING) << "Ignoring the value of TF_FORCE_GPU_ALLOW_GROWTH because "
                    "force_memory_growth was requested by the device.";
    return true;
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

bool PluggableDeviceBFCAllocator::GetGarbageCollectionValue() {
  const char* enable_gpu_garbage_collection =
      std::getenv("TF_ENABLE_GPU_GARBAGE_COLLECTION");
  if (enable_gpu_garbage_collection == nullptr) {
    // By default, turn on the memory garbage collection
    return true;
  }
  if (strcmp("false", enable_gpu_garbage_collection) == 0) {
    return false;
  } else if (strcmp("true", enable_gpu_garbage_collection) == 0) {
    return true;
  }

  LOG(ERROR)
      << "The TF_ENABLE_GPU_GARBAGE_COLLECTION environment variable is set but"
      << " could not be parsed: \"" << enable_gpu_garbage_collection << "\"."
      << " Valid values are \"true\" or \"false\"."
      << " Using the default value \"true\".";
  return true;
}

PluggableDeviceBFCAllocator::PluggableDeviceBFCAllocator(
    tsl::SubAllocator* sub_allocator, size_t total_memory, const string& name,
    bool force_memory_growth_requested)
    : PluggableDeviceBFCAllocator(sub_allocator, total_memory, GPUOptions(),
                                  name, force_memory_growth_requested) {}

PluggableDeviceBFCAllocator::PluggableDeviceBFCAllocator(
    tsl::SubAllocator* sub_allocator, size_t total_memory,
    const GPUOptions& gpu_options, const string& name,
    bool force_memory_growth_requested)
    : BFCAllocator(absl::WrapUnique(sub_allocator), total_memory, name, [&] {
        BFCAllocator::Options o;
        o.allow_growth = PluggableDeviceBFCAllocator::GetAllowGrowthValue(
            gpu_options, force_memory_growth_requested);
        o.garbage_collection =
            PluggableDeviceBFCAllocator::GetGarbageCollectionValue();
        return o;
      }()) {}

}  // namespace tensorflow
