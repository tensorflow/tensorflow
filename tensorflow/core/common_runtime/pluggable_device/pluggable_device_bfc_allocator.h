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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_BFC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_BFC_ALLOCATOR_H_

#include <cstddef>

#include "xla/tsl/framework/allocator.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A PluggableDevice memory allocator that implements a 'best-fit with
// coalescing' algorithm
class PluggableDeviceBFCAllocator : public BFCAllocator {
 public:
  PluggableDeviceBFCAllocator(tsl::SubAllocator* sub_allocator,
                              size_t total_memory, const string& name,
                              bool force_memory_growth_requested);
  PluggableDeviceBFCAllocator(tsl::SubAllocator* sub_allocator,
                              size_t total_memory,
                              const GPUOptions& gpu_options, const string& name,
                              bool force_memory_growth_requested);
  ~PluggableDeviceBFCAllocator() override = default;

  PluggableDeviceBFCAllocator(const PluggableDeviceBFCAllocator&) = delete;
  void operator=(const PluggableDeviceBFCAllocator&) = delete;

 private:
  static bool GetAllowGrowthValue(const GPUOptions& gpu_options,
                                  bool force_memory_growth_requested);
  static bool GetGarbageCollectionValue();
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_BFC_ALLOCATOR_H_
