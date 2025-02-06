/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_MANAGED_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_MANAGED_ALLOCATOR_H_

#include <string>

#include "xla/tsl/framework/allocator.h"

namespace tensorflow {

// An allocator for CUDA unified memory. Memory allocated with this allocator
// can be accessed from both host and device. CUDA transparently migrates dirty
// pages, which can be slow. Therefore, this allocator is intended for
// convenience in functional tests only.
class GpuManagedAllocator : public tsl::Allocator {
 public:
  std::string Name() override { return "GpuManagedAllocator"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_MANAGED_ALLOCATOR_H_
