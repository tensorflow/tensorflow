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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_

#include <memory>
#include <optional>
#include <string>

#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/bfc_allocator.h"
#include "xla/tsl/platform/macros.h"

namespace tensorflow {

// A GPU memory allocator that implements a 'best-fit with coalescing'
// algorithm.
class GPUBFCAllocator : public tsl::BFCAllocator {
 public:
  // See BFCAllocator::Options.
  struct Options {
    // Overridden by TF_FORCE_GPU_ALLOW_GROWTH if that envvar is set.
    bool allow_growth = false;

    // If nullopt, defaults to TF_ENABLE_GPU_GARBAGE_COLLECTION, or true if that
    // envvar is not present.
    //
    // Note:
    //
    //  - BFCAllocator defaults garbage_collection to false, not true.
    //  - this is not the same override behavior as TF_FORCE_GPU_ALLOW_GROWTH.
    std::optional<bool> garbage_collection;

    double fragmentation_fraction = 0;
    bool allow_retry_on_failure = true;
  };

  GPUBFCAllocator(std::unique_ptr<tsl::SubAllocator> sub_allocator,
                  size_t total_memory, const std::string& name,
                  const Options& opts);

  ~GPUBFCAllocator() override {}

  GPUBFCAllocator(const GPUBFCAllocator&) = delete;
  void operator=(const GPUBFCAllocator&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
