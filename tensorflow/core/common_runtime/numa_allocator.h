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

// A simple CPU allocator that intercepts malloc/free calls from MKL library
// and redirects them to Tensorflow allocator

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NUMA_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NUMA_ALLOCATOR_H_

#include <cstdlib>
#include <string>
#include <numa.h>

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/visitable_allocator.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

class NumaSubAllocator : public SubAllocator {
 public:
  NumaSubAllocator(int node_id) : node_id_(node_id){
    if(numa_available() < 0) 
      std::cout << "The system does not support NUMA\n";
  }

  ~NumaSubAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    assert(alignment < 4098);
    return numa_alloc_onnode(num_bytes, node_id_);
  }
  void Free(void* ptr, size_t num_bytes) override 
  { 
    numa_free(ptr, num_bytes);
  }

  private:
   int node_id_;
};

class NumaAllocator : public VisitableAllocator {
 public:
  // Constructor and other standard functions

  /// Environment variable that user can set to upper bound on memory allocation
  static constexpr const char* kMaxLimitStr = "TF_MKL_ALLOC_MAX_BYTES";

  /// Default upper limit on allocator size - 64GB
  static const size_t kDefaultMaxLimit = 64LL << 30;

  NumaAllocator(int node_id) : node_id_(node_id) 
  { 
    TF_CHECK_OK(Initialize());
  }

  ~NumaAllocator() override { delete allocator_; }

  Status Initialize() {
    VLOG(2) << "NumaAllocator: In NumaAllocator";

    // Set upper bound on memory allocation to physical RAM available on the
    // CPU unless explicitly specified by user
    uint64 max_mem_bytes = kDefaultMaxLimit;
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    max_mem_bytes =
        (uint64)sysconf(_SC_PHYS_PAGES) * (uint64)sysconf(_SC_PAGESIZE);
#endif
    char* user_mem_bytes = getenv(kMaxLimitStr);

    if (user_mem_bytes != NULL) {
      uint64 user_val = 0;
      if (!strings::safe_strtou64(user_mem_bytes, &user_val)) {
        return errors::InvalidArgument("Invalid memory limit (", user_mem_bytes,
                                       ") specified for MKL allocator through ",
                                       kMaxLimitStr);
      }
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
      if (user_val > max_mem_bytes) {
        LOG(WARNING) << "The user specified a memory limit " << kMaxLimitStr
                     << "=" << user_val
                     << " greater than available physical memory: "
                     << max_mem_bytes
                     << ". This could significantly reduce performance!";
      }
#endif
      max_mem_bytes = user_val;
    }

    VLOG(1) << "NumaAllocator: Setting max_mem_bytes: " << max_mem_bytes;
    allocator_ = new BFCAllocator(new NumaSubAllocator(node_id_), max_mem_bytes,
                                  kAllowGrowth, kName);

    return Status::OK();
  }

  inline string Name() override { return kName; }

  inline void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return allocator_->AllocateRaw(alignment, num_bytes);
  }

  inline void DeallocateRaw(void* ptr) override {
    allocator_->DeallocateRaw(ptr);
  }

  void GetStats(AllocatorStats* stats) override { allocator_->GetStats(stats); }

  void ClearStats() override { allocator_->ClearStats(); }

  void AddAllocVisitor(Visitor visitor) override {
    allocator_->AddAllocVisitor(visitor);
  }

  void AddFreeVisitor(Visitor visitor) override {
    allocator_->AddFreeVisitor(visitor);
  }

 private:

  /// Do we allow growth in BFC Allocator
  static const bool kAllowGrowth = true;

  /// Name
  static constexpr const char* kName = "Numa";

  /// The alignment that we need for the allocations
  static const size_t kAlignment = 64;

  VisitableAllocator* allocator_;  // owned by this class

  int node_id_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NUMA_ALLOCATOR_H_
