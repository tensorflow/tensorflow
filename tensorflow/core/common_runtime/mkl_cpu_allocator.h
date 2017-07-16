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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_MKL_CPU_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_MKL_CPU_ALLOCATOR_H_

#ifdef INTEL_MKL

#include <string>
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mem.h"

#include "third_party/mkl/include/i_malloc.h"

namespace tensorflow {

class MklSubAllocator : public SubAllocator {
 public:
  ~MklSubAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    return port::AlignedMalloc(num_bytes, alignment);
  }
  void Free(void* ptr, size_t num_bytes) override { port::AlignedFree(ptr); }
};

/// CPU allocator for MKL that wraps BFC allocator and intercepts
/// and redirects memory allocation calls from MKL.
class MklCPUAllocator : public Allocator {
 public:
  // Constructor and other standard functions

  MklCPUAllocator() {
    VLOG(2) << "MklCPUAllocator: In MklCPUAllocator";
    allocator_ =
        new BFCAllocator(new MklSubAllocator, kMaxMemSize, kAllowGrowth, kName);

    // For redirecting all allocations from MKL to this allocator
    // From: http://software.intel.com/en-us/node/528565
    i_malloc = MallocHook;
    i_calloc = CallocHook;
    i_realloc = ReallocHook;
    i_free = FreeHook;
  }

  ~MklCPUAllocator() override { delete allocator_; }

  inline string Name() override { return kName; }

  inline void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return allocator_->AllocateRaw(alignment, num_bytes);
  }

  inline void DeallocateRaw(void* ptr) override {
    allocator_->DeallocateRaw(ptr);
  }

 private:
  // Hooks provided by this allocator for memory allocation routines from MKL

  static inline void* MallocHook(size_t size) {
    VLOG(2) << "MklCPUAllocator: In MallocHook";
    return cpu_allocator()->AllocateRaw(kAlignment, size);
  }

  static inline void FreeHook(void* ptr) {
    VLOG(2) << "MklCPUAllocator: In FreeHook";
    cpu_allocator()->DeallocateRaw(ptr);
  }

  static inline void* CallocHook(size_t num, size_t size) {
    Status s = Status(error::Code::UNIMPLEMENTED,
                      "Unimplemented case for hooking MKL function.");
    TF_CHECK_OK(s);  // way to assert with an error message
  }

  static inline void* ReallocHook(void* ptr, size_t size) {
    Status s = Status(error::Code::UNIMPLEMENTED,
                      "Unimplemented case for hooking MKL function.");
    TF_CHECK_OK(s);  // way to assert with an error message
  }

  // TODO(jbobba): We should ideally move this into CPUOptions in config.proto.
  /// Memory limit - 64GB
  static const size_t kMaxMemSize =
      static_cast<size_t>(64) * 1024 * 1024 * 1024;

  /// Do we allow growth in BFC Allocator
  static const bool kAllowGrowth = true;

  /// Name
  static constexpr const char* kName = "mklcpu";

  /// The alignment that we need for the allocations
  static const size_t kAlignment = 64;

  Allocator* allocator_;  // owned by this class
};

}  // namespace tensorflow

#endif  // INTEL_MKL

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_MKL_CPU_ALLOCATOR_H_
