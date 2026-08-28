/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_BUFFER_REGISTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_BUFFER_REGISTRY_H_

// This header is Objective-C++ only: it names `id<MTLBuffer>` directly and can
// only be included from a .mm translation unit.

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <map>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace tensorflow {
namespace metal {

// Maps device addresses back to the MTLBuffer that backs them.
//
// TensorFlow hands plugins an opaque `void*` in SP_DeviceMemoryBase and then
// treats it as a real address: the BFC allocator carves sub-allocations out of
// a region by pointer arithmetic, and kernels receive interior pointers. A
// plugin that returned an `id<MTLBuffer>` in that field would break the moment
// core added an offset to it.
//
// Apple Silicon is a unified memory architecture, so we can satisfy both
// contracts at once: allocate with MTLResourceStorageModeShared and hand core
// the buffer's `contents` pointer, which is a genuine CPU-addressable address
// in the same physical memory the GPU reads. Pointer arithmetic on it is valid.
// This registry is what lets us go the other way, recovering the (buffer,
// offset) pair a Metal encoder needs from an arbitrary interior pointer.
//
// A side effect worth naming: because the address is host-visible, host/device
// transfers degenerate into memcpy. That removes the copy overhead that
// dominates small-model performance on Mac, rather than merely amortizing it.
//
// All methods are thread-safe.
class MetalBufferRegistry {
 public:
  static MetalBufferRegistry& Global();

  // Takes ownership of one reference on `buffer` and returns the address core
  // should see. Returns nullptr if `buffer` is nil or not host-visible.
  void* Register(id<MTLBuffer> buffer);

  // Resolves an arbitrary address inside a registered allocation. On success
  // sets `*buffer` to the owning buffer (not retained, valid until the
  // matching Unregister) and `*offset` to the byte offset of `address` within
  // it. Returns false if `address` belongs to no live allocation.
  bool Lookup(const void* address, id<MTLBuffer>* buffer,
              size_t* offset) const;

  // Releases the allocation whose base address is exactly `address`. Returns
  // false if there is no such allocation. Interior pointers are rejected:
  // core always deallocates with the base address it was given.
  bool Unregister(void* address);

  // Snapshot of the counters SP_StreamExecutor::get_allocator_stats reports.
  struct Stats {
    int64_t num_allocs = 0;
    int64_t bytes_in_use = 0;
    int64_t peak_bytes_in_use = 0;
    int64_t largest_alloc_size = 0;
  };
  Stats GetStats() const;

 private:
  MetalBufferRegistry() = default;
  MetalBufferRegistry(const MetalBufferRegistry&) = delete;
  MetalBufferRegistry& operator=(const MetalBufferRegistry&) = delete;

  struct Entry {
    id<MTLBuffer> buffer;
    size_t size;
  };

  mutable absl::Mutex mu_;
  // Keyed by base address so that Lookup can binary-search for the allocation
  // containing an interior pointer.
  std::map<uintptr_t, Entry> allocations_ ABSL_GUARDED_BY(mu_);
  Stats stats_ ABSL_GUARDED_BY(mu_);
};

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_BUFFER_REGISTRY_H_
