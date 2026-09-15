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

#include "tensorflow/core/common_runtime/metal/metal_buffer_registry.h"

#include <cstddef>
#include <cstdint>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"

namespace tensorflow {
namespace metal {

MetalBufferRegistry& MetalBufferRegistry::Global() {
  static MetalBufferRegistry* registry = new MetalBufferRegistry();
  return *registry;
}

void* MetalBufferRegistry::Register(id<MTLBuffer> buffer) {
  if (buffer == nil) return nullptr;

  // `contents` is null for MTLResourceStorageModePrivate buffers. Refusing
  // them here keeps the invariant the whole backend relies on: every address
  // core sees is host-addressable, so pointer arithmetic on it is meaningful.
  void* address = [buffer contents];
  if (address == nullptr) {
    LOG(ERROR) << "Metal: refusing to register a buffer with no host-visible "
                  "contents; the Metal backend requires "
                  "MTLResourceStorageModeShared allocations.";
    return nullptr;
  }

  const size_t size = [buffer length];
  [buffer retain];

  absl::MutexLock lock(&mu_);
  auto [it, inserted] =
      allocations_.emplace(reinterpret_cast<uintptr_t>(address),
                           Entry{buffer, size});
  if (!inserted) {
    // Metal must not hand back an address that is still live; if it does, our
    // model of the address space is wrong and silently overwriting the entry
    // would leak the previous buffer and corrupt later lookups.
    LOG(ERROR) << "Metal: duplicate registration for address " << address;
    [buffer release];
    return nullptr;
  }

  stats_.num_allocs++;
  stats_.bytes_in_use += static_cast<int64_t>(size);
  if (stats_.bytes_in_use > stats_.peak_bytes_in_use) {
    stats_.peak_bytes_in_use = stats_.bytes_in_use;
  }
  if (static_cast<int64_t>(size) > stats_.largest_alloc_size) {
    stats_.largest_alloc_size = static_cast<int64_t>(size);
  }
  return address;
}

bool MetalBufferRegistry::Lookup(const void* address, id<MTLBuffer>* buffer,
                                 size_t* offset) const {
  if (address == nullptr) return false;
  const uintptr_t key = reinterpret_cast<uintptr_t>(address);

  absl::MutexLock lock(&mu_);
  // Greatest base address <= key, then a range check. This is what makes
  // interior pointers, the ones the BFC allocator and kernels actually pass
  // around, resolvable back to a buffer.
  auto it = allocations_.upper_bound(key);
  if (it == allocations_.begin()) return false;
  --it;

  const uintptr_t base = it->first;
  if (key - base >= it->second.size) return false;

  if (buffer != nullptr) *buffer = it->second.buffer;
  if (offset != nullptr) *offset = static_cast<size_t>(key - base);
  return true;
}

bool MetalBufferRegistry::Unregister(void* address) {
  if (address == nullptr) return false;

  id<MTLBuffer> buffer = nil;
  {
    absl::MutexLock lock(&mu_);
    auto it = allocations_.find(reinterpret_cast<uintptr_t>(address));
    if (it == allocations_.end()) return false;
    buffer = it->second.buffer;
    stats_.bytes_in_use -= static_cast<int64_t>(it->second.size);
    allocations_.erase(it);
  }
  // Released outside the lock: -release can run arbitrary teardown and we do
  // not want it holding up concurrent allocations.
  [buffer release];
  return true;
}

MetalBufferRegistry::Stats MetalBufferRegistry::GetStats() const {
  absl::MutexLock lock(&mu_);
  return stats_;
}

}  // namespace metal
}  // namespace tensorflow
