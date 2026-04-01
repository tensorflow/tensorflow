/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_CACHE_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_CACHE_H_

#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/tsl/util/tied_ref.h"

namespace xla::gpu {

// Owns multicast and symmetric memories used by executable.
// This cache is needed to prevent destruction of the multicast objects and
// unregistering the symmetric memories before the executable is done using
// them. The class is thread-safe.
class CollectiveMemoryCache {
 public:
  void AddMulticastMemory(
      tsl::TiedRef<stream_executor::gpu::MulticastMemory> multicast_memory);

 private:
  absl::Mutex mutex_;
  std::vector<tsl::TiedRef<stream_executor::gpu::MulticastMemory>>
      multicast_memories_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_CACHE_H_
