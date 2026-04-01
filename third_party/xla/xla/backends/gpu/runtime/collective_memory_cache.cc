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

#include "xla/backends/gpu/runtime/collective_memory_cache.h"

#include <algorithm>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/tsl/util/tied_ref.h"

namespace xla::gpu {

void CollectiveMemoryCache::AddMulticastMemory(
    tsl::TiedRef<stream_executor::gpu::MulticastMemory> multicast_memory) {
  absl::MutexLock lock(mutex_);
  multicast_memories_.erase(
      std::remove_if(
          multicast_memories_.begin(), multicast_memories_.end(),
          [](tsl::TiedRef<stream_executor::gpu::MulticastMemory>&
                 multicast_memory) { return multicast_memory.Expired(); }),
      multicast_memories_.end());
  multicast_memories_.push_back(std::move(multicast_memory));
}

}  // namespace xla::gpu
