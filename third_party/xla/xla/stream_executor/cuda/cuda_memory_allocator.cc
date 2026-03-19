/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_memory_allocator.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// Per-process registry of collective allocator factories.
static absl::Mutex collective_allocators_mu(absl::kConstInit);
static absl::NoDestructor<
    absl::flat_hash_map<CollectiveAllocatorType, CollectiveAllocatorFactory>>
    collective_allocators ABSL_GUARDED_BY(collective_allocators_mu);

namespace {
// Instead of failing early we return a memory allocator that always fails when
// asked to allocate collective memory.
//
// TODO(patrios): We should fail early, but in open source builds something is
// wrong with linking order and allocators are not registered.
class NoCollectiveMemoryAllocator : public MemoryAllocator {
 public:
  explicit NoCollectiveMemoryAllocator(CollectiveAllocatorType allocator_type)
      : allocator_type_(allocator_type) {}

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) override {
    return absl::UnimplementedError(absl::StrCat(
        "No collective memory allocator registered for ", allocator_type_));
  }

 private:
  CollectiveAllocatorType allocator_type_;
};
}  // namespace

void RegisterCollectiveAllocatorFactory(
    CollectiveAllocatorType allocator_type,
    absl::AnyInvocable<std::unique_ptr<MemoryAllocator>(StreamExecutor*)>
        allocator_factory) {
  VLOG(1) << "Registering collective allocator factory for "
          << absl::StrCat(allocator_type);
  absl::MutexLock lock(collective_allocators_mu);
  collective_allocators->insert({allocator_type, std::move(allocator_factory)});
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
CreateCollectiveMemoryAllocator(StreamExecutor* executor,
                                CollectiveAllocatorType allocator_type) {
  absl::MutexLock lock(collective_allocators_mu);
  auto it = collective_allocators->find(allocator_type);
  if (it == collective_allocators->end()) {
    return std::make_unique<NoCollectiveMemoryAllocator>(allocator_type);
  }
  return it->second(executor);
}

}  // namespace stream_executor::gpu
