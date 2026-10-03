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

#ifndef XLA_BACKENDS_GPU_RUNTIME_LOCK_FREE_KERNEL_CACHE_H_
#define XLA_BACKENDS_GPU_RUNTIME_LOCK_FREE_KERNEL_CACHE_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

namespace se = ::stream_executor;

// A thread-safe, lock-free read cache for kernel instances keyed by
// `se::StreamExecutor*`.
//
// Designed for execution hot paths (such as `GetKernelAndArgs` and steady-state
// `Initialize`) where acquiring a mutex introduces multi-GPU contention and
// cache line bouncing.
//
// Reads (`Find`) are completely lock-free via an atomic acquire load of an
// immutable snapshot followed by a linear scan over its small inlined vector of
// entries (see `Snapshot` below).
//
// Writes (`GetOrCreate`) synchronize via a mutex, perform double-checked
// locking, invoke the loader callback if the kernel is not yet loaded, and
// publish a new immutable snapshot with release semantics.
class LockFreeKernelCache {
 public:
  struct Entry {
    se::StreamExecutor* executor = nullptr;
    se::Kernel* kernel = nullptr;
  };

  // Up to 8 entries are stored inline, which avoids heap allocation for typical
  // multi-GPU-per-process configurations. Larger configurations are also
  // supported: `absl::InlinedVector` dynamically spills to the heap once the
  // inline capacity is exceeded.
  using Snapshot = absl::InlinedVector<Entry, 8>;

  LockFreeKernelCache() = default;
  ~LockFreeKernelCache() = default;

  LockFreeKernelCache(const LockFreeKernelCache&) = delete;
  LockFreeKernelCache& operator=(const LockFreeKernelCache&) = delete;
  LockFreeKernelCache(LockFreeKernelCache&&) = delete;
  LockFreeKernelCache& operator=(LockFreeKernelCache&&) = delete;

  // Lock-free lookup of a loaded kernel for `executor`.
  // Returns nullptr if not loaded or if `executor` is null.
  se::Kernel* Find(se::StreamExecutor* executor) const {
    if (executor == nullptr) {
      return nullptr;
    }
    const Snapshot* snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr) {
      return nullptr;
    }
    for (const Entry& entry : *snapshot) {
      if (entry.executor == executor) {
        return entry.kernel;
      }
    }
    return nullptr;
  }

  // Looks up the kernel for `executor`. If not present, locks `mutex_`, loads
  // the kernel via `loader`, stores the new kernel, and atomically publishes a
  // new snapshot. Returns InvalidArgumentError if `executor` is null.
  template <typename LoaderFn>
  absl::StatusOr<se::Kernel*> GetOrCreate(se::StreamExecutor* executor,
                                          LoaderFn&& loader) {
    if (executor == nullptr) {
      return absl::InvalidArgumentError("Executor must not be null");
    }

    // Fast path: lock-free check in steady-state (steps > 0).
    if (se::Kernel* kernel = Find(executor)) {
      return kernel;
    }

    absl::MutexLock lock(mutex_);

    // Double check under lock (step 0, in case another thread loaded it).
    const Snapshot* current = snapshot_.load(std::memory_order_relaxed);
    if (current != nullptr) {
      for (const Entry& entry : *current) {
        if (entry.executor == executor) {
          return entry.kernel;
        }
      }
    }

    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel, loader());
    se::Kernel* kernel_ptr = kernel.get();

    auto new_snapshot = std::make_unique<Snapshot>();
    if (current != nullptr) {
      *new_snapshot = *current;
    }
    new_snapshot->push_back(Entry{executor, kernel_ptr});

    // Retain ownership of kernel and snapshot across the lifetime of the cache.
    kernels_.push_back(std::move(kernel));
    const Snapshot* new_snapshot_ptr = new_snapshot.get();
    snapshots_.push_back(std::move(new_snapshot));

    // Atomically publish the new snapshot.
    snapshot_.store(new_snapshot_ptr, std::memory_order_release);

    return kernel_ptr;
  }

  // Returns true if the cache contains a kernel for `executor`.
  bool Contains(se::StreamExecutor* executor) const {
    return Find(executor) != nullptr;
  }

  bool contains(se::StreamExecutor* executor) const {
    return Contains(executor);
  }

  // Returns the number of loaded kernels.
  size_t size() const noexcept {
    const Snapshot* snapshot = snapshot_.load(std::memory_order_acquire);
    return snapshot != nullptr ? snapshot->size() : 0;
  }

  // Returns whether the cache is empty.
  bool empty() const noexcept { return size() == 0; }

 private:
  // Active immutable snapshot published to lock-free readers.
  std::atomic<const Snapshot*> snapshot_{nullptr};

  // Mutex protecting mutations on the cold path (step 0 initialization).
  absl::Mutex mutex_;

  // Master ownership of loaded kernels.
  std::vector<std::unique_ptr<se::Kernel>> kernels_ ABSL_GUARDED_BY(mutex_);

  // All allocated snapshots retained until cache destruction to guarantee
  // that concurrent lock-free readers never access dangling pointers.
  std::vector<std::unique_ptr<Snapshot>> snapshots_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_LOCK_FREE_KERNEL_CACHE_H_
