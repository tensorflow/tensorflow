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
#ifndef XLA_PYTHON_TRANSFER_STREAMING_IFRT_H_
#define XLA_PYTHON_TRANSFER_STREAMING_IFRT_H_

#include <atomic>
#include <deque>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/transfer/streaming.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace aux {

inline constexpr uint64_t kCpuPageSize = 4096;

// Maps a preallocated buffer into device memory and i
absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>> MapPjrtMemory(
    xla::ifrt::Client* client, void* data, size_t buffer_size,
    std::shared_ptr<void> owner);

absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>> AllocateAndMapPjrtMemory(
    xla::ifrt::Client* client, size_t buffer_size);

// An structure which represents a single copy of a chunk out of a buffer
// with an assigned 'buffer_id'.
struct DmaCopyChunk {
  xla::ifrt::ArrayRef arr;
  xla::PjRtBuffer* buffer;
  size_t buffer_id;
  size_t offset;
  size_t size;

  // Divides an IFRT array up evenly for copying.
  static absl::StatusOr<std::vector<DmaCopyChunk>> DivideBufferCopiesEvenly(
      xla::ifrt::ArrayRef arr, size_t xfer_size, size_t buffer_id);
};

// Copies into subdivisions of scratch asyncly in parallel calling on_done
// sequentially when the copy has finished.
class PremappedCopierState {
 public:
  PremappedCopierState(std::shared_ptr<absl::Span<uint8_t>> scratch,
                       size_t max_num_parallel_copies, size_t xfer_size);
  struct WorkQueueItem {
    DmaCopyChunk work;
    void* dest_buffer;
    size_t seq_id;
    bool is_ready;
    absl::AnyInvocable<void(PremappedCopierState* state, void* buf,
                            const DmaCopyChunk& chunk) &&>
        on_done;
  };
  using WorkList = absl::InlinedVector<WorkQueueItem*, 8>;
  // on_done callback must schedule a call to ReturnBuffer at some point in the
  // future. Since on_done can be called from the TPU thread, avoid doing any
  // serious work (or even calling ReturnBuffer).
  void ScheduleCopy(
      const DmaCopyChunk& blob,
      absl::AnyInvocable<void(PremappedCopierState* state, void* buf,
                              const DmaCopyChunk& chunk) &&>
          on_done);

  // Allows buffer to be reused.
  void ReturnBuffer(void* buffer);

 private:
  void StartWorkUnlocked(const WorkList& work_list);
  WorkList FindWorkLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void FlushReadyWorkItemsInOrder() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::Mutex mu_;
  size_t base_seq_id_ ABSL_GUARDED_BY(mu_) = 0;
  size_t read_seq_id_ ABSL_GUARDED_BY(mu_) = 0;
  size_t num_parallel_copies_ = 0;
  std::deque<WorkQueueItem> work_queue_ ABSL_GUARDED_BY(mu_);
  std::shared_ptr<absl::Span<uint8_t>> scratch_;
  size_t max_num_parallel_copies_;
  size_t xfer_size_;
  size_t max_copies_;
  std::vector<void*> available_copy_offsets_ ABSL_GUARDED_BY(mu_);
};

// Creates a ChunkDestination for a buffer_index of an
// AsyncHostToDeviceTransferManager.
tsl::RCReference<ChunkDestination> MakeDmaDestination(
    std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager> atm,
    int buffer_index, size_t transfer_size);

namespace internal {

// A semaphore which calls a callback with [false]*N + [true]
// for some sequence of values. Callbacks may happen async, but the
// last callback will wait for all previous decrements to finish.
class IsLastSemaphore {
 public:
  explicit IsLastSemaphore(size_t value)
      : guard_counter_(value), counter_(value) {}

  template <typename T>
  auto DoWork(size_t value, T&& cb) -> decltype(cb(false)) {
    bool is_last = guard_counter_.fetch_sub(value) - value == 0;
    if (is_last && counter_.fetch_sub(value) - value != 0) {
      // Wait if we happen to slip in between guard_counter and counter.
      absl::MutexLock l(&mu_);
      auto cond = [this]() { return counter_.load() == 0; };
      mu_.Await(absl::Condition(&cond));
    }
    auto cleanup = absl::MakeCleanup([&]() {
      if (!is_last && (counter_.fetch_sub(value) - value) == 0) {
        // Wake any waiters.
        absl::MutexLock l(&mu_);
      }
    });
    return cb(is_last);
  }

 private:
  absl::Mutex mu_;
  std::atomic<ssize_t> guard_counter_;
  std::atomic<ssize_t> counter_;
};

}  // namespace internal
}  // namespace aux

#endif  // XLA_PYTHON_TRANSFER_STREAMING_IFRT_H_
