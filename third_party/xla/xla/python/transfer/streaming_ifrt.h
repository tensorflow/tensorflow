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
#include "xla/pjrt/raw_buffer.h"
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
  absl::AnyInvocable<xla::PjRtFuture<>(void* dst, int64_t offset,
                                       int64_t transfer_size)>
      copy_fn;
  size_t buffer_id;
  size_t offset;
  size_t size;

  static DmaCopyChunk Make(xla::ifrt::ArrayRef arr, xla::PjRtBuffer* buffer,
                           size_t buffer_id, size_t offset, size_t size) {
    return DmaCopyChunk{
        [arr, buffer](void* dst, int64_t offset,
                      int64_t transfer_size) -> xla::PjRtFuture<> {
          return buffer->CopyRawToHost(dst, offset, transfer_size);
        },
        buffer_id, offset, size};
  }

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
      DmaCopyChunk blob,
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

// A PullTable::Entry impl for a list of raw_buffer + ready_future.
class RawBufferEntry : public PullTable::Entry {
 public:
  struct BufferRef {
    // TODO(parkers): Technically this should be a use-ref instead of a
    // ready_future + buffer, but there is no PJRT api for this.
    xla::PjRtFuture<> ready_future;
    tsl::RCReference<xla::PjRtRawBuffer> buffer;
    size_t buf_size;
  };

  explicit RawBufferEntry(std::vector<BufferRef> arrs,
                          std::shared_ptr<PremappedCopierState> state,
                          size_t xfer_size);
  bool Handle(tsl::RCReference<ConnectionState> state,
              const SocketTransferPullRequest& req,
              size_t base_req_id) override;

 private:
  absl::Mutex mu_;
  size_t num_consumed_bufs_ = 0;
  std::vector<BufferRef> arrs_;
  std::shared_ptr<PremappedCopierState> state_;
  size_t xfer_size_;
};

// A PullTable::Entry impl for a list of pjrt buffers.
class PjRtBufferEntry : public PullTable::Entry {
 public:
  struct BufferRef {
    std::shared_ptr<xla::PjRtBuffer> buffer;
    size_t buf_size;
  };
  explicit PjRtBufferEntry(std::vector<BufferRef> arrs,
                           std::shared_ptr<PremappedCopierState> state,
                           size_t xfer_size);
  bool Handle(tsl::RCReference<ConnectionState> state,
              const SocketTransferPullRequest& req,
              size_t base_req_id) override;

 private:
  absl::Mutex mu_;
  size_t num_consumed_bufs_ = 0;
  std::vector<BufferRef> arrs_;
  std::shared_ptr<PremappedCopierState> state_;
  size_t xfer_size_;
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
  auto DoWork(size_t value, T&& cb) -> absl::Status {
    bool is_last;
    {
      absl::MutexLock l(&mu_);
      if (is_poisoned_) {
        return absl::OkStatus();
      }
      guard_counter_ -= value;
      is_last = guard_counter_ == 0;
      if (is_last) {
        // Wait if we happen to slip in between guard_counter and counter.
        auto cond = [this, value]() { return counter_ == value; };
        mu_.Await(absl::Condition(&cond));
      }
    }
    auto cleanup = absl::MakeCleanup([&]() {
      absl::MutexLock l(&mu_);
      counter_ -= value;
    });
    return cb(is_last);
  }

  void Poison() {
    absl::MutexLock l(&mu_);
    is_poisoned_ = true;
    auto cond = [this]() { return counter_ == guard_counter_; };
    mu_.Await(absl::Condition(&cond));
  }

 private:
  absl::Mutex mu_;
  bool is_poisoned_ = false;
  ssize_t guard_counter_;
  ssize_t counter_;
};

}  // namespace internal
}  // namespace aux

#endif  // XLA_PYTHON_TRANSFER_STREAMING_IFRT_H_
