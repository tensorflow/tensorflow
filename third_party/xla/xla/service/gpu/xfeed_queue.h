/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_XFEED_QUEUE_H_
#define XLA_SERVICE_GPU_XFEED_QUEUE_H_

#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

// TODO(b/30467474) Once GPU outfeed implementation settles, consider
// folding back the cpu and gpu outfeed implementations into a generic
// one if possible.

// Manages a thread-safe queue of buffers.
template <typename BufferType>
class XfeedQueue {
 public:
  // Adds a tree of buffers to the queue. The individual buffers correspond to
  // the elements of a tuple and may be nullptr if the buffer is a tuple index
  // buffer.
  void EnqueueDestination(BufferType buffers) {
    absl::MutexLock l(&mu_);
    enqueued_buffers_.push_back(std::move(buffers));

    EnqueueHook();
  }

  // Blocks until the queue is non-empty, then returns the buffer at the head of
  // the queue.
  BufferType BlockingGetNextDestination() {
    for (const auto& callback : before_get_next_dest_callbacks_) {
      callback();
    }

    bool became_empty;
    BufferType current_buffer;
    {
      absl::MutexLock l(&mu_,
                        absl::Condition(this, &XfeedQueue::IsBufferEnqueued));
      current_buffer = std::move(enqueued_buffers_.front());
      enqueued_buffers_.pop_front();
      DequeueHook();
      became_empty = enqueued_buffers_.empty();
    }
    if (became_empty) {
      for (const auto& callback : on_empty_callbacks_) {
        callback();
      }
    }
    return current_buffer;
  }

  void RegisterOnEmptyCallback(std::function<void()> callback) {
    on_empty_callbacks_.push_back(std::move(callback));
  }
  void RegisterBeforeGetNextDestinationCallback(
      std::function<void()> callback) {
    before_get_next_dest_callbacks_.push_back(std::move(callback));
  }

  virtual ~XfeedQueue() = default;

 protected:
  virtual void DequeueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {}
  virtual void EnqueueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {}

  absl::Mutex mu_;

  // The queue of trees of buffers. Buffer* queue contents are not owned.
  std::deque<BufferType> enqueued_buffers_ ABSL_GUARDED_BY(mu_);

 private:
  // Returns true if there is a buffer in the queue.
  bool IsBufferEnqueued() const ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return !enqueued_buffers_.empty();
  }

  // List of callbacks which will be called when 'enqueued_buffers_' becomes
  // empty.
  std::vector<std::function<void()>> on_empty_callbacks_;

  // List of callbacks which will be called before BlockingGetNextDestination()
  // is called. This lets you e.g. call EnqueueDestination() for each call to
  // BlockingGetNextDestination().
  std::vector<std::function<void()>> before_get_next_dest_callbacks_;
};

// Like XfeedQueue but with a maximum capacity.  Clients can call
// `BlockUntilEnqueueSlotAvailable` to block until there are fewer than
// `max_pending_xfeeds_` capacity pending infeed items.
//
// We introduce a separate `BlockUntilEnqueueSlotAvailable` (as opposed to
// overriding `EnqueueDestination` to block) because we want to block before we
// copy the buffer to GPU memory, in order to bound the memory consumption due
// to pending infeeds.
template <typename BufferType>
class BlockingXfeedQueue : public XfeedQueue<BufferType> {
 public:
  explicit BlockingXfeedQueue(int max_pending_xfeeds)
      : max_pending_xfeeds_(max_pending_xfeeds) {}

  void BlockUntilEnqueueSlotAvailable() {
    absl::MutexLock l{
        &this->mu_,
        absl::Condition(this, &BlockingXfeedQueue::IsEnqueueSlotAvailable)};

    pending_buffers_++;
  }

 protected:
  void EnqueueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) override {
    pending_buffers_--;
  }

  void DequeueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) override {}

 private:
  const int max_pending_xfeeds_;

  bool IsEnqueueSlotAvailable() const ABSL_SHARED_LOCKS_REQUIRED(this->mu_) {
    VLOG(2) << "Capacity "
            << (pending_buffers_ + this->enqueued_buffers_.size())
            << " >= max capacity " << max_pending_xfeeds_;
    return pending_buffers_ + this->enqueued_buffers_.size() <
           max_pending_xfeeds_;
  }

  // Keeps track of the number of buffers reserved but not added to
  // enqueued_buffers_.
  int pending_buffers_ ABSL_GUARDED_BY(this->mu_) = 0;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_XFEED_QUEUE_H_
