/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XFEED_QUEUE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XFEED_QUEUE_H_

#include <deque>
#include <functional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/thread_annotations.h"

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
    tensorflow::mutex_lock l(mu_);
    enqueued_buffers_.push_back(std::move(buffers));
    enqueue_cv_.notify_one();

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
      tensorflow::mutex_lock l(mu_);
      while (enqueued_buffers_.empty()) {
        enqueue_cv_.wait(l);
      }
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

  virtual ~XfeedQueue() {}

 protected:
  virtual void DequeueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {}
  virtual void EnqueueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {}

  tensorflow::mutex mu_;

  // The queue of trees of buffers. Buffer* queue contents are not owned.
  std::deque<BufferType> enqueued_buffers_ ABSL_GUARDED_BY(mu_);

 private:
  // Condition variable that is signaled every time a buffer is enqueued.
  tensorflow::condition_variable enqueue_cv_;

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
    tensorflow::mutex_lock l{this->mu_};
    while (pending_buffers_ + this->enqueued_buffers_.size() >=
           max_pending_xfeeds_) {
      VLOG(2) << "Capacity "
              << (pending_buffers_ + this->enqueued_buffers_.size())
              << " >= max capacity " << max_pending_xfeeds_;
      dequeue_cv_.wait(l);
    }

    pending_buffers_++;
  }

 protected:
  void EnqueueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) override {
    pending_buffers_--;
  }

  void DequeueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) override {
    dequeue_cv_.notify_one();
  }

 private:
  const int max_pending_xfeeds_;

  // Condition variable that is signaled every time a buffer is dequeued.
  tensorflow::condition_variable dequeue_cv_;

  // Keeps track of the number of buffers reserved but not added to
  // enqueued_buffers_.
  int pending_buffers_ ABSL_GUARDED_BY(this->mu_) = 0;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XFEED_QUEUE_H_
