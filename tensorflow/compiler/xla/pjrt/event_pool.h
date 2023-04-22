/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_EVENT_POOL_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_EVENT_POOL_H_

#include <memory>
#include <stack>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace xla {

class EventPool {
 public:
  class Handle {
   public:
    Handle() = default;
    ~Handle();

    Handle(const Handle&) = delete;
    Handle(Handle&&) = default;
    Handle& operator=(const Handle&) = delete;
    Handle& operator=(Handle&&) = default;

    // There is a total order on events handed out by the event pool. The most
    // useful aspect of this total order is that two events returned by
    // ThenAllocateAndRecordEvent on the same stream can be compared to see
    // which was recorded earlier on that stream.
    // Valid sequence numbers are > 0.
    inline bool operator<(const Handle& rhs) const {
      return sequence_number_ < rhs.sequence_number_;
    }
    inline bool operator>(const Handle& rhs) const { return rhs < *this; }
    inline bool operator<=(const Handle& rhs) const { return !(*this > rhs); }
    inline bool operator>=(const Handle& rhs) const { return !(*this < rhs); }

    se::Event* event() const { return event_.get(); }
    uint64 sequence_number() const { return sequence_number_; }

   private:
    friend class EventPool;

    EventPool* pool_ = nullptr;
    std::unique_ptr<se::Event> event_;
    uint64 sequence_number_;
  };

  // Initializes a new EventPool. If `allow_reuse` is true, then events will be
  // returned to the pool when their handles are deleted and made available to
  // subsequent allocations. Reuse only works on the GPU platform.
  explicit EventPool(bool allow_reuse);

  // Allocates a new (or reused) event from the pool, and records the event on
  // `stream`.
  //
  // Reuse is only possible on GPU. Event allocation and recording are coupled
  // in a single operation because on GPU it is recording an event that makes it
  // a "new" event. According to the CUDA documentation it is safe to call
  // cudaEventRecord even if that event may still be in use on the device; APIs
  // such as cudaStreamWaitEvent capture the state of the event at the time of
  // the host-side call and are not affected by a later host-side
  // cudaEventRecord.
  StatusOr<Handle> ThenAllocateAndRecordEvent(se::Stream* stream);

  // Version of ThenAllocateAndRecordEvent split into two phases; this is
  // sometimes helpful if we want to avoid failures by preallocating events.
  StatusOr<Handle> AllocateEvent(se::StreamExecutor* executor);
  void ThenRecordEvent(se::Stream* stream, EventPool::Handle& handle);

 private:
  const bool allow_reuse_;

  absl::Mutex mu_;
  std::stack<std::unique_ptr<se::Event>> free_events_ TF_GUARDED_BY(mu_);
  uint64 next_sequence_number_ TF_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_EVENT_POOL_H_
