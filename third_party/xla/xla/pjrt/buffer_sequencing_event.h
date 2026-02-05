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

#ifndef XLA_PJRT_BUFFER_SEQUENCING_EVENT_H_
#define XLA_PJRT_BUFFER_SEQUENCING_EVENT_H_

#include <cstdint>
#include <functional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/event_pool.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

// A BufferSequencingEvent keeps track of dependencies of a buffer on each
// stream it has been used on.
//
// Each logical buffer in an XLA computation may be defined (i.e., written to)
// at most once. We call the operation that writes the buffer's value on some
// stream (e.g., a transfer or compute kernel) the buffer's definition event.
//
// After the operation that populates the value of a buffer has been enqueued on
// 'stream', SetSequencingEvent() should also be called to trigger the
// definition event after the operation has completed.
//
// After the buffer is read on 'stream' another event should be added so that
// it is possible to sequence buffer donation after all reads have completed.
//
// Since different streams are not necessarily synchronized with one another,
// if we wish to consume the value of the buffer on a different stream, we
// should first call WaitForEventOnStream(stream), which add a cross-stream
// from 'stream' to the buffer's definition event, causing 'stream' to pause
// until the definition event has been triggered, if needed. Operations on
// 'stream' may then assume that the buffer is valid and its contents correspond
// to the desired buffer.
//
// The dependency logic caches the set of streams at the tail of which the
// definition event is known to have occurred; waiting for the same event on the
// same stream causes no additional waiting.
class BufferSequencingEvent : tsl::AsyncPayload::KeepOnError {
 public:
  struct EventState {
    // An event that is triggered when the content of one or more buffers has
    // been read or written. If this event is used as a definition event and is
    // nullptr, it is assumed that the buffer's content is always defined for
    // example because it uses storage borrowed from elsewhere.
    EventPool::Handle event;

    se::Stream* definition_stream;
  };

  explicit BufferSequencingEvent(AsyncWorkRunner* async_work_runner)
      : async_work_runner_(async_work_runner),
        event_(tsl::MakeUnconstructedAsyncValueRef<EventState>()) {}

  explicit BufferSequencingEvent(AsyncWorkRunner* async_work_runner,
                                 tsl::AsyncValueRef<EventState> event)
      : async_work_runner_(async_work_runner), event_(event) {}

  static tsl::AsyncValueRef<BufferSequencingEvent> Create(
      AsyncWorkRunner* async_work_runner) {
    return tsl::MakeConstructedAsyncValueRef<BufferSequencingEvent>(
        async_work_runner);
  }

  // Sets the sequencing event to 'event', which is recorded on 'stream'. Must
  // be called at most once. Unblocks any other host threads that are blocked in
  // WaitForEventOnStream.
  // Do not call directly, use: PjRtStreamExecutorClient::AllocateAndRecordEvent
  // or PjRtStreamExecutorClient::ThenRecordEvent.
  void SetSequencingEvent(EventPool::Handle event, se::Stream* stream);

  // Adds synchronization events to 'stream' that wait for this event to be
  // defined on 'stream'. Does nothing if the event is already known to have
  // occurred by the tail of 'stream'. If SetSequencingEvent has not yet been
  // called, blocks the calling thread until the event has been recorded.
  void WaitForEventOnStream(se::Stream* stream);

  // Same as WaitForEventOnStream, but takes a raw platform-specific
  // stream. Currently on implemented for CUDA and ROCM GPU, where stream is a
  // GpuStreamHandle (e.g. a cudaStream_t).
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream);

  // Returns true if the event is known by the host to have already occurred. If
  // SetSequencingEvent has not yet been called, blocks the calling thread
  // until the event has been recorded.
  bool IsComplete();

  // Compares the sequence numbers of two recorded events. It is illegal to call
  // the comparison operators unless both events have been recorded.
  inline bool operator<(const BufferSequencingEvent& rhs) const {
    return sequence_number() < rhs.sequence_number();
  }
  inline bool operator>(const BufferSequencingEvent& rhs) const {
    return rhs < *this;
  }
  inline bool operator<=(const BufferSequencingEvent& rhs) const {
    return !(*this > rhs);
  }
  inline bool operator>=(const BufferSequencingEvent& rhs) const {
    return !(*this < rhs);
  }

  // Executes the `task` if the event is ready; otherwise adds the `task`
  // callback to `event_` async value, to be executed when it becomes
  // available.
  void ExecuteOrAddToFutureTasks(const std::string& task_name,
                                 std::function<void()> task);

  bool IsDefined() { return event_.IsAvailable(); }

  // Do not call directly. Use PjRtStreamExecutorClient::SetEventAsError.
  void SetDefinedStatus(absl::Status status);

  absl::Status GetDefinedStatus() {
    CHECK(event_.IsAvailable());
    if (const auto* error = event_.GetErrorIfPresent()) {
      return *error;
    }
    return absl::OkStatus();
  }

  bool IsPredeterminedError() { return event_.IsError(); }

  // Returns true if either:
  // 1. The event IsPredeterminedError
  // Or:
  // 2. The event is known to have occurred by the tail of 'stream'.
  // If SetSequencingEvent and SetDefinedStatus has not yet been called,
  // blocks the calling thread until either of those 2 happens.
  bool IsPredeterminedErrorOrDefinedOn(se::Stream* stream);

  se::Stream* definition_stream() const { return event_->definition_stream; }

  const tsl::AsyncValueRef<EventState>& event() { return event_; }

 private:
  uint64_t sequence_number() const;

  mutable absl::Mutex mu_;
  // A list of all streams for which the buffer's content is known to be defined
  // at the tail of the queue, i.e., for any newly enqueued command.
  absl::InlinedVector<se::Stream*, 2> streams_defined_on_ ABSL_GUARDED_BY(mu_);

  AsyncWorkRunner* async_work_runner_;

  // Indicates if the buffer is in an error status. And error status is used to
  // propagate the error to the buffer consumers.
  tsl::AsyncValueRef<EventState> event_;
};

using BufferSequencingEventRef = tsl::AsyncValueRef<BufferSequencingEvent>;

}  // namespace xla

#endif  // XLA_PJRT_BUFFER_SEQUENCING_EVENT_H_
