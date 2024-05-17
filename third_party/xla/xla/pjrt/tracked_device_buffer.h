/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_PJRT_TRACKED_DEVICE_BUFFER_H_
#define XLA_PJRT_TRACKED_DEVICE_BUFFER_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "xla/pjrt/event_pool.h"
#include "xla/service/executable.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/threadpool.h"

namespace xla {

// A BufferSequencingEvent keeps track of dependencies of a buffer on each
// stream it has been used on.
//
// Each logical buffer in an XLA computation may be defined (i.e., written to)
// at most once. We call the operation that writes the buffer's value on some
// stream (e.g., a transfer or compute kernel) the buffer's definition event.
//
// After the operation that populates the value of a buffer has been enqueued on
// 'stream', RecordOnStream(stream) should also be called to trigger the
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
class BufferSequencingEvent {
 public:
  explicit BufferSequencingEvent(tsl::thread::ThreadPool* thread_pool)
      : thread_pool_(thread_pool),
        defined_status_(tsl::MakeUnconstructedAsyncValueRef<Status>()) {}

  // Sets the sequencing event to 'event', which is recorded on 'stream'. Must
  // be called at most once. Unblocks any other host threads that are blocked in
  // WaitForEventOnStream.
  void SetSequencingEvent(EventPool::Handle event, se::Stream* stream);

  // Adds synchronization events to 'stream' that wait for this event to be
  // defined on 'stream'. Does nothing if the event is already known to have
  // occurred by the tail of 'stream'. If RecordOnStream has not yet been
  // called, blocks the calling thread until the event has been recorded.
  void WaitForEventOnStream(se::Stream* stream);

  // Same as WaitForEventOnStream, but takes a raw platform-specific
  // stream. Currently on implemented for CUDA and ROCM GPU, where stream is a
  // GpuStreamHandle (e.g. a cudaStream_t).
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream);

  // Returns true if the event is known to have occurred by the tail of
  // 'stream'. If RecordOnStream has not yet been called, blocks the calling
  // thread until the event has been recorded.
  bool DefinedOn(se::Stream* stream);

  // Returns true if the event is known by the host to have already occurred. If
  // RecordOnStream has not yet been called, blocks the calling thread until the
  // event has been recorded.
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

  inline bool operator==(int number) const {
    return sequence_number() == number;
  }

  // Executes the `task` if the event is ready; otherwise adds the `task`
  // callback to `on_ready_tasks_callback_` that can not be executed until the
  // the event is ready.
  void ExecuteOrAddToFutureTasks(const std::string& task_name,
                                 std::function<void()> task);

  // Executes all the callbacks in `on_ready_tasks_callback_`. Those callbacks
  // can only proceed until the event is ready.
  void ExecuteFutureTasks();

  bool IsDefined() {
    absl::MutexLock lock(&mu_);
    return defined_status_.IsConcrete();
  }

  void SetDefinedStatus(absl::Status status) {
    {
      absl::MutexLock lock(&mu_);
      defined_status_.emplace(status);
    }

    this->ExecuteFutureTasks();
  }

  absl::Status GetDefinedStatus() {
    absl::MutexLock lock(&mu_);
    CHECK(defined_status_.IsConcrete());
    return defined_status_.get();
  }

  bool IsPredeterminedError() {
    absl::MutexLock lock(&mu_);
    return defined_status_.IsConcrete() && !defined_status_.get().ok();
  }

 private:
  bool EventHasBeenRecorded() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  uint64_t sequence_number() const;

  // An event that is triggered when the content of one or more buffers has been
  // read or written. If this event is used as a definition event and is
  // nullptr, it is assumed that the buffer's content is always defined for
  // example because it uses storage borrowed from elsewhere.
  EventPool::Handle event_;

  // Cache of event_->sequence_number that avoids synchronization overhead.
  // TODO(phawkins): In fact, event_->sequence_number is unused beyond the
  // initial population of sequence_number_, and we could remove it if we
  // refactored the EventPool API.
  std::atomic<uint64_t> sequence_number_{0};

  mutable absl::Mutex mu_;
  // A list of all streams for which the buffer's content is known to be defined
  // at the tail of the queue, i.e., for any newly enqueued command.
  absl::InlinedVector<se::Stream*, 2> streams_defined_on_ ABSL_GUARDED_BY(mu_);

  // A map of the task name and callback to execute when the
  // TrackedDeviceBuffer's `definition_events_` are all recorded and ready to be
  // consumed by other tasks.
  absl::flat_hash_map<std::string, std::function<void()>>
      on_ready_tasks_callback_ ABSL_GUARDED_BY(mu_);

  tsl::thread::ThreadPool* thread_pool_;

  // Indicates if the buffer is in an error status. And error status is used to
  // propagate the error to the buffer consumers.
  tsl::AsyncValueRef<absl::Status> defined_status_ ABSL_GUARDED_BY(mu_);
};

// Class that represents a tuple of device buffers. Like a ScopedShapedBuffer it
// owns all of the device memory in the tuple. It also tracks the definition and
// usage of the memory on streams, to allow for synchronized usage and deletion
// of memory under all of the allocation model semantics.
class TrackedDeviceBuffer {
 public:
  // Helper object to keep track of usage of the buffer on streams.
  struct StreamAndEvent {
    // A stream the buffer has been used on.
    se::Stream* stream;
    // An event that is later than the most recent usage of the buffer on
    // stream.
    std::shared_ptr<BufferSequencingEvent> event;
    // True if and only if a reference to the buffer is kept live until after
    // the host knows that event is complete.
    bool reference_held;
  };

  // Converts a ScopedShapedBuffer into a TrackedDeviceBuffer. Takes ownership
  // of the buffers of the shaped_buffer.
  static std::shared_ptr<TrackedDeviceBuffer> FromScopedShapedBuffer(
      ScopedShapedBuffer* shaped_buffer,
      absl::Span<const std::shared_ptr<BufferSequencingEvent>>
          definition_events);

  // Builds a ShapedBuffer view onto the buffers of 'tree'.
  ShapedBuffer AsShapedBuffer(const Shape& on_device_shape) const;

  // Adds the owned device buffers in order to 'iterator'. Used to add the
  // buffers to an ExecutionInput. We require but do not verify that 'iterator'
  // when passed in is pointing to a sub-tuple of the ExecutionInput whose
  // on_device_shape matches that of the TrackedDeviceBuffer. 'end' is used to
  // check that 'iterator' doesn't run out of bounds.
  void AddToInputAsImmutable(
      ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
      const ShapeTree<MaybeOwningDeviceMemory>::iterator& end) const;

  // Adds the owned device buffers in order to 'iterator', marking them as
  // available to be donated. If donation succeeds, i.e., execution_input is
  // subsequently successfully enqueued to a computation,
  // this->ReleaseDeviceMemory() must be called to avoid freeing the device
  // memory twice. We require but do not verify that 'iterator' when passed in
  // is pointing to a sub-tuple of execution_input whose on_device_shape matches
  // that of the TrackedDeviceBuffer. 'end' is used to check that 'iterator'
  // doesn't run out of bounds.
  void AddToInputAsDonated(
      ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
      const ShapeTree<MaybeOwningDeviceMemory>::iterator& end,
      ExecutionInput* execution_input,
      se::DeviceMemoryAllocator* allocator) const;

  se::DeviceMemoryAllocator* allocator() const { return allocator_; }
  int device_ordinal() const { return device_ordinal_; }
  absl::InlinedVector<se::DeviceMemoryBase, 1>& device_memory() {
    return device_memory_;
  }
  const absl::InlinedVector<se::DeviceMemoryBase, 1>& device_memory() const {
    return device_memory_;
  }
  absl::Span<const std::shared_ptr<BufferSequencingEvent>> definition_events()
      const {
    return definition_events_;
  }
  absl::Span<const StreamAndEvent> usage_events() const {
    return usage_events_;
  }

  // Relinquishes ownership of the buffer's device memory, e.g., after the
  // buffer is passed to a computation that aliases its inputs to outputs.
  void ReleaseDeviceMemory() { device_memory_.clear(); }

  // Indicates that the buffer has been used on a stream.
  //
  //   usage_stream:   a stream that the buffer was used on.
  //   event:          an event that has been recorded on usage_stream after the
  //                   buffer was used.
  //   reference_held: true if and only if the caller has caused a memory
  //                   reference to *this to stay live until after the host
  //                   is sure that the usage (transfer or execution) has
  //                   completed.
  void AddUsageEvent(se::Stream* usage_stream,
                     std::shared_ptr<BufferSequencingEvent> event,
                     bool reference_held);

  using StreamAndEventContainer = absl::InlinedVector<StreamAndEvent, 3>;
  // Returns the set of streams that the buffer was used on, and for each stream
  // an event later than the last use of the buffer. After
  // LockUseAndTransferUsageEvents is called it is illegal to use the buffer on
  // any stream and, e.g. AddUsageHold will CHECK fail.
  StreamAndEventContainer LockUseAndTransferUsageEvents();

  TrackedDeviceBuffer() : in_use_(true) {}
  TrackedDeviceBuffer(se::DeviceMemoryAllocator* allocator, int device_ordinal,
                      absl::Span<se::DeviceMemoryBase const> device_memory,
                      absl::Span<const std::shared_ptr<BufferSequencingEvent>>
                          definition_events,
                      absl::AnyInvocable<void() &&> on_delete_callback);
  ~TrackedDeviceBuffer();

 private:
  // Are the buffers in device_memory_ owned? If so, which allocator and device
  // ordinal? May be nullptr, indicating the buffers are not owned.
  se::DeviceMemoryAllocator* allocator_;
  int device_ordinal_;

  // Each host-side buffer may have several buffers on-device.
  absl::InlinedVector<se::DeviceMemoryBase, 1> device_memory_;

  // Events that are triggered when the content of one or more buffers is ready
  // during multistream execution. May be nullptr, which is used in the
  // single-stream execution case where events are not necessary for buffer
  // event sequencing. All events must be triggered before the buffers can be
  // used.
  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 2>
      definition_events_;

  // in_use_ starts out true, and is set to false when the buffer is released
  // from its owning PjRtBuffer. Once in_use_ is false, the buffer may no
  // longer be used on any stream.
  bool in_use_;
  // Set of streams that the buffer has ever been used on, see comment on
  // StreamAndEvent.
  StreamAndEventContainer usage_events_;

  // A callback to call when the TrackedDeviceBuffer is about to be destroyed.
  absl::AnyInvocable<void() &&> on_delete_callback_;
};

// Populates 'events' with the set of buffer events for buffer. If
// get_usage_events=true populates with the latest usage events, otherwise
// populates with the definition events.
void GetDeviceBufferEvents(const TrackedDeviceBuffer& buffer,
                           bool get_usage_events,
                           absl::flat_hash_set<BufferSequencingEvent*>* events);

// Waits for all of the definition events in a buffer on 'stream'.
void WaitForBufferDefinitionEventsOnStream(const TrackedDeviceBuffer& buffer,
                                           se::Stream* stream);

}  // namespace xla

#endif  // XLA_PJRT_TRACKED_DEVICE_BUFFER_H_
