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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_SHARED_DEVICE_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_SHARED_DEVICE_BUFFER_H_

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/python/event_pool.h"
#include "tensorflow/compiler/xla/python/local_device_state.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/stream.h"

namespace xla {

// A BufferDefinitionEvent describes whether a buffer is valid from the
// viewpoint of each of stream that may access it.
//
// Each logical buffer in an XLA computation may be defined (i.e., written to)
// at most once. We call the operation that writes the buffer's value on some
// stream (e.g., a transfer or compute kernel) the buffer's definition event.
//
// After the operation that populates the value of a buffer has been enqueued on
// 'stream', RecordOnStream(stream) should also be called to trigger the
// definition event after the operation has completed.
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
//
// TODO(misard) Rename this BufferSequencingEvent now that it is used for Usage
// events as well.
class BufferDefinitionEvent {
 public:
  BufferDefinitionEvent() = default;

  // Sets the definition event of the buffer to 'event', which is recorded
  // on 'stream'. Must be called at most once. Unblocks any other host threads
  // are blocked in WaitForEventOnStream.
  void SetDefinitionEvent(EventPool::Handle event, se::Stream* stream);

  // Adds synchronization events to 'stream' that wait for this event to be
  // defined on 'stream'. Does nothing if the event is already known to have
  // occurred by the tail of 'stream'. If RecordOnStream has not yet been
  // called, blocks the calling thread until the event has been recorded.
  void WaitForEventOnStream(se::Stream* stream);

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
  inline bool operator<(const BufferDefinitionEvent& rhs) const {
    return sequence_number() < rhs.sequence_number();
  }
  inline bool operator>(const BufferDefinitionEvent& rhs) const {
    return rhs < *this;
  }
  inline bool operator<=(const BufferDefinitionEvent& rhs) const {
    return !(*this > rhs);
  }
  inline bool operator>=(const BufferDefinitionEvent& rhs) const {
    return !(*this < rhs);
  }

 private:
  bool EventHasBeenRecorded() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  uint64 sequence_number() const;

  // An event that is triggered when the content of one or more buffers is
  // ready. If this event is nullptr, it is assumed that the buffer's content is
  // always defined.
  EventPool::Handle event_;

  mutable absl::Mutex mu_;
  // A list of all streams for which the buffer's content is known to be defined
  // at the tail of the queue, i.e., for any newly enqueued command.
  absl::InlinedVector<se::Stream*, 2> streams_defined_on_ TF_GUARDED_BY(mu_);
};

// Class that represents a tuple of device buffers. Like a ScopedShapedBuffer it
// owns all of the device memory in the tuple. It also tracks the definition and
// usage of the memory on streams, to allow for synchronized usage and deletion
// of memory under all of the allocation model semantics.
class SharedDeviceBuffer {
 public:
  // Converts a ScopedShapedBuffer into a SharedDeviceBuffer. Takes ownership of
  // the buffers of the shaped_buffer.
  static std::shared_ptr<SharedDeviceBuffer> FromScopedShapedBuffer(
      ScopedShapedBuffer* shaped_buffer,
      absl::Span<const std::shared_ptr<BufferDefinitionEvent>>
          definition_events);

  // Helper class to retain a "hold" on a SharedDeviceBuffer while it is being
  // enqueued on a stream. If the enqueue completes successfully the hold
  // should be released using a call to Convert. If the ScopedUsage is deleted
  // without Convert being called, e.g., on error, the hold is dropped.
  // Deletion of a buffer will block until all ScopedUsage objects referencing
  // it are either deleted or have their Convert methods called.
  class ScopedUsage {
   public:
    ScopedUsage() = default;
    ~ScopedUsage();
    ScopedUsage(ScopedUsage&&) = default;
    ScopedUsage(const ScopedUsage&) = delete;
    ScopedUsage& operator=(const ScopedUsage&) = delete;

    ScopedUsage& Acquire(std::shared_ptr<SharedDeviceBuffer> parent);
    std::shared_ptr<SharedDeviceBuffer> Release();
    void Transfer(std::shared_ptr<SharedDeviceBuffer> parent);

    bool IsValid() { return parent_ != nullptr; }
    SharedDeviceBuffer* operator->() const { return parent_.get(); }
    const SharedDeviceBuffer& operator*() const { return *parent_; }
    std::shared_ptr<SharedDeviceBuffer> buffer_reference() const {
      return parent_;
    }

    // Converts the usage hold into a usage event.
    //
    //   usage_stream:   a stream that the buffer was used on.
    //   event:          an event that has been recorded on usage_stream after
    //                   the buffer was used.
    //   reference_held: true if and only if the caller has caused a memory
    //                   reference to *this to stay live until after the host
    //                   is sure that the usage (transfer or execution) has
    //                   completed.
    void Convert(se::Stream* usage_stream,
                 std::shared_ptr<BufferDefinitionEvent> event,
                 bool reference_held);

   private:
    std::shared_ptr<SharedDeviceBuffer> parent_;
  };

  // Increments the count of external frameworks, e.g., Numpy, that the buffer
  // is shared with. Operations that require exclusive access, such as update in
  // place, will fail if any external references are held.
  void AddExternalReference();

  // Decrements the count of external frameworks that the buffer is shared with.
  void DropExternalReference();

  // Builds a ShapedBuffer view onto the buffers of 'tree'. We require but do
  // not verify that TransferManager::HostShapeToDeviceShape(on_host_shape) ==
  // on_device_shape().
  ShapedBuffer AsShapedBuffer(const Shape& on_host_shape,
                              const Shape& on_device_shape,
                              se::Platform* platform) const;

  se::DeviceMemoryAllocator* allocator() const { return allocator_; }
  int device_ordinal() const { return device_ordinal_; }
  absl::InlinedVector<se::DeviceMemoryBase, 1>& device_memory() {
    return device_memory_;
  }
  const absl::InlinedVector<se::DeviceMemoryBase, 1>& device_memory() const {
    return device_memory_;
  }
  absl::Span<const std::shared_ptr<BufferDefinitionEvent>> definition_events()
      const {
    return definition_events_;
  }

  // Helper object to keep track of usage of the buffer on streams.
  struct StreamAndEvent {
    // A stream the buffer has been used on.
    se::Stream* stream;
    // An event that is later than the most recent usage of the buffer on
    // stream.
    std::shared_ptr<BufferDefinitionEvent> event;
    // True if and only if a reference to the buffer is kept live until after
    // the host knows that event is complete.
    bool reference_held;
  };
  using StreamAndEventContainer = absl::InlinedVector<StreamAndEvent, 3>;
  // Returns the set of streams that the buffer was used on, and for each stream
  // an event later than the last use of the buffer. After
  // LockUseAndTransferUsageEvents is called it is illegal to use the buffer on
  // any stream and, e.g. AddUsageHold will CHECK fail.
  StreamAndEventContainer LockUseAndTransferUsageEvents();

  SharedDeviceBuffer()
      : in_use_(true), usage_holds_(0), external_references_(0) {}
  SharedDeviceBuffer(se::DeviceMemoryAllocator* allocator, int device_ordinal,
                     absl::Span<se::DeviceMemoryBase const> device_memory,
                     absl::Span<const std::shared_ptr<BufferDefinitionEvent>>
                         definition_events,
                     std::function<void()> on_delete_callback);
  ~SharedDeviceBuffer();

 private:
  friend class ScopedUsage;

  // Indicates that the buffer is going to be used on a stream. Deletion of
  // the buffer will block until there are no remaining ScopedUsage objects.
  void AddUsageHold();

  // Indicates that a previous usage hold can be discarded, e.g., because of an
  // error while an action was being enqueued on a stream.
  void DropUsageHold();

  // Indicates that a previous usage hold can be converted into a usage event.
  //
  //   usage_stream:   a stream that the buffer was used on.
  //   event:          an event that has been recorded on usage_stream after the
  //                   buffer was used.
  //   reference_held: true if and only if the caller has caused a memory
  //                   reference to *this to stay live until after the host
  //                   is sure that the usage (transfer or execution) has
  //                   completed.
  void ConvertUsageHold(se::Stream* usage_stream,
                        std::shared_ptr<BufferDefinitionEvent> event,
                        bool reference_held);

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
  absl::InlinedVector<std::shared_ptr<BufferDefinitionEvent>, 2>
      definition_events_;

  absl::Mutex mu_;
  // in_use_ starts out true, and is set to false when the buffer is released
  // from its owning PyLocalBuffer. Once in_use_ is false, the buffer may no
  // longer be used on any stream.
  bool in_use_ TF_GUARDED_BY(mu_);
  // Count of operations that are currently enqueuing the buffer onto a stream.
  int usage_holds_ TF_GUARDED_BY(mu_);
  // Set of streams that the buffer has ever been used on, see comment on
  // StreamAndEvent.
  StreamAndEventContainer usage_events_ TF_GUARDED_BY(mu_);
  // Count of external frameworks that hold a reference to this buffer.
  int external_references_ TF_GUARDED_BY(mu_);

  // A callback to call when the SharedDeviceBuffer is about to be destroyed.
  std::function<void()> on_delete_callback_;
};

// Populates 'events' with the set of buffer definition events for buffer.
void GetDeviceBufferDefinitionEvents(
    const SharedDeviceBuffer& buffer,
    absl::flat_hash_set<BufferDefinitionEvent*>* events);

// Waits for all of the definition events in a buffer on 'stream'.
void WaitForBufferDefinitionEventsOnStream(const SharedDeviceBuffer& buffer,
                                           se::Stream* stream);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_SHARED_DEVICE_BUFFER_H_
