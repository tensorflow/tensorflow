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

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/python/event_pool.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {

// A BufferDefinitionEvent describes whether a buffer is valid from the
// viewpoint of each of stream that may access it.
//
// Each logical buffer in an XLA computation may be defined (i.e., written to)
// at most once, although the same physical piece of memory may be reused for
// multiple logical buffers. We call the operation that writes the buffer's
// value on some stream (e.g., a transfer or compute kernel) the buffer's
// definition event.
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

 private:
  bool EventHasBeenRecorded() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // An event that is triggered when the content of one or more buffers is
  // ready. If this event is nullptr, it is assumed that the buffer's content is
  // always defined.
  EventPool::Handle event_;

  absl::Mutex mu_;

  // A list of all streams for which the buffer's content is known to be defined
  // at the tail of the queue, i.e., for any newly enqueued command.
  absl::InlinedVector<se::Stream*, 2> streams_defined_on_ TF_GUARDED_BY(mu_);
};

// Class that represents a node in a reference-counted DAG of device buffers.
// Unlike a ShapedBuffer, which owns none of its buffers, and
// ScopedShapedBuffer, which owns an entire buffer tree, the reference counting
// in a SharedDeviceBuffer DAG is done at the level of individual device
// buffers. Reference counting buffer individually is more convenient when
// manipulating on-device tuples where a tuple and its elements may have
// different lifetimes.
class SharedDeviceBuffer {
 public:
  // Converts a ScopedShapedBuffer into a Buffer tree. Takes ownership of the
  // buffers of the shaped_buffer.
  static std::shared_ptr<SharedDeviceBuffer> FromScopedShapedBuffer(
      ScopedShapedBuffer* shaped_buffer,
      absl::Span<const std::shared_ptr<BufferDefinitionEvent>>
          definition_events);

  // Makes a tuple buffer. Does not initialize the tuple table.
  static StatusOr<std::shared_ptr<SharedDeviceBuffer>> MakeTuple(
      std::vector<std::shared_ptr<SharedDeviceBuffer>> children,
      const Shape& on_host_shape, TransferManager* transfer_manager,
      se::DeviceMemoryAllocator* allocator, int device_ordinal,
      absl::Span<const std::shared_ptr<BufferDefinitionEvent>>
          definition_events);

  // Makes an uninitialized array buffer.
  static StatusOr<std::shared_ptr<SharedDeviceBuffer>> MakeArray(
      Shape on_device_shape, TransferManager* transfer_manager,
      se::DeviceMemoryAllocator* allocator, int device_ordinal,
      absl::Span<const std::shared_ptr<BufferDefinitionEvent>>
          definition_events);

  // Builds a ShapedBuffer view onto the buffers of 'tree'. We require but do
  // not verify that TransferManager::HostShapeToDeviceShape(on_host_shape) ==
  // on_device_shape().
  ShapedBuffer AsShapedBuffer(const Shape& on_host_shape,
                              const Shape& on_device_shape,
                              se::Platform* platform) const;

  const std::vector<std::shared_ptr<SharedDeviceBuffer>>& children() const {
    return children_;
  }
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

  SharedDeviceBuffer() = default;
  SharedDeviceBuffer(se::DeviceMemoryAllocator* allocator, int device_ordinal,
                     absl::Span<se::DeviceMemoryBase const> device_memory,
                     std::vector<std::shared_ptr<SharedDeviceBuffer>> children,
                     absl::Span<const std::shared_ptr<BufferDefinitionEvent>>
                         definition_events,
                     std::function<void()> on_delete_callback);
  SharedDeviceBuffer(absl::Span<se::OwningDeviceMemory> device_memory,
                     std::vector<std::shared_ptr<SharedDeviceBuffer>> children,
                     absl::Span<const std::shared_ptr<BufferDefinitionEvent>>
                         definition_events);
  ~SharedDeviceBuffer();

 private:
  // Are the buffers in device_memory_ owned? If so, which allocator and device
  // ordinal? May be nullptr, indicating the buffers are not owned.
  se::DeviceMemoryAllocator* allocator_;
  int device_ordinal_;

  // Each host-side buffer may have several buffers on-device.
  absl::InlinedVector<se::DeviceMemoryBase, 1> device_memory_;
  std::vector<std::shared_ptr<SharedDeviceBuffer>> children_;

  // An event that is triggered when the content of one or more buffers is
  // ready during multistream execution. May be nullptr, which is used in the
  // single-stream execution case where events are not necessary for buffer
  // event sequencing.
  absl::InlinedVector<std::shared_ptr<BufferDefinitionEvent>, 2>
      definition_events_;

  // A callback to call when the SharedDeviceBuffer is about to be destroyed.
  std::function<void()> on_delete_callback_;
};

// Populates 'events' with the set of buffer definition events for all buffers
// in the buffer DAG rooted at 'buffer'.
void GetDeviceBufferDefinitionEvents(
    const SharedDeviceBuffer& buffer,
    absl::flat_hash_set<BufferDefinitionEvent*>* events);

// Waits for all of the buffer definition events in a buffer DAG on 'stream'.
void WaitForBufferDefinitionEventsOnStream(const SharedDeviceBuffer& buffer,
                                           se::Stream* stream);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_SHARED_DEVICE_BUFFER_H_
