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
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/service/executable.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

class RawSEDeviceMemory {
 public:
  explicit RawSEDeviceMemory(se::DeviceAddressBase value) : value_(value) {}

  virtual ~RawSEDeviceMemory() = default;

  const se::DeviceAddressBase& mem() const { return value_; }

  void* opaque() const { return value_.opaque(); }

  // TODO(parkers): Donate this ref-counted object instead of the underlying
  // buffer.
  virtual void UnsafeReleaseMemory() = 0;

  // Builds a ShapedBuffer which points to mem() of shape on_device_shape.
  ShapedBuffer AsShapedBuffer(PjRtDevice* device,
                              const Shape& on_device_shape) const;

  static tsl::AsyncValueRef<RawSEDeviceMemory> Create(
      se::DeviceAddressBase value, LocalDeviceState* local_device,
      se::DeviceAddressAllocator* allocator);
  static tsl::AsyncValueRef<RawSEDeviceMemory> CreateDelayedMemory();
  static void ConstructDelayed(tsl::AsyncValueRef<RawSEDeviceMemory> buf,
                               se::DeviceAddressBase value,
                               LocalDeviceState* local_device,
                               se::DeviceAddressAllocator* allocator);
  static tsl::AsyncValueRef<RawSEDeviceMemory> CreateForeign(
      se::DeviceAddressBase value,
      absl::AnyInvocable<void() &&> on_delete_callback);

  // Returns a definition event (or nullptr if the definition is known to be in
  // the past).
  virtual absl::StatusOr<BufferSequencingEventRef> GetDefinitionEvent(
      AsyncWorkRunner* async_work_runner, bool nullptr_if_past) const {
    return BufferSequencingEventRef();
  }

 private:
  se::DeviceAddressBase value_;
};

// Class that represents a tuple of device buffers. Like a ScopedShapedBuffer it
// owns all of the device memory in the tuple. It also tracks the definition and
// usage of the memory on streams, to allow for synchronized usage and deletion
// of memory under all of the allocation model semantics.
class TrackedDeviceBuffer : public AbstractTrackedDeviceBuffer {
 public:
  // Helper object to keep track of usage of the buffer on streams.
  struct StreamAndEvent {
    // An event that is later than the most recent usage of the buffer on
    // stream.
    BufferSequencingEventRef event;
    // True if and only if a reference to the buffer is kept live until after
    // the host knows that event is complete.
    bool reference_held;
  };

  // Adds the owned device buffers in order to 'iterator'. Used to add the
  // buffers to an ExecutionInput. We require but do not verify that 'iterator'
  // when passed in is pointing to a sub-tuple of the ExecutionInput whose
  // on_device_shape matches that of the TrackedDeviceBuffer. 'end' is used to
  // check that 'iterator' doesn't run out of bounds.
  void AddToInputAsImmutable(
      ShapeTree<MaybeOwningDeviceAddress>::iterator* iterator,
      const ShapeTree<MaybeOwningDeviceAddress>::iterator& end) const;

  // Adds the owned device buffers in order to 'iterator', marking them as
  // available to be donated. If donation succeeds, i.e., execution_input is
  // subsequently successfully enqueued to a computation,
  // this->ReleaseDeviceMemory() must be called to avoid freeing the device
  // memory twice. We require but do not verify that 'iterator' when passed in
  // is pointing to a sub-tuple of execution_input whose on_device_shape matches
  // that of the TrackedDeviceBuffer. 'end' is used to check that 'iterator'
  // doesn't run out of bounds.
  void AddToInputAsDonated(
      ShapeTree<MaybeOwningDeviceAddress>::iterator* iterator,
      const ShapeTree<MaybeOwningDeviceAddress>::iterator& end,
      ExecutionInput* execution_input,
      se::DeviceAddressAllocator* allocator) const;

  const absl::InlinedVector<BufferSequencingEventRef, 2>& definition_events()
      const {
    return definition_events_;
  }
  absl::Span<const StreamAndEvent> usage_events() const {
    return usage_events_;
  }

  // Only to be called by ScopedHold to mark a successful donation.
  void ConfirmDonation() override;

  // Indicates that the buffer has been used on a stream.
  //
  //   usage_stream:   a stream that the buffer was used on.
  //   event:          an event that has been recorded on usage_stream after the
  //                   buffer was used.
  //   reference_held: true if and only if the caller has caused a memory
  //                   reference to *this to stay live until after the host
  //                   is sure that the usage (transfer or execution) has
  //                   completed.
  void AddUsageEvent(BufferSequencingEventRef event, bool reference_held);

  using StreamAndEventContainer = absl::InlinedVector<StreamAndEvent, 3>;
  // Returns the set of streams that the buffer was used on, and for each stream
  // an event later than the last use of the buffer. After
  // LockUseAndTransferUsageEvents is called it is illegal to use the buffer on
  // any stream and, e.g. AddUsageHold will CHECK fail.
  StreamAndEventContainer LockUseAndTransferUsageEvents();

  TrackedDeviceBuffer(
      PjRtDevice* device, tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
      absl::Span<const BufferSequencingEventRef> definition_events);
  ~TrackedDeviceBuffer() override;

  std::vector<tsl::RCReference<tsl::AsyncValue>> GetAsyncValueDefinitionEvents()
      override;

  std::vector<tsl::RCReference<tsl::AsyncValue>>
  GetAsyncValueDefinitionAndUsageEvents() override;

  void AddUsageEvent(tsl::RCReference<PjRtDeviceEvent> event) override;

  void Delete(PjRtMemorySpace* memory_space) override;

  absl::Status WaitUntilBufferReadyOnStream(std::intptr_t stream) override {
    for (const BufferSequencingEventRef& event : definition_events()) {
      TF_RETURN_IF_ERROR(event->WaitForEventOnExternalStream(stream));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<AbstractTrackedDeviceBuffer>>
  CloneWithControlDependency(PjRtMemorySpace* memory_space,
                             Future<> dependency) override;

  Future<> GetReadyFuture(PjRtMemorySpace* memory_space) override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> GetDefinitionEvent(
      PjRtMemorySpace* memory_space) override;

  bool AddDefinitionEventsToSet(PjRtDeviceEventSet& events) override;

  void AddUsageEventsToSet(PjRtDeviceEventSet& events) override;

 private:
  PjRtDevice* device_;
  // Events that are triggered when the content of one or more buffers is ready
  // during multistream execution. May be nullptr, which is used in the
  // single-stream execution case where events are not necessary for buffer
  // event sequencing. All events must be triggered before the buffers can be
  // used.
  absl::InlinedVector<BufferSequencingEventRef, 2> definition_events_;

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

// Waits for all of the definition events in a buffer on 'stream'.
void WaitForBufferDefinitionEventsOnStream(
    absl::Span<const BufferSequencingEventRef> definition_events,
    se::Stream* stream);

}  // namespace xla

#endif  // XLA_PJRT_TRACKED_DEVICE_BUFFER_H_
