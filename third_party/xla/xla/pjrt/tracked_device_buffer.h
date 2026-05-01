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

  static tsl::AsyncValueRef<RawSEDeviceMemory> CreateSlice(
      tsl::AsyncValueRef<RawSEDeviceMemory> base, size_t offset, size_t size);
  // Returns a definition event (or nullptr if the definition is known to be in
  // the past).
  virtual absl::StatusOr<BufferSequencingEventRef> GetDefinitionEvent(
      AsyncWorkRunner* async_work_runner, bool nullptr_if_past) const {
    return BufferSequencingEventRef();
  }

 private:
  se::DeviceAddressBase value_;
};

// PjRtDeviceEventSet that coalesces events to try and maintain the minimal
// wait set for all added events. For stream based events it inspects the
// stream id to determine if a new event is redundant or not.
class PjRtStreamExecutorUsageEventSet : public PjRtDeviceEventSet {
 public:
  PjRtStreamExecutorUsageEventSet() = default;

  void AddEvent(PjRtDeviceEventRef event) override;

  void AddEvent(BufferSequencingEventRef event, bool reference_held);

  void AppendTo(
      std::vector<tsl::RCReference<tsl::AsyncValue>>& events) override;

  void AppendTo(PjRtDeviceEventSet& events) override;

  std::unique_ptr<PjRtDeviceEventSet> Clone() const override;

 private:
  // Helper object to keep track of usage of the buffer on streams.
  struct StreamAndEvent {
    BufferSequencingEventRef event;
    bool reference_held;
  };

  using StreamAndEventContainer = absl::InlinedVector<StreamAndEvent, 3>;
  // Set of streams that the buffer has ever been used on, see comment on
  // StreamAndEvent.
  StreamAndEventContainer usage_events_;
};

// Class that represents a tuple of device buffers. Like a ScopedShapedBuffer it
// owns all of the device memory in the tuple. It also tracks the definition and
// usage of the memory on streams, to allow for synchronized usage and deletion
// of memory under all of the allocation model semantics.
class TrackedDeviceBuffer : public AbstractTrackedDeviceBuffer {
 public:
  TrackedDeviceBuffer(
      PjRtDevice* device, PjRtRawBufferRef raw_buffer,
      absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events,
      std::unique_ptr<PjRtDeviceEventSet> usage_events = nullptr);
  ~TrackedDeviceBuffer() override;

  void Delete(PjRtMemorySpace* memory_space) override;

  absl::Status WaitUntilBufferReadyOnStream(std::intptr_t stream) override {
    for (const auto& event : definition_events()) {
      TF_RETURN_IF_ERROR(event.down_cast<BufferSequencingEvent>()
                             ->WaitForEventOnExternalStream(stream));
    }
    return absl::OkStatus();
  }

  std::unique_ptr<AbstractTrackedDeviceBuffer> Clone(
      absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events,
      std::unique_ptr<PjRtDeviceEventSet> usage_events) const override {
    return std::make_unique<TrackedDeviceBuffer>(device_, raw_buffer(),
                                                 std::move(definition_events),
                                                 std::move(usage_events));
  }

 private:
  PjRtDevice* device_;
  // in_use_ starts out true, and is set to false when the buffer is released
  // from its owning PjRtBuffer. Once in_use_ is false, the buffer may no
  // longer be used on any stream.
  bool in_use_;
};

// Waits for all of the definition events in a buffer on 'stream'.
void WaitForBufferDefinitionEventsOnStream(
    absl::Span<const BufferSequencingEventRef> definition_events,
    se::Stream* stream);

}  // namespace xla

#endif  // XLA_PJRT_TRACKED_DEVICE_BUFFER_H_
