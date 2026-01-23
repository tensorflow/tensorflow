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

#include "xla/pjrt/tracked_device_buffer.h"

#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/se_raw_buffer.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/casts.h"

namespace xla {

ShapedBuffer RawSEDeviceMemory::AsShapedBuffer(
    PjRtDevice* device, const Shape& on_device_shape) const {
  ShapedBuffer shaped_buffer(on_device_shape, device->local_device_id().value(),
                             device->local_hardware_id().value());
  ShapeTree<se::DeviceAddressBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  CHECK(iterator != shaped_buffer.buffers().end());
  iterator->second = mem();
  ++iterator;
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

class AllocatedRawSEDeviceMemory : public RawSEDeviceMemory {
 public:
  AllocatedRawSEDeviceMemory(se::DeviceAddressBase value,
                             LocalDeviceState* local_device,
                             se::DeviceAddressAllocator* allocator)
      : RawSEDeviceMemory(value),
        allocator_(allocator),
        local_device_(local_device) {
    if (local_device_->allocation_model() ==
        LocalDeviceState::kComputeSynchronized) {
      sync_point_ = local_device_->GetNextComputeStreamSyncPoint();
    }
  }

  ~AllocatedRawSEDeviceMemory() override {
    if (allocator_) {
      absl::Status status = allocator_->Deallocate(
          local_device_->local_device_id().value(), mem());
      if (!status.ok()) {
        LOG(ERROR) << "Buffer deallocation failed: " << status;
      }
    }
  }

  void UnsafeReleaseMemory() override { allocator_ = nullptr; }

  absl::StatusOr<BufferSequencingEventRef> GetDefinitionEvent(
      AsyncWorkRunner* async_work_runner, bool nullptr_if_past) const override {
    if (sync_point_ != std::numeric_limits<size_t>::max()) {
      return local_device_->GetEventForComputeStreamSyncPoint(
          sync_point_, async_work_runner, nullptr_if_past);
    }
    return BufferSequencingEventRef();
  }

 private:
  se::DeviceAddressAllocator* allocator_;
  LocalDeviceState* local_device_;
  size_t sync_point_ = std::numeric_limits<size_t>::max();
};

tsl::AsyncValueRef<RawSEDeviceMemory> RawSEDeviceMemory::Create(
    se::DeviceAddressBase value, LocalDeviceState* local_device,
    se::DeviceAddressAllocator* allocator) {
  return tsl::MakeAvailableAsyncValueRef<AllocatedRawSEDeviceMemory>(
      value, local_device, allocator);
}

/*static*/ void RawSEDeviceMemory::ConstructDelayed(
    tsl::AsyncValueRef<RawSEDeviceMemory> buf, se::DeviceAddressBase value,
    LocalDeviceState* local_device, se::DeviceAddressAllocator* allocator) {
  tsl::Cast<AllocatedRawSEDeviceMemory>(buf).emplace(value, local_device,
                                                     allocator);
}

/*static*/ tsl::AsyncValueRef<RawSEDeviceMemory>
RawSEDeviceMemory::CreateDelayedMemory() {
  return tsl::MakeUnconstructedAsyncValueRef<AllocatedRawSEDeviceMemory>();
}

class ForeignRawSEDeviceMemory : public RawSEDeviceMemory {
 public:
  ForeignRawSEDeviceMemory(se::DeviceAddressBase value,
                           absl::AnyInvocable<void() &&> on_delete_callback)
      : RawSEDeviceMemory(value),
        on_delete_callback_(std::move(on_delete_callback)) {}

  ~ForeignRawSEDeviceMemory() override { std::move(on_delete_callback_)(); }

  void UnsafeReleaseMemory() override {
    LOG(FATAL) << "ForeignRawSEDeviceMemory cannot be donated.";
  }

 private:
  absl::AnyInvocable<void() &&> on_delete_callback_;
};

tsl::AsyncValueRef<RawSEDeviceMemory> RawSEDeviceMemory::CreateForeign(
    se::DeviceAddressBase value,
    absl::AnyInvocable<void() &&> on_delete_callback) {
  return tsl::MakeAvailableAsyncValueRef<ForeignRawSEDeviceMemory>(
      value, std::move(on_delete_callback));
}

TrackedDeviceBuffer::TrackedDeviceBuffer(
    PjRtDevice* device, tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
    absl::Span<const BufferSequencingEventRef> definition_events)
    : AbstractTrackedDeviceBuffer(std::move(raw_buffer)),
      device_(device),
      definition_events_(std::make_move_iterator(definition_events.begin()),
                         std::make_move_iterator(definition_events.end())),
      in_use_(true) {}

TrackedDeviceBuffer::~TrackedDeviceBuffer() = default;

void TrackedDeviceBuffer::ConfirmDonation() {
  // As a sanity check ensure no more usage events can be added to the buffer.
  LockUseAndTransferUsageEvents();
  // Release the memory so that no new usage is possible.
  ReleaseDeviceMemory();
}

void TrackedDeviceBuffer::AddUsageEvent(BufferSequencingEventRef event,
                                        bool reference_held) {
  CHECK(in_use_);

  // If the event is 0, it means that the event is not recorded yet and the task
  // related to this event is deferred, so just add it.
  if (!event->IsDefined()) {
    usage_events_.push_back({event, reference_held});
    return;
  }
  auto* usage_stream = event->definition_stream();

  for (auto& existing : usage_events_) {
    // If the existing event is 0, it means that the event is not recorded yet
    // and the task related to this event is deferred, so don't replace it.
    if (!existing.event->IsDefined()) continue;
    if (existing.event->definition_stream() == usage_stream) {
      if (*existing.event < *event) {
        existing.event = event;
        existing.reference_held = reference_held;
      }
      return;
    }
  }
  usage_events_.push_back({event, reference_held});
}

absl::StatusOr<std::unique_ptr<AbstractTrackedDeviceBuffer>>
TrackedDeviceBuffer::CloneWithControlDependency(PjRtMemorySpace* memory_space,
                                                Future<> dependency) {
  auto* se_client =
      tensorflow::down_cast<PjRtStreamExecutorClient*>(memory_space->client());

  // Copy all the data in the existing tracked_buffer.
  const auto& original_definition_events = definition_events();
  absl::InlinedVector<BufferSequencingEventRef, 4> definition_events;

  auto definition_event_for_status =
      BufferSequencingEvent::Create(se_client->async_work_runner());
  // definition_event_for_status must be the first one so that it blocks other
  // actions like D2H transfer from execution before the buffer is ready.
  definition_events.push_back(definition_event_for_status);
  definition_events.insert(definition_events.end(),
                           original_definition_events.begin(),
                           original_definition_events.end());

  auto new_device_buffer = std::make_unique<TrackedDeviceBuffer>(
      device_, raw_buffer(), std::move(definition_events));

  auto* device = tensorflow::down_cast<PjRtStreamExecutorDevice*>(
      memory_space->devices()[0]);
  LocalDeviceState* local_device = device->local_device_state();
  dependency.OnReady(
      [definition_event_for_status = std::move(definition_event_for_status),
       local_device, client = se_client](absl::Status status) mutable {
        // Forward the absl::Status from the supplied dependency to the
        // definition event.
        if (!status.ok()) {
          client->SetEventAsError(definition_event_for_status, status);
          return;
        }
        auto stream = local_device->BorrowStreamFromPool();
        CHECK_OK(client->AllocateAndRecordEvent(definition_event_for_status,
                                                local_device, stream.get()));
        local_device->ReturnStreamToPool(std::move(stream));
      });
  return new_device_buffer;
}

Future<> TrackedDeviceBuffer::GetReadyFuture(PjRtMemorySpace* memory_space) {
  auto [promise, future] = MakePromise<>();
  std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events;
  definition_events.reserve(definition_events_.size());
  for (const auto& event : definition_events_) {
    definition_events.push_back(event.CopyRCRef());
  }
  absl::Span<tsl::RCReference<tsl::AsyncValue> const> definition_events_span =
      definition_events;
  tsl::RunWhenReady(
      definition_events_span,
      [promise = std::move(promise),
       definition_events = std::move(definition_events)]() mutable {
        for (auto& event : definition_events) {
          if (const absl::Status* error = event->GetErrorIfPresent()) {
            promise.Set(*error);
            return;
          }
        }
        promise.Set();
      });
  return future;
}

void TrackedDeviceBuffer::Delete(PjRtMemorySpace* memory_space) {
  std::unique_ptr<TrackedDeviceBuffer> device_buffer(this);
  // All events already hold onto refs to the buffer to ensure liveness so there
  // is no work to do.
}

TrackedDeviceBuffer::StreamAndEventContainer
TrackedDeviceBuffer::LockUseAndTransferUsageEvents() {
  CHECK(in_use_);
  in_use_ = false;
  return std::move(usage_events_);
}

std::vector<tsl::RCReference<tsl::AsyncValue>>
TrackedDeviceBuffer::GetAsyncValueDefinitionEvents() {
  std::vector<tsl::RCReference<tsl::AsyncValue>> avs;
  avs.reserve(definition_events_.size());
  for (const auto& ev : definition_events_) {
    avs.push_back(ev.CopyRCRef());
  }
  return avs;
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
TrackedDeviceBuffer::GetDefinitionEvent(PjRtMemorySpace* memory_space) {
  if (definition_events_.size() != 1) {
    return absl::InternalError(
        "GetMergedDefinitionEvent only supported on TPU for buffers with "
        "exactly 1 definition event.");
  }
  return tsl::MakeRef<PjRtStreamExecutorDeviceEvent>(definition_events_[0]);
}

void TrackedDeviceBuffer::AddUsageEvent(
    tsl::RCReference<PjRtDeviceEvent> event) {
  if (event) {
    AddUsageEvent(
        tensorflow::down_cast<PjRtStreamExecutorDeviceEvent*>(event.get())
            ->event(),
        true);
  }
}

void GetDeviceBufferEvents(
    const TrackedDeviceBuffer& buffer, bool get_usage_events,
    absl::flat_hash_set<BufferSequencingEvent*>* events) {
  if (get_usage_events) {
    for (const auto& e : buffer.usage_events()) {
      events->insert(&*e.event);
    }
  } else {
    for (const auto& e : buffer.definition_events()) {
      events->insert(&*e);
    }
  }
}

void WaitForBufferDefinitionEventsOnStream(
    absl::Span<const BufferSequencingEventRef> definition_events,
    se::Stream* stream) {
  if (definition_events.size() <= 1) {
    for (const auto& event : definition_events) {
      event->WaitForEventOnStream(stream);
    }
  } else {
    absl::flat_hash_set<BufferSequencingEvent*> events;
    for (const auto& event : definition_events) {
      if (events.emplace(&*event).second) {
        event->WaitForEventOnStream(stream);
      }
    }
  }
}

}  // namespace xla
