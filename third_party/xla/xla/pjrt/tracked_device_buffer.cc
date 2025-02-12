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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/executable.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/event.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"

namespace xla {

void BufferSequencingEvent::SetSequencingEvent(EventPool::Handle event,
                                               se::Stream* stream) {
  {
    absl::MutexLock lock(&mu_);
    defined_status_.emplace(absl::OkStatus());
    CHECK(!event_.event());
    event_ = std::move(event);
    CHECK(streams_defined_on_.empty());
    streams_defined_on_.push_back(stream);
    sequence_number_.store(event_.sequence_number(), std::memory_order_seq_cst);
  }
  this->ExecuteFutureTasks();
}

bool BufferSequencingEvent::EventHasBeenRecorded() const {
  return event_.event() != nullptr;
}

bool BufferSequencingEvent::IsDefinedNoLock() const {
  return defined_status_.IsConcrete();
}

uint64_t BufferSequencingEvent::sequence_number() const {
  uint64_t seq = sequence_number_.load(std::memory_order_seq_cst);
  return seq;
}

void BufferSequencingEvent::WaitForEventOnStream(se::Stream* stream) {
  absl::MutexLock lock(&mu_);

  // We cannot wait for an event until ThenRecordEvent has been called; on GPU
  // newly created events are deemed to have already happened past.
  mu_.Await(
      absl::Condition(this, &BufferSequencingEvent::EventHasBeenRecorded));

  // The set of defined streams is expected to be very small indeed (usually
  // 1-2), so a simple linear scan should be fast enough.
  if (std::find(streams_defined_on_.begin(), streams_defined_on_.end(),
                stream) != streams_defined_on_.end()) {
    // stream is in streams_defined_on_; it doesn't need to be waited on.
    return;
  }

  stream->WaitFor(event_.event()).IgnoreError();
  streams_defined_on_.push_back(stream);
}

absl::Status BufferSequencingEvent::WaitForEventOnExternalStream(
    std::intptr_t stream) {
  absl::MutexLock lock(&mu_);

  // We cannot wait for an event until ThenRecordEvent has been called; on GPU
  // newly created events are deemed to have already happened past.
  // TODO(skyewm): do we need this? WaitForEventOnExternalStream is only
  // implemented for GPU.
  mu_.Await(
      absl::Condition(this, &BufferSequencingEvent::EventHasBeenRecorded));

  return event_.event()->WaitForEventOnExternalStream(stream);
}

bool BufferSequencingEvent::IsPredeterminedErrorOrDefinedOn(
    se::Stream* stream) {
  absl::MutexLock lock(&mu_);
  // IsDefined would be true for both a defined buffer and an error buffer.
  // Can't use BufferSequencingEvent::EventHasBeenRecorded here since that's
  // only true for a non-error buffer(i.e. defined buffer).
  mu_.Await(absl::Condition(this, &BufferSequencingEvent::IsDefinedNoLock));

  if (defined_status_.IsConcrete() && !defined_status_.get().ok()) {
    // IsPredeterminedError
    return true;
  }

  // The set of defined streams is expected to be very small indeed (usually
  // 1-2), so a simple linear scan should be fast enough.
  return std::find(streams_defined_on_.begin(), streams_defined_on_.end(),
                   stream) != streams_defined_on_.end();
}

bool BufferSequencingEvent::IsComplete() {
  absl::MutexLock lock(&mu_);

  // We cannot wait for an event until ThenRecordEvent has been called; on
  // GPU newly created events are deemed to have already happened past.
  mu_.Await(
      absl::Condition(this, &BufferSequencingEvent::EventHasBeenRecorded));

  return event_.event()->PollForStatus() == se::Event::Status::kComplete;
}

void BufferSequencingEvent::ExecuteOrAddToFutureTasks(
    const std::string& task_name, std::function<void()> task) {
  tsl::profiler::TraceMeProducer producer(
      "BufferSequencingEvent::ExecuteOrAddToFutureTasks",
      tsl::profiler::ContextType::kPjRt);
  uint64_t context_id = producer.GetContextId();
  auto wrapped_task = [task = std::move(task), context_id]() {
    tsl::profiler::TraceMeConsumer consumer("BufferSequencingEvent::Execute",
                                            tsl::profiler::ContextType::kPjRt,
                                            context_id);
    task();
  };
  {
    absl::MutexLock lock(&mu_);
    if (!defined_status_.IsConcrete()) {
      on_ready_tasks_callback_[task_name] = std::move(wrapped_task);
      return;
    }
    // Release the lock to avoid deadlock, in the case where the
    // thread_pool_->Schedule() executes wrapped_task inline.
    // This is rare but could happen. The callbacks could potentially try to
    // acquire the mutex of this BufferSequencingEvent.
  }
  thread_pool_->Schedule(std::move(wrapped_task));
}

void BufferSequencingEvent::ExecuteFutureTasks() {
  absl::flat_hash_map<std::string, std::function<void()>>
      on_ready_tasks_callback;
  {
    absl::MutexLock lock(&mu_);
    on_ready_tasks_callback = std::move(on_ready_tasks_callback_);
    // Release the lock to avoid deadlock, in the case where the
    // thread_pool_->Schedule() executes call_all_task_callbacks inline.
    // This is rare but could happen. The callbacks could potentially try to
    // acquire the mutex of this BufferSequencingEvent.
  }
  auto call_all_task_callbacks = [on_ready_tasks_callback =
                                      std::move(on_ready_tasks_callback)]() {
    for (auto& [task_name, task_callback] : on_ready_tasks_callback) {
      task_callback();
    }
  };
  thread_pool_->Schedule(std::move(call_all_task_callbacks));
}

/* static */ std::shared_ptr<TrackedDeviceBuffer>
TrackedDeviceBuffer::FromScopedShapedBuffer(
    ScopedShapedBuffer* shaped_buffer,
    absl::Span<const std::shared_ptr<BufferSequencingEvent>> definition_events,
    PjRtDevice* device) {
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer->buffers().begin();
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(1);

  ShapeUtil::ForEachSubshape(
      shaped_buffer->on_device_shape(), [&](const Shape&, const ShapeIndex&) {
        CHECK(iterator != shaped_buffer->buffers().end());
        buffers.push_back(iterator->second);
        iterator->second = se::DeviceMemoryBase();
        ++iterator;
      });
  CHECK(iterator == shaped_buffer->buffers().end());
  return std::make_shared<TrackedDeviceBuffer>(
      shaped_buffer->memory_allocator(), device,
      absl::Span<se::DeviceMemoryBase>(buffers), definition_events,
      /*on_delete_callback=*/nullptr);
}

ShapedBuffer TrackedDeviceBuffer::AsShapedBuffer(
    const Shape& on_device_shape) const {
  ShapedBuffer shaped_buffer(on_device_shape,
                             device_->local_device_id().value(),
                             device_->local_hardware_id().value());
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  for (const se::DeviceMemoryBase& buf : device_memory_) {
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = buf;
    ++iterator;
  }
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

// See comment on ExecutionInput in xla/service/executable.h to understand
// the meaning of owned/unowned in that class.

void TrackedDeviceBuffer::AddToInputAsImmutable(
    ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
    const ShapeTree<MaybeOwningDeviceMemory>::iterator& end) const {
  for (const se::DeviceMemoryBase& buf : device_memory_) {
    CHECK(*iterator != end);
    // Set buffers to be case (1) in the comment on ExecutionInput.
    (*iterator)->second = MaybeOwningDeviceMemory(buf);
    ++(*iterator);
  }
}

void TrackedDeviceBuffer::AddToInputAsDonated(
    ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
    const ShapeTree<MaybeOwningDeviceMemory>::iterator& end,
    ExecutionInput* execution_input,
    se::DeviceMemoryAllocator* allocator) const {
  for (const se::DeviceMemoryBase& buf : device_memory_) {
    CHECK(*iterator != end);
    // Set buffers to be case (2) in the comment on ExecutionInput.
    (*iterator)->second = MaybeOwningDeviceMemory(se::OwningDeviceMemory(
        buf, device_->local_device_id().value(), allocator));
    execution_input->SetUnownedIndex((*iterator)->first);
    ++(*iterator);
  }
}

TrackedDeviceBuffer::TrackedDeviceBuffer(
    se::DeviceMemoryAllocator* allocator, PjRtDevice* device,
    absl::Span<se::DeviceMemoryBase const> device_memory,
    absl::Span<const std::shared_ptr<BufferSequencingEvent>> definition_events,
    absl::AnyInvocable<void() &&> on_delete_callback)
    : allocator_(allocator),
      device_(device),
      device_memory_(device_memory.begin(), device_memory.end()),
      definition_events_(std::make_move_iterator(definition_events.begin()),
                         std::make_move_iterator(definition_events.end())),
      in_use_(true),
      on_delete_callback_(std::move(on_delete_callback)) {}

TrackedDeviceBuffer::~TrackedDeviceBuffer() {
  if (allocator_) {
    for (const se::DeviceMemoryBase& buffer : device_memory_) {
      absl::Status status =
          allocator_->Deallocate(device_->local_device_id().value(), buffer);
      if (!status.ok()) {
        LOG(ERROR) << "Buffer deallocation failed: " << status;
      }
    }
  }
  if (on_delete_callback_) {
    std::move(on_delete_callback_)();
  }
}

void TrackedDeviceBuffer::AddUsageEvent(
    se::Stream* usage_stream, std::shared_ptr<BufferSequencingEvent> event,
    bool reference_held) {
  CHECK(in_use_);

  // If the event is 0, it means that the event is not recorded yet and the task
  // related to this event is deferred, so just add it.
  if (*event == 0) {
    usage_events_.push_back({usage_stream, event, reference_held});
    return;
  }

  for (auto& existing : usage_events_) {
    // If the existing event is 0, it means that the event is not recorded yet
    // and the task related to this event is deferred, so don't replace it.
    if (*existing.event == 0) continue;
    if (existing.stream == usage_stream) {
      if (*existing.event < *event) {
        existing.event = event;
        existing.reference_held = reference_held;
      }
      return;
    }
  }
  usage_events_.push_back({usage_stream, event, reference_held});
}

TrackedDeviceBuffer::StreamAndEventContainer
TrackedDeviceBuffer::LockUseAndTransferUsageEvents() {
  CHECK(in_use_);
  in_use_ = false;
  return std::move(usage_events_);
}

void GetDeviceBufferEvents(
    const TrackedDeviceBuffer& buffer, bool get_usage_events,
    absl::flat_hash_set<BufferSequencingEvent*>* events) {
  if (get_usage_events) {
    for (const auto& e : buffer.usage_events()) {
      events->insert(e.event.get());
    }
  } else {
    for (const auto& e : buffer.definition_events()) {
      events->insert(e.get());
    }
  }
}

void WaitForBufferDefinitionEventsOnStream(const TrackedDeviceBuffer& buffer,
                                           se::Stream* stream) {
  absl::flat_hash_set<BufferSequencingEvent*> events;
  GetDeviceBufferEvents(buffer, /*get_usage_events=*/false, &events);
  for (BufferSequencingEvent* event : events) {
    event->WaitForEventOnStream(stream);
  }
}

}  // namespace xla
