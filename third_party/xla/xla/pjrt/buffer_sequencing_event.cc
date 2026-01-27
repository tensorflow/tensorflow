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

#include "xla/pjrt/buffer_sequencing_event.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/event_pool.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"

namespace xla {

void BufferSequencingEvent::SetSequencingEvent(EventPool::Handle event,
                                               se::Stream* stream) {
  EventState state;
  state.event = std::move(event);
  state.definition_stream = stream;
  event_.emplace(std::move(state));
}

void BufferSequencingEvent::SetDefinedStatus(absl::Status status) {
  CHECK(!status.ok());
  event_.SetError(status);
}

uint64_t BufferSequencingEvent::sequence_number() const {
  return event_->event.sequence_number();
}

void BufferSequencingEvent::WaitForEventOnStream(se::Stream* stream) {
  // We cannot wait for an event until ThenRecordEvent has been called; on GPU
  // newly created events are deemed to have already happened past.
  tsl::BlockUntilReady(event_);

  if (event_.IsError()) {
    return;
  }
  if (event_->definition_stream == stream) {
    return;
  }

  absl::MutexLock lock(mu_);
  // The set of defined streams is expected to be very small indeed (usually
  // 1-2), so a simple linear scan should be fast enough.
  if (std::find(streams_defined_on_.begin(), streams_defined_on_.end(),
                stream) != streams_defined_on_.end()) {
    // stream is in streams_defined_on_; it doesn't need to be waited on.
    return;
  }

  stream->WaitFor(event_->event.event()).IgnoreError();
  streams_defined_on_.push_back(stream);
}

absl::Status BufferSequencingEvent::WaitForEventOnExternalStream(
    std::intptr_t stream) {
  tsl::BlockUntilReady(event_);
  if (const auto* error = event_.GetErrorIfPresent()) {
    return *error;
  }
  return event_->event.event()->WaitForEventOnExternalStream(stream);
}

bool BufferSequencingEvent::IsPredeterminedErrorOrDefinedOn(
    se::Stream* stream) {
  tsl::BlockUntilReady(event_);
  CHECK(event_.IsAvailable());

  // IsPredeterminedError
  if (event_.IsError()) {
    return true;
  }

  if (event_->definition_stream == stream) {
    return true;
  }

  // The set of defined streams is expected to be very small indeed (usually
  // 1-2), so a simple linear scan should be fast enough.
  absl::MutexLock lock(mu_);
  return absl::c_find(streams_defined_on_, stream) != streams_defined_on_.end();
}

bool BufferSequencingEvent::IsComplete() {
  tsl::BlockUntilReady(event_);
  if (event_.IsError()) {
    return true;
  }

  return event_->event.event()->PollForStatus() == se::Event::Status::kComplete;
}

void BufferSequencingEvent::ExecuteOrAddToFutureTasks(
    const std::string& task_name, std::function<void()> task) {
  tsl::profiler::TraceMeProducer producer(
      "BufferSequencingEvent::ExecuteOrAddToFutureTasks",
      tsl::profiler::ContextType::kPjRt);

  auto traced_task = [task = std::move(task),
                      context_id = producer.GetContextId()]() {
    tsl::profiler::TraceMeConsumer consumer("BufferSequencingEvent::Execute",
                                            tsl::profiler::ContextType::kPjRt,
                                            context_id);
    task();
  };

  // Execute the `task` when definition event becomes available. If it's already
  // available, the task will be executed immediately.
  event_.AndThen([this, traced_task = std::move(traced_task)]() mutable {
    async_work_runner_->Schedule(std::move(traced_task));
  });
}

}  // namespace xla
