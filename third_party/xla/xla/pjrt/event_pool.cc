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

#include "xla/pjrt/event_pool.h"

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/statusor.h"

namespace xla {

EventPool::Handle::~Handle() {
  if (pool_ && event_) {
    absl::MutexLock lock(&pool_->mu_free_events_);
    pool_->free_events_.push(std::move(event_));
  }
}

EventPool::EventPool(bool allow_reuse)
    : allow_reuse_(allow_reuse), next_sequence_number_(1) {}

absl::StatusOr<EventPool::Handle> EventPool::AllocateEvent(
    se::StreamExecutor* executor) {
  Handle event;

  if (allow_reuse_) {
    event.pool_ = this;
    absl::MutexLock lock(&mu_free_events_);
    if (!free_events_.empty()) {
      event.event_ = std::move(free_events_.top());
      free_events_.pop();
    }
  }
  if (!event.event_) {
    TF_ASSIGN_OR_RETURN(event.event_, executor->CreateEvent());
  }
  return event;
}

void EventPool::ThenRecordEvent(se::Stream* stream, EventPool::Handle& handle) {
  absl::MutexLock lock(&mu_sequence_number_);
  stream->RecordEvent(handle.event_.get()).IgnoreError();
  handle.sequence_number_ = next_sequence_number_++;
}

absl::StatusOr<EventPool::Handle> EventPool::ThenAllocateAndRecordEvent(
    se::Stream* stream) {
  TF_ASSIGN_OR_RETURN(EventPool::Handle handle,
                      AllocateEvent(stream->parent()));
  ThenRecordEvent(stream, handle);
  return handle;
}

}  // namespace xla
