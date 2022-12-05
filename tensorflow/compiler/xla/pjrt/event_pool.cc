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

#include "tensorflow/compiler/xla/pjrt/event_pool.h"

#include <memory>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

EventPool::Handle::~Handle() {
  if (pool_ && event_) {
    absl::MutexLock lock(&pool_->mu_);
    pool_->free_events_.push(std::move(event_));
  }
}

EventPool::EventPool(bool allow_reuse)
    : allow_reuse_(allow_reuse), next_sequence_number_(1) {}

StatusOr<EventPool::Handle> EventPool::AllocateEvent(
    se::StreamExecutor* executor) {
  Handle event;

  if (allow_reuse_) {
    event.pool_ = this;
    absl::MutexLock lock(&mu_);
    if (!free_events_.empty()) {
      event.event_ = std::move(free_events_.top());
      free_events_.pop();
    }
  }
  if (!event.event_) {
    event.event_ = std::make_unique<se::Event>(executor);
    TF_RET_CHECK(event.event_->Init()) << "Event initialization failed";
  }
  return event;
}

void EventPool::ThenRecordEvent(se::Stream* stream, EventPool::Handle& handle) {
  absl::MutexLock lock(&mu_);
  stream->ThenRecordEvent(handle.event_.get());
  handle.sequence_number_ = next_sequence_number_++;
}

StatusOr<EventPool::Handle> EventPool::ThenAllocateAndRecordEvent(
    se::Stream* stream) {
  TF_ASSIGN_OR_RETURN(EventPool::Handle handle,
                      AllocateEvent(stream->parent()));
  ThenRecordEvent(stream, handle);
  return handle;
}

}  // namespace xla
