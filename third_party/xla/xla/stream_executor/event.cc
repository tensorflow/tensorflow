/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/event.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/stream_executor/event_interface.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

Event::Event(StreamExecutor* stream_exec)
    : stream_exec_(stream_exec),
      implementation_(
          stream_exec_->implementation()->CreateEventImplementation()) {}

Event::~Event() {
  // Deal with nullptr implementation_, as this event may have been std::moved.
  if (stream_exec_ && implementation_) {
    auto status = stream_exec_->DeallocateEvent(this);
    if (!status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
}

Event::Event(Event&&) = default;
Event& Event::operator=(Event&&) = default;

bool Event::Init() {
  auto status = stream_exec_->AllocateEvent(this);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
    return false;
  }

  return true;
}

Event::Status Event::PollForStatus() {
  return stream_exec_->PollForEventStatus(this);
}

absl::Status Event::WaitForEventOnExternalStream(std::intptr_t stream) {
  return stream_exec_->WaitForEventOnExternalStream(stream, this);
}

}  // namespace stream_executor
