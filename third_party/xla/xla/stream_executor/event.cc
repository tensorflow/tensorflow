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
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/stream_executor/stream_executor_interface.h"

namespace stream_executor {

Event::Event(StreamExecutorInterface* stream_exec)
    : stream_exec_(stream_exec) {}

Event::Status Event::PollForStatus() {
  return stream_exec_->PollForEventStatus(this);
}

absl::Status Event::WaitForEventOnExternalStream(std::intptr_t stream) {
  return stream_exec_->WaitForEventOnExternalStream(stream, this);
}

}  // namespace stream_executor
