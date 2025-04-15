/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_HOST_HOST_EVENT_H_
#define XLA_STREAM_EXECUTOR_HOST_HOST_EVENT_H_

#include <memory>

#include "absl/synchronization/notification.h"
#include "xla/stream_executor/event.h"

namespace stream_executor {

// This class is a host-side implementation of the Event interface. It is
// intended to be used with the HostStream implementation.
class HostEvent : public Event {
 public:
  HostEvent() : notification_(std::make_shared<absl::Notification>()) {}

  std::shared_ptr<absl::Notification>& notification() { return notification_; }

  Status PollForStatus() override {
    return notification_->HasBeenNotified() ? Event::Status::kComplete
                                            : Event::Status::kPending;
  }

 private:
  // We use a std::shared_ptr here because the client may delete the HostEvent
  // object while there are still RecordEvent and WaitForEvent callbacks pending
  // on a stream.
  std::shared_ptr<absl::Notification> notification_;
};
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_EVENT_H_
