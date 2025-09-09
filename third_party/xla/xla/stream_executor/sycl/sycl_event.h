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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_EVENT_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_EVENT_H_

#include <sycl/sycl.hpp>

#include "absl/status/statusor.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {

// This class implements the Event class for SYCL devices.
class SyclEvent : public Event {
 public:
  Event::Status PollForStatus() override;

  // Waits for the event to complete on the specified stream.
  static absl::Status WaitStreamOnEvent(StreamExecutor* executor,
                                        sycl::queue* stream_handle,
                                        const sycl::event& event);

  // Waits for the event to complete on an external stream.
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream) override;

  // Creates a SyclEvent instance and initializes it with a default
  // constructed sycl::event that has no dependencies and associated commands.
  static absl::StatusOr<SyclEvent> Create(StreamExecutor* executor);

  sycl::event GetEvent() const { return event_; }

  // We don't need a destructor for sycl::event since it is handled by the SYCL
  // runtime.
  ~SyclEvent() = default;

  // Ensure SyclEvent is moveable but not copyable.
  SyclEvent(const SyclEvent&) = delete;
  SyclEvent& operator=(const SyclEvent&) = delete;
  SyclEvent(SyclEvent&& other) noexcept;
  SyclEvent& operator=(SyclEvent&& other) noexcept;

 private:
  explicit SyclEvent(StreamExecutor* executor, const sycl::event& event)
      : executor_(executor), event_(event) {}

  // The Executor used to which this object and sycl::event are bound.
  StreamExecutor* executor_;

  // The underlying SYCL event.
  sycl::event event_;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_EVENT_H_
