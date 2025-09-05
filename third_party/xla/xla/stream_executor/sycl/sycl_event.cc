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

#include "xla/stream_executor/sycl/sycl_event.h"

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/event.h"

namespace stream_executor {
namespace gpu {

Event::Status SyclEvent::PollForStatus() {
  try {
    auto event_status =
        event_.get_info<sycl::info::event::command_execution_status>();
    switch (event_status) {
      case sycl::info::event_command_status::submitted: {
        VLOG(2)
            << "Command is submitted to the queue but not yet running on the "
               "device.";
        return Event::Status::kPending;
      }
      case sycl::info::event_command_status::running: {
        VLOG(2) << "Command has started running on the device but has not yet "
                   "completed.";
        return Event::Status::kPending;
      }
      case sycl::info::event_command_status::complete: {
        VLOG(2) << "Command has finished running on the device.";
        return Event::Status::kComplete;
      }
      default: {
        LOG(ERROR) << "Event status is unknown: "
                   << static_cast<int>(event_status);
        return Event::Status::kUnknown;
      }
    }
  } catch (const sycl::exception& e) {
    LOG(ERROR) << "SYCL exception while polling event status: " << e.what()
               << " (error code: " << e.code() << ")";
    return Event::Status::kError;
  }
}

absl::Status SyclEvent::WaitStreamOnEvent(StreamExecutor* executor,
                                          sycl::queue* stream_handle,
                                          const sycl::event& event) {
  // No need to call executor->Activate() since the SYCL context need not
  // be activated explicitly.
  if (stream_handle == nullptr) {
    return absl::InternalError(
        "WaitStreamOnEvent: Stream handle is not initialized.");
  }
  std::vector<sycl::event> event_list{event};
  stream_handle->submit([&](sycl::handler& cgh) {
    cgh.depends_on(event_list);
    cgh.host_task([=]() {});
  });
  return absl::OkStatus();
}

absl::Status SyclEvent::WaitForEventOnExternalStream(std::intptr_t stream) {
  sycl::queue* queue_ptr = absl::bit_cast<sycl::queue*>(stream);
  return WaitStreamOnEvent(executor_, queue_ptr, event_);
}

absl::StatusOr<SyclEvent> SyclEvent::Create(StreamExecutor* executor) {
  // SYCL reports synchronous (host-side) errors via exceptions, so we catch
  // them and return an error status.
  try {
    // Initialize with a default-constructed sycl::event.
    return SyclEvent(executor, sycl::event());
  } catch (const sycl::exception& e) {
    LOG(ERROR) << "SYCL exception while creating event: " << e.what()
               << " (error code: " << e.code() << ")";
    return absl::InternalError(
        absl::StrCat("Failed to create SYCL event: ", e.what()));
  }
}

SyclEvent::SyclEvent(SyclEvent&& other) noexcept
    : executor_(other.executor_), event_(other.event_) {
  other.executor_ = nullptr;
}

SyclEvent& SyclEvent::operator=(SyclEvent&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  executor_ = other.executor_;
  event_ = other.event_;
  other.executor_ = nullptr;
  return *this;
}

}  // namespace gpu
}  // namespace stream_executor
