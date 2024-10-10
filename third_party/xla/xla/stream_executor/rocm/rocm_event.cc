/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_event.h"

#include <cstdint>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {
namespace {
absl::Status WaitStreamOnEvent(Context* context, hipStream_t stream,
                               hipEvent_t event) {
  ScopedActivateContext activation{context};
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipStreamWaitEvent(stream, event, 0 /* = flags */),
               "could not wait stream on event"));
  return absl::OkStatus();
}

enum class EventFlags { kDefault, kDisableTiming };
absl::StatusOr<hipEvent_t> InitEvent(Context* context, EventFlags flags) {
  int hipflags;
  switch (flags) {
    case EventFlags::kDefault:
      hipflags = hipEventDefault;
      break;
    case EventFlags::kDisableTiming:
      hipflags = hipEventDisableTiming | hipEventReleaseToSystem;
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(hipflags);
  }

  ScopedActivateContext activated{context};
  hipEvent_t event;
  hipError_t res = wrap::hipEventCreateWithFlags(&event, hipflags);

  if (res == hipSuccess) {
    return event;
  }
  if (res == hipErrorMemoryAllocation) {
    return absl::ResourceExhaustedError(
        "could not create ROCM event: out of device memory");
  }
  return absl::FailedPreconditionError(
      absl::StrCat("could not create ROCM event: ", ToString(res)));
}

void DestroyEvent(Context* context, hipEvent_t event) {
  if (event == nullptr) {
    return;
  }

  ScopedActivateContext activated{context};
  hipError_t res = wrap::hipEventDestroy(event);

  if (res != hipSuccess) {
    LOG(ERROR) << absl::StrFormat(
        "error destroying ROCM event in device %d: %s",
        context->device_ordinal(), ToString(res));
  }
}

}  // namespace

Event::Status RocmEvent::PollForStatus() {
  ScopedActivateContext activated(context_);
  hipError_t res = wrap::hipEventQuery(handle_);

  if (res == hipSuccess) {
    return Event::Status::kComplete;
  } else if (res == hipErrorNotReady) {
    return Event::Status::kPending;
  }

  return Event::Status::kError;
}

absl::Status RocmEvent::WaitForEventOnExternalStream(std::intptr_t stream) {
  return WaitStreamOnEvent(context_, absl::bit_cast<hipStream_t>(stream),
                           handle_);
}

absl::StatusOr<RocmEvent> RocmEvent::Create(Context* context,
                                            bool allow_timing) {
  TF_ASSIGN_OR_RETURN(
      hipEvent_t event_handle,
      InitEvent(context, allow_timing ? EventFlags::kDefault
                                      : EventFlags::kDisableTiming));

  return RocmEvent(context, event_handle);
}

RocmEvent::~RocmEvent() { DestroyEvent(context_, handle_); }

RocmEvent::RocmEvent(RocmEvent&& other)
    : context_(other.context_), handle_(other.handle_) {
  other.context_ = nullptr;
  other.handle_ = nullptr;
}

RocmEvent& RocmEvent::operator=(RocmEvent&& other) {
  if (this == &other) {
    return *this;
  }

  DestroyEvent(context_, handle_);

  context_ = other.context_;
  handle_ = other.handle_;
  other.context_ = nullptr;
  other.handle_ = nullptr;
  return *this;
}
}  // namespace gpu
}  // namespace stream_executor
