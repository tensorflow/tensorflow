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
#include "absl/status/status.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "tsl/platform/errors.h"

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
}  // namespace

Event::Status RocmEvent::PollForStatus() {
  ScopedActivateContext activated(context());
  hipError_t res = wrap::hipEventQuery(gpu_event());

  if (res == hipSuccess) {
    return Event::Status::kComplete;
  } else if (res == hipErrorNotReady) {
    return Event::Status::kPending;
  }

  return Event::Status::kError;
}

absl::Status RocmEvent::WaitForEventOnExternalStream(std::intptr_t stream) {
  return WaitStreamOnEvent(context(), absl::bit_cast<hipStream_t>(stream),
                           gpu_event());
}

}  // namespace gpu
}  // namespace stream_executor
