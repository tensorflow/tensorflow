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

#include "xla/stream_executor/rocm/rocm_timer.h"

#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<float> GetEventElapsedTime(Context* context, hipEvent_t start,
                                          hipEvent_t stop) {
  ScopedActivateContext activated{context};
  // The stop event must have completed in order for hipEventElapsedTime to
  // work.
  hipError_t res = wrap::hipEventSynchronize(stop);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to synchronize the stop event: " << ToString(res);
    return false;
  }
  float elapsed_milliseconds;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipEventElapsedTime(&elapsed_milliseconds, start, stop),
               "failed to get elapsed time between events"));

  return elapsed_milliseconds;
}
}  // namespace

RocmTimer::RocmTimer(Context* context, std::unique_ptr<RocmEvent> start_event,
                     std::unique_ptr<RocmEvent> stop_event, GpuStream* stream)
    : context_(context),
      stream_(stream),
      start_event_(std::move(start_event)),
      stop_event_(std::move(stop_event)) {}

absl::StatusOr<absl::Duration> RocmTimer::GetElapsedDuration() {
  if (is_stopped_) {
    return absl::InternalError("Measuring inactive timer");
  }
  TF_RETURN_IF_ERROR(stream_->RecordEvent(stop_event_.get()));
  TF_ASSIGN_OR_RETURN(float elapsed_milliseconds,
                      GetEventElapsedTime(context_, start_event_->GetHandle(),
                                          stop_event_->GetHandle()));
  is_stopped_ = true;
  return absl::Milliseconds(elapsed_milliseconds);
}
}  // namespace stream_executor::gpu
