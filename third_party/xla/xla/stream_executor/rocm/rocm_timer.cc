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
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<float> GetEventElapsedTime(StreamExecutor* executor,
                                          hipEvent_t start, hipEvent_t stop) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
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

RocmTimer::RocmTimer(StreamExecutor* executor, RocmEvent start_event,
                     RocmEvent stop_event, Stream* stream)
    : executor_(executor),
      stream_(stream),
      start_event_(std::move(start_event)),
      stop_event_(std::move(stop_event)) {}

absl::StatusOr<absl::Duration> RocmTimer::GetElapsedDuration() {
  if (is_stopped_) {
    return absl::FailedPreconditionError("Measuring inactive timer");
  }
  TF_RETURN_IF_ERROR(stream_->RecordEvent(&stop_event_));
  TF_ASSIGN_OR_RETURN(float elapsed_milliseconds,
                      GetEventElapsedTime(executor_, start_event_.GetHandle(),
                                          stop_event_.GetHandle()));
  is_stopped_ = true;
  return absl::Milliseconds(elapsed_milliseconds);
}

absl::StatusOr<RocmTimer> RocmTimer::Create(StreamExecutor* executor,
                                            Stream* stream) {
  TF_ASSIGN_OR_RETURN(RocmEvent start_event,
                      RocmEvent::Create(executor, /*allow_timing=*/true));
  TF_ASSIGN_OR_RETURN(RocmEvent stop_event,
                      RocmEvent::Create(executor, /*allow_timing=*/true));
  TF_RETURN_IF_ERROR(stream->RecordEvent(&start_event));
  return RocmTimer(executor, std::move(start_event), std::move(stop_event),
                   stream);
}
}  // namespace stream_executor::gpu
