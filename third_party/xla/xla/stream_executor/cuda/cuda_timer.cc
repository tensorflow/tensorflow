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

#include "xla/stream_executor/cuda/cuda_timer.h"

#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/cuda/delay_kernel.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {

namespace {
absl::StatusOr<float> GetEventElapsedTime(Context* context, CUevent start,
                                          CUevent stop) {
  ScopedActivateContext activated{context};
  // The stop event must have completed in order for cuEventElapsedTime to
  // work.
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuEventSynchronize(stop)));

  float elapsed_milliseconds;

  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuEventElapsedTime(&elapsed_milliseconds, start, stop)));

  return elapsed_milliseconds;
}

}  // namespace

CudaTimer::CudaTimer(Context* context, CudaEvent start_event,
                     CudaEvent stop_event, GpuStream* stream,
                     GpuSemaphore semaphore)
    : semaphore_(std::move(semaphore)),
      context_(context),
      stream_(stream),
      start_event_(std::move(start_event)),
      stop_event_(std::move(stop_event)) {}

CudaTimer::~CudaTimer() {
  if (semaphore_ && !is_stopped_) {
    // Signal the delay kernel that it can exit
    *semaphore_ = GpuSemaphoreState::kRelease;
    // Wait for the delay kernel to exit before destroying the value that it is
    // watching.
    absl::Status result = stream_->BlockHostUntilDone();
    if (!result.ok()) {
      LOG(ERROR) << result.message();
    }
  }
}

absl::StatusOr<absl::Duration> CudaTimer::GetElapsedDuration() {
  if (is_stopped_) {
    return absl::FailedPreconditionError("Measuring inactive timer");
  }
  TF_RETURN_IF_ERROR(stream_->RecordEvent(&stop_event_));
  // If we launched the delay kernel then check if it already timed out.
  if (semaphore_) {
    if (*semaphore_ == GpuSemaphoreState::kTimedOut) {
      // The delay kernel did not achieve the intended result.
      LOG(ERROR) << "Delay kernel timed out: measured time has sub-optimal "
                    "accuracy. There may be a missing warmup execution, please "
                    "investigate in Nsight Systems.";
    } else {
      // Signal that the kernel can exit
      *semaphore_ = GpuSemaphoreState::kRelease;
    }
  }
  TF_ASSIGN_OR_RETURN(float elapsed_milliseconds,
                      GetEventElapsedTime(context_, start_event_.GetHandle(),
                                          stop_event_.GetHandle()));
  is_stopped_ = true;
  return absl::Milliseconds(elapsed_milliseconds);
}

absl::StatusOr<CudaTimer> CudaTimer::Create(Context* context, GpuStream* stream,
                                            TimerType timer_type) {
  GpuSemaphore semaphore{};

  if (timer_type == TimerType::kDelayKernel) {
    TF_ASSIGN_OR_RETURN(semaphore, LaunchDelayKernel(stream));
  }

  TF_ASSIGN_OR_RETURN(CudaEvent start_event,
                      CudaEvent::Create(context, /*allow_timing=*/true));
  TF_ASSIGN_OR_RETURN(CudaEvent stop_event,
                      CudaEvent::Create(context, /*allow_timing=*/true));

  TF_RETURN_IF_ERROR(stream->RecordEvent(&start_event));

  return CudaTimer(context, std::move(start_event), std::move(stop_event),
                   stream, std::move(semaphore));
}

}  // namespace stream_executor::gpu
