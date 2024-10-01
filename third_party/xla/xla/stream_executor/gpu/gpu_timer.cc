/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/gpu_timer.h"

#include <memory>
#include <random>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_event.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

namespace {

bool return_random_durations = false;

absl::Duration RandomDuration() {
  static absl::Mutex mu(absl::kConstInit);
  static std::mt19937 rng ABSL_GUARDED_BY(mu);
  std::uniform_real_distribution<float> distribution(10, 1000);
  absl::MutexLock l(&mu);
  return absl::Microseconds(distribution(rng));
}

}  // namespace

void GpuTimer::ReturnRandomDurationsForTesting() {
  return_random_durations = true;
}

GpuTimer::~GpuTimer() {
  if (semaphore_ && !is_stopped_) {
    // Signal the delay kernel that it can exit
    *semaphore_ = GpuSemaphoreState::kRelease;
    // Wait for the delay kernel to exit before destroying the value that it is
    // watching.
    absl::Status status =
        GpuDriver::SynchronizeStream(context_, stream_->gpu_stream());
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
  }
  start_event_.reset();
  stop_event_.reset();
}

absl::StatusOr<absl::Duration> GpuTimer::GetElapsedDuration() {
  if (is_stopped_) {
    return absl::InternalError("Measuring inactive timer");
  }
  TF_RETURN_IF_ERROR(stop_event_->Record(stream_->gpu_stream()));
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
  TF_ASSIGN_OR_RETURN(
      float elapsed_milliseconds,
      GpuDriver::GetEventElapsedTime(context_, start_event_->gpu_event(),
                                     stop_event_->gpu_event()));
  is_stopped_ = true;
  if (return_random_durations) {
    return RandomDuration();
  }
  return absl::Milliseconds(elapsed_milliseconds);
}

}  // namespace gpu
}  // namespace stream_executor
