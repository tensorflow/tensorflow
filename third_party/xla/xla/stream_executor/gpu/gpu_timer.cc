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

#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>
#include <string_view>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/utility/utility.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_timer_kernel.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/stream.h"
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

bool ShouldLaunchDelayKernel() {
  // Only launch the delay kernel if CUDA_LAUNCH_BLOCKING is not set to 1.
  static bool value = [] {
    const char* blocking = std::getenv("CUDA_LAUNCH_BLOCKING");
    return !blocking || std::string_view{blocking} != "1";
  }();
  return value;
}

absl::Status CreateGpuTimerParts(GpuStream* stream, bool use_delay_kernel,
                                 GpuContext* context,
                                 GpuEventHandle& start_event,
                                 GpuEventHandle& stop_event,
                                 GpuSemaphore& semaphore) {
  TF_RETURN_IF_ERROR(GpuDriver::InitEvent(context, &start_event,
                                          GpuDriver::EventFlags::kDefault));
  TF_RETURN_IF_ERROR(GpuDriver::InitEvent(context, &stop_event,
                                          GpuDriver::EventFlags::kDefault));
  CHECK(start_event != nullptr && stop_event != nullptr);
  if (!use_delay_kernel) {
    LOG(WARNING)
        << "Skipping the delay kernel, measurement accuracy will be reduced";
  }

  if (use_delay_kernel && ShouldLaunchDelayKernel()) {
    TF_ASSIGN_OR_RETURN(bool is_supported, DelayKernelIsSupported(stream));

    if (is_supported) {
      TF_ASSIGN_OR_RETURN(semaphore, LaunchDelayKernel(stream));
    }
  }

  // The start event goes after the delay kernel in the stream
  TF_RETURN_IF_ERROR(
      GpuDriver::RecordEvent(context, start_event, stream->gpu_stream()));
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<std::unique_ptr<EventBasedTimer>>
GpuTimer::CreateEventBasedTimer(GpuStream* stream, GpuContext* context,
                                bool use_delay_kernel) {
  GpuEventHandle start_event = nullptr;
  GpuEventHandle stop_event = nullptr;
  GpuSemaphore semaphore{};
  TF_RETURN_IF_ERROR(CreateGpuTimerParts(stream, use_delay_kernel, context,
                                         start_event, stop_event, semaphore));
  return std::make_unique<GpuTimer>(context, start_event, stop_event, stream,
                                    std::move(semaphore));
}

/*static*/ void GpuTimer::ReturnRandomDurationsForTesting() {
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
  if (start_event_ != nullptr) {
    absl::Status status = GpuDriver::DestroyEvent(context_, &start_event_);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
  }
  if (stop_event_ != nullptr) {
    absl::Status status = GpuDriver::DestroyEvent(context_, &stop_event_);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
  }
}

absl::StatusOr<absl::Duration> GpuTimer::GetElapsedDuration() {
  if (is_stopped_) {
    return absl::InternalError("Measuring inactive timer");
  }
  TF_RETURN_IF_ERROR(
      GpuDriver::RecordEvent(context_, stop_event_, stream_->gpu_stream()));
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
  float elapsed_milliseconds = NAN;
  if (!GpuDriver::GetEventElapsedTime(context_, &elapsed_milliseconds,
                                      start_event_, stop_event_)) {
    return absl::InternalError("Error stopping the timer");
  }
  is_stopped_ = true;
  if (return_random_durations) {
    return RandomDuration();
  }
  return absl::Milliseconds(elapsed_milliseconds);
}

}  // namespace gpu
}  // namespace stream_executor
