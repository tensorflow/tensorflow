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
#include <optional>
#include <random>
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
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
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

/*deprecated*/ /*static*/ absl::StatusOr<GpuTimer> GpuTimer::Create(
    GpuStream* stream) {
  GpuExecutor* parent = stream->parent();
  GpuContext* context = parent->gpu_context();
  GpuEventHandle start_event;
  TF_RETURN_IF_ERROR(GpuDriver::InitEvent(context, &start_event,
                                          GpuDriver::EventFlags::kDefault));
  GpuEventHandle stop_event;
  TF_RETURN_IF_ERROR(GpuDriver::InitEvent(context, &stop_event,
                                          GpuDriver::EventFlags::kDefault));
  CHECK(start_event != nullptr && stop_event != nullptr);
  TF_RETURN_IF_ERROR(GpuDriver::RecordEvent(parent->gpu_context(), start_event,
                                            stream->gpu_stream()));
  return absl::StatusOr<GpuTimer>{absl::in_place, parent, start_event,
                                  stop_event, stream};
}

/*deprecated*/ /*static*/ absl::StatusOr<std::optional<GpuTimer>>
GpuTimer::CreateIfNeeded(GpuStream* stream, bool is_needed) {
  if (is_needed) {
    TF_ASSIGN_OR_RETURN(GpuTimer t, GpuTimer::Create(stream));
    return {std::make_optional(std::move(t))};
  }
  return std::nullopt;
}

[[deprecated("So it can quietly call a deprecated method")]] /*static*/ absl::
    StatusOr<GpuTimer>
    GpuTimer::Create(Stream* stream) {
  return GpuTimer::Create(AsGpuStream(stream));
}

[[deprecated("So it can quietly call a deprecated method")]] /*static*/ absl::
    StatusOr<std::optional<GpuTimer>>
    GpuTimer::CreateIfNeeded(Stream* stream, bool is_needed) {
  return GpuTimer::CreateIfNeeded(AsGpuStream(stream), is_needed);
}

/*static*/ void GpuTimer::ReturnRandomDurationsForTesting() {
  return_random_durations = true;
}

GpuTimer::~GpuTimer() {
  GpuContext* context = parent_->gpu_context();
  if (start_event_ != nullptr) {
    absl::Status status = GpuDriver::DestroyEvent(context, &start_event_);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
  }
  if (stop_event_ != nullptr) {
    absl::Status status = GpuDriver::DestroyEvent(context, &stop_event_);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
  }
}

absl::StatusOr<absl::Duration> GpuTimer::GetElapsedDuration() {
  if (is_stopped_) {
    return absl::InternalError("Measuring inactive timer");
  }
  TF_RETURN_IF_ERROR(GpuDriver::RecordEvent(parent_->gpu_context(), stop_event_,
                                            stream_->gpu_stream()));
  float elapsed_milliseconds = NAN;
  if (!GpuDriver::GetEventElapsedTime(parent_->gpu_context(),
                                      &elapsed_milliseconds, start_event_,
                                      stop_event_)) {
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
