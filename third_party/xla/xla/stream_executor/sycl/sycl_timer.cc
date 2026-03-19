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

#include "xla/stream_executor/sycl/sycl_timer.h"

#include <level_zero/ze_api.h>

#include "xla/stream_executor/activate_context.h"

constexpr int kMsecInSec = 1000;

namespace stream_executor::sycl {

namespace {

absl::StatusOr<float> GetEventElapsedTime(StreamExecutor* executor,
                                          const ::sycl::event& start,
                                          const ::sycl::event& stop) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  // Get the native Level Zero event handles.
  ze_event_handle_t start_event =
      ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(start);
  ze_event_handle_t end_event =
      ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(stop);

  // Synchronize the events to ensure they are complete.
  ze_result_t sync_result = zeEventHostSynchronize(end_event, UINT64_MAX);
  if (sync_result != ZE_RESULT_SUCCESS) {
    return absl::InternalError(
        "GetEventElapsedTime: Failed to synchronize end event");
  }

  // Query the kernel timestamps for the start and end events.
  ze_kernel_timestamp_result_t start_timestamp{}, end_timestamp{};
  if (zeEventQueryKernelTimestamp(start_event, &start_timestamp) !=
          ZE_RESULT_SUCCESS ||
      zeEventQueryKernelTimestamp(end_event, &end_timestamp) !=
          ZE_RESULT_SUCCESS) {
    return absl::InternalError(
        "GetEventElapsedTime: Failed to query kernel timestamps for events");
  }

  // Get the frequency and mask for the device to convert timestamps to
  // milliseconds.
  // We assume that all SYCL devices have the same frequency and mask, so
  // we use kDefaultDeviceOrdinal.
  TF_ASSIGN_OR_RETURN(SyclTimerProperties timer_props,
                      SyclGetTimerProperties(kDefaultDeviceOrdinal));

  const uint64_t kernel_start_time = start_timestamp.global.kernelStart;
  const uint64_t kernel_end_time = end_timestamp.global.kernelEnd;
  uint64_t elapsed_ticks;
  if (kernel_start_time < kernel_end_time) {
    elapsed_ticks = kernel_end_time - kernel_start_time;
  } else {
    elapsed_ticks = (timer_props.timestamp_mask + 1ull) + kernel_end_time -
                    kernel_start_time;
  }
  float elapsed_milliseconds =
      static_cast<float>(elapsed_ticks) * kMsecInSec / timer_props.frequency_hz;

  VLOG(1) << "Frequency: " << timer_props.frequency_hz
          << ", mask: " << timer_props.timestamp_mask;
  VLOG(1) << "The duration between start and stop events is "
          << elapsed_milliseconds << " ms.";
  return elapsed_milliseconds;
}

}  // namespace

SyclTimer::SyclTimer(StreamExecutor* executor, SyclEvent start_event,
                     SyclEvent stop_event, Stream* stream)
    : executor_(executor),
      stream_(stream),
      start_event_(std::move(start_event)),
      stop_event_(std::move(stop_event)) {}

absl::StatusOr<absl::Duration> SyclTimer::GetElapsedDuration() {
  if (is_timer_stopped_) {
    return absl::FailedPreconditionError("Measuring inactive timer");
  }
  TF_RETURN_IF_ERROR(stream_->RecordEvent(&stop_event_));
  TF_ASSIGN_OR_RETURN(float elapsed_milliseconds,
                      GetEventElapsedTime(executor_, start_event_.GetEvent(),
                                          stop_event_.GetEvent()));
  is_timer_stopped_ = true;
  return absl::Milliseconds(elapsed_milliseconds);
}

absl::StatusOr<SyclTimer> SyclTimer::Create(StreamExecutor* executor,
                                            Stream* stream) {
  TF_ASSIGN_OR_RETURN(SyclEvent start_event, SyclEvent::Create(executor));
  TF_ASSIGN_OR_RETURN(SyclEvent stop_event, SyclEvent::Create(executor));
  TF_RETURN_IF_ERROR(stream->RecordEvent(&start_event));
  return SyclTimer(executor, std::move(start_event), std::move(stop_event),
                   stream);
}

}  // namespace stream_executor::sycl
