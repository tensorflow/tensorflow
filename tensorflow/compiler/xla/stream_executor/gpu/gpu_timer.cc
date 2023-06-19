/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_timer.h"

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/tsl/platform/status.h"

namespace stream_executor {
namespace gpu {

tsl::StatusOr<GpuTimer> GpuTimer::Create(GpuStream* stream) {
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
  return tsl::StatusOr<GpuTimer>{absl::in_place, parent, start_event,
                                 stop_event, stream};
}

GpuTimer::~GpuTimer() {
  GpuContext* context = parent_->gpu_context();
  tsl::Status status = GpuDriver::DestroyEvent(context, &start_event_);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  status = GpuDriver::DestroyEvent(context, &stop_event_);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
}

float GpuTimer::GetElapsedMilliseconds() const {
  CHECK(elapsed_milliseconds_.has_value());
  return *elapsed_milliseconds_;
}

tsl::Status GpuTimer::Stop() {
  TF_RETURN_IF_ERROR(GpuDriver::RecordEvent(parent_->gpu_context(), stop_event_,
                                            stream_->gpu_stream()));

  if (elapsed_milliseconds_.has_value()) {
    return absl::InternalError("Timer already stoppped");
  }
  float elapsed_milliseconds = NAN;
  if (!GpuDriver::GetEventElapsedTime(parent_->gpu_context(),
                                      &elapsed_milliseconds, start_event_,
                                      stop_event_)) {
    return absl::InternalError("Error stopping the timer");
  }
  elapsed_milliseconds_ = elapsed_milliseconds;
  return tsl::OkStatus();
}

}  // namespace gpu
}  // namespace stream_executor
