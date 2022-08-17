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
#include "tensorflow/compiler/xla/stream_executor/lib/status.h"

namespace stream_executor {
namespace gpu {

bool GpuTimer::Init() {
  CHECK(start_event_ == nullptr && stop_event_ == nullptr);
  GpuContext* context = parent_->gpu_context();
  port::Status status = GpuDriver::InitEvent(context, &start_event_,
                                             GpuDriver::EventFlags::kDefault);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return false;
  }

  status = GpuDriver::InitEvent(context, &stop_event_,
                                GpuDriver::EventFlags::kDefault);
  if (!status.ok()) {
    LOG(ERROR) << status;
    status = GpuDriver::DestroyEvent(context, &start_event_);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
    return false;
  }

  CHECK(start_event_ != nullptr && stop_event_ != nullptr);
  return true;
}

void GpuTimer::Destroy() {
  GpuContext* context = parent_->gpu_context();
  port::Status status = GpuDriver::DestroyEvent(context, &start_event_);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  status = GpuDriver::DestroyEvent(context, &stop_event_);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
}

float GpuTimer::GetElapsedMilliseconds() const {
  CHECK(start_event_ != nullptr && stop_event_ != nullptr);
  // TODO(leary) provide a way to query timer resolution?
  // CUDA docs say a resolution of about 0.5us
  float elapsed_milliseconds = NAN;
  (void)GpuDriver::GetEventElapsedTime(
      parent_->gpu_context(), &elapsed_milliseconds, start_event_, stop_event_);
  return elapsed_milliseconds;
}

bool GpuTimer::Start(GpuStream* stream) {
  port::Status status = GpuDriver::RecordEvent(
      parent_->gpu_context(), start_event_, stream->gpu_stream());
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool GpuTimer::Stop(GpuStream* stream) {
  port::Status status = GpuDriver::RecordEvent(
      parent_->gpu_context(), stop_event_, stream->gpu_stream());
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

}  // namespace gpu
}  // namespace stream_executor
