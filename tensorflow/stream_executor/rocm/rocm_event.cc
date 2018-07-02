/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/rocm/rocm_event.h"

#include "tensorflow/stream_executor/rocm/rocm_gpu_executor.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace stream_executor {
namespace rocm {

ROCMEvent::ROCMEvent(ROCMExecutor* parent)
    : parent_(parent), rocm_event_(nullptr) {}

ROCMEvent::~ROCMEvent() {}

port::Status ROCMEvent::Init() {
  return ROCMDriver::CreateEvent(parent_->device_ordinal(), &rocm_event_,
                                 ROCMDriver::EventFlags::kDisableTiming);
}

port::Status ROCMEvent::Destroy() {
  return ROCMDriver::DestroyEvent(parent_->device_ordinal(), &rocm_event_);
}

port::Status ROCMEvent::Record(ROCMStream* stream) {
  return ROCMDriver::RecordEvent(parent_->device_ordinal(), rocm_event_,
                                 stream->rocm_stream());
}

Event::Status ROCMEvent::PollForStatus() {
  port::StatusOr<hipError_t> status =
      ROCMDriver::QueryEvent(parent_->device_ordinal(), rocm_event_);
  if (!status.ok()) {
    LOG(ERROR) << "Error polling for event status: "
               << status.status().error_message();
    return Event::Status::kError;
  }

  switch (status.ValueOrDie()) {
    case hipSuccess:
      return Event::Status::kComplete;
    case hipErrorNotReady:
      return Event::Status::kPending;
    default:
      LOG(INFO) << "Error condition returned for event status: "
                << status.ValueOrDie();
      return Event::Status::kError;
  }
}

const hipEvent_t& ROCMEvent::rocm_event() {
  return rocm_event_;
}

}  // namespace rocm
}  // namespace stream_executor
