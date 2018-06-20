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

#include "tensorflow/stream_executor/cuda/cuda_event.h"

#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace stream_executor {
namespace cuda {

CUDAEvent::CUDAEvent(CUDAExecutor* parent)
    : parent_(parent), cuda_event_(nullptr) {}

CUDAEvent::~CUDAEvent() {}

port::Status CUDAEvent::Init() {
  return CUDADriver::CreateEvent(parent_->cuda_context(), &cuda_event_,
                                 CUDADriver::EventFlags::kDisableTiming);
}

port::Status CUDAEvent::Destroy() {
  return CUDADriver::DestroyEvent(parent_->cuda_context(), &cuda_event_);
}

port::Status CUDAEvent::Record(CUDAStream* stream) {
  return CUDADriver::RecordEvent(parent_->cuda_context(), cuda_event_,
                                 stream->cuda_stream());
}

Event::Status CUDAEvent::PollForStatus() {
  port::StatusOr<CUresult> status =
      CUDADriver::QueryEvent(parent_->cuda_context(), cuda_event_);
  if (!status.ok()) {
    LOG(ERROR) << "Error polling for event status: "
               << status.status().error_message();
    return Event::Status::kError;
  }

  switch (status.ValueOrDie()) {
    case CUDA_SUCCESS:
      return Event::Status::kComplete;
    case CUDA_ERROR_NOT_READY:
      return Event::Status::kPending;
    default:
      LOG(INFO) << "Error condition returned for event status: "
                << status.ValueOrDie();
      return Event::Status::kError;
  }
}

const CUevent& CUDAEvent::cuda_event() {
  return cuda_event_;
}

}  // namespace cuda
}  // namespace stream_executor
