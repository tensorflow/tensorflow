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

#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_event.h"

#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

Event::Status GpuEvent::PollForStatus() {
  tsl::StatusOr<CUresult> status =
      GpuDriver::QueryEvent(parent_->gpu_context(), gpu_event_);
  if (!status.ok()) {
    LOG(ERROR) << "Error polling for event status: "
               << status.status().error_message();
    return Event::Status::kError;
  }

  switch (status.value()) {
    case CUDA_SUCCESS:
      return Event::Status::kComplete;
    case CUDA_ERROR_NOT_READY:
      return Event::Status::kPending;
    default:
      LOG(INFO) << "Error condition returned for event status: "
                << status.value();
      return Event::Status::kError;
  }
}

}  // namespace gpu
}  // namespace stream_executor
