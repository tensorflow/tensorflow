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

#include "tensorflow/stream_executor/rocm/rocm_stream.h"

#include "tensorflow/stream_executor/rocm/rocm_gpu_executor.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/stream.h"

namespace stream_executor {
namespace rocm {

bool ROCMStream::Init() {
  if (!ROCMDriver::CreateStream(parent_->device_ordinal(), &rocm_stream_)) {
    return false;
  }
  return ROCMDriver::CreateEvent(parent_->device_ordinal(), &completed_event_,
                                 ROCMDriver::EventFlags::kDisableTiming)
      .ok();
}

void ROCMStream::Destroy() {
  if (completed_event_ != nullptr) {
    port::Status status =
        ROCMDriver::DestroyEvent(parent_->device_ordinal(), &completed_event_);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
    }
  }

  ROCMDriver::DestroyStream(parent_->device_ordinal(), &rocm_stream_);
}

bool ROCMStream::IsIdle() const {
  return ROCMDriver::IsStreamIdle(parent_->device_ordinal(), rocm_stream_);
}

ROCMStream *AsROCMStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return static_cast<ROCMStream *>(stream->implementation());
}

hipStream_t AsROCMStreamValue(Stream *stream) {
  DCHECK(stream != nullptr);
  return AsROCMStream(stream)->rocm_stream();
}

}  // namespace rocm
}  // namespace stream_executor
