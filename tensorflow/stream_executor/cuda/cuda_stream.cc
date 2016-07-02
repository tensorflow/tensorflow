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

#include "tensorflow/stream_executor/cuda/cuda_stream.h"

#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/stream.h"

namespace perftools {
namespace gputools {
namespace cuda {

bool CUDAStream::Init() {
  if (!CUDADriver::CreateStream(parent_->cuda_context(), &cuda_stream_)) {
    return false;
  }
  return CUDADriver::CreateEvent(parent_->cuda_context(), &completed_event_,
                                 CUDADriver::EventFlags::kDisableTiming)
      .ok();
}

void CUDAStream::Destroy() {
  if (completed_event_ != nullptr) {
    port::Status status =
        CUDADriver::DestroyEvent(parent_->cuda_context(), &completed_event_);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
    }
  }

  CUDADriver::DestroyStream(parent_->cuda_context(), &cuda_stream_);
}

bool CUDAStream::IsIdle() const {
  return CUDADriver::IsStreamIdle(parent_->cuda_context(), cuda_stream_);
}

CUDAStream *AsCUDAStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return static_cast<CUDAStream *>(stream->implementation());
}

CUstream AsCUDAStreamValue(Stream *stream) {
  DCHECK(stream != nullptr);
  return AsCUDAStream(stream)->cuda_stream();
}

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools
