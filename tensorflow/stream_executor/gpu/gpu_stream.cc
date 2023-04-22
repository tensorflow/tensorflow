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

#include "tensorflow/stream_executor/gpu/gpu_stream.h"

#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/stream.h"

namespace stream_executor {
namespace gpu {

bool GpuStream::Init() {
  if (!GpuDriver::CreateStream(parent_->gpu_context(), &gpu_stream_,
                               priority_)) {
    return false;
  }
  return GpuDriver::InitEvent(parent_->gpu_context(), &completed_event_,
                              GpuDriver::EventFlags::kDisableTiming)
      .ok();
}

void GpuStream::Destroy() {
  if (completed_event_ != nullptr) {
    port::Status status =
        GpuDriver::DestroyEvent(parent_->gpu_context(), &completed_event_);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
    }
  }

  GpuDriver::DestroyStream(parent_->gpu_context(), &gpu_stream_);
}

bool GpuStream::IsIdle() const {
  return GpuDriver::IsStreamIdle(parent_->gpu_context(), gpu_stream_);
}

GpuStream* AsGpuStream(Stream* stream) {
  DCHECK(stream != nullptr);
  return static_cast<GpuStream*>(stream->implementation());
}

GpuStreamHandle AsGpuStreamValue(Stream* stream) {
  DCHECK(stream != nullptr);
  return AsGpuStream(stream)->gpu_stream();
}

}  // namespace gpu
}  // namespace stream_executor
