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

#include "tensorflow/stream_executor/gpu/gpu_event.h"

#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace stream_executor {
namespace gpu {

GpuEvent::GpuEvent(GpuExecutor* parent)
    : parent_(parent), gpu_event_(nullptr) {}

GpuEvent::~GpuEvent() {}

port::Status GpuEvent::Init() {
  return GpuDriver::CreateEvent(parent_->gpu_context(), &gpu_event_,
                                GpuDriver::EventFlags::kDisableTiming);
}

port::Status GpuEvent::Destroy() {
  return GpuDriver::DestroyEvent(parent_->gpu_context(), &gpu_event_);
}

port::Status GpuEvent::Record(GpuStream* stream) {
  return GpuDriver::RecordEvent(parent_->gpu_context(), gpu_event_,
                                stream->gpu_stream());
}

GpuEventHandle GpuEvent::gpu_event() { return gpu_event_; }

}  // namespace gpu
}  // namespace stream_executor
