/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/gpu_event.h"

#include "absl/status/status.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"

namespace stream_executor {
namespace gpu {

GpuEvent::GpuEvent(GpuExecutor* parent)
    : parent_(parent), gpu_event_(nullptr) {}

GpuEvent::~GpuEvent() {}

absl::Status GpuEvent::Init() {
  return GpuDriver::InitEvent(parent_->gpu_context(), &gpu_event_,
                              GpuDriver::EventFlags::kDisableTiming);
}

absl::Status GpuEvent::Destroy() {
  return GpuDriver::DestroyEvent(parent_->gpu_context(), &gpu_event_);
}

absl::Status GpuEvent::Record(GpuStream* stream) {
  return GpuDriver::RecordEvent(parent_->gpu_context(), gpu_event_,
                                stream->gpu_stream());
}

GpuEventHandle GpuEvent::gpu_event() { return gpu_event_; }

}  // namespace gpu
}  // namespace stream_executor
