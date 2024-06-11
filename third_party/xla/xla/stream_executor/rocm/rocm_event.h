/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_EVENT_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_EVENT_H_

#include "xla/stream_executor/gpu/gpu_event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"

namespace stream_executor::gpu {

// This class implements Event::PollForStatus for ROCm devices.
class RocmEvent : public GpuEvent {
 public:
  explicit RocmEvent(GpuExecutor *executor) : GpuEvent(executor) {}

  Event::Status PollForStatus() override;
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_EVENT_H_
