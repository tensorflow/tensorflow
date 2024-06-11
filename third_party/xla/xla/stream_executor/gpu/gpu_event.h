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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_EVENT_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_EVENT_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"

namespace stream_executor {
namespace gpu {

// GpuEvent wraps a GpuEventHandle in the platform-independent Event interface.
class GpuEvent : public Event {
 public:
  explicit GpuEvent(GpuExecutor* parent);

  ~GpuEvent() override;

  // Populates the CUDA-platform-specific elements of this object.
  absl::Status Init();

  // Deallocates any platform-specific elements of this object. This is broken
  // out (not part of the destructor) to allow for error reporting.
  absl::Status Destroy();

  // Inserts the event at the current position into the specified stream.
  absl::Status Record(GpuStream* stream);

  // The underlying CUDA event element.
  GpuEventHandle gpu_event();

  absl::Status WaitForEventOnExternalStream(std::intptr_t stream) override;

 protected:
  GpuExecutor* parent() const { return parent_; }

 private:
  // The Executor used to which this object and GpuEventHandle are bound.
  GpuExecutor* parent_;

  // The underlying CUDA event element.
  GpuEventHandle gpu_event_;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_EVENT_H_
