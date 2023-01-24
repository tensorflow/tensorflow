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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_EVENT_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_EVENT_H_

#include "tensorflow/compiler/xla/stream_executor/event.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/tsl/platform/status.h"

namespace stream_executor {
namespace gpu {

// GpuEvent wraps a GpuEventHandle in the platform-independent EventInterface
// interface.
class GpuEvent : public internal::EventInterface {
 public:
  explicit GpuEvent(GpuExecutor* parent);

  ~GpuEvent() override;

  // Populates the CUDA-platform-specific elements of this object.
  tsl::Status Init();

  // Deallocates any platform-specific elements of this object. This is broken
  // out (not part of the destructor) to allow for error reporting.
  tsl::Status Destroy();

  // Inserts the event at the current position into the specified stream.
  tsl::Status Record(GpuStream* stream);

  // Polls the CUDA platform for the event's current status.
  Event::Status PollForStatus();

  // The underlying CUDA event element.
  GpuEventHandle gpu_event();

 private:
  // The Executor used to which this object and GpuEventHandle are bound.
  GpuExecutor* parent_;

  // The underlying CUDA event element.
  GpuEventHandle gpu_event_;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_EVENT_H_
