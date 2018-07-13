/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_EVENT_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_EVENT_H_

#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/rocm/rocm_driver.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"

namespace stream_executor {
namespace rocm {

// ROCMEvent wraps a hipEvent_t in the platform-independent EventInterface
// interface.
class ROCMEvent : public internal::EventInterface {
 public:
  explicit ROCMEvent(ROCMExecutor* parent);

  ~ROCMEvent() override;

  // Populates the ROCM-platform-specific elements of this object.
  port::Status Init();

  // Deallocates any platform-specific elements of this object. This is broken
  // out (not part of the destructor) to allow for error reporting.
  port::Status Destroy();

  // Inserts the event at the current position into the specified stream.
  port::Status Record(ROCMStream* stream);

  // Polls the ROCM platform for the event's current status.
  Event::Status PollForStatus();

  // The underlying ROCM event element.
  const hipEvent_t& rocm_event();

 private:
  // The Executor used to which this object and hipEvent_t are bound.
  ROCMExecutor* parent_;

  // The underlying ROCM event element.
  hipEvent_t rocm_event_;
};
}  // namespace rocm
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_EVENT_H_
