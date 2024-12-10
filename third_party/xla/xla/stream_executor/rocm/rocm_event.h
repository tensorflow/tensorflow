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

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// This class implements Event for ROCm devices.
class RocmEvent : public Event {
 public:
  Event::Status PollForStatus() override;
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream) override;

  // Creates a new RocmEvent. If allow_timing is false, the event will not
  // support timing, which is cheaper to create.
  static absl::StatusOr<RocmEvent> Create(StreamExecutor* executor,
                                          bool allow_timing);

  hipEvent_t GetHandle() const { return handle_; }

  ~RocmEvent() override;
  RocmEvent(const RocmEvent&) = delete;
  RocmEvent& operator=(const RocmEvent&) = delete;
  RocmEvent(RocmEvent&& other);
  RocmEvent& operator=(RocmEvent&& other);

 private:
  explicit RocmEvent(StreamExecutor* executor, hipEvent_t handle)
      : executor_(executor), handle_(handle) {}

  // The Executor used to which this object and hipEvent_t are bound.
  StreamExecutor* executor_;

  // The underlying CUDA event handle.
  hipEvent_t handle_;
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_EVENT_H_
