/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_EVENT_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_EVENT_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/context.h"

namespace stream_executor::gpu {

class GpuContext;

// This class implements Event::PollForStatus for CUDA devices.
class CudaEvent : public Event {
 public:
  Event::Status PollForStatus() override;
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream) override;

  // Creates a new CudaEvent. If allow_timing is false, the event will not
  // support timing, which is cheaper to create.
  static absl::StatusOr<CudaEvent> Create(Context* context, bool allow_timing);

  CUevent GetHandle() const { return handle_; }

  ~CudaEvent() override;
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&& other);
  CudaEvent& operator=(CudaEvent&& other);

 private:
  explicit CudaEvent(Context* context, CUevent handle)
      : context_(context), handle_(handle) {}

  // The Context used to which this object and GpuEventHandle are bound.
  Context* context_;

  // The underlying CUDA event handle.
  CUevent handle_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_EVENT_H_
