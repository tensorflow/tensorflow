/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_TIMER_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_TIMER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/gpu/gpu_stream.h"

namespace stream_executor::gpu {
class CudaTimer : public EventBasedTimer {
 public:
  CudaTimer(Context* context, std::unique_ptr<CudaEvent> start_event,
            std::unique_ptr<CudaEvent> stop_event, GpuStream* stream,
            GpuSemaphore semaphore);
  ~CudaTimer() override;

  absl::StatusOr<absl::Duration> GetElapsedDuration() override;

 private:
  GpuSemaphore semaphore_;
  bool is_stopped_ = false;
  Context* context_;
  GpuStream* stream_;
  std::unique_ptr<CudaEvent> start_event_;
  std::unique_ptr<CudaEvent> stop_event_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_TIMER_H_
