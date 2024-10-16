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

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// This class implements EventBasedTimer for CUDA devices.
class CudaTimer : public EventBasedTimer {
 public:
  ~CudaTimer() override;
  CudaTimer(CudaTimer&&) = default;
  CudaTimer& operator=(CudaTimer&&) = default;

  absl::StatusOr<absl::Duration> GetElapsedDuration() override;

  enum class TimerType {
    kDelayKernel,
    kEventBased,
  };
  static absl::StatusOr<CudaTimer> Create(StreamExecutor* executor,
                                          Stream* stream, TimerType timer_type);

 private:
  CudaTimer(StreamExecutor* executor, CudaEvent start_event,
            CudaEvent stop_event, Stream* stream, GpuSemaphore semaphore);

  GpuSemaphore semaphore_;
  bool is_stopped_ = false;
  StreamExecutor* executor_;
  Stream* stream_;
  CudaEvent start_event_;
  CudaEvent stop_event_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_TIMER_H_
