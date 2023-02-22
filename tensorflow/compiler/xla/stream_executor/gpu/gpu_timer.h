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

// Defines the GpuTimer type - the CUDA-specific implementation of the generic
// StreamExecutor Timer interface.

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_TIMER_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_TIMER_H_

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace gpu {

class GpuExecutor;
class GpuStream;

// Wraps a pair of GpuEventHandles in order to satisfy the platform-independent
// TimerInterface -- both a start and a stop event are present which may be
// recorded in a stream.
class GpuTimer : public internal::TimerInterface {
 public:
  explicit GpuTimer(GpuExecutor* parent)
      : parent_(parent), start_event_(nullptr), stop_event_(nullptr) {}

  // Note: teardown needs to be explicitly handled in this API by a call to
  // StreamExecutor::DeallocateTimer(), which invokes Destroy().
  // TODO(csigg): Change to RAII.
  ~GpuTimer() override {}

  // Allocates the platform-specific pieces of the timer, called as part of
  // StreamExecutor::AllocateTimer().
  bool Init();

  // Deallocates the platform-specific pieces of the timer, called as part of
  // StreamExecutor::DeallocateTimer().
  void Destroy();

  // Records the "timer start" event at the current point in the stream.
  bool Start(GpuStream* stream);

  // Records the "timer stop" event at the current point in the stream.
  bool Stop(GpuStream* stream);

  // Returns the elapsed time, in milliseconds, between the start and stop
  // events.
  float GetElapsedMilliseconds() const;

  // See Timer::Microseconds().
  // TODO(leary) make this into an error code interface...
  uint64_t Microseconds() const override {
    return GetElapsedMilliseconds() * 1e3;
  }

  // See Timer::Nanoseconds().
  uint64_t Nanoseconds() const override {
    return GetElapsedMilliseconds() * 1e6;
  }

 private:
  GpuExecutor* parent_;
  GpuEventHandle start_event_;  // Event recorded to indicate the "start"
                                // timestamp executing in a stream.
  GpuEventHandle stop_event_;   // Event recorded to indicate the "stop"
                                // timestamp executing in a stream.
};

struct GpuTimerDeleter {
  void operator()(GpuTimer* t) {
    t->Destroy();
    delete t;
  }
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_TIMER_H_
