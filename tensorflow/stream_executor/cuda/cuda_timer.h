/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Defines the CUDATimer type - the CUDA-specific implementation of the generic
// StreamExecutor Timer interface.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_TIMER_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_TIMER_H_

#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"

namespace stream_executor {
namespace cuda {

class CUDAExecutor;
class CUDAStream;

// Wraps a pair of CUevents in order to satisfy the platform-independent
// TimerInferface -- both a start and a stop event are present which may be
// recorded in a stream.
class CUDATimer : public internal::TimerInterface {
 public:
  explicit CUDATimer(CUDAExecutor *parent)
      : parent_(parent), start_event_(nullptr), stop_event_(nullptr) {}

  // Note: teardown needs to be explicitly handled in this API by a call to
  // StreamExecutor::DeallocateTimer(), which invokes Destroy().
  // TODO(csigg): Change to RAII.
  ~CUDATimer() override {}

  // Allocates the platform-specific pieces of the timer, called as part of
  // StreamExecutor::AllocateTimer().
  bool Init();

  // Deallocates the platform-specific pieces of the timer, called as part of
  // StreamExecutor::DeallocateTimer().
  void Destroy();

  // Records the "timer start" event at the current point in the stream.
  bool Start(CUDAStream *stream);

  // Records the "timer stop" event at the current point in the stream.
  bool Stop(CUDAStream *stream);

  // Returns the elapsed time, in milliseconds, between the start and stop
  // events.
  float GetElapsedMilliseconds() const;

  // See Timer::Microseconds().
  // TODO(leary) make this into an error code interface...
  uint64 Microseconds() const override {
    return GetElapsedMilliseconds() * 1e3;
  }

  // See Timer::Nanoseconds().
  uint64 Nanoseconds() const override { return GetElapsedMilliseconds() * 1e6; }

 private:
  CUDAExecutor *parent_;
  CUevent start_event_;  // Event recorded to indicate the "start" timestamp
                         // executing in a stream.
  CUevent stop_event_;   // Event recorded to indicate the "stop" timestamp
                         // executing in a stream.
};

struct TimerDeleter {
  void operator()(CUDATimer *t) {
    t->Destroy();
    delete t;
  }
};

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_TIMER_H_
