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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_TIMER_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_TIMER_H_

#include <optional>

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace gpu {

class GpuExecutor;
class GpuStream;

// Timer is started once it's created, and can only be stopped once.
class GpuTimer {
 public:
  static tsl::StatusOr<GpuTimer> Create(GpuStream* stream);

  explicit GpuTimer(GpuExecutor* parent, GpuEventHandle start_event,
                    GpuEventHandle stop_event, GpuStream* stream)
      : parent_(parent),
        start_event_(start_event),
        stop_event_(stop_event),
        stream_(stream) {}

  ~GpuTimer();

  // Records the "timer stop" event at the current point in the stream.
  // Returns "error status" if called multiple times.
  tsl::Status Stop();

  // Returns the elapsed time, in milliseconds, between the start and stop
  // events.
  float GetElapsedMilliseconds() const;

  uint64_t Microseconds() const { return GetElapsedMilliseconds() * 1e3; }

  uint64_t Nanoseconds() const { return GetElapsedMilliseconds() * 1e6; }

 private:
  GpuExecutor* parent_;
  GpuEventHandle start_event_ =
      nullptr;  // Event recorded to indicate the "start"
                // timestamp executing in a stream.
  GpuEventHandle stop_event_ =
      nullptr;  // Event recorded to indicate the "stop"
                // timestamp executing in a stream.
  GpuStream* stream_;
  std::optional<float> elapsed_milliseconds_;

  SE_DISALLOW_COPY_AND_ASSIGN(GpuTimer);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_TIMER_H_
