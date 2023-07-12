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
#include <utility>

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace gpu {

class GpuExecutor;
class GpuStream;

// Timer is started once it's created, and is stopped once read.
class GpuTimer {
 public:
  static tsl::StatusOr<GpuTimer> Create(GpuStream* stream);

  // An ugly but a very convenient helper: creates a timer only when we need
  // one, but always returns an object. If `is_needed` is false, returns an
  // empty optional, acts like `Create` otherwise.
  static tsl::StatusOr<std::optional<GpuTimer>> CreateIfNeeded(
      GpuStream* stream, bool is_needed);

  explicit GpuTimer(GpuExecutor* parent, GpuEventHandle start_event,
                    GpuEventHandle stop_event, GpuStream* stream)
      : parent_(parent),
        start_event_(start_event),
        stop_event_(stop_event),
        stream_(stream) {}

  GpuTimer(GpuTimer&& other)
      : parent_(other.parent_),
        start_event_(std::exchange(other.start_event_, nullptr)),
        stop_event_(std::exchange(other.stop_event_, nullptr)),
        stream_(other.stream_) {}

  ~GpuTimer();

  // Stops the timer on the first call and returns the elapsed duration.
  // Subsequent calls error out.
  tsl::StatusOr<absl::Duration> GetElapsedDuration();

 private:
  GpuExecutor* parent_;
  GpuEventHandle start_event_ = nullptr;
  GpuEventHandle stop_event_ = nullptr;
  GpuStream* stream_;
  bool is_stopped_ = false;

  SE_DISALLOW_COPY_AND_ASSIGN(GpuTimer);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_TIMER_H_
