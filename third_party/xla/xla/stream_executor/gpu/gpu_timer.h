/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_TIMER_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_TIMER_H_

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_event.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"

namespace xla {
namespace gpu {
class DeterminismTest;
}
}  // namespace xla

namespace stream_executor {
namespace gpu {

class GpuStream;

// When a timer is created it launches a delay kernel into the given stream and
// queues a start event immediately afterwards. This delay kernel blocks
// execution on the stream until GetElapsedDuration() is called, at which point
// an end event is queued and the delay kernel exits. This allows the device
// execution time of the tasks queued to the stream while the timer is active
// to be measured more accurately.
class GpuTimer : public EventBasedTimer {
 public:
  GpuTimer(Context* context, std::unique_ptr<GpuEvent> start_event,
           std::unique_ptr<GpuEvent> stop_event, GpuStream* stream,
           GpuSemaphore semaphore = {})
      : context_(context),
        start_event_(std::move(start_event)),
        stop_event_(std::move(stop_event)),
        stream_(stream),
        semaphore_(std::move(semaphore)) {}

  GpuTimer(GpuTimer&& other)
      : context_(other.context_),
        start_event_(std::exchange(other.start_event_, nullptr)),
        stop_event_(std::exchange(other.stop_event_, nullptr)),
        stream_(other.stream_),
        semaphore_(std::move(other.semaphore_)) {}

  GpuTimer& operator=(GpuTimer&& other) {
    if (this != &other) {
      context_ = other.context_;
      start_event_ = std::exchange(other.start_event_, nullptr);
      stop_event_ = std::exchange(other.stop_event_, nullptr);
      stream_ = other.stream_;
      semaphore_ = std::move(other.semaphore_);
    }
    return *this;
  }

  ~GpuTimer() override;

  absl::StatusOr<absl::Duration> GetElapsedDuration() override;

 private:
  Context* context_;
  std::unique_ptr<GpuEvent> start_event_;
  std::unique_ptr<GpuEvent> stop_event_;
  GpuStream* stream_;
  GpuSemaphore semaphore_;
  bool is_stopped_ = false;

  GpuTimer(const GpuTimer&) = delete;
  void operator=(const GpuTimer&) = delete;

  // If called, all timers will return random durations instead of the actual
  // duration the timer took. Used for testing only.
  static void ReturnRandomDurationsForTesting();
  friend class ::xla::gpu::DeterminismTest;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_TIMER_H_
