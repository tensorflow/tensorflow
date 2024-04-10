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
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_timer_kernel.h"
#include "xla/stream_executor/gpu/gpu_types.h"

namespace xla {
namespace gpu {
class DeterminismTest;
}
}  // namespace xla

namespace stream_executor {
namespace gpu {

class GpuExecutor;
class GpuStream;

// When a timer is created it launches a delay kernel into the given stream and
// queues a start event immediately afterwards. This delay kernel blocks
// execution on the stream until GetElapsedDuration() is called, at which point
// an end event is queued and the delay kernel exits. This allows the device
// execution time of the tasks queued to the stream while the timer is active
// to be measured more accurately.
class GpuTimer {
 public:
  class GpuSemaphore {
   public:
    GpuSemaphore() = default;
    static absl::StatusOr<GpuSemaphore> Create(StreamExecutor* executor);
    explicit operator bool() const { return bool{ptr_}; }
    GpuSemaphoreState& operator*() {
      return *static_cast<GpuSemaphoreState*>(ptr_->opaque());
    }
    DeviceMemory<GpuSemaphoreState> device();

   private:
    explicit GpuSemaphore(std::unique_ptr<MemoryAllocation> alloc)
        : ptr_{std::move(alloc)} {}
    std::unique_ptr<MemoryAllocation> ptr_;
  };
  static absl::StatusOr<GpuTimer> Create(Stream* stream, bool use_delay_kernel);
  [[deprecated("Pass Stream* not GpuStream*")]] static absl::StatusOr<GpuTimer>
  Create(GpuStream* stream);

  // An ugly but a very convenient helper: creates a timer only when we need
  // one, but always returns an object. If `is_needed` is false, returns an
  // empty optional, acts like `Create` otherwise.
  static absl::StatusOr<std::optional<GpuTimer>> CreateIfNeeded(
      Stream* stream, bool use_delay_kernel, bool is_needed);
  [[deprecated("Pass Stream* not GpuStream*")]] static absl::StatusOr<
      std::optional<GpuTimer>>
  CreateIfNeeded(GpuStream* stream, bool is_needed);

  explicit GpuTimer(GpuExecutor* parent, GpuEventHandle start_event,
                    GpuEventHandle stop_event, GpuStream* stream,
                    GpuSemaphore semaphore = {})
      : parent_(parent),
        start_event_(start_event),
        stop_event_(stop_event),
        stream_(stream),
        semaphore_(std::move(semaphore)) {}

  GpuTimer(GpuTimer&& other)
      : parent_(other.parent_),
        start_event_(std::exchange(other.start_event_, nullptr)),
        stop_event_(std::exchange(other.stop_event_, nullptr)),
        stream_(other.stream_),
        semaphore_(std::move(other.semaphore_)) {}

  GpuTimer& operator=(GpuTimer&& other) {
    if (this != &other) {
      parent_ = other.parent_;
      start_event_ = std::exchange(other.start_event_, nullptr);
      stop_event_ = std::exchange(other.stop_event_, nullptr);
      stream_ = other.stream_;
      semaphore_ = std::move(other.semaphore_);
    }
    return *this;
  }

  ~GpuTimer();

  // Stops the timer on the first call and returns the elapsed duration.
  // Subsequent calls error out.
  absl::StatusOr<absl::Duration> GetElapsedDuration();

 private:
  GpuExecutor* parent_;
  GpuEventHandle start_event_ = nullptr;
  GpuEventHandle stop_event_ = nullptr;
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
