/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_TIMER_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_TIMER_H_

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_event.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

namespace stream_executor::sycl {

// This class implements a timer for SYCL streams by measuring the elapsed
// time between two SYCL events using Level-Zero backend timestamps.
//
// It uses two SYCL events: one for the start time and one for the stop time.
// The timer does not take ownership of the SYCL events, so the caller must
// ensure that the events remain valid for the lifetime of this SyclTimer.
//
// The timer is not thread-safe and should be used in a single-threaded context.
class SyclTimer : public EventBasedTimer {
 public:
  // Move constructor and move assignment operator.
  SyclTimer(SyclTimer&&) = default;
  SyclTimer& operator=(SyclTimer&&) = default;

  // Delete copy constructor and copy assignment operator.
  SyclTimer(const SyclTimer&) = delete;
  SyclTimer& operator=(const SyclTimer&) = delete;

  ~SyclTimer() override = default;

  // Stops the timer (via is_timer_stopped_) on the first call and returns the
  // elapsed duration. Subsequent calls error out.
  absl::StatusOr<absl::Duration> GetElapsedDuration() override;

  // Creates a SyclTimer instance.
  // The SyclTimer does not initialize the start and stop events.
  // The caller must ensure that the events are initialized before use.
  static absl::StatusOr<SyclTimer> Create(StreamExecutor* executor,
                                          Stream* stream);

 private:
  SyclTimer(StreamExecutor* executor, SyclEvent start_event,
            SyclEvent stop_event, Stream* stream);

  bool is_timer_stopped_ = false;
  StreamExecutor* executor_;
  Stream* stream_;
  SyclEvent start_event_;
  SyclEvent stop_event_;
};
}  // namespace stream_executor::sycl

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_TIMER_H_
