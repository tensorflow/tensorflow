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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_TIMER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_TIMER_H_

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

class RocmTimer : public EventBasedTimer {
 public:
  RocmTimer(RocmTimer&&) = default;
  RocmTimer& operator=(RocmTimer&&) = default;

  absl::StatusOr<absl::Duration> GetElapsedDuration() override;

  static absl::StatusOr<RocmTimer> Create(StreamExecutor* executor,
                                          Stream* stream);

 private:
  RocmTimer(StreamExecutor* executor, RocmEvent start_event,
            RocmEvent stop_event, Stream* stream);

  bool is_stopped_ = false;
  StreamExecutor* executor_;
  Stream* stream_;
  RocmEvent start_event_;
  RocmEvent stop_event_;
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_TIMER_H_
