/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_TIMER_H_
#define TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_TIMER_H_

#include <chrono>

#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace host {

class HostTimer : public internal::TimerInterface {
 public:
  HostTimer() {}
  ~HostTimer() override {}

  // Begins the timer at the present point in the stream.
  bool Start(Stream* stream);

  // Stops the timer at the present point in the stream.
  bool Stop(Stream* stream);

  // Returns the most recent value recorded for a start/stopcycle, in
  // microseconds.
  uint64_t Microseconds() const override;

  // Returns the most recent value recorded for a start/stopcycle, in
  // nanoseconds.
  uint64_t Nanoseconds() const override;

 private:
  using clock = std::chrono::high_resolution_clock;

  clock::time_point start_time_;
  clock::duration duration_;

  // Actually starts (rather than enqueues starting) the timer.
  void StartNow();

  // Actually stops (rather than enqueues stopping) the timer.
  void StopNow();
};

}  // namespace host
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_TIMER_H_
