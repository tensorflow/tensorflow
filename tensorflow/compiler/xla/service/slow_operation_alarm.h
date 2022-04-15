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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SLOW_OPERATION_ALARM_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SLOW_OPERATION_ALARM_H_

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <tuple>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// This RAII object asynchronously prints a warning if it's alive for more than
// a certain amount of time.
class SlowOperationAlarm {
 public:
  // If `counter` is not null, this alarm will throttle itself to logging
  // once-every-power-of-two occurrences. The counter must outlive this object.
  SlowOperationAlarm(absl::Duration timeout, std::string msg,
                     std::atomic<int64_t>* counter = nullptr);
  SlowOperationAlarm(absl::Duration timeout,
                     std::function<std::string()> msg_fn,
                     std::atomic<int64_t>* counter = nullptr);
  ~SlowOperationAlarm();

  // Not copyable or movable, because the constructor stores a pointer to `this`
  // into a global variable.
  SlowOperationAlarm(const SlowOperationAlarm&) = delete;
  SlowOperationAlarm(const SlowOperationAlarm&&) = delete;
  SlowOperationAlarm& operator=(const SlowOperationAlarm&) = delete;
  SlowOperationAlarm& operator=(const SlowOperationAlarm&&) = delete;

  absl::Time deadline() const { return deadline_; }
  std::string msg() const { return msg_fn_(); }
  std::atomic<int64_t>* counter() { return counter_; }
  void cancel() { UnscheduleAlarm(this); }
  // Has the alarm fired?  If appropriate, consider cancel()'ing first, to avoid
  // a race.
  bool fired() const { return fired_.load(); }

 private:
  static void AlarmLoop();
  static void ScheduleAlarm(SlowOperationAlarm* alarm);
  static void UnscheduleAlarm(const SlowOperationAlarm* alarm);

  absl::Time deadline_;
  std::function<std::string()> msg_fn_;
  std::atomic<bool> fired_{false};
  // counter_ may be null.  If it's not, this alarm prints something only once
  // every power of two occurrences.
  std::atomic<int64_t>* counter_;
};

// Returns an object which prints a warning about slow compilation after a
// certain amount of time.
//
// In debug builds, recommends building with -c opt.
//
// In opt builds, recommends filing a bug.
//
// This is throttled to once-every-power-of-two occurrences, globally.
ABSL_MUST_USE_RESULT std::unique_ptr<SlowOperationAlarm> SlowCompilationAlarm(
    absl::string_view msg = "");

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SLOW_OPERATION_ALARM_H_
