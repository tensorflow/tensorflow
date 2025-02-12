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

#ifndef XLA_SERVICE_SLOW_OPERATION_ALARM_H_
#define XLA_SERVICE_SLOW_OPERATION_ALARM_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace xla {

// This RAII object asynchronously prints a warning if it's alive for more than
// a certain amount of time.
class SlowOperationAlarm {
 public:
  // If `counter` is not null, this alarm will throttle itself to logging
  // once-every-power-of-two occurrences. The counter must outlive this object.
  SlowOperationAlarm(absl::Duration timeout, std::string msg,
                     std::atomic<int64_t>* counter = nullptr,
                     absl::string_view context = "");
  SlowOperationAlarm(absl::Duration timeout,
                     std::function<std::string()> msg_fn,
                     std::atomic<int64_t>* counter = nullptr,
                     absl::string_view context = "");
  ~SlowOperationAlarm();

  // Not copyable or movable, because the constructor stores a pointer to `this`
  // into a global variable.
  SlowOperationAlarm(const SlowOperationAlarm&) = delete;
  SlowOperationAlarm(const SlowOperationAlarm&&) = delete;
  SlowOperationAlarm& operator=(const SlowOperationAlarm&) = delete;
  SlowOperationAlarm& operator=(const SlowOperationAlarm&&) = delete;

  absl::Time deadline() const { return deadline_; }
  std::atomic<int64_t>* counter() { return counter_; }
  void cancel() { UnscheduleAlarm(this); }
  // Has the alarm fired?  If appropriate, consider cancel()'ing first, to avoid
  // a race.
  bool fired() const { return fired_.load(); }

 private:
  static void AlarmLoop();
  static void ScheduleAlarm(SlowOperationAlarm* alarm);
  static void UnscheduleAlarm(const SlowOperationAlarm* alarm);

  absl::Time start_;
  absl::Time deadline_;
  std::string context_;
  std::function<std::string()> msg_fn_;
  std::atomic<bool> fired_{false};
  // counter_ may be null.  If it's not, this alarm prints something only once
  // every power of two occurrences.
  std::atomic<int64_t>* counter_;
  // If the alarm has fired the result of calling msg_fn_ is cached into msg_
  // so that it can be reused in the destructor.
  std::string msg_;
};

// Returns an object which prints a warning about slow compilation after a
// certain amount of time. It will also print the total lifetime duration of
// the returned object when it goes out of scope.
//
// In debug builds, recommends building with -c opt.
//
// In opt builds, recommends filing a bug.
//
// This is throttled to once-every-power-of-two occurrences, globally.
//
// `context` is an additional message prepended to the alarm.
[[nodiscard]] std::unique_ptr<SlowOperationAlarm> SlowCompilationAlarm(
    absl::string_view context);

}  // namespace xla

#endif  // XLA_SERVICE_SLOW_OPERATION_ALARM_H_
