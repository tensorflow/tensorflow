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

#include "tensorflow/compiler/xla/service/slow_operation_alarm.h"

#include <list>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace {

absl::Mutex mu(absl::kConstInit);
absl::CondVar* ready;
absl::once_flag init_flag;
std::list<SlowOperationAlarm*>* outstanding_alarms ABSL_PT_GUARDED_BY(mu) =
    nullptr;

void AlarmLoop() {
  while (true) {
    absl::MutexLock lock(&mu);

    // Fire any alarms which are ready.
    absl::Time now = absl::Now();
    for (auto it = outstanding_alarms->begin();
         it != outstanding_alarms->end();) {
      auto next = std::next(it);
      auto* alarm = *it;
      // Fire the alarm if applicable.
      if (alarm->deadline() <= now) {
        outstanding_alarms->erase(it);
        int64 count =
            alarm->counter() == nullptr ? 0 : alarm->counter()->fetch_add(1);
        // If the alarm has a counter, only fire if the count is a power of 2.
        if (count == 0 || (count & (count - 1)) == 0) {
          // We fire alarms with LOG(ERROR) because otherwise it might not show
          // up without --logtostderr.
          LOG(ERROR) << alarm->msg();
        }
      }
      it = next;
    }

    if (outstanding_alarms->empty()) {
      ready->Wait(&mu);
      continue;
    }

    SlowOperationAlarm* next_alarm = *absl::c_min_element(
        *outstanding_alarms,
        [](const SlowOperationAlarm* a, const SlowOperationAlarm* b) {
          return a->deadline() < b->deadline();
        });
    ready->WaitWithDeadline(&mu, next_alarm->deadline());
  }
}

void ScheduleAlarm(SlowOperationAlarm* alarm) {
  absl::call_once(init_flag, [] {
    ready = new absl::CondVar();
    outstanding_alarms = new std::list<SlowOperationAlarm*>();
    (void)tensorflow::Env::Default()->StartThread(
        tensorflow::ThreadOptions(), "SlowOperationAlarm", [] { AlarmLoop(); });
  });

  absl::MutexLock lock(&mu);
  outstanding_alarms->push_back(alarm);
  ready->Signal();
}

void UnscheduleAlarm(const SlowOperationAlarm* alarm) {
  absl::MutexLock lock(&mu);
  CHECK(outstanding_alarms != nullptr);
  auto it = absl::c_find(*outstanding_alarms, alarm);
  if (it != outstanding_alarms->end()) {
    outstanding_alarms->erase(it);
  }
}

}  // namespace

SlowOperationAlarm::SlowOperationAlarm(absl::Duration timeout, string msg,
                                       std::atomic<int64>* counter /*=nullptr*/)
    : deadline_(absl::Now() + timeout),
      msg_(std::move(msg)),
      counter_(counter) {
  ScheduleAlarm(this);
}

SlowOperationAlarm::~SlowOperationAlarm() { UnscheduleAlarm(this); }

std::unique_ptr<SlowOperationAlarm> SlowCompilationAlarm() {
  // Pass a counter to these alarms so they only log once every power-of-two
  // occurrences.
  static auto* counter = new std::atomic<int64>(0);

  const char* separator = "\n********************************";
#if NDEBUG
  return absl::make_unique<SlowOperationAlarm>(
      absl::Duration(absl::Minutes(2)),
      absl::StrCat(
          separator,
          "\nVery slow compile?  If you want to file a bug, run with envvar "
          "XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.",
          separator),
      counter);
#else
  return absl::make_unique<SlowOperationAlarm>(
      absl::Duration(absl::Seconds(10)),
      absl::StrCat(
          separator,
          "\nSlow compile?  XLA was built without compiler optimizations, "
          "which can be slow.  Try rebuilding with -c opt.",
          separator),
      counter);
#endif
}

}  // namespace xla
