/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/internal/traceme_recorder.h"

#include <atomic>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {
namespace {

MATCHER_P(Named, name, "") { return arg.name == name; }

constexpr static uint64 kNanosInSec = 1000000000;

TEST(RecorderTest, SingleThreaded) {
  uint64 start_time = Env::Default()->NowNanos();
  uint64 end_time = start_time + kNanosInSec;

  TraceMeRecorder::Record({1, "before", start_time, end_time});
  TraceMeRecorder::Start(/*level=*/1);
  TraceMeRecorder::Record({2, "during1", start_time, end_time});
  TraceMeRecorder::Record({3, "during2", start_time, end_time});
  auto results = TraceMeRecorder::Stop();
  TraceMeRecorder::Record({4, "after", start_time, end_time});

  ASSERT_EQ(results.size(), 1);
  EXPECT_THAT(results[0].events,
              ::testing::ElementsAre(Named("during1"), Named("during2")));
}

void SpinNanos(int nanos) {
  uint64 deadline = Env::Default()->NowNanos() + nanos;
  while (Env::Default()->NowNanos() < deadline) {
  }
}

// Checks the functional behavior of the recorder, when used from several
// unsynchronized threads.
//
// Each thread records a stream of events.
//   Thread 0: activity=0, activity=1, activity=2, ...
//   Thread 1: activity=0, activity=1, activity=2, ...
//   ...
//
// We turn the recorder on and off repeatedly in sessions, expecting to see:
//   - data from every thread (eventually - maybe not every session)
//   - unbroken sessions: a consecutive sequence of IDs from each thread
//   - gaps between sessions: a thread's IDs should be non-consecutive overall
TEST(RecorderTest, Multithreaded) {
  constexpr static int kNumThreads = 4;

  // Start several threads writing events.
  tensorflow::Notification start;
  tensorflow::Notification stop;
  thread::ThreadPool pool(Env::Default(), "testpool", kNumThreads);
  std::atomic<int> thread_count = {0};
  for (int i = 0; i < kNumThreads; i++) {
    pool.Schedule([&start, &stop, &thread_count, i] {
      uint64 j = 0;
      bool was_active = false;
      auto record_event = [&j, i]() {
        uint64 start_time = Env::Default()->NowNanos();
        uint64 end_time = start_time + kNanosInSec;
        TraceMeRecorder::Record({/*activity_id=*/j++,
                                 /*name=*/strings::StrCat(i), start_time,
                                 end_time});
      };
      thread_count.fetch_add(1, std::memory_order_relaxed);
      start.WaitForNotification();
      while (!stop.HasBeenNotified()) {
        // Mimicking production usage, we guard with a racy check.
        // In principle this isn't needed, but a feedback loop can form:
        // 1) many events accumulate while the recorder is off
        // 2) clearing/analyzing these events is slow
        // 3) while clearing, more events are accumulating, causing 1
        if (TraceMeRecorder::Active()) {
          record_event();
          was_active = true;
        }
        // Record some events after the recorder is no longer active to simulate
        // point 1 and 3.
        if (was_active && !TraceMeRecorder::Active()) {
          record_event();
          record_event();
          was_active = false;
        }
        // This snowballs into OOM in some configurations, causing flakiness.
        // Keep this big enough to prevent OOM and small enough such that
        // each thread records at least one event.
        SpinNanos(10);
      }
    });
  }

  // For each thread, keep track of which events we've seen.
  struct {
    bool split_session = false;
    bool overlapping_sessions = false;
    std::set<uint64> events;
  } thread_state[kNumThreads];
  // We expect each thread to eventually have multiple events, not all in a
  // contiguous range.
  auto done = [&thread_state] {
    for (const auto& t : thread_state) {
      if (t.events.size() < 2) return false;
    }
    return true;
  };

  // Wait while all the threads are spun up.
  while (thread_count.load(std::memory_order_relaxed) < kNumThreads) {
    LOG(INFO) << "Waiting for all threads to spin up...";
    Env::Default()->SleepForMicroseconds(1 * EnvTime::kMillisToMicros);
  }

  // We will probably be done after two iterations (with each thread getting
  // some events each iteration). No guarantees as all the threads might not get
  // scheduled in a session, so try for a while.
  start.Notify();
  constexpr static int kMaxIters = 100;
  for (int iters = 0; iters < kMaxIters && !done(); ++iters) {
    LOG(INFO) << "Looping until convergence, iteration: " << iters;
    TraceMeRecorder::Start(/*level=*/1);
    Env::Default()->SleepForMicroseconds(100 * EnvTime::kMillisToMicros);
    auto results = TraceMeRecorder::Stop();
    for (const auto& thread : results) {
      if (thread.events.empty()) continue;
      std::istringstream ss(thread.events.front().name);
      int thread_index = 0;
      ss >> thread_index;
      auto& state = thread_state[thread_index];

      std::set<uint64> session_events;
      uint64 current = 0;
      for (const auto& event : thread.events) {
        session_events.emplace(event.activity_id);
        // Session events should be contiguous.
        if (current != 0 && event.activity_id != current + 1) {
          state.split_session = true;
        }
        current = event.activity_id;
      }

      for (const auto& event : session_events) {
        auto result = state.events.emplace(event);
        if (!result.second) {
          // Session events should not overlap with those from previous
          // sessions.
          state.overlapping_sessions = true;
        }
      }
    }
    Env::Default()->SleepForMicroseconds(1 * EnvTime::kMillisToMicros);
  }
  stop.Notify();

  for (const auto& thread : thread_state) {
    EXPECT_FALSE(thread.split_session)
        << "Expected contiguous events in a session";
    EXPECT_FALSE(thread.overlapping_sessions) << "Expected disjoint sessions";
    EXPECT_GT(thread.events.size(), 1)
        << "Expected gaps in thread events between sessions";
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
