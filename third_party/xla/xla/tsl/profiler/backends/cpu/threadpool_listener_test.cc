/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/tsl/profiler/backends/cpu/threadpool_listener.h"

#include <cstdint>
#include <memory>

#include "absl/log/check.h"
#include "absl/synchronization/notification.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/profiler/backends/cpu/traceme_recorder.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "tsl/platform/tracing.h"

namespace tsl {
namespace profiler {
namespace {

TEST(ThreadpoolListenerTest, StartAndStopWhileThreadEmitsEvents) {
  absl::Notification thread_started;
  absl::Notification stop_thread;

  std::unique_ptr<Thread> worker(
      Env::Default()->StartThread(ThreadOptions(), "event_emitter", [&]() {
        thread_started.Notify();
        while (!stop_thread.HasBeenNotified()) {
          uint64_t id = tracing::GetUniqueArg();
          tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);
          tracing::ScopedRegion region(tracing::EventCategory::kRunClosure, id);
        }
      }));

  thread_started.WaitForNotification();

  ThreadpoolProfilerInterface listener;
  TraceMeRecorder::Start(/*level=*/1);
  EXPECT_OK(listener.Start());
  SleepForMillis(10);
  EXPECT_OK(listener.Stop());
  TraceMeRecorder::Events events = TraceMeRecorder::Stop();
  EXPECT_FALSE(events.empty());

  stop_thread.Notify();
  worker.reset();
}

ThreadpoolProfilerInterface* g_listener = nullptr;

void StartListener(const benchmark::State& state) {
  TraceMeRecorder::Start(/*level=*/1);
  g_listener = new ThreadpoolProfilerInterface();
  CHECK_OK(g_listener->Start());
}

void StopListener(const benchmark::State& state) {
  CHECK_OK(g_listener->Stop());
  delete g_listener;
  g_listener = nullptr;
  TraceMeRecorder::Stop();
}

void BM_RecordEvent(benchmark::State& state) {
  for (auto _ : state) {
    tracing::RecordEvent(tracing::EventCategory::kScheduleClosure,
                         tracing::GetUniqueArg());
  }
}

BENCHMARK(BM_RecordEvent)
    ->Setup(StartListener)
    ->Teardown(StopListener)
    ->ThreadRange(1, 8);

}  // namespace
}  // namespace profiler
}  // namespace tsl
