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

#include "xla/service/rendezvous.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

absl::Duration Timeout() { return absl::Seconds(10); }
absl::Duration Terminate() { return absl::Seconds(10); }

tsl::thread::ThreadPool CreateThreadPool(int32_t size) {
  return tsl::thread::ThreadPool(tsl::Env::Default(), "rendezvous_test", size);
}

TEST(RendezvousTest, OneParticipant) {
  auto result =
      RendezvousSingle<int32_t>("rendezvous_test", 0, 1, [] { return 42; });
  ASSERT_EQ(*result, 42);
}

TEST(RendezvousTest, TwoParticipants) {
  absl::BlockingCounter counter(2);
  std::vector<std::shared_ptr<int32_t>> results(2);

  auto task = [&](int32_t id) {
    return [&, id] {
      results[id] =
          RendezvousSingle<int32_t>("rendezvous_test", 0, 2, [] { return 42; });
      counter.DecrementCount();
    };
  };

  auto thread_pool = CreateThreadPool(2);
  thread_pool.Schedule(task(0));
  thread_pool.Schedule(task(1));
  counter.Wait();

  ASSERT_EQ(results.size(), 2);
  ASSERT_EQ(*results[0], 42);
  ASSERT_EQ(*results[1], 42);
}

TEST(RendezvousTest, TwoParticipantsWithValues) {
  absl::BlockingCounter counter(2);
  std::vector<std::shared_ptr<int32_t>> results(2);

  auto accumulate = [](absl::Span<const int32_t* const> values) {
    int32_t result = 0;
    for (const int32_t* value : values) result += *value;
    return result;
  };

  auto task = [&](int32_t id) {
    return [&, id] {
      results[id] =
          RendezvousSingle<int32_t>("rendezvous_test", 0, id, 2, accumulate);
      counter.DecrementCount();
    };
  };

  auto thread_pool = CreateThreadPool(2);
  thread_pool.Schedule(task(0));
  thread_pool.Schedule(task(1));
  counter.Wait();

  ASSERT_EQ(results.size(), 2);
  ASSERT_EQ(*results[0], 1);
  ASSERT_EQ(*results[1], 1);
}

TEST(RendezvousTest, RepeatRendezvous) {
  auto thread_pool = CreateThreadPool(2);

  for (int32_t i = 0; i < 10; ++i) {
    absl::BlockingCounter counter(2);

    auto task = [&] {
      RendezvousSingle<int32_t>("rendezvous_test", i, 2, [] { return 42; });
      counter.DecrementCount();
    };

    thread_pool.Schedule(task);
    thread_pool.Schedule(task);
    counter.Wait();
  }
}

TEST(RendezvousTest, ReturningStatusOr) {
  absl::BlockingCounter counter(2);
  std::vector<absl::StatusOr<std::shared_ptr<int32_t>>> results(2);

  auto task = [&](int32_t id) {
    return [&, id] {
      results[id] = RendezvousSingle<absl::StatusOr<int32_t>>(
          "rendezvous_test", 0, 2, [] { return 42; });
      counter.DecrementCount();
    };
  };

  auto thread_pool = CreateThreadPool(2);
  thread_pool.Schedule(task(0));
  thread_pool.Schedule(task(1));
  counter.Wait();

  ASSERT_EQ(results.size(), 2);
  ASSERT_EQ(**results[0], 42);
  ASSERT_EQ(**results[1], 42);
}

TEST(RendezvousTest, RendezvousSingleFlag) {
  RendezvousSingleFlag flag;

  auto thread_pool = CreateThreadPool(2);
  int32_t num_executed = 0;

  absl::BlockingCounter round_0(2);
  absl::BlockingCounter round_1(2);

  auto task = [&](absl::BlockingCounter& counter) {
    return [&] {
      RendezvousSingle<int32_t>(
          flag, "rendezvous_test", 0, 2, [&] { return ++num_executed; },
          Timeout(), Terminate());
      counter.DecrementCount();
    };
  };

  // Execute rendezvous a first time.
  thread_pool.Schedule(task(round_0));
  thread_pool.Schedule(task(round_0));
  round_0.Wait();

  ASSERT_EQ(num_executed, 1);

  // Execute rendezvous a second time.
  thread_pool.Schedule(task(round_1));
  thread_pool.Schedule(task(round_1));
  round_1.Wait();

  // Check that we did not execute it second time.
  ASSERT_EQ(num_executed, 1);
}

TEST(RendezvousTest, RendezvousSingleFlagRace) {
  RendezvousSingleFlag flag;

  static constexpr int32_t kNumRendezvous = 16;
  static constexpr int32_t kNumThreads = 8;

  auto thread_pool = CreateThreadPool(kNumRendezvous * kNumThreads);

  auto task = [&](int32_t key) {
    return [&, key] {
      RendezvousSingle(flag, "key: " + std::to_string(key), key, kNumThreads,
                       Timeout(), Terminate());
    };
  };

  for (int32_t key = 0; key < kNumRendezvous; ++key) {
    for (int32_t thread = 0; thread < kNumThreads; ++thread) {
      thread_pool.Schedule(task(key));
    }
  }
}

TEST(RendezvousTest, RendezvousSingleFlagRaceWithBarriers) {
  RendezvousSingleFlag flag;

  static constexpr int32_t kNumRendezvous = 16;
  static constexpr int32_t kNumThreads = 8;

  auto thread_pool = CreateThreadPool(kNumRendezvous * kNumThreads);

  // We use barriers and notifications to make sure all 128 threads start
  // rendezvous at the same time to detect potential deadlocks and data races.
  absl::BlockingCounter participants_ready(kNumRendezvous * kNumThreads);
  absl::Notification participants_notification;
  absl::BlockingCounter participants_done(kNumRendezvous * kNumThreads);

  auto task = [&](int32_t key) {
    return [&, key] {
      participants_ready.DecrementCount();
      participants_notification.WaitForNotification();
      RendezvousSingle(flag, "key: " + std::to_string(key), key, kNumThreads,
                       Timeout(), Terminate());
      participants_done.DecrementCount();
    };
  };

  for (int32_t key = 0; key < kNumRendezvous; ++key) {
    for (int32_t thread = 0; thread < kNumThreads; ++thread) {
      thread_pool.Schedule(task(key));
    }
  }

  participants_notification.Notify();
  participants_ready.Wait();
  participants_done.Wait();
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_Rendezvous(benchmark::State& state) {
  int64_t num_threads = state.range(0);
  auto thread_pool = CreateThreadPool(num_threads);

  for (auto _ : state) {
    absl::BlockingCounter counter(num_threads);
    for (int64_t i = 0; i < num_threads; ++i) {
      thread_pool.Schedule([&] {
        RendezvousSingle<int32_t>("rendezvous_test", 0, num_threads,
                                  [] { return 42; });
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }
}

static void BM_RendezvousWithValues(benchmark::State& state) {
  int64_t num_threads = state.range(0);
  auto thread_pool = CreateThreadPool(num_threads);

  for (auto _ : state) {
    absl::BlockingCounter counter(num_threads);
    for (int64_t i = 0; i < num_threads; ++i) {
      thread_pool.Schedule([&] {
        int32_t value = i;
        RendezvousSingle<int32_t>("rendezvous_test", 0, value, num_threads,
                                  [](auto) { return 42; });
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }
}

BENCHMARK(BM_Rendezvous)
    ->MeasureProcessCPUTime()
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32);

BENCHMARK(BM_RendezvousWithValues)
    ->MeasureProcessCPUTime()
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32);

}  // namespace
}  // namespace xla
