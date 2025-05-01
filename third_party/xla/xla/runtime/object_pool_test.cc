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

#include "xla/runtime/object_pool.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

using IntPool = ObjectPool<std::unique_ptr<int32_t>>;

TEST(ObjectPoolTest, GetOrCreate) {
  int32_t counter = 0;
  IntPool pool([&]() -> absl::StatusOr<std::unique_ptr<int32_t>> {
    return std::make_unique<int32_t>(counter++);
  });

  TF_ASSERT_OK_AND_ASSIGN(auto obj0, pool.GetOrCreate());
  ASSERT_EQ(**obj0, 0);

  TF_ASSERT_OK_AND_ASSIGN(auto obj1, pool.GetOrCreate());
  ASSERT_EQ(**obj1, 1);

  auto destroy = [](IntPool::BorrowedObject obj) {};
  destroy(std::move(obj0));
  destroy(std::move(obj1));

  TF_ASSERT_OK_AND_ASSIGN(auto obj2, pool.GetOrCreate());
  ASSERT_EQ(**obj2, 1);
  ASSERT_EQ(counter, 2);
}

TEST(ObjectPoolTest, GetOrCreateUnderContention) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);

  // We concurrently mutate counter field to detect races under tsan and track
  // the number of concurrent users to detect races in our test code.
  struct Obj {
    int64_t counter{0};
    std::atomic<int64_t> users{0};
  };

  absl::Mutex mutex;
  std::vector<Obj*> objs;

  ObjectPool<std::unique_ptr<Obj>> pool(
      [&]() -> absl::StatusOr<std::unique_ptr<Obj>> {
        absl::MutexLock lock(&mutex);
        auto obj = std::make_unique<Obj>();
        objs.push_back(obj.get());
        return obj;
      });

  size_t num_tasks = 100;
  size_t num_iters = 100;

  absl::BlockingCounter blocking_counter(num_tasks);

  for (int32_t t = 0; t < num_tasks; ++t) {
    threads.Schedule([&] {
      for (int32_t i = 0; i < num_iters; ++i) {
        TF_ASSERT_OK_AND_ASSIGN(auto obj, pool.GetOrCreate());
        CHECK_EQ((*obj)->users.fetch_add(1), 0);
        ASSERT_GE((*obj)->counter++, 0);
        CHECK_EQ((*obj)->users.fetch_sub(1), 1);
      }
      blocking_counter.DecrementCount();
    });
  }

  blocking_counter.Wait();

  // We should create at most one object for each thread in the pool.
  EXPECT_GT(objs.size(), 0);
  EXPECT_LE(objs.size(), threads.NumThreads());

  // Check that the sum of counters matches the number of executed operations.
  int64_t sum = 0;
  absl::c_for_each(objs, [&](Obj* obj) { sum += obj->counter; });
  EXPECT_EQ(sum, num_tasks * num_iters);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks.
//===----------------------------------------------------------------------===//

static void BM_GetOrCreate(benchmark::State& state) {
  IntPool pool([cnt = 0]() mutable -> absl::StatusOr<std::unique_ptr<int32_t>> {
    return std::make_unique<int32_t>(cnt++);
  });

  for (auto _ : state) {
    auto obj = pool.GetOrCreate();
    benchmark::DoNotOptimize(obj);
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_GetOrCreate);

static void BM_GetOrCreateUnderContention(benchmark::State& state) {
  size_t num_threads = state.range(0);
  size_t num_iters = state.range(1);

  tsl::thread::ThreadPool threads(tsl::Env::Default(), "bench", num_threads);

  IntPool pool([cnt = 0]() mutable -> absl::StatusOr<std::unique_ptr<int32_t>> {
    return std::make_unique<int32_t>(cnt++);
  });

  for (auto _ : state) {
    absl::BlockingCounter blocking_counter(num_threads);

    for (int32_t t = 0; t < num_threads; ++t) {
      threads.Schedule([&] {
        for (int32_t i = 0; i < num_iters; ++i) {
          auto obj = pool.GetOrCreate();
          (***obj)++;
        }
        blocking_counter.DecrementCount();
      });
    }

    blocking_counter.Wait();
  }

  state.SetItemsProcessed(state.iterations() * num_threads * num_iters);
}

BENCHMARK(BM_GetOrCreateUnderContention)
    ->MeasureProcessCPUTime()
    ->ArgPair(1, 1000)
    ->ArgPair(2, 1000)
    ->ArgPair(4, 1000)
    ->ArgPair(8, 1000);

}  // namespace
}  // namespace xla
