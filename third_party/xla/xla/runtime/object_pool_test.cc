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
#include "absl/status/status.h"
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

struct NonDefaultConstructible {
  explicit NonDefaultConstructible(int32_t value) : value(value) {}

  NonDefaultConstructible(NonDefaultConstructible&&) = default;
  NonDefaultConstructible& operator=(NonDefaultConstructible&&) = default;

  int32_t value;
};

TEST(ObjectPoolTest, GetOrCreate) {
  int32_t counter = 0;
  IntPool pool([&]() -> absl::StatusOr<std::unique_ptr<int32_t>> {
    return std::make_unique<int32_t>(counter++);
  });

  ASSERT_OK_AND_ASSIGN(auto obj0, pool.GetOrCreate());
  ASSERT_EQ(**obj0, 0);

  ASSERT_OK_AND_ASSIGN(auto obj1, pool.GetOrCreate());
  ASSERT_EQ(**obj1, 1);

  auto destroy = [](IntPool::BorrowedObject obj) {};
  destroy(std::move(obj0));
  destroy(std::move(obj1));

  ASSERT_OK_AND_ASSIGN(auto obj2, pool.GetOrCreate());
  ASSERT_EQ(**obj2, 1);
  ASSERT_EQ(counter, 2);
}

TEST(ObjectPoolTest, Clear) {
  int32_t counter = 0;
  IntPool pool([&]() -> absl::StatusOr<std::unique_ptr<int32_t>> {
    return std::make_unique<int32_t>(counter++);
  });

  ASSERT_OK_AND_ASSIGN(auto obj0, pool.GetOrCreate());
  ASSERT_EQ(**obj0, 0);

  ASSERT_OK_AND_ASSIGN(auto obj1, pool.GetOrCreate());
  ASSERT_EQ(**obj1, 1);

  auto destroy = [](IntPool::BorrowedObject obj) {};
  destroy(std::move(obj0));
  destroy(std::move(obj1));

  pool.Clear();

  ASSERT_OK_AND_ASSIGN(auto obj, pool.GetOrCreate());
  ASSERT_EQ(**obj, 2);
  ASSERT_EQ(counter, 3);
  EXPECT_EQ(pool.num_created(), 1);
  EXPECT_EQ(pool.num_available(), 0);
}

TEST(ObjectPoolTest, ConstructorPreallocatesWithBuilderArgs) {
  ObjectPool<NonDefaultConstructible, int32_t> pool(
      [&](int32_t value) { return NonDefaultConstructible(value); }, 2, 42);

  EXPECT_EQ(pool.num_created(), 2);
  EXPECT_EQ(pool.num_available(), 2);

  ASSERT_OK_AND_ASSIGN(auto obj0, pool.Get());
  ASSERT_OK_AND_ASSIGN(auto obj1, pool.Get());
  EXPECT_EQ(obj0->value, 42);
  EXPECT_EQ(obj1->value, 42);
  EXPECT_EQ(pool.num_available(), 0);
}

TEST(ObjectPoolTest, Preallocate) {
  int32_t counter = 0;
  IntPool pool([&]() -> absl::StatusOr<std::unique_ptr<int32_t>> {
    return std::make_unique<int32_t>(counter++);
  });

  absl::Status status = pool.Preallocate(3);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(counter, 3);
  EXPECT_EQ(pool.num_created(), 3);
  EXPECT_EQ(pool.num_available(), 3);

  status = pool.Preallocate(2);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(counter, 5);
  EXPECT_EQ(pool.num_created(), 5);
  EXPECT_EQ(pool.num_available(), 5);

  std::vector<IntPool::BorrowedObject> borrowed;
  std::vector<int32_t> values;
  for (int32_t i = 0; i < 5; ++i) {
    ASSERT_OK_AND_ASSIGN(auto object, pool.Get());
    values.push_back(**object);
    borrowed.push_back(std::move(object));
  }

  absl::c_sort(values);
  EXPECT_EQ(values, (std::vector<int32_t>{0, 1, 2, 3, 4}));
  EXPECT_EQ(counter, 5);
  EXPECT_EQ(pool.num_available(), 0);
  EXPECT_EQ(pool.Get().status().code(), absl::StatusCode::kResourceExhausted);
}

TEST(ObjectPoolTest, PreallocateReturnsBuilderErrorWithoutChangingPool) {
  int32_t counter = 0;
  IntPool pool([&]() -> absl::StatusOr<std::unique_ptr<int32_t>> {
    if (counter == 2) {
      return absl::InternalError("failed to create object");
    }
    return std::make_unique<int32_t>(counter++);
  });

  absl::Status status = pool.Preallocate(1);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(counter, 1);
  EXPECT_EQ(pool.num_created(), 1);
  EXPECT_EQ(pool.num_available(), 1);

  status = pool.Preallocate(3);
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(counter, 2);
  EXPECT_EQ(pool.num_created(), 1);
  EXPECT_EQ(pool.num_available(), 1);

  ASSERT_OK_AND_ASSIGN(auto object, pool.Get());
  EXPECT_EQ(**object, 0);
  EXPECT_EQ(pool.Get().status().code(), absl::StatusCode::kResourceExhausted);
}

TEST(ObjectPoolTest, Get) {
  int32_t counter = 0;
  IntPool pool([&]() -> absl::StatusOr<std::unique_ptr<int32_t>> {
    return std::make_unique<int32_t>(counter++);
  });

  EXPECT_EQ(pool.num_created(), 0);
  EXPECT_EQ(pool.num_available(), 0);

  auto empty = pool.Get();
  EXPECT_EQ(empty.status().code(), absl::StatusCode::kResourceExhausted);

  ASSERT_OK_AND_ASSIGN(auto obj0, pool.GetOrCreate());
  ASSERT_EQ(**obj0, 0);
  EXPECT_EQ(pool.num_created(), 1);
  EXPECT_EQ(pool.num_available(), 0);

  auto borrowed = pool.Get();
  EXPECT_EQ(borrowed.status().code(), absl::StatusCode::kResourceExhausted);

  auto destroy = [](IntPool::BorrowedObject obj) {};
  destroy(std::move(obj0));
  EXPECT_EQ(pool.num_available(), 1);

  ASSERT_OK_AND_ASSIGN(auto obj1, pool.Get());
  EXPECT_EQ(**obj1, 0);
  EXPECT_EQ(counter, 1);
  EXPECT_EQ(pool.num_available(), 0);
}

TEST(ObjectPoolTest, BorrowedObjectMoveAssignmentReturnsPreviousObject) {
  int32_t counter = 0;
  IntPool pool([&]() -> absl::StatusOr<std::unique_ptr<int32_t>> {
    return std::make_unique<int32_t>(counter++);
  });

  ASSERT_OK_AND_ASSIGN(auto obj0, pool.GetOrCreate());
  ASSERT_OK_AND_ASSIGN(auto obj1, pool.GetOrCreate());

  obj1 = std::move(obj0);
  EXPECT_EQ(pool.num_available(), 1);

  ASSERT_OK_AND_ASSIGN(auto obj2, pool.Get());
  EXPECT_EQ(**obj1, 0);
  EXPECT_EQ(**obj2, 1);
  EXPECT_EQ(counter, 2);
}

TEST(ObjectPoolTest, SupportsNonDefaultConstructibleObjects) {
  ObjectPool<NonDefaultConstructible, int32_t> pool(
      [](int32_t value) -> absl::StatusOr<NonDefaultConstructible> {
        return NonDefaultConstructible(value);
      });

  ASSERT_OK_AND_ASSIGN(auto obj0, pool.GetOrCreate(42));
  EXPECT_EQ(obj0->value, 42);

  auto destroy =
      [](ObjectPool<NonDefaultConstructible, int32_t>::BorrowedObject obj) {};
  destroy(std::move(obj0));

  ASSERT_OK_AND_ASSIGN(auto obj1, pool.Get());
  EXPECT_EQ(obj1->value, 42);
  EXPECT_EQ(pool.num_created(), 1);
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
        absl::MutexLock lock(mutex);
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
        ASSERT_OK_AND_ASSIGN(auto obj, pool.GetOrCreate());
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
    ->ArgPair(8, 1000)
    ->ArgPair(16, 1000)
    ->ArgPair(32, 1000);

}  // namespace
}  // namespace xla
