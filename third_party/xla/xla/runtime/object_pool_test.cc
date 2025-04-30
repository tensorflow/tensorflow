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

#include "absl/status/statusor.h"
#include "absl/synchronization/blocking_counter.h"
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

  std::atomic<int32_t> counter = 0;
  IntPool pool([&]() -> absl::StatusOr<std::unique_ptr<int32_t>> {
    return std::make_unique<int32_t>(counter++);
  });

  size_t num_tasks = 10;
  absl::BlockingCounter blocking_counter(num_tasks);

  for (int32_t t = 0; t < num_tasks; ++t) {
    threads.Schedule([&] {
      for (int32_t i = 0; i < 100; ++i) {
        TF_ASSERT_OK_AND_ASSIGN(auto obj, pool.GetOrCreate());
        ASSERT_GE(**obj, 0);
      }
      blocking_counter.DecrementCount();
    });
  }

  blocking_counter.Wait();

  // We should create at most one object for each thread in the pool.
  EXPECT_LE(counter, 8);
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
}

BENCHMARK(BM_GetOrCreate);

}  // namespace
}  // namespace xla
