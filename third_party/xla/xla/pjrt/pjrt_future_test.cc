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

#include "xla/pjrt/pjrt_future.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {

TEST(PjRtFutureTest, StatusConstructedFuture) {
  PjRtFuture<> future = PjRtFuture<>(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::OkStatus());
}

TEST(PjRtFutureTest, ValueConstructedFuture) {
  PjRtFuture<int32_t> future = PjRtFuture<int32_t>(42);
  EXPECT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::StatusOr<int32_t>(42));
}

TEST(PjRtFutureTest, StatelessFuture) {
  auto [promise, future] = PjRtFuture<>::MakePromise();

  EXPECT_FALSE(future.IsReady());
  promise.Set();
  EXPECT_TRUE(future.IsReady());

  EXPECT_EQ(future.Await(), absl::OkStatus());

  future.OnReady(
      [](absl::Status status) { EXPECT_EQ(status, absl::OkStatus()); });
}

TEST(PjRtFutureTest, StatefulFutureToStateless) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();
  PjRtFuture<> ready_future = future.GetReadyFuture();

  EXPECT_FALSE(ready_future.IsReady());
  promise.Set(42);
  EXPECT_EQ(ready_future.Await(), absl::OkStatus());
}

TEST(PjRtFutureTest, StatefulFutureToStatelessError) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();
  PjRtFuture<> ready_future = future.GetReadyFuture();

  EXPECT_FALSE(ready_future.IsReady());
  promise.Set(absl::InternalError("test"));
  EXPECT_EQ(ready_future.Await(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, MoveOnlyFutureToStateless) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();
  PjRtFuture<> ready_future = future.GetReadyFuture();

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(ready_future.IsReady());

  promise.Set(std::make_unique<int32_t>(42));
  EXPECT_EQ(ready_future.Await(), absl::OkStatus());
}

TEST(PjRtFutureTest, MoveOnlyFutureToStatelessError) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();
  PjRtFuture<> ready_future = future.GetReadyFuture();

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(ready_future.IsReady());

  promise.Set(absl::InternalError("test"));
  EXPECT_EQ(ready_future.Await(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, CopyableFuture) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();

  PjRtFuture<int32_t> copy_constructed(future);
  PjRtFuture<int32_t> copy_assigned = future;

  EXPECT_FALSE(copy_constructed.IsReady());
  EXPECT_FALSE(copy_assigned.IsReady());
  promise.Set(42);
  EXPECT_TRUE(copy_constructed.IsReady());
  EXPECT_TRUE(copy_assigned.IsReady());
}

TEST(PjRtFutureTest, MoveConstructedFuture) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();

  PjRtFuture<std::unique_ptr<int32_t>> move_constructed(std::move(future));

  EXPECT_FALSE(move_constructed.IsReady());
  promise.Set(std::make_unique<int32_t>(42));
  EXPECT_TRUE(move_constructed.IsReady());
}

TEST(PjRtFutureTest, MoveAssignedFuture) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();

  PjRtFuture<std::unique_ptr<int32_t>> move_assigned = std::move(future);

  EXPECT_FALSE(move_assigned.IsReady());
  promise.Set(std::make_unique<int32_t>(42));
  EXPECT_TRUE(move_assigned.IsReady());
}

TEST(PjRtFutureTest, AwaitMoveOnlyFuture) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();

  promise.Set(std::make_unique<int32_t>(42));

  EXPECT_EQ(**future.Await(), 42);
  EXPECT_EQ(**std::move(future).Await(), 42);
}

TEST(PjRtFutureTest, OnReadyRvalueFuture) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();

  promise.Set(42);

  std::move(future).OnReady(
      [](absl::StatusOr<int32_t> value) { EXPECT_EQ(*value, 42); });
}

TEST(PjRtFutureTest, OnReadyMoveOnlyFuture) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();

  promise.Set(std::make_unique<int32_t>(42));

  std::move(future).OnReady([](absl::StatusOr<std::unique_ptr<int32_t>> value) {
    EXPECT_EQ(**value, 42);
  });
}

TEST(PjRtFutureTest, UnlinkedPromiseIsUnique) {
  auto [promise, future] = PjRtFuture<>::MakePromise();
  EXPECT_FALSE(promise.IsUniqueReference());
  future = {};
  EXPECT_TRUE(promise.IsUniqueReference());
}

TEST(PjRtFutureTest, PromiseIsUnique) {
  auto [promise, future] = PjRtFuture<>::MakePromise();

  // Future is linked to the promise object.
  EXPECT_FALSE(promise.IsUniqueReference());

  // Future is destroyed, but we added a callback to underlying value.
  future.OnReady([](const absl::Status&) {});
  future = {};
  EXPECT_FALSE(promise.IsUniqueReference());

  // Once promise is fulfilled, the callback is executed, and because we
  // destroyed the future, the underlying value is not referenced by anyone
  // else, and the promise becomes unique.
  promise.Set();
  EXPECT_TRUE(promise.IsUniqueReference());
}

TEST(PjRtFutureTest, MapCopyableFuture) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();
  PjRtFuture<float> mapped = future.Map([](int32_t v) { return v * 2.0f; });

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(42);
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(*future.Await(), 42);
  EXPECT_EQ(*mapped.Await(), 84.0f);

  PjRtFuture<int32_t> mapped_again =
      std::move(mapped).Map([](float v) -> int32_t { return v; });
  EXPECT_EQ(*mapped_again.Await(), 84);
}

TEST(PjRtFutureTest, MapCopyableFutureError) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();
  PjRtFuture<float> mapped = future.Map([](int32_t v) { return v * 2.0f; });

  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, MapMoveOnlyFuture) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();

  PjRtFuture<std::unique_ptr<float>> mapped =
      std::move(future).Map([](std::unique_ptr<int32_t> v) {
        return std::make_unique<float>(*v * 2.0f);
      });

  EXPECT_FALSE(mapped.IsReady());

  promise.Set(std::make_unique<int32_t>(42));

  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(**mapped.Await(), 84.0f);
}

TEST(PjRtFutureTest, MapMoveOnlyFutureError) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();
  PjRtFuture<std::unique_ptr<float>> mapped =
      std::move(future).Map([](std::unique_ptr<int32_t> v) {
        return std::make_unique<float>(*v * 2.0f);
      });

  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, MapCopyableWithInplaceConstructor) {
  struct Struct {
    explicit Struct(int32_t v) : v(v) {}
    int32_t v;
  };

  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();
  PjRtFuture<Struct> mapped = future.Map<Struct>([](int32_t v) { return v; });

  promise.Set(42);
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await()->v, 42);
}

TEST(PjRtFutureTest, MapMoveOnlyWithInplaceConstructor) {
  struct Struct {
    explicit Struct(int32_t v) : v(v) {}
    int32_t v;
  };

  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();
  PjRtFuture<Struct> mapped = std::move(future).Map<Struct>(
      [](std::unique_ptr<int32_t> v) { return *v; });

  promise.Set(std::make_unique<int32_t>(42));
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await()->v, 42);
}

TEST(PjRtFutureTest, MapUnusedResult) {
  auto [promise, future] = PjRtFuture<int>::MakePromise();

  bool called = false;
  future.Map([&](int) {
    called = true;
    return 2;
  });
  promise.Set(1);
  EXPECT_FALSE(called);
}

TEST(PjRtFutureTest, MapStatusUnusedResult) {
  auto [promise, future] = PjRtFuture<>::MakePromise();

  bool called = false;
  future.Map([&]() {
    called = true;
    return 2;
  });
  promise.Set();
  EXPECT_FALSE(called);
}

TEST(PjRtFutureTest, TryMapCopyableFuture) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();
  PjRtFuture<float> mapped = future.TryMap(
      [](int32_t v) -> absl::StatusOr<float> { return v * 2.0f; });

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(42);
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(*future.Await(), 42);
  EXPECT_EQ(*mapped.Await(), 84.0f);

  PjRtFuture<int32_t> mapped_again = std::move(mapped).TryMap(
      [](float v) -> absl::StatusOr<int32_t> { return v; });
  EXPECT_EQ(*mapped_again.Await(), 84);
}

TEST(PjRtFutureTest, TryMapCopyableFutureForwardError) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();
  PjRtFuture<float> mapped = future.TryMap(
      [](int32_t v) -> absl::StatusOr<float> { return v * 2.0f; });

  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, TryMapCopyableFutureCreateError) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();
  PjRtFuture<float> mapped =
      future.TryMap([](int32_t v) -> absl::StatusOr<float> {
        return absl::InternalError("test");
      });

  promise.Set(42);
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, TryMapMoveOnlyFuture) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();

  PjRtFuture<std::unique_ptr<float>> mapped = std::move(future).TryMap(
      [](std::unique_ptr<int32_t> v) -> absl::StatusOr<std::unique_ptr<float>> {
        return std::make_unique<float>(*v * 2.0f);
      });

  EXPECT_FALSE(mapped.IsReady());

  promise.Set(std::make_unique<int32_t>(42));

  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(**mapped.Await(), 84.0f);
}

TEST(PjRtFutureTest, TryMapMoveOnlyFutureForwardError) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();

  PjRtFuture<std::unique_ptr<float>> mapped = std::move(future).TryMap(
      [](std::unique_ptr<int32_t> v) -> absl::StatusOr<std::unique_ptr<float>> {
        return std::make_unique<float>(*v * 2.0f);
      });

  EXPECT_FALSE(mapped.IsReady());

  promise.Set(absl::InternalError("test"));

  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, TryMapMoveOnlyFutureCreateError) {
  auto [promise, future] = PjRtFuture<std::unique_ptr<int32_t>>::MakePromise();

  PjRtFuture<std::unique_ptr<float>> mapped = std::move(future).TryMap(
      [](std::unique_ptr<int32_t> v) -> absl::StatusOr<std::unique_ptr<float>> {
        return absl::InternalError("test");
      });

  EXPECT_FALSE(mapped.IsReady());

  promise.Set(std::make_unique<int32_t>(42));

  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, TryMapUnusedResult) {
  auto [promise, future] = PjRtFuture<int>::MakePromise();

  bool called = false;
  future.TryMap([&](int) -> absl::StatusOr<int> {
    called = true;
    return 2;
  });
  promise.Set(1);
  EXPECT_FALSE(called);
}

TEST(PjRtFutureTest, TryMapStatusUnusedResult) {
  auto [promise, future] = PjRtFuture<>::MakePromise();

  bool called = false;
  future.TryMap([&]() -> absl::StatusOr<int> {
    called = true;
    return 2;
  });
  promise.Set();
  EXPECT_FALSE(called);
}

TEST(PjRtFutureTest, StatelessError) {
  auto [promise, future] = PjRtFuture<>::MakePromise();

  EXPECT_FALSE(future.IsReady());
  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(future.IsReady());

  absl::Status status = future.Await();
  EXPECT_EQ(status, absl::InternalError("test"));

  future.OnReady([](absl::Status status) {
    EXPECT_EQ(status, absl::InternalError("test"));
  });
}

TEST(PjRtFutureTest, StatelessImmediate) {
  PjRtFuture<> ok_future(absl::OkStatus());
  PjRtFuture<> error_future(absl::InternalError("test"));

  EXPECT_TRUE(ok_future.IsReady());
  EXPECT_TRUE(error_future.IsReady());

  EXPECT_EQ(ok_future.Await(), absl::OkStatus());
  EXPECT_EQ(error_future.Await(), absl::InternalError("test"));

  ok_future.OnReady(
      [](absl::Status status) { EXPECT_EQ(status, absl::OkStatus()); });

  error_future.OnReady([](absl::Status status) {
    EXPECT_EQ(status, absl::InternalError("test"));
  });
}

TEST(PjRtFutureTest, MapStatelessFuture) {
  auto [promise, future] = PjRtFuture<>::MakePromise();
  PjRtFuture<float> mapped = future.Map([]() { return 42.0f; });

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(future.Await(), absl::OkStatus());
  EXPECT_EQ(*mapped.Await(), 42.0f);
}

TEST(PjRtFutureTest, MapStatelessFutureError) {
  auto [promise, future] = PjRtFuture<>::MakePromise();
  PjRtFuture<float> mapped = future.Map([]() { return 42.0f; });

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(future.Await(), absl::InternalError("test"));
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, TryMapStatelessFuture) {
  auto [promise, future] = PjRtFuture<>::MakePromise();
  PjRtFuture<float> mapped =
      future.TryMap([]() -> absl::StatusOr<float> { return 42.0f; });

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(future.Await(), absl::OkStatus());
  EXPECT_EQ(*mapped.Await(), 42.0f);
}

TEST(PjRtFutureTest, TryMapStatelessFutureForwardError) {
  auto [promise, future] = PjRtFuture<>::MakePromise();
  PjRtFuture<float> mapped =
      future.TryMap([]() -> absl::StatusOr<float> { return 42.0f; });

  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, TryMapStatelessFutureCreateError) {
  auto [promise, future] = PjRtFuture<>::MakePromise();
  PjRtFuture<float> mapped = future.TryMap(
      []() -> absl::StatusOr<float> { return absl::InternalError("test"); });

  promise.Set(absl::OkStatus());
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(PjRtFutureTest, MapToStatelessFuture) {
  auto [promise, future] = PjRtFuture<>::MakePromise();
  PjRtFuture<float> mapped = future.MapTo(42.0f);

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(future.Await(), absl::OkStatus());
  EXPECT_EQ(*mapped.Await(), 42.0f);
}

TEST(PjRtFutureTest, StatefulFuture) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();

  EXPECT_FALSE(future.IsReady());
  promise.Set(42);
  EXPECT_TRUE(future.IsReady());

  future.OnReady([](absl::StatusOr<int32_t> value) { EXPECT_EQ(*value, 42); });
}

TEST(PjRtFutureTest, StatusFuture) {
  auto [promise, future] = PjRtFuture<>::MakePromise();

  EXPECT_FALSE(future.IsReady());
  promise.Set(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());

  future.OnReady(
      [](absl::Status status) { EXPECT_EQ(status, absl::OkStatus()); });
}

TEST(PjRtFutureTest, StatusOrFuture) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise();

  EXPECT_FALSE(future.IsReady());
  promise.Set(42);
  EXPECT_TRUE(future.IsReady());

  future.OnReady([](absl::StatusOr<int32_t> value) { EXPECT_EQ(*value, 42); });
}

TEST(PjRtFutureTest, JoinFutures) {
  auto empty_join = JoinFutures({});
  EXPECT_TRUE(empty_join.IsReady());
  EXPECT_EQ(empty_join.Await(), absl::OkStatus());

  auto [promise0, future0] = PjRtFuture<>::MakePromise();
  auto [promise1, future1] = PjRtFuture<>::MakePromise();

  std::vector<PjRtFuture<>> futures0 = {future0};
  std::vector<PjRtFuture<>> futures1 = {future0, future1};

  auto join_one = JoinFutures(futures0);
  EXPECT_FALSE(join_one.IsReady());

  auto join_two = JoinFutures(futures1);
  EXPECT_FALSE(join_two.IsReady());

  promise0.Set();
  EXPECT_TRUE(join_one.IsReady());
  EXPECT_FALSE(join_two.IsReady());
  EXPECT_EQ(join_one.Await(), absl::OkStatus());

  promise1.Set();
  EXPECT_TRUE(join_two.IsReady());
  EXPECT_EQ(join_two.Await(), absl::OkStatus());
}

TEST(PjRtFutureTest, JoinErrors) {
  auto empty_join = JoinFutures({});
  EXPECT_TRUE(empty_join.IsReady());
  EXPECT_EQ(empty_join.Await(), absl::OkStatus());

  auto [promise0, future0] = PjRtFuture<>::MakePromise();
  auto [promise1, future1] = PjRtFuture<>::MakePromise();

  std::vector<PjRtFuture<>> futures0 = {future0};
  std::vector<PjRtFuture<>> futures1 = {future0, future1};

  auto join_one = JoinFutures(futures0);
  EXPECT_FALSE(join_one.IsReady());

  auto join_two = JoinFutures(futures1);
  EXPECT_FALSE(join_two.IsReady());

  promise0.Set(absl::InternalError("error #0"));
  EXPECT_TRUE(join_one.IsReady());
  EXPECT_FALSE(join_two.IsReady());
  EXPECT_EQ(join_one.Await(), absl::InternalError("error #0"));

  promise1.Set(absl::InternalError("error #1"));
  EXPECT_TRUE(join_two.IsReady());
  EXPECT_EQ(join_two.Await(), absl::InternalError("error #0"));
}

TEST(PjRtFutureTest, WithProfiling) {
  auto [promise, future] = PjRtFuture<int32_t>::MakePromise(
      [&] { return PjRtFutureHelpers::ProfilingKeys{}; },
      [&](PjRtFutureHelpers::ProfilingKeys) {});

  auto update_profiling = PjRtFutureHelpers::WithProfiling(
      std::move(future), [&] { return PjRtFutureHelpers::ProfilingKeys{}; },
      [&](PjRtFutureHelpers::ProfilingKeys) {});

  EXPECT_FALSE(update_profiling.IsReady());

  promise.Set(42);

  EXPECT_TRUE(update_profiling.IsReady());
  EXPECT_EQ(*update_profiling.Await(), 42);
}

TEST(PjRtFutureTest, MakeSharedPromise) {
  {  // Stateless future.
    auto [promise, future] = PjRtFuture<>::MakePromise();

    auto shared_promise = std::move(promise).ToShared();
    shared_promise->Set();

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_FALSE(static_cast<bool>(promise));

    EXPECT_TRUE(future.IsReady());
    EXPECT_EQ(future.Await(), absl::OkStatus());
  }

  {  // Stateful future.
    auto [promise, future] = PjRtFuture<int32_t>::MakePromise();

    auto shared_promise = std::move(promise).ToShared();
    shared_promise->Set(42);

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_FALSE(static_cast<bool>(promise));

    EXPECT_TRUE(future.IsReady());
    EXPECT_EQ(*future.Await(), 42);
  }
}

//===----------------------------------------------------------------------===//
// Performance benchmarks.
//===----------------------------------------------------------------------===//

static void BM_CreateOkFuture(benchmark::State& state) {
  for (auto _ : state) {
    PjRtFuture<> future(absl::OkStatus());
    benchmark::DoNotOptimize(future);
  }
}

static void BM_StatelessMapTo(benchmark::State& state) {
  std::shared_ptr<float> value = std::make_shared<float>(42.0f);

  for (auto _ : state) {
    PjRtFuture<> future(absl::OkStatus());
    PjRtFuture<std::shared_ptr<float>> mapped = future.MapTo(value);
    benchmark::DoNotOptimize(mapped);
  }
}

BENCHMARK(BM_CreateOkFuture);
BENCHMARK(BM_StatelessMapTo);

}  // namespace xla
