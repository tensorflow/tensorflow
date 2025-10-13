/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tsl/concurrency/future.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"

namespace tsl {

using ::absl_testing::IsOk;
using ::testing::Not;

// Inline executor that counts the number of tasks executed.
struct CountingExecutor : public Executor {
  void Execute(Task task) final {
    ++num_tasks;
    std::move(task)();
  }

  int32_t num_tasks = 0;
};

TEST(FutureTest, StatusConstructedFuture) {
  Future<> future = Future<>(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::OkStatus());
}

TEST(FutureTest, ValueConstructedFuture) {
  Future<int32_t> future = Future<int32_t>(42);
  EXPECT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::StatusOr<int32_t>(42));
}

TEST(FutureTest, StatelessFuture) {
  auto [promise, future] = Future<>::MakePromise();

  EXPECT_FALSE(future.IsReady());
  promise.Set();
  EXPECT_TRUE(future.IsReady());

  EXPECT_EQ(future.Await(), absl::OkStatus());

  future.OnReady(
      [](absl::Status status) { EXPECT_EQ(status, absl::OkStatus()); });
}

TEST(FutureTest, CreateFutureFromPromise) {
  auto [promise, _] = Future<int32_t>::MakePromise();
  Future<int32_t> future = promise.future();

  EXPECT_FALSE(future.IsReady());
  promise.Set(42);
  EXPECT_EQ(*future.Await(), 42);
}

TEST(FutureTest, StatefulFutureToStateless) {
  auto [promise, future] = Future<int32_t>::MakePromise();
  Future<> ready_future = future.GetReadyFuture();

  EXPECT_FALSE(ready_future.IsReady());
  promise.Set(42);
  EXPECT_EQ(ready_future.Await(), absl::OkStatus());
}

TEST(FutureTest, StatefulFutureToStatelessError) {
  auto [promise, future] = Future<int32_t>::MakePromise();
  Future<> ready_future = future.GetReadyFuture();

  EXPECT_FALSE(ready_future.IsReady());
  promise.Set(absl::InternalError("test"));
  EXPECT_EQ(ready_future.Await(), absl::InternalError("test"));
}

TEST(FutureTest, MoveOnlyFutureToStateless) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();
  Future<> ready_future = future.GetReadyFuture();

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(ready_future.IsReady());

  promise.Set(std::make_unique<int32_t>(42));
  EXPECT_EQ(ready_future.Await(), absl::OkStatus());
}

TEST(FutureTest, MoveOnlyFutureToStatelessError) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();
  Future<> ready_future = future.GetReadyFuture();

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(ready_future.IsReady());

  promise.Set(absl::InternalError("test"));
  EXPECT_EQ(ready_future.Await(), absl::InternalError("test"));
}

TEST(FutureTest, CopyableFuture) {
  auto [promise, future] = Future<int32_t>::MakePromise();

  Future<int32_t> copy_constructed(future);
  Future<int32_t> copy_assigned = future;

  EXPECT_FALSE(copy_constructed.IsReady());
  EXPECT_FALSE(copy_assigned.IsReady());
  promise.Set(42);
  EXPECT_TRUE(copy_constructed.IsReady());
  EXPECT_TRUE(copy_assigned.IsReady());
}

TEST(FutureTest, MoveConstructedFuture) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();

  Future<std::unique_ptr<int32_t>> move_constructed(std::move(future));

  EXPECT_FALSE(move_constructed.IsReady());
  promise.Set(std::make_unique<int32_t>(42));
  EXPECT_TRUE(move_constructed.IsReady());
}

TEST(FutureTest, MoveAssignedFuture) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();

  Future<std::unique_ptr<int32_t>> move_assigned = std::move(future);

  EXPECT_FALSE(move_assigned.IsReady());
  promise.Set(std::make_unique<int32_t>(42));
  EXPECT_TRUE(move_assigned.IsReady());
}

TEST(FutureTest, AwaitMoveOnlyFuture) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();

  promise.Set(std::make_unique<int32_t>(42));

  EXPECT_EQ(**future.Await(), 42);
  EXPECT_EQ(**std::move(future).Await(), 42);
}

TEST(FutureTest, OnReadyRvalueFuture) {
  auto [promise, future] = Future<int32_t>::MakePromise();

  promise.Set(42);

  std::move(future).OnReady(
      [](absl::StatusOr<int32_t> value) { EXPECT_EQ(*value, 42); });
}

TEST(FutureTest, OnReadyMoveOnlyFuture) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();

  promise.Set(std::make_unique<int32_t>(42));

  std::move(future).OnReady([](absl::StatusOr<std::unique_ptr<int32_t>> value) {
    EXPECT_EQ(**value, 42);
  });
}

TEST(FutureTest, PromiseNotSet) {
  Future<> future;
  {
    Promise<> promise;
    std::tie(promise, future) = Future<>::MakePromise();
  }
  ASSERT_TRUE(future.IsReady());
  EXPECT_THAT(future.Await(), Not(IsOk()));
}

TEST(FutureTest, PromiseSetTwice) {
  auto [promise, future] = Future<>::MakePromise();
  promise.Set();
  EXPECT_DEATH(promise.Set(), "Promise must not be fulfilled more than once");
}

TEST(FutureTest, UnlinkedPromiseIsUnique) {
  auto [promise, future] = Future<>::MakePromise();
  EXPECT_FALSE(promise.IsUniqueReference());
  future = {};
  EXPECT_TRUE(promise.IsUniqueReference());
}

TEST(FutureTest, PromiseIsUnique) {
  auto [promise, future] = Future<>::MakePromise();

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

TEST(FutureTest, MapCopyableFuture) {
  auto [promise, future] = Future<int32_t>::MakePromise();
  Future<float> mapped = future.Map([](int32_t v) { return v * 2.0f; });

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(42);
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(*future.Await(), 42);
  EXPECT_EQ(*mapped.Await(), 84.0f);

  Future<int32_t> mapped_again =
      std::move(mapped).Map([](float v) -> int32_t { return v; });
  EXPECT_EQ(*mapped_again.Await(), 84);
}

TEST(FutureTest, MapCopyableFutureError) {
  auto [promise, future] = Future<int32_t>::MakePromise();
  Future<float> mapped = future.Map([](int32_t v) { return v * 2.0f; });

  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(FutureTest, MapMoveOnlyFuture) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();

  Future<std::unique_ptr<float>> mapped =
      std::move(future).Map([](std::unique_ptr<int32_t> v) {
        return std::make_unique<float>(*v * 2.0f);
      });

  EXPECT_FALSE(mapped.IsReady());

  promise.Set(std::make_unique<int32_t>(42));

  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(**mapped.Await(), 84.0f);
}

TEST(FutureTest, MapMoveOnlyFutureError) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();
  Future<std::unique_ptr<float>> mapped =
      std::move(future).Map([](std::unique_ptr<int32_t> v) {
        return std::make_unique<float>(*v * 2.0f);
      });

  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(FutureTest, MapCopyableWithInplaceConstructor) {
  struct Struct {
    Struct(int32_t v) : v(v) {}  // NOLINT
    int32_t v;
  };

  auto [promise, future] = Future<int32_t>::MakePromise();
  Future<Struct> mapped = future.Map<Struct>([](int32_t v) { return v; });

  promise.Set(42);
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await()->v, 42);
}

TEST(FutureTest, MapMoveOnlyWithInplaceConstructor) {
  struct Struct {
    Struct(int32_t v) : v(v) {}  // NOLINT
    int32_t v;
  };

  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();
  Future<Struct> mapped = std::move(future).Map<Struct>(
      [](std::unique_ptr<int32_t> v) { return *v; });

  promise.Set(std::make_unique<int32_t>(42));
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await()->v, 42);
}

TEST(FutureTest, MapStatelessUnusedResult) {
  auto [promise, future] = Future<>::MakePromise();

  bool called = false;
  // We intentionally drop returned future to test that promise will not
  // execute map functor.
  (void)future.Map([&]() { called = true; });
  promise.Set(absl::OkStatus());
  EXPECT_FALSE(called);
}

TEST(FutureTest, MapStatelessOnExecutorUnusedResult) {
  auto [promise, future] = Future<>::MakePromise();

  CountingExecutor executor;
  bool called = false;
  // We intentionally drop returned future to test that promise will not
  // execute map functor.
  (void)future.Map(executor, [&]() { called = true; });
  promise.Set(absl::OkStatus());
  EXPECT_FALSE(called);
  EXPECT_EQ(executor.num_tasks, 0);
}

TEST(FutureTest, MapStatefulUnusedResult) {
  auto [promise, future] = Future<int32_t>::MakePromise();

  bool called = false;
  // We intentionally drop returned future to test that promise will not
  // execute map functor.
  (void)future.Map([&](int) { called = true; });
  promise.Set(1);
  EXPECT_FALSE(called);
}

TEST(FutureTest, MapStatefulOnExecutorUnusedResult) {
  auto [promise, future] = Future<int32_t>::MakePromise();

  CountingExecutor executor;
  bool called = false;
  // We intentionally drop returned future to test that promise will not
  // execute map functor.
  (void)future.Map(executor, [&](int32_t) { called = true; });
  promise.Set(1);
  EXPECT_FALSE(called);
  EXPECT_EQ(executor.num_tasks, 0);
}

TEST(FutureTest, MapStatefulRvalueOnExecutorUnusedResult) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();

  CountingExecutor executor;
  bool called = false;
  // We intentionally drop returned future to test that promise will not
  // execute map functor.
  (void)std::move(future).Map(executor, [&](auto) { called = true; });
  promise.Set(std::make_unique<int32_t>(1));
  EXPECT_FALSE(called);
  EXPECT_EQ(executor.num_tasks, 0);
}

TEST(FutureTest, TryMapCopyableFutureToStateless) {
  auto [promise, future] = Future<int32_t>::MakePromise();
  promise.Set(42);

  {
    Future<> mapped = future.Map([](int32_t) { return absl::OkStatus(); });
    EXPECT_EQ(mapped.Await(), absl::OkStatus());
  }

  {
    Future<> mapped =
        future.Map([](int32_t) { return absl::InternalError("test"); });
    EXPECT_EQ(mapped.Await(), absl::InternalError("test"));
  }
}

TEST(FutureTest, TryMapCopyableFuture) {
  auto [promise, future] = Future<int32_t>::MakePromise();
  Future<float> mapped =
      future.Map([](int32_t v) -> absl::StatusOr<float> { return v * 2.0f; });

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(42);
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(*future.Await(), 42);
  EXPECT_EQ(*mapped.Await(), 84.0f);

  Future<int32_t> mapped_again = std::move(mapped).Map(
      [](float v) -> absl::StatusOr<int32_t> { return v; });
  EXPECT_EQ(*mapped_again.Await(), 84);
}

TEST(FutureTest, TryMapCopyableFutureForwardError) {
  auto [promise, future] = Future<int32_t>::MakePromise();
  Future<float> mapped =
      future.Map([](int32_t v) -> absl::StatusOr<float> { return v * 2.0f; });

  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(FutureTest, TryMapCopyableFutureCreateError) {
  auto [promise, future] = Future<int32_t>::MakePromise();
  Future<float> mapped = future.Map([](int32_t v) -> absl::StatusOr<float> {
    return absl::InternalError("test");
  });

  promise.Set(42);
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(FutureTest, TryMapMoveOnlyFutureToStateless) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();
  promise.Set(std::make_unique<int32_t>(42));

  Future<> mapped = std::move(future).Map(
      [](std::unique_ptr<int32_t>) { return absl::OkStatus(); });
  EXPECT_EQ(mapped.Await(), absl::OkStatus());
}

TEST(FutureTest, TryMapMoveOnlyFuture) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();

  Future<std::unique_ptr<float>> mapped = std::move(future).Map(
      [](std::unique_ptr<int32_t> v) -> absl::StatusOr<std::unique_ptr<float>> {
        return std::make_unique<float>(*v * 2.0f);
      });

  EXPECT_FALSE(mapped.IsReady());

  promise.Set(std::make_unique<int32_t>(42));

  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(**mapped.Await(), 84.0f);
}

TEST(FutureTest, TryMapMoveOnlyFutureForwardError) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();

  Future<std::unique_ptr<float>> mapped = std::move(future).Map(
      [](std::unique_ptr<int32_t> v) -> absl::StatusOr<std::unique_ptr<float>> {
        return std::make_unique<float>(*v * 2.0f);
      });

  EXPECT_FALSE(mapped.IsReady());

  promise.Set(absl::InternalError("test"));

  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(FutureTest, MapFutureCopies) {
  auto [promise, future] = Future<std::shared_ptr<int32_t>>::MakePromise();
  promise.Set(std::make_shared<int32_t>(42));

  Future<std::shared_ptr<int32_t>> future0 = future;
  Future<std::shared_ptr<int32_t>> future1 = future;

  Future<> future2 = std::move(future0).Map(
      [](std::shared_ptr<int32_t>) { return absl::OkStatus(); });
  Future<> future3 = std::move(future1).Map(
      [](std::shared_ptr<int32_t>) { return absl::OkStatus(); });

  EXPECT_EQ(future2.Await(), absl::OkStatus());
  EXPECT_EQ(future3.Await(), absl::OkStatus());

  // Check that future holds a valid shared pointer, and it was not actually
  // moved to any of the functors.
  EXPECT_EQ(**future.Await(), 42);
}

TEST(FutureTest, TryMapMoveOnlyFutureCreateError) {
  auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();

  Future<std::unique_ptr<float>> mapped = std::move(future).Map(
      [](std::unique_ptr<int32_t> v) -> absl::StatusOr<std::unique_ptr<float>> {
        return absl::InternalError("test");
      });

  EXPECT_FALSE(mapped.IsReady());

  promise.Set(std::make_unique<int32_t>(42));

  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(FutureTest, MapWithVoidFunctor) {
  {
    auto [promise, future] = Future<>::MakePromise();
    promise.Set(absl::OkStatus());

    Future<> mapped = future.Map([] {});
    EXPECT_EQ(mapped.Await(), absl::OkStatus());
  }

  {
    auto [promise, future] = Future<int32_t>::MakePromise();
    promise.Set(42);

    Future<> mapped = future.Map([](int32_t value) { EXPECT_EQ(value, 42); });
    EXPECT_EQ(mapped.Await(), absl::OkStatus());
  }

  {
    auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();
    promise.Set(std::make_unique<int32_t>(42));

    Future<> mapped = std::move(future).Map(
        [](std::unique_ptr<int32_t> value) { EXPECT_EQ(*value, 42); });
    EXPECT_EQ(mapped.Await(), absl::OkStatus());
  }
}

TEST(FutureTest, MapDoesNotCopy) {
  static int32_t counter = 0;

  // A trivial class that counts how many times the copy constructor is called.
  struct Data {
    Data() = default;

    Data(const Data& other) { ++counter; }
    Data(Data&& other) {}

    Data& operator=(Data& other) = delete;
    Data& operator=(Data&& other) = delete;
  };

  auto [promise, future] = Future<Data>::MakePromise();

  Future<> m0 = future.Map([](const Data& data) {});
  Future<> m1 = future.Map([](Data data) {});
  Future<> m2 = std::move(future).Map([](const Data& data) {});

  promise.Set(Data{});

  EXPECT_EQ(m0.Await(), absl::OkStatus());
  EXPECT_EQ(m1.Await(), absl::OkStatus());
  EXPECT_EQ(m2.Await(), absl::OkStatus());

  EXPECT_EQ(counter, 1);
};

TEST(FutureTest, MapOnExecutorDoesNotCopy) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 4);
  tsl::Executor* executor = thread_pool.AsExecutor();

  static int32_t counter = 0;

  // A trivial class that counts how many times the copy constructor is called.
  struct Data {
    Data() = default;

    Data(const Data& other) { ++counter; }
    Data(Data&& other) {}

    Data& operator=(Data& other) = delete;
    Data& operator=(Data&& other) = delete;
  };

  auto [promise, future] = Future<Data>::MakePromise();

  Future<> m0 = future.Map(*executor, [](const Data& data) {});
  Future<> m1 = future.Map(*executor, [](Data data) {});
  Future<> m2 = std::move(future).Map(*executor, [](const Data& data) {});

  promise.Set(Data{});

  EXPECT_EQ(m0.Await(), absl::OkStatus());
  EXPECT_EQ(m1.Await(), absl::OkStatus());
  EXPECT_EQ(m2.Await(), absl::OkStatus());

  EXPECT_EQ(counter, 1);
};

TEST(FutureTest, StatelessError) {
  auto [promise, future] = Future<>::MakePromise();

  EXPECT_FALSE(future.IsReady());
  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(future.IsReady());

  absl::Status status = future.Await();
  EXPECT_EQ(status, absl::InternalError("test"));

  future.OnReady([](absl::Status status) {
    EXPECT_EQ(status, absl::InternalError("test"));
  });
}

TEST(FutureTest, StatelessImmediate) {
  Future<> ok_future(absl::OkStatus());
  Future<> error_future(absl::InternalError("test"));

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

TEST(FutureTest, MapStatelessFuture) {
  auto [promise, future] = Future<>::MakePromise();
  Future<float> mapped = future.Map([]() { return 42.0f; });

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(future.Await(), absl::OkStatus());
  EXPECT_EQ(*mapped.Await(), 42.0f);
}

TEST(FutureTest, MapStatelessToStatus) {
  auto [promise, future] = Future<>::MakePromise();
  promise.Set(absl::OkStatus());

  {
    Future<> mapped = future.Map([] { return absl::OkStatus(); });
    EXPECT_TRUE(mapped.IsReady());
    EXPECT_EQ(mapped.Await(), absl::OkStatus());
  }

  {
    Future<> mapped = future.Map([] { return absl::InternalError("test"); });
    EXPECT_TRUE(mapped.IsReady());
    EXPECT_EQ(mapped.Await(), absl::InternalError("test"));
  }
}

TEST(FutureTest, MapStatelessErrorToStatus) {
  auto [promise, future] = Future<>::MakePromise();
  promise.Set(absl::InternalError("test"));

  Future<> mapped = future.Map([] { return absl::OkStatus(); });
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await(), absl::InternalError("test"));
}

TEST(FutureTest, MapStatelessFutureError) {
  auto [promise, future] = Future<>::MakePromise();
  Future<float> mapped = future.Map([]() { return 42.0f; });

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(future.Await(), absl::InternalError("test"));
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(FutureTest, MapStatelessFutureToStatusOr) {
  auto [promise, future] = Future<>::MakePromise();
  Future<float> mapped =
      future.Map([]() -> absl::StatusOr<float> { return 42.0f; });

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(future.Await(), absl::OkStatus());
  EXPECT_EQ(*mapped.Await(), 42.0f);
}

TEST(FutureTest, MapStatelessFutureForwardError) {
  auto [promise, future] = Future<>::MakePromise();
  Future<float> mapped =
      future.Map([]() -> absl::StatusOr<float> { return 42.0f; });

  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(FutureTest, MapStatelessFutureCreateError) {
  auto [promise, future] = Future<>::MakePromise();
  Future<float> mapped = future.Map(
      []() -> absl::StatusOr<float> { return absl::InternalError("test"); });

  promise.Set(absl::OkStatus());
  EXPECT_TRUE(mapped.IsReady());
  EXPECT_EQ(mapped.Await().status(), absl::InternalError("test"));
}

TEST(FutureTest, MapToStatelessFuture) {
  auto [promise, future] = Future<>::MakePromise();
  Future<float> mapped = future.MapTo(42.0f);

  EXPECT_FALSE(future.IsReady());
  EXPECT_FALSE(mapped.IsReady());

  promise.Set(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());
  EXPECT_TRUE(mapped.IsReady());

  EXPECT_EQ(future.Await(), absl::OkStatus());
  EXPECT_EQ(*mapped.Await(), 42.0f);
}

TEST(FutureTest, StatefulFuture) {
  auto [promise, future] = Future<int32_t>::MakePromise();

  EXPECT_FALSE(future.IsReady());
  promise.Set(42);
  EXPECT_TRUE(future.IsReady());

  future.OnReady([](absl::StatusOr<int32_t> value) { EXPECT_EQ(*value, 42); });
}

TEST(FutureTest, StatusFuture) {
  auto [promise, future] = Future<>::MakePromise();

  EXPECT_FALSE(future.IsReady());
  promise.Set(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());

  future.OnReady(
      [](absl::Status status) { EXPECT_EQ(status, absl::OkStatus()); });
}

TEST(FutureTest, StatusOrFuture) {
  auto [promise, future] = Future<int32_t>::MakePromise();

  EXPECT_FALSE(future.IsReady());
  promise.Set(42);
  EXPECT_TRUE(future.IsReady());

  future.OnReady([](absl::StatusOr<int32_t> value) { EXPECT_EQ(*value, 42); });
}

TEST(FutureTest, JoinFutures) {
  auto empty_join = JoinFutures({});
  EXPECT_TRUE(empty_join.IsReady());
  EXPECT_EQ(empty_join.Await(), absl::OkStatus());

  auto [promise0, future0] = Future<>::MakePromise();
  auto [promise1, future1] = Future<>::MakePromise();

  std::vector<Future<>> futures0 = {future0};
  std::vector<Future<>> futures1 = {future0, future1};

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

TEST(FutureTest, JoinErrors) {
  auto empty_join = JoinFutures({});
  EXPECT_TRUE(empty_join.IsReady());
  EXPECT_EQ(empty_join.Await(), absl::OkStatus());

  auto [promise0, future0] = Future<>::MakePromise();
  auto [promise1, future1] = Future<>::MakePromise();

  std::vector<Future<>> futures0 = {future0};
  std::vector<Future<>> futures1 = {future0, future1};

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

TEST(FutureTest, WithProfiling) {
  auto [promise, future] = Future<int32_t>::MakePromise(
      [&] { return FutureHelpers::ProfilingKeys{}; },
      [&](FutureHelpers::ProfilingKeys) {});

  auto update_profiling = FutureHelpers::WithProfiling(
      std::move(future), [&] { return FutureHelpers::ProfilingKeys{}; },
      [&](FutureHelpers::ProfilingKeys) {});

  EXPECT_FALSE(update_profiling.IsReady());

  promise.Set(42);

  EXPECT_TRUE(update_profiling.IsReady());
  EXPECT_EQ(*update_profiling.Await(), 42);
}

TEST(FutureTest, MakeSharedPromise) {
  {  // Stateless future.
    auto [promise, future] = Future<>::MakePromise();

    auto shared_promise = std::move(promise).ToShared();
    shared_promise->Set();

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_FALSE(static_cast<bool>(promise));

    EXPECT_TRUE(future.IsReady());
    EXPECT_EQ(future.Await(), absl::OkStatus());
  }

  {  // Stateful future.
    auto [promise, future] = Future<int32_t>::MakePromise();

    auto shared_promise = std::move(promise).ToShared();
    shared_promise->Set(42);

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_FALSE(static_cast<bool>(promise));

    EXPECT_TRUE(future.IsReady());
    EXPECT_EQ(*future.Await(), 42);
  }
}

TEST(FutureTest, MakeOnStateless) {
  InlineExecutor e;

  {
    auto future = Future<>::MakeOn(e, [] { return absl::OkStatus(); });
    EXPECT_TRUE(future.IsReady());
    EXPECT_EQ(future.Await(), absl::OkStatus());
  }

  {
    auto future =
        Future<>::MakeOn(e, [] { return absl::InternalError("test"); });
    EXPECT_TRUE(future.IsReady());
    EXPECT_EQ(future.Await(), absl::InternalError("test"));
  }
}

TEST(FutureTest, MakeOnStateful) {
  InlineExecutor executor;

  struct Foo {
    Foo(int32_t value) : value(value) {}  // NOLINT
    int32_t value;
  };

  {
    auto future = Future<int32_t>::MakeOn(executor, [] { return 42; });
    EXPECT_TRUE(future.IsReady());
    EXPECT_EQ(*future.Await(), 42);
  }

  {
    auto future = Future<Foo>::MakeOn(executor, [] { return 42; });
    EXPECT_TRUE(future.IsReady());
    EXPECT_EQ(future.Await()->value, 42);
  }

  {
    auto future = Future<std::unique_ptr<int32_t>>::MakeOn(
        executor, [] { return std::make_unique<int32_t>(42); });
    EXPECT_TRUE(future.IsReady());
    EXPECT_EQ(**future.Await(), 42);
  }

  {
    auto future = Future<int32_t>::MakeOn(
        executor, [] { return absl::InternalError("test"); });
    EXPECT_TRUE(future.IsReady());
    EXPECT_EQ(future.Await().status(), absl::InternalError("test"));
  }
}

TEST(FutureTest, OnReadyOnExecutor) {
  Future<> future0(absl::OkStatus());
  future0.OnReady(InlineExecutor::Instance(), [](absl::Status status) {
    ASSERT_EQ(status, absl::OkStatus());
  });

  Future<int32_t> future1(42);
  future1.OnReady(InlineExecutor::Instance(),
                  [](absl::StatusOr<int32_t> x) { ASSERT_EQ(*x, 42); });

  Future<std::unique_ptr<int32_t>> future2(std::make_unique<int32_t>(42));
  std::move(future2).OnReady(
      InlineExecutor::Instance(),
      [](absl::StatusOr<std::unique_ptr<int32_t>> x) { ASSERT_EQ(**x, 42); });
}

TEST(FutureTest, MapOnExecutor) {
  Future<> future0(absl::OkStatus());
  Future<int32_t> mapped0 =
      future0.Map(InlineExecutor::Instance(), [] { return 42; });
  EXPECT_EQ(*mapped0.Await(), 42);

  Future<int32_t> future1(42);
  Future<int32_t> mapped1 =
      future1.Map(InlineExecutor::Instance(), [](int32_t x) { return x + 1; });
  EXPECT_EQ(*mapped1.Await(), 43);

  Future<std::unique_ptr<int32_t>> future2(std::make_unique<int32_t>(42));
  Future<int32_t> mapped2 =
      std::move(future2).Map(InlineExecutor::Instance(),
                             [](std::unique_ptr<int32_t> x) { return *x + 1; });
  EXPECT_EQ(*mapped2.Await(), 43);
}

TEST(FutureTest, MapStatelessOnThreadPoolExecutor) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 4);

  std::vector<Future<>> mapped;
  std::atomic<int32_t> counter = 0;

  {  // Create mapped future in a nested scope to make sure that `promise` and
    // `future` are destroyed before the end of the test.
    auto [promise, future] = Future<>::MakePromise();
    for (size_t i = 0; i < 100; ++i) {
      mapped.push_back(
          future.Map(*thread_pool.AsExecutor(), [&] { ++counter; }));
    }
    promise.Set();
  }

  EXPECT_EQ(tsl::JoinFutures(mapped).Await(), absl::OkStatus());
  EXPECT_EQ(counter, 100);
}

TEST(FutureTest, MapStatefulOnThreadPoolExecutor) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 4);

  std::vector<Future<>> mapped;
  std::atomic<int32_t> counter = 0;

  {  // Create mapped future in a nested scope to make sure that `promise` and
    // `future` are destroyed before the end of the test.
    auto [promise, future] = Future<int32_t>::MakePromise();
    for (size_t i = 0; i < 100; ++i) {
      mapped.push_back(future.Map(*thread_pool.AsExecutor(),
                                  [&](int32_t value) { counter += value; }));
    }
    promise.Set(1);
  }

  EXPECT_EQ(tsl::JoinFutures(mapped).Await(), absl::OkStatus());
  EXPECT_EQ(counter, 100);
}

TEST(FutureTest, MapMoveOnlyOnThreadPoolExecutor) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 4);

  std::vector<Future<>> mapped;
  std::atomic<int32_t> counter = 0;

  {  // Create mapped future in a nested scope to make sure that `promise` and
    // `future` are destroyed before the end of the test.
    auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();
    for (size_t i = 0; i < 100; ++i) {
      mapped.push_back(future.Map(
          *thread_pool.AsExecutor(),
          [&](const std::unique_ptr<int32_t>& value) { counter += *value; }));
    }
    promise.Set(std::make_unique<int32_t>(1));
  }

  EXPECT_EQ(tsl::JoinFutures(mapped).Await(), absl::OkStatus());
  EXPECT_EQ(counter, 100);
}

TEST(FutureTest, MapMoveOnlyRvalueOnThreadPoolExecutor) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 4);

  std::vector<Future<>> mapped;
  std::atomic<int32_t> counter = 0;

  {  // Create mapped future in a nested scope to make sure that `promise` and
    // `future` are destroyed before the end of the test.
    for (size_t i = 0; i < 100; ++i) {
      auto [promise, future] = Future<std::unique_ptr<int32_t>>::MakePromise();
      mapped.push_back(std::move(future).Map(
          *thread_pool.AsExecutor(),
          [&](std::unique_ptr<int32_t> value) { counter += *value; }));
      promise.Set(std::make_unique<int32_t>(1));
    }
  }

  EXPECT_EQ(tsl::JoinFutures(mapped).Await(), absl::OkStatus());
  EXPECT_EQ(counter, 100);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks.
//===----------------------------------------------------------------------===//

static void BM_CreateOkFuture(benchmark::State& state) {
  for (auto _ : state) {
    Future<> future(absl::OkStatus());
    benchmark::DoNotOptimize(future);
  }
}

static void BM_CopyFuture(benchmark::State& state) {
  Future<> future(absl::OkStatus());

  for (auto _ : state) {
    Future<> copy = future;
    benchmark::DoNotOptimize(copy);
  }
}

static void BM_MapStatelessFuture(benchmark::State& state) {
  Future<> future(absl::OkStatus());

  for (auto _ : state) {
    Future<int32_t> mapped = future.Map([] { return 42; });
    benchmark::DoNotOptimize(mapped);
  }
}

static void BM_TryMapStatelessFuture(benchmark::State& state) {
  Future<> future(absl::OkStatus());

  for (auto _ : state) {
    Future<int32_t> mapped =
        future.Map([]() -> absl::StatusOr<int32_t> { return 42; });
    benchmark::DoNotOptimize(mapped);
  }
}

static void BM_MapToFromStatelessFuture(benchmark::State& state) {
  Future<> future(absl::OkStatus());

  for (auto _ : state) {
    Future<int32_t> mapped = future.MapTo(42);
    benchmark::DoNotOptimize(mapped);
  }
}

static void BM_MapStatefulFuture(benchmark::State& state) {
  Future<int32_t> future(42);

  for (auto _ : state) {
    Future<int32_t> mapped = future.Map([](int32_t x) { return x + 1; });
    benchmark::DoNotOptimize(mapped);
  }
}

static void BM_TryMapStatefulFuture(benchmark::State& state) {
  Future<int32_t> future(42);

  for (auto _ : state) {
    Future<int32_t> mapped =
        future.Map([](int32_t x) -> absl::StatusOr<int32_t> { return x + 1; });
    benchmark::DoNotOptimize(mapped);
  }
}

static void BM_CreateAndMapStatelessFuture(benchmark::State& state) {
  Future<> future(absl::OkStatus());

  for (auto _ : state) {
    auto [promise, future] = Future<>::MakePromise();
    Future<int32_t> mapped = future.Map([] { return 42; });
    promise.Set(absl::OkStatus());
    benchmark::DoNotOptimize(mapped);
  }
}

BENCHMARK(BM_CreateOkFuture);
BENCHMARK(BM_CopyFuture);
BENCHMARK(BM_MapStatelessFuture);
BENCHMARK(BM_TryMapStatelessFuture);
BENCHMARK(BM_MapToFromStatelessFuture);
BENCHMARK(BM_MapStatefulFuture);
BENCHMARK(BM_TryMapStatefulFuture);
BENCHMARK(BM_CreateAndMapStatelessFuture);

}  // namespace tsl
