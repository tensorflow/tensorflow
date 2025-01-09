/* Copyright 2022 Google LLC. All Rights Reserved.

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

#include "xla/tsl/concurrency/async_value_ref.h"

#include <any>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace tsl {

class WrappedInt32 {
 public:
  explicit WrappedInt32(int32_t value) : value_(value) {}
  int32_t value() const { return value_; }

 private:
  int32_t value_;
};

constexpr int32_t kTestValue = 42;

TEST(AsyncValueRefTest, MakeUnconstructedStatusOrOfAny) {
  auto value = MakeUnconstructedAsyncValueRef<absl::StatusOr<std::any>>();
  EXPECT_TRUE(value.IsUnavailable());
}

TEST(AsyncValueRefTest, MakeUnconstructedStatusOr) {
  auto value = MakeUnconstructedAsyncValueRef<absl::StatusOr<int32_t>>();
  EXPECT_TRUE(value.IsUnavailable());
}

TEST(AsyncValueRefTest, MakeConstructedStatusOr) {
  auto value = MakeConstructedAsyncValueRef<absl::StatusOr<int32_t>>(42);
  EXPECT_TRUE(value.IsUnavailable());
}

TEST(AsyncValueRefTest, MakeAvailableStatusOr) {
  auto value = MakeAvailableAsyncValueRef<absl::StatusOr<int32_t>>(42);
  EXPECT_TRUE(value.IsAvailable());
  EXPECT_EQ(**value, 42);
}

TEST(AsyncValueRefTest, ImplicitStatusConversion) {
  auto error = []() -> AsyncValueRef<WrappedInt32> {
    return absl::InternalError("Error");
  }();

  EXPECT_TRUE(error.IsAvailable());
  EXPECT_TRUE(error.IsError());
  EXPECT_EQ(error.GetError(), absl::InternalError("Error"));
}

TEST(AsyncValueRefTest, ImplicitStatusConversionWithStatusOrPayloadAndStatus) {
  auto status = []() -> absl::StatusOr<absl::StatusOr<int32_t>> {
    return absl::InternalError("Error");
  }();

  auto error = []() -> AsyncValueRef<absl::StatusOr<int32_t>> {
    return absl::InternalError("Error");
  }();

  // Check that AsyncValueRef<absl::StatusOr<T>> behavior is consistent with
  // absl::StatusOr<absl::StatusOr<T>> for implicit error conversion.

  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.status(), absl::InternalError("Error"));

  EXPECT_TRUE(error.IsError());
  EXPECT_EQ(error.GetError(), absl::InternalError("Error"));
}

TEST(AsyncValueRefTest, ValueCheck) {
  auto wrapped_int_value = MakeAvailableAsyncValueRef<WrappedInt32>(kTestValue);
  EXPECT_EQ(wrapped_int_value.get().value(), kTestValue);
  EXPECT_EQ(wrapped_int_value->value(), kTestValue);
  EXPECT_EQ((*wrapped_int_value).value(), kTestValue);
}

TEST(AsyncValueRefTest, ValueCheckFromRCReference) {
  auto wrapped_int_value = MakeAvailableAsyncValueRef<WrappedInt32>(kTestValue);
  RCReference<AsyncValue> generic_value = std::move(wrapped_int_value);
  EXPECT_EQ(generic_value->get<WrappedInt32>().value(), kTestValue);
}

TEST(AsyncValueRefTest, ValueCheckFromAliasedRCReference) {
  auto wrapped_int_value = MakeAvailableAsyncValueRef<WrappedInt32>(kTestValue);
  RCReference<AsyncValue> generic_value = std::move(wrapped_int_value);
  AsyncValueRef<WrappedInt32> aliased_int_value(std::move(generic_value));
  EXPECT_EQ(aliased_int_value.get().value(), kTestValue);
  EXPECT_EQ(aliased_int_value->value(), kTestValue);
  EXPECT_EQ((*aliased_int_value).value(), kTestValue);
}

TEST(AsyncValueRefTest, ConstructedToError) {
  auto value = MakeConstructedAsyncValueRef<int32_t>(kTestValue);

  EXPECT_FALSE(value.IsConcrete());
  EXPECT_FALSE(value.IsAvailable());

  value.AndThen([] {});
  value.SetError(absl::InternalError("test error"));

  EXPECT_TRUE(value.IsAvailable());
  EXPECT_FALSE(value.IsConcrete());
  EXPECT_TRUE(value.IsError());
}

TEST(AsyncValueRefTest, ConstructedToConcrete) {
  auto value = MakeConstructedAsyncValueRef<int32_t>(kTestValue);

  EXPECT_FALSE(value.IsConcrete());
  EXPECT_FALSE(value.IsAvailable());

  value.AndThen([] {});
  value.SetStateConcrete();

  EXPECT_TRUE(value.IsAvailable());
  EXPECT_TRUE(value.IsConcrete());
  EXPECT_FALSE(value.IsError());

  EXPECT_EQ(kTestValue, value.get());
}

TEST(AsyncValueRefTest, UnconstructedEmplace) {
  auto value = MakeUnconstructedAsyncValueRef<int32_t>();

  EXPECT_FALSE(value.IsConcrete());
  EXPECT_FALSE(value.IsAvailable());

  value.AndThen([] {});

  value.emplace(kTestValue);
  EXPECT_TRUE(value.IsAvailable());
  EXPECT_TRUE(value.IsConcrete());

  EXPECT_EQ(kTestValue, value.get());
}

TEST(AsyncValueRefTest, CopyRef) {
  auto value = MakeAvailableAsyncValueRef<int32_t>(kTestValue);

  EXPECT_TRUE(value.IsConcrete());

  EXPECT_TRUE(value.IsUnique());
  auto copied_value = value.CopyRef();
  EXPECT_FALSE(value.IsUnique());

  EXPECT_EQ(value.GetAsyncValue(), copied_value.GetAsyncValue());
}

TEST(AsyncValueRefTest, AndThen) {
  AsyncValueRef<int32_t> ref = MakeUnconstructedAsyncValueRef<int32_t>();

  EXPECT_FALSE(ref.IsConcrete());
  EXPECT_FALSE(ref.IsAvailable());

  bool executed = false;
  ref.AndThen([&]() { executed = true; });

  ref.emplace(42);
  EXPECT_TRUE(executed);
}

TEST(AsyncValueRefTest, AndThenError) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  auto error = absl::InternalError("test error");
  ref.SetError(error);

  ref.AndThen([&](absl::Status status) { EXPECT_EQ(status, error); });
}

TEST(AsyncValueRefTest, AndThenNoError) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);
  ref.AndThen([](absl::Status status) { EXPECT_TRUE(status.ok()); });
}

TEST(AsyncValueRefTest, AndThenStatusOrError) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  auto error = absl::InternalError("test error");
  ref.SetError(error);

  ref.AndThen([&](absl::StatusOr<int32_t*> v) {
    EXPECT_FALSE(v.ok());
    EXPECT_EQ(v.status(), error);
  });
}

TEST(AsyncValueRefTest, AndThenStatusOrNoError) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);
  ref.AndThen([&](absl::StatusOr<int32_t*> v) { EXPECT_EQ(**v, 42); });
}

TEST(AsyncValueRefTest, Nullptr) {
  // Test constructing from nullptr.
  AsyncValueRef<int> av_int = nullptr;
  EXPECT_FALSE(av_int);

  // Test assignment to nullptr.
  AsyncValueRef<int> av_int2 = MakeConstructedAsyncValueRef<int>(kTestValue);
  EXPECT_TRUE(av_int2);
  av_int2 = nullptr;
  EXPECT_FALSE(av_int2);
}

TEST(AsyncValueRefTest, MapAvailable) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);

  AsyncValueRef<float> mapped_to_float =
      ref.Map([](int32_t value) -> float { return value; });
  EXPECT_TRUE(mapped_to_float.IsAvailable());
  EXPECT_EQ(mapped_to_float.get(), 42.0f);
}

TEST(AsyncValueRefTest, MapUnvailable) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  AsyncValueRef<float> mapped_to_float =
      ref.Map([](int32_t value) -> float { return value; });

  EXPECT_FALSE(mapped_to_float.IsAvailable());
  ref.SetStateConcrete();

  EXPECT_TRUE(mapped_to_float.IsAvailable());
  EXPECT_EQ(mapped_to_float.get(), 42.0f);
}

TEST(AsyncValueRefTest, MapToNonMoveable) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);

  AsyncValueRef<std::atomic<int32_t>> mapped_to_atomic =
      ref.Map<std::atomic<int32_t>>([](int32_t value) { return value; });
  EXPECT_TRUE(mapped_to_atomic.IsAvailable());
  EXPECT_EQ(mapped_to_atomic->load(), 42);
}

TEST(AsyncValueRefTest, MapError) {
  AsyncValueRef<int32_t> ref =
      MakeErrorAsyncValueRef(absl::InternalError("error"));

  AsyncValueRef<float> mapped_to_float =
      ref.Map([](int32_t value) -> float { return value; });
  EXPECT_TRUE(mapped_to_float.IsError());
  EXPECT_EQ(mapped_to_float.GetError(), absl::InternalError("error"));
}

TEST(AsyncValueRefTest, MapUnvailableError) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  AsyncValueRef<float> mapped_to_float =
      ref.Map([](int32_t value) -> float { return value; });

  EXPECT_FALSE(mapped_to_float.IsAvailable());
  ref.SetError(absl::InternalError("error"));

  EXPECT_TRUE(mapped_to_float.IsError());
  EXPECT_EQ(mapped_to_float.GetError(), absl::InternalError("error"));
}

TEST(AsyncValueRefTest, MapMultipleTimes) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);

  auto plus_one = [](int32_t value) { return value + 1; };
  AsyncValueRef<int32_t> mapped = ref.Map(plus_one)
                                      .Map(plus_one)
                                      .Map(plus_one)
                                      .Map(plus_one)
                                      .Map(plus_one)
                                      .Map(plus_one);

  EXPECT_TRUE(mapped.IsAvailable());
  EXPECT_EQ(mapped.get(), 42 + 6);
}

TEST(AsyncValuePtrTest, MapToStatus) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);

  AsyncValueRef<absl::Status> mapped_to_status =
      ref.Map([](int32_t value) -> absl::Status { return absl::OkStatus(); });
  EXPECT_TRUE(mapped_to_status.IsAvailable());
  EXPECT_EQ(mapped_to_status.get(), absl::OkStatus());
}

TEST(AsyncValueRefTest, MapToStatusOr) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);

  AsyncValueRef<absl::StatusOr<float>> mapped_to_float =
      ref.Map([](int32_t value) -> absl::StatusOr<float> { return value; });
  EXPECT_TRUE(mapped_to_float.IsAvailable());
  EXPECT_EQ(*mapped_to_float.get(), 42.0f);
}

TEST(AsyncValueRefTest, TryMap) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);

  AsyncValueRef<float> mapped_to_float =
      ref.TryMap([](int32_t value) -> absl::StatusOr<float> { return value; });
  EXPECT_TRUE(mapped_to_float.IsAvailable());
  EXPECT_EQ(mapped_to_float.get(), 42.0f);
}

TEST(AsyncValueRefTest, TryMapError) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);

  AsyncValueRef<float> mapped_to_float =
      ref.TryMap([](int32_t value) -> absl::StatusOr<float> {
        return absl::InternalError("error");
      });
  EXPECT_TRUE(mapped_to_float.IsError());
  EXPECT_EQ(mapped_to_float.GetError(), absl::InternalError("error"));
}

TEST(AsyncValueRefTest, TryMapConstructible) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);

  struct X {
    explicit X(float value) : value(value) {}
    float value;
  };

  AsyncValueRef<X> mapped_to_x = ref.TryMap<X>(
      [](int32_t value) -> absl::StatusOr<float> { return value; });
  EXPECT_TRUE(mapped_to_x.IsAvailable());
  EXPECT_EQ(mapped_to_x->value, 42.0f);
}

TEST(AsyncValueRefTest, FlatMapAvailable) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);

  AsyncValueRef<float> fmapped_to_float = ref.FlatMap([](int32_t value) {
    return MakeAvailableAsyncValueRef<float>(static_cast<float>(value));
  });

  EXPECT_TRUE(fmapped_to_float.IsAvailable());
  EXPECT_EQ(fmapped_to_float.get(), 42.0f);
}

TEST(AsyncValueRefTest, FlatMapUnavailable) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  AsyncValueRef<float> fmapped_to_float = ref.FlatMap([](int32_t value) {
    return MakeAvailableAsyncValueRef<float>(static_cast<float>(value));
  });

  EXPECT_FALSE(fmapped_to_float.IsAvailable());
  ref.SetStateConcrete();

  EXPECT_TRUE(fmapped_to_float.IsAvailable());
  EXPECT_EQ(fmapped_to_float.get(), 42.0f);
}

TEST(AsyncValueRefTest, FlatMapAvailableError) {
  AsyncValueRef<int32_t> ref =
      MakeErrorAsyncValueRef(absl::InternalError("error"));

  AsyncValueRef<float> fmapped_to_float = ref.FlatMap([](int32_t value) {
    return MakeAvailableAsyncValueRef<float>(static_cast<float>(value));
  });

  EXPECT_TRUE(fmapped_to_float.IsError());
  EXPECT_EQ(fmapped_to_float.GetError(), absl::InternalError("error"));
}

TEST(AsyncValueRefTest, FlatMapUnavailableError) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  AsyncValueRef<float> fmapped_to_float = ref.FlatMap([](int32_t value) {
    return MakeAvailableAsyncValueRef<float>(static_cast<float>(value));
  });

  EXPECT_FALSE(fmapped_to_float.IsAvailable());
  ref.SetError(absl::InternalError("error"));

  EXPECT_TRUE(fmapped_to_float.IsError());
  EXPECT_EQ(fmapped_to_float.GetError(), absl::InternalError("error"));
}

struct DeferredExecutor : public AsyncValue::Executor {
  void Execute(Task task) final { tasks.push_back(std::move(task)); }

  size_t Quiesce() {
    size_t n = 0;
    while (!tasks.empty()) {
      Task task = std::move(tasks.back());
      tasks.pop_back();
      task();
      ++n;
    }
    return n;
  }

  std::vector<Task> tasks;
};

TEST(AsyncValueRefTest, MakeAsyncValueRef) {
  DeferredExecutor executor;

  {  // Make AsyncValueRef from a function that returns a value.
    AsyncValueRef<float> ref =
        MakeAsyncValueRef<float>(executor, []() -> float { return 42.0f; });

    EXPECT_FALSE(ref.IsAvailable());
    EXPECT_EQ(executor.Quiesce(), 1);

    EXPECT_TRUE(ref.IsAvailable());
    EXPECT_EQ(ref.get(), 42.0f);
  }

  {  // Make AsyncValueRef with automatic type inference.
    AsyncValueRef<float> ref =
        MakeAsyncValueRef(executor, []() -> float { return 42.0f; });

    EXPECT_FALSE(ref.IsAvailable());
    EXPECT_EQ(executor.Quiesce(), 1);

    EXPECT_TRUE(ref.IsAvailable());
    EXPECT_EQ(ref.get(), 42.0f);
  }

  {  // Make AsyncValueRef from a function that returns a StatusOr value.
    AsyncValueRef<float> ref = TryMakeAsyncValueRef<float>(
        executor, []() -> absl::StatusOr<float> { return 42.0f; });

    EXPECT_FALSE(ref.IsAvailable());
    EXPECT_EQ(executor.Quiesce(), 1);

    EXPECT_TRUE(ref.IsAvailable());
    EXPECT_EQ(ref.get(), 42.0f);
  }

  {  // Make AsyncValueRef from a function that returns a StatusOr value with
     // automatic type inference.
    AsyncValueRef<float> ref = TryMakeAsyncValueRef(
        executor, []() -> absl::StatusOr<float> { return 42.0f; });

    EXPECT_FALSE(ref.IsAvailable());
    EXPECT_EQ(executor.Quiesce(), 1);

    EXPECT_TRUE(ref.IsAvailable());
    EXPECT_EQ(ref.get(), 42.0f);
  }

  {  // Make AsyncValueRef from a function that returns a StatusOr error.
    AsyncValueRef<float> ref = TryMakeAsyncValueRef<float>(
        executor,
        []() -> absl::StatusOr<float> { return absl::InternalError("test"); });

    EXPECT_FALSE(ref.IsAvailable());
    EXPECT_EQ(executor.Quiesce(), 1);

    EXPECT_TRUE(ref.IsError());
    EXPECT_EQ(ref.GetError(), absl::InternalError("test"));
  }

  {  // Make AsyncValueRef from a function that returns a StatusOr error with
     // automatic type inference.
    AsyncValueRef<float> ref = TryMakeAsyncValueRef(
        executor,
        []() -> absl::StatusOr<float> { return absl::InternalError("test"); });

    EXPECT_FALSE(ref.IsAvailable());
    EXPECT_EQ(executor.Quiesce(), 1);

    EXPECT_TRUE(ref.IsError());
    EXPECT_EQ(ref.GetError(), absl::InternalError("test"));
  }
}

TEST(AsyncValueRefTest, MapAvailableOnExecutor) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);

  DeferredExecutor executor;
  AsyncValueRef<float> mapped_to_float =
      ref.Map(executor, [](int32_t value) -> float { return value; });

  EXPECT_FALSE(mapped_to_float.IsAvailable());
  EXPECT_EQ(executor.Quiesce(), 1);

  EXPECT_TRUE(mapped_to_float.IsAvailable());
  EXPECT_EQ(mapped_to_float.get(), 42.0f);
}

TEST(AsyncValueRefTest, MapErrorOnExecutor) {
  AsyncValueRef<int32_t> ref =
      MakeErrorAsyncValueRef(absl::InternalError("error"));

  DeferredExecutor executor;
  AsyncValueRef<float> mapped_to_float =
      ref.Map(executor, [](int32_t value) -> float { return value; });

  EXPECT_FALSE(mapped_to_float.IsAvailable());
  EXPECT_EQ(executor.Quiesce(), 1);

  EXPECT_TRUE(mapped_to_float.IsError());
  EXPECT_EQ(mapped_to_float.GetError(), absl::InternalError("error"));
}

TEST(AsyncValueRefTest, MapUnavailableOnExecutor) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  DeferredExecutor executor;
  AsyncValueRef<float> mapped_to_float =
      ref.Map(executor, [](int32_t value) -> float { return value; });

  ref.SetStateConcrete();
  ref.release()->DropRef();

  EXPECT_FALSE(mapped_to_float.IsAvailable());
  EXPECT_EQ(executor.Quiesce(), 1);

  EXPECT_TRUE(mapped_to_float.IsAvailable());
  EXPECT_EQ(mapped_to_float.get(), 42.0f);
}

TEST(AsyncValueRefTest, TryMapOnExecutor) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  DeferredExecutor executor;
  AsyncValueRef<float> mapped_to_float = ref.TryMap(
      executor, [](int32_t value) -> absl::StatusOr<float> { return value; });

  ref.SetStateConcrete();
  ref.release()->DropRef();

  EXPECT_FALSE(mapped_to_float.IsAvailable());
  EXPECT_EQ(executor.Quiesce(), 1);

  EXPECT_TRUE(mapped_to_float.IsAvailable());
  EXPECT_EQ(mapped_to_float.get(), 42.0f);
}

TEST(AsyncValueRefTest, TryMapErrorOnExecutor) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  DeferredExecutor executor;
  AsyncValueRef<float> mapped_to_float =
      ref.TryMap(executor, [](int32_t value) -> absl::StatusOr<float> {
        return absl::InternalError("error");
      });

  ref.SetStateConcrete();
  ref.release()->DropRef();

  EXPECT_FALSE(mapped_to_float.IsAvailable());
  EXPECT_EQ(executor.Quiesce(), 1);

  EXPECT_TRUE(mapped_to_float.IsError());
  EXPECT_EQ(mapped_to_float.GetError(), absl::InternalError("error"));
}

TEST(AsyncValueRefTest, FlatMapAvailableOnExecutor) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  DeferredExecutor executor;
  AsyncValueRef<float> fmapped_to_float =
      ref.FlatMap(executor, [](int32_t value) {
        return MakeAvailableAsyncValueRef<float>(static_cast<float>(value));
      });

  ref.SetStateConcrete();
  ref.release()->DropRef();

  EXPECT_FALSE(fmapped_to_float.IsAvailable());
  EXPECT_EQ(executor.Quiesce(), 1);

  EXPECT_TRUE(fmapped_to_float.IsAvailable());
  EXPECT_EQ(fmapped_to_float.get(), 42.0f);
}

TEST(AsyncValueRefTest, FlatMapDeferredAsyncValueOnExecutor) {
  DeferredExecutor executor0;
  DeferredExecutor executor1;

  // Use non-copyable std::unique_ptr<int32_t> to make sure that we don't
  // accidentally copy the value into the FlatMap functor.

  {  // Use a regular FlatMap.
    AsyncValueRef<float> fmapped_to_float =
        MakeAsyncValueRef<std::unique_ptr<int32_t>>(executor0, [] {
          return std::make_unique<int32_t>(42);
        }).FlatMap([&](AsyncValuePtr<std::unique_ptr<int32_t>> ptr) {
          return MakeAsyncValueRef<float>(
              executor1, [ref = ptr.CopyRef()] { return **ref; });
        });

    EXPECT_FALSE(fmapped_to_float.IsAvailable());
    EXPECT_EQ(executor0.Quiesce(), 1);

    EXPECT_FALSE(fmapped_to_float.IsAvailable());
    EXPECT_EQ(executor1.Quiesce(), 1);

    EXPECT_TRUE(fmapped_to_float.IsAvailable());
    EXPECT_EQ(fmapped_to_float.get(), 42.0f);
  }

  {  // Use a FlatMap that itself executed on given executor.
    AsyncValueRef<float> fmapped_to_float =
        MakeAsyncValueRef<std::unique_ptr<int32_t>>(executor0, [] {
          return std::make_unique<int32_t>(42);
        }).FlatMap(executor1, [&](AsyncValuePtr<std::unique_ptr<int32_t>> ptr) {
          return MakeAsyncValueRef<float>(
              executor1, [ref = ptr.CopyRef()] { return **ref; });
        });

    EXPECT_FALSE(fmapped_to_float.IsAvailable());
    EXPECT_EQ(executor0.Quiesce(), 1);

    EXPECT_FALSE(fmapped_to_float.IsAvailable());
    EXPECT_EQ(executor1.Quiesce(), 2);

    EXPECT_TRUE(fmapped_to_float.IsAvailable());
    EXPECT_EQ(fmapped_to_float.get(), 42.0f);
  }
}

TEST(AsyncValueRefTest, BlockUntilReady) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);
  BlockUntilReady(ref);
}

TEST(AsyncValueRefTest, RunWhenReady) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);
  bool executed = false;
  RunWhenReady(absl::MakeConstSpan({ref}), [&] { executed = true; });
  EXPECT_TRUE(executed);
}

namespace {
// We create a hierarchy of classes with different alignment requirements so
// that we can test that they all can be safely accessed via AsyncValueRef<T>
// references using different types. We also use this hierarchy to test
// LLVM-style casting APIs (Isa, DynCast, Cast).
struct A {
  alignas(16) int32_t a;
};
struct B : public A {
  alignas(32) int32_t b;
};
struct C : public B {
  alignas(64) int32_t c;
};
struct D : public B {
  alignas(64) int32_t d;
};
}  // namespace

TEST(AsyncValueRefTest, AlignedPayload) {
  AsyncValueRef<D> d_ref = MakeAvailableAsyncValueRef<D>();
  d_ref->a = 1;
  d_ref->b = 2;
  d_ref->d = 3;

  EXPECT_EQ(d_ref->a, 1);
  EXPECT_EQ(d_ref->b, 2);
  EXPECT_EQ(d_ref->d, 3);

  AsyncValueRef<B> b_ref = d_ref.CopyRef();
  EXPECT_EQ(b_ref->a, 1);
  EXPECT_EQ(b_ref->b, 2);

  AsyncValueRef<A> a_ref = d_ref.CopyRef();
  EXPECT_EQ(a_ref->a, 1);
}

TEST(AsyncValueRefTest, Isa) {
  // Empty async reference always returns false for any Isa<T>.
  AsyncValueRef<A> null_ref;
  EXPECT_FALSE(Isa<A>(null_ref));

  AsyncValueRef<A> a_ref = MakeAvailableAsyncValueRef<A>();
  AsyncValueRef<A> b_ref = MakeAvailableAsyncValueRef<B>();
  AsyncValueRef<A> c_ref = MakeAvailableAsyncValueRef<C>();
  AsyncValueRef<A> d_ref = MakeAvailableAsyncValueRef<D>();

  EXPECT_TRUE(Isa<A>(a_ref));
  EXPECT_TRUE(Isa<B>(b_ref));
  EXPECT_TRUE(Isa<C>(c_ref));
  EXPECT_TRUE(Isa<D>(d_ref));

  // Error async value is Isa<T> of any type in the hierarchy.
  AsyncValueRef<A> err = MakeErrorAsyncValueRef(absl::InternalError("error"));
  EXPECT_TRUE(Isa<A>(err));
  EXPECT_TRUE(Isa<B>(err));
  EXPECT_TRUE(Isa<C>(err));
  EXPECT_TRUE(Isa<D>(err));

  // If the value was constructed with a concrete type it should return true
  // for Isa<T> even if it was set to error later but only if types match.S
  AsyncValueRef<A> a_err = MakeConstructedAsyncValueRef<A>();
  AsyncValueRef<B> b_err = MakeConstructedAsyncValueRef<B>();
  a_err.SetError(absl::InternalError("error"));
  b_err.SetError(absl::InternalError("error"));

  EXPECT_TRUE(Isa<A>(a_err));
  EXPECT_TRUE(Isa<B>(b_err));

  // Indirect async value is Isa<T> only if it would be a no-op cast.
  auto indirect = MakeIndirectAsyncValue();
  AsyncValueRef<A> c_indirect(indirect);
  EXPECT_TRUE(Isa<A>(c_indirect));
  EXPECT_FALSE(Isa<C>(c_indirect));

  // After forwarding indirect async value to a concrete one it correctly
  // returns true from Isa<T> check.
  indirect->ForwardTo(c_ref.CopyRCRef());
  EXPECT_TRUE(Isa<A>(c_indirect));
  EXPECT_TRUE(Isa<C>(c_indirect));

  // Typed indirect async value correctly handled by Isa<T>.
  auto typed_indirect = MakeIndirectAsyncValue<C>();
  AsyncValueRef<A> c_typed_indirect(indirect);
  EXPECT_TRUE(Isa<A>(c_typed_indirect));
  EXPECT_TRUE(Isa<C>(c_typed_indirect));

  // Forwarding does not change anything for typed indirect async value.
  typed_indirect->ForwardTo(c_ref.CopyRCRef());
  EXPECT_TRUE(Isa<A>(c_typed_indirect));
  EXPECT_TRUE(Isa<C>(c_typed_indirect));

  // Typed indirect async value with error correctly handled by Isa<T>.
  auto typed_indirect_err = MakeIndirectAsyncValue<C>();
  AsyncValueRef<A> c_typed_indirect_err(typed_indirect_err);
  EXPECT_TRUE(Isa<A>(c_typed_indirect.AsPtr()));
  EXPECT_TRUE(Isa<C>(c_typed_indirect.AsPtr()));

  // After indirect async value is set to error it should still return true
  // from Isa<T> checks.
  typed_indirect_err->SetError(absl::InternalError("error"));
  EXPECT_TRUE(Isa<A>(c_typed_indirect_err.AsPtr()));
  EXPECT_TRUE(Isa<C>(c_typed_indirect_err.AsPtr()));
}

TEST(AsyncValueRefTest, DynCast) {
  AsyncValueRef<A> a_ref = MakeAvailableAsyncValueRef<A>();
  AsyncValueRef<A> b_ref = MakeAvailableAsyncValueRef<B>();
  AsyncValueRef<A> c_ref = MakeAvailableAsyncValueRef<C>();
  AsyncValueRef<A> d_ref = MakeAvailableAsyncValueRef<D>();

  EXPECT_TRUE(DynCast<A>(a_ref));
  EXPECT_TRUE(DynCast<B>(b_ref));
  EXPECT_TRUE(DynCast<C>(c_ref));
  EXPECT_TRUE(DynCast<D>(d_ref));

  // No-op casts are always successful.
  EXPECT_TRUE(DynCast<A>(c_ref));

  // We don't support casting to base (C inherits from B) because we can't do
  // that safely relying just on AsyncValue type id. For safe conversion to base
  // we need to introduce some kind of traits to the type hierarchy or rely on
  // builtin `dynamic_cast` (will work only for constructed values).
  EXPECT_FALSE(DynCast<B>(c_ref));

  // Types are unrelated, although they have same base.
  EXPECT_FALSE(DynCast<C>(d_ref));

  // Error async value can be DynCast to any type in the hierarchy.
  AsyncValueRef<A> err = MakeErrorAsyncValueRef(absl::InternalError("error"));
  EXPECT_TRUE(DynCast<A>(err));
  EXPECT_TRUE(DynCast<B>(err));
  EXPECT_TRUE(DynCast<C>(err));
  EXPECT_TRUE(DynCast<D>(err));

  // If the value was constructed with a concrete type it should DynCast
  // successfully even it it was set to error later but only if types match.
  AsyncValueRef<A> a_err = MakeConstructedAsyncValueRef<A>();
  AsyncValueRef<B> b_err = MakeConstructedAsyncValueRef<B>();
  a_err.SetError(absl::InternalError("error"));
  b_err.SetError(absl::InternalError("error"));

  EXPECT_TRUE(DynCast<A>(a_err));
  EXPECT_TRUE(DynCast<B>(b_err));
  EXPECT_FALSE(DynCast<C>(a_err));

  // Indirect async value can't be DynCast until it's forwarded unless it's a
  // no-op DynCast to the same type.
  auto indirect = MakeIndirectAsyncValue();
  AsyncValueRef<A> c_indirect(indirect);
  EXPECT_TRUE(DynCast<A>(c_indirect));
  EXPECT_FALSE(DynCast<C>(c_indirect));

  // After forwarding indirect async value to a concrete one it can be DynCast
  // to a concrete type.
  indirect->ForwardTo(c_ref.CopyRCRef());
  EXPECT_TRUE(DynCast<A>(c_indirect));
  EXPECT_TRUE(DynCast<C>(c_indirect));

  // Typed indirect async value correctly handled by DynCast<T>.
  auto typed_indirect = MakeIndirectAsyncValue<C>();
  AsyncValueRef<A> c_typed_indirect(indirect);
  EXPECT_TRUE(DynCast<A>(c_typed_indirect));
  EXPECT_TRUE(DynCast<C>(c_typed_indirect));

  // Forwarding does not change anything for typed indirect async value.
  typed_indirect->ForwardTo(c_ref.CopyRCRef());
  EXPECT_TRUE(DynCast<A>(c_typed_indirect));
  EXPECT_TRUE(DynCast<C>(c_typed_indirect));
}

TEST(AsyncValueRefTest, Cast) {
  AsyncValueRef<A> a_ref = MakeAvailableAsyncValueRef<A>();
  AsyncValueRef<A> b_ref = MakeAvailableAsyncValueRef<B>();
  AsyncValueRef<A> c_ref = MakeAvailableAsyncValueRef<C>();
  AsyncValueRef<A> d_ref = MakeAvailableAsyncValueRef<D>();

  EXPECT_TRUE(Cast<A>(a_ref));
  EXPECT_TRUE(Cast<B>(b_ref));
  EXPECT_TRUE(Cast<C>(c_ref));
  EXPECT_TRUE(Cast<D>(d_ref));

  EXPECT_TRUE(Cast<A>(c_ref));

  // Error async value can be Cast to any type in the hierarchy.
  AsyncValueRef<A> err = MakeErrorAsyncValueRef(absl::InternalError("error"));
  EXPECT_TRUE(Cast<A>(err));
  EXPECT_TRUE(Cast<B>(err));
  EXPECT_TRUE(Cast<C>(err));
  EXPECT_TRUE(Cast<D>(err));

  // If the value was constructed with a concrete type it should Cast
  // successfully even it it was set to error later but only if types match.
  AsyncValueRef<A> a_err = MakeConstructedAsyncValueRef<A>();
  AsyncValueRef<B> b_err = MakeConstructedAsyncValueRef<B>();
  a_err.SetError(absl::InternalError("error"));
  b_err.SetError(absl::InternalError("error"));

  EXPECT_TRUE(Cast<A>(a_err));
  EXPECT_TRUE(Cast<B>(b_err));

  // Indirect async value can't be Cast until it's forwarded unless it's a
  // no-op Cast to the same type.
  auto indirect = MakeIndirectAsyncValue();
  AsyncValueRef<A> c_indirect(indirect);
  EXPECT_TRUE(Cast<A>(c_indirect));

  // After forwarding indirect async value to a concrete one it can be Cast
  // to a concrete type.
  indirect->ForwardTo(c_ref.CopyRCRef());
  EXPECT_TRUE(Cast<A>(c_indirect));
  EXPECT_TRUE(Cast<C>(c_indirect));

  // Typed indirect async value correctly handled by Cast<T>.
  auto typed_indirect = MakeIndirectAsyncValue<C>();
  AsyncValueRef<A> c_typed_indirect(indirect);
  EXPECT_TRUE(Cast<A>(c_typed_indirect));
  EXPECT_TRUE(Cast<C>(c_typed_indirect));

  // Forwarding does not change anything for typed indirect async value.
  typed_indirect->ForwardTo(c_ref.CopyRCRef());
  EXPECT_TRUE(Cast<A>(c_typed_indirect));
  EXPECT_TRUE(Cast<C>(c_typed_indirect));
}

TEST(AsyncValueRefTest, RecursiveOwnership) {
  // This is a test for recursive ownership of AsyncValue:
  //   (1) AsyncValueRef owned by a State object
  //   (2) State object owned by AsyncValue::AndThen callback.
  //
  // We check that setting async value state concrete and then running all
  // AndThen callbacks doesn't cause an asan error.
  struct State {
    explicit State(AsyncValueRef<int32_t> value) : value(std::move(value)) {}
    AsyncValueRef<int32_t> value;
  };

  AsyncValueRef<int32_t> value = MakeConstructedAsyncValueRef<int32_t>(42);
  auto state = std::make_unique<State>(std::move(value));

  State* state_ptr = state.get();
  int64_t counter = 0;

  // Enqueue callbacks.
  state_ptr->value.AndThen([&, value = 1] { counter += value; });
  state_ptr->value.AndThen([&, value = 2] { counter += value; });
  state_ptr->value.AndThen([&, value = 3] { counter += value; });

  // Move state ownership to the callback.
  state_ptr->value.AndThen([state = std::move(state)] {});

  // Run all callbacks and as a side effect destroy the `state` object.
  state_ptr->value.SetStateConcrete();
  EXPECT_EQ(counter, 1 + 2 + 3);
}

TEST(AsyncValueRefTest, CountDownSuccess) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);

  CountDownAsyncValueRef<int32_t> count_down_ref(ref, 2);
  CountDownAsyncValueRef<int32_t> count_down_ref_copy = count_down_ref;

  EXPECT_FALSE(ref.IsAvailable());

  EXPECT_FALSE(count_down_ref.CountDown());
  EXPECT_FALSE(ref.IsAvailable());

  EXPECT_TRUE(count_down_ref_copy.CountDown());
  EXPECT_TRUE(ref.IsAvailable());
  EXPECT_EQ(*ref, 42);
}

TEST(AsyncValueRefTest, CountDownError) {
  CountDownAsyncValueRef<int32_t> count_down_ref(2);
  AsyncValueRef<int32_t> ref = count_down_ref.AsRef();

  CountDownAsyncValueRef<int32_t> count_down_ref_copy = count_down_ref;

  EXPECT_FALSE(ref.IsAvailable());

  EXPECT_FALSE(count_down_ref.CountDown(absl::InternalError("error")));
  EXPECT_FALSE(ref.IsAvailable());

  EXPECT_TRUE(count_down_ref_copy.CountDown());
  EXPECT_TRUE(ref.IsError());
  EXPECT_EQ(ref.GetError(), absl::InternalError("error"));
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

template <size_t size>
static void BM_MakeConstructed(benchmark::State& state) {
  for (auto _ : state) {
    auto ref = MakeConstructedAsyncValueRef<std::array<char, size>>();
    benchmark::DoNotOptimize(ref);
  }
}

BENCHMARK(BM_MakeConstructed<1>);
BENCHMARK(BM_MakeConstructed<4>);
BENCHMARK(BM_MakeConstructed<8>);
BENCHMARK(BM_MakeConstructed<16>);
BENCHMARK(BM_MakeConstructed<32>);
BENCHMARK(BM_MakeConstructed<64>);
BENCHMARK(BM_MakeConstructed<128>);
BENCHMARK(BM_MakeConstructed<256>);

static void BM_CountDownSuccess(benchmark::State& state) {
  size_t n = state.range(0);

  for (auto _ : state) {
    auto ref = MakeConstructedAsyncValueRef<int32_t>(42);
    CountDownAsyncValueRef<int32_t> count_down_ref(ref, n);
    for (size_t i = 0; i < n; ++i) {
      count_down_ref.CountDown();
    }
  }
}

BENCHMARK(BM_CountDownSuccess)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

static void BM_CountDownError(benchmark::State& state) {
  size_t n = state.range(0);

  absl::Status error = absl::InternalError("error");

  for (auto _ : state) {
    auto ref = MakeConstructedAsyncValueRef<int32_t>(42);
    CountDownAsyncValueRef<int32_t> count_down_ref(ref, n);
    for (size_t i = 0; i < n; ++i) {
      count_down_ref.CountDown(error);
    }
  }
}

BENCHMARK(BM_CountDownError)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

}  // namespace tsl
