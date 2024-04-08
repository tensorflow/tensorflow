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

#include "tsl/concurrency/async_value_ref.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/platform/test.h"

namespace tsl {

class WrappedInt32 {
 public:
  explicit WrappedInt32(int32_t value) : value_(value) {}
  int32_t value() const { return value_; }

 private:
  int32_t value_;
};

constexpr int32_t kTestValue = 42;

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

TEST(AsyncValueRefTest, AndThenError) {
  auto value = MakeConstructedAsyncValueRef<int32_t>(kTestValue);

  auto diag = absl::InternalError("test error");
  value.AndThen([&](absl::Status status) { EXPECT_EQ(status, diag); });

  value.SetError(diag);
}

TEST(AsyncValueRefTest, AndThenNoError) {
  auto value = MakeConstructedAsyncValueRef<int32_t>(kTestValue);

  value.AndThen([](absl::Status status) { EXPECT_TRUE(status.ok()); });

  value.SetStateConcrete();
}

TEST(AsyncValueRefTest, AndThenStatusOrError) {
  auto value = MakeConstructedAsyncValueRef<int32_t>(kTestValue);

  auto diag = absl::InternalError("test error");
  value.AndThen([&](absl::StatusOr<int32_t*> v) {
    EXPECT_FALSE(v.ok());
    EXPECT_EQ(v.status(), diag);
  });

  value.SetError(diag);
}

TEST(AsyncValueRefTest, PtrAndThenStatusOrError) {
  auto value = MakeConstructedAsyncValueRef<int32_t>(kTestValue);

  auto diag = absl::InternalError("test error");
  value.AsPtr().AndThen([&](absl::StatusOr<int32_t*> v) {
    EXPECT_FALSE(v.ok());
    EXPECT_EQ(v.status(), diag);
  });

  value.SetError(diag);
}

TEST(AsyncValueRefTest, AndThenStatusOrNoError) {
  auto value = MakeConstructedAsyncValueRef<int32_t>(kTestValue);

  value.AndThen([](absl::StatusOr<int32_t*> v) {
    EXPECT_TRUE(v.ok());
    EXPECT_EQ(**v, kTestValue);
  });

  value.SetStateConcrete();
}

TEST(AsyncValueRefTest, PtrAndThenStatusOrNoError) {
  auto value = MakeConstructedAsyncValueRef<int32_t>(kTestValue);

  value.AsPtr().AndThen([](absl::StatusOr<int32_t*> v) {
    EXPECT_TRUE(v.ok());
    EXPECT_EQ(**v, kTestValue);
  });

  value.SetStateConcrete();
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

namespace {
struct A {
  virtual ~A() = default;
};
struct B : public A {};
struct C : public B {};
struct D : public A {};
}  // namespace

TEST(AsyncValueRefTest, Isa) {
  // Empty async reference always returns false for any Isa<T>().
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
}

}  // namespace tsl
