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
#include "absl/types/span.h"
#include "tsl/concurrency/async_value.h"
#include "tsl/concurrency/ref_count.h"
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
}

}  // namespace tsl
