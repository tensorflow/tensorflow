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

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tsl/concurrency/async_value_ref.h"
#include "tsl/platform/test.h"

namespace tsl {

TEST(AsyncValuePtrTest, Construct) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  EXPECT_EQ(ptr.get(), 42);
}

TEST(AsyncValuePtrTest, CopyRef) {
  AsyncValueRef<int32_t> ref0 = MakeAvailableAsyncValueRef<int32_t>(42);
  AsyncValuePtr<int32_t> ptr = ref0.AsPtr();

  EXPECT_TRUE(ref0.IsUnique());  // pointer doesn't change the reference count

  AsyncValueRef<int32_t> ref1 = ptr.CopyRef();

  EXPECT_FALSE(ref0.IsUnique());
  EXPECT_FALSE(ref1.IsUnique());
}

TEST(AsyncValuePtrTest, Emplace) {
  AsyncValueRef<int32_t> ref = MakeUnconstructedAsyncValueRef<int32_t>();
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  EXPECT_FALSE(ptr.IsConcrete());
  EXPECT_FALSE(ptr.IsAvailable());

  ptr.emplace(42);
  EXPECT_EQ(ptr.get(), 42);
}

TEST(AsyncValuePtrTest, SetError) {
  AsyncValueRef<int32_t> ref = MakeUnconstructedAsyncValueRef<int32_t>();
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  EXPECT_FALSE(ptr.IsConcrete());
  EXPECT_FALSE(ptr.IsAvailable());

  ptr.SetError(absl::InternalError("test error"));

  EXPECT_TRUE(ptr.IsAvailable());
  EXPECT_TRUE(ptr.IsError());
}

TEST(AsyncValuePtrTest, AndThen) {
  AsyncValueRef<int32_t> ref = MakeUnconstructedAsyncValueRef<int32_t>();
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  EXPECT_FALSE(ptr.IsConcrete());
  EXPECT_FALSE(ptr.IsAvailable());

  bool executed = false;
  ptr.AndThen([&]() { executed = true; });

  ptr.emplace(42);
  EXPECT_TRUE(executed);
}

TEST(AsyncValuePtrTest, AndThenError) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  auto error = absl::InternalError("test error");
  ptr.SetError(error);
  ptr.AndThen([&](absl::Status status) { EXPECT_EQ(status, error); });
}

TEST(AsyncValuePtrTest, AndThenNoError) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  ptr.AndThen([](absl::Status status) { EXPECT_TRUE(status.ok()); });
}

TEST(AsyncValuePtrTest, AndThenStatusOrError) {
  AsyncValueRef<int32_t> ref = MakeConstructedAsyncValueRef<int32_t>(42);
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  auto error = absl::InternalError("test error");
  ptr.SetError(error);

  ptr.AndThen([&](absl::StatusOr<int32_t*> v) {
    EXPECT_FALSE(v.ok());
    EXPECT_EQ(v.status(), error);
  });
}

TEST(AsyncValuePtrTest, AndThenStatusOrNoError) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();

  ptr.AndThen([&](absl::StatusOr<int32_t*> v) { EXPECT_EQ(**v, 42); });
}

TEST(AsyncValuePtrTest, BlockUntilReady) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();
  BlockUntilReady(ptr);
}

TEST(AsyncValuePtrTest, RunWhenReady) {
  AsyncValueRef<int32_t> ref = MakeAvailableAsyncValueRef<int32_t>(42);
  AsyncValuePtr<int32_t> ptr = ref.AsPtr();
  bool executed = false;
  RunWhenReady(absl::MakeConstSpan({ptr}), [&] { executed = true; });
  EXPECT_TRUE(executed);
}

namespace {
struct A {
  virtual ~A() = default;
};
struct B : public A {};
struct C : public B {};
struct D : public A {};
}  // namespace

TEST(AsyncValuePtrTest, Isa) {
  // Empty async pointer always returns false for any Isa<T>.
  AsyncValuePtr<A> null_ptr;
  EXPECT_FALSE(Isa<A>(null_ptr));

  AsyncValueRef<A> a_ref = MakeAvailableAsyncValueRef<A>();
  AsyncValueRef<A> b_ref = MakeAvailableAsyncValueRef<B>();
  AsyncValueRef<A> c_ref = MakeAvailableAsyncValueRef<C>();
  AsyncValueRef<A> d_ref = MakeAvailableAsyncValueRef<D>();

  EXPECT_TRUE(Isa<A>(a_ref.AsPtr()));
  EXPECT_TRUE(Isa<B>(b_ref.AsPtr()));
  EXPECT_TRUE(Isa<C>(c_ref.AsPtr()));
  EXPECT_TRUE(Isa<D>(d_ref.AsPtr()));

  // Error async value is Isa<T> of any type in the hierarchy.
  AsyncValueRef<A> err = MakeErrorAsyncValueRef(absl::InternalError("error"));
  EXPECT_TRUE(Isa<A>(err.AsPtr()));
  EXPECT_TRUE(Isa<B>(err.AsPtr()));
  EXPECT_TRUE(Isa<C>(err.AsPtr()));
  EXPECT_TRUE(Isa<D>(err.AsPtr()));

  // If the value was constructed with a concrete type it should return true
  // for Isa<T> even if it was set to error later but only if types match.
  AsyncValueRef<A> a_err = MakeConstructedAsyncValueRef<A>();
  AsyncValueRef<B> b_err = MakeConstructedAsyncValueRef<B>();
  a_err.SetError(absl::InternalError("error"));
  b_err.SetError(absl::InternalError("error"));

  EXPECT_TRUE(Isa<A>(a_err.AsPtr()));
  EXPECT_TRUE(Isa<B>(b_err.AsPtr()));

  // Indirect async value is Isa<T> only if it would be a no-op cast.
  auto indirect = MakeIndirectAsyncValue();
  AsyncValueRef<A> c_indirect(indirect);
  EXPECT_TRUE(Isa<A>(c_indirect.AsPtr()));
  EXPECT_FALSE(Isa<C>(c_indirect.AsPtr()));

  // After forwarding indirect async value to a concrete one it correctly
  // returns true from Isa<T> check.
  indirect->ForwardTo(c_ref.CopyRCRef());
  EXPECT_TRUE(Isa<A>(c_indirect.AsPtr()));
  EXPECT_TRUE(Isa<C>(c_indirect.AsPtr()));
}

TEST(AsyncValuePtrTest, DynCast) {
  AsyncValueRef<A> a_ref = MakeAvailableAsyncValueRef<A>();
  AsyncValueRef<A> b_ref = MakeAvailableAsyncValueRef<B>();
  AsyncValueRef<A> c_ref = MakeAvailableAsyncValueRef<C>();
  AsyncValueRef<A> d_ref = MakeAvailableAsyncValueRef<D>();

  EXPECT_TRUE(DynCast<A>(a_ref.AsPtr()));
  EXPECT_TRUE(DynCast<B>(b_ref.AsPtr()));
  EXPECT_TRUE(DynCast<C>(c_ref.AsPtr()));
  EXPECT_TRUE(DynCast<D>(d_ref.AsPtr()));

  // No-op casts are always successful.
  EXPECT_TRUE(DynCast<A>(c_ref.AsPtr()));

  // We don't support casting to base (C inherits from B) because we can't do
  // that safely relying just on AsyncValue type id. For safe conversion to base
  // we need to introduce some kind of traits to the type hierarchy or rely on
  // builtin `dynamic_cast` (will work only for constructed values).
  EXPECT_FALSE(DynCast<B>(c_ref.AsPtr()));

  // Types are unrelated, although they have same base.
  EXPECT_FALSE(DynCast<C>(d_ref.AsPtr()));

  // Error async value can be DynCast to any type in the hierarchy.
  AsyncValueRef<A> err = MakeErrorAsyncValueRef(absl::InternalError("error"));
  EXPECT_TRUE(DynCast<A>(err.AsPtr()));
  EXPECT_TRUE(DynCast<B>(err.AsPtr()));
  EXPECT_TRUE(DynCast<C>(err.AsPtr()));
  EXPECT_TRUE(DynCast<D>(err.AsPtr()));

  // If the value was constructed with a concrete type it should DynCast
  // successfully even it it was set to error later but only if types match.
  AsyncValueRef<A> a_err = MakeConstructedAsyncValueRef<A>();
  AsyncValueRef<B> b_err = MakeConstructedAsyncValueRef<B>();
  a_err.SetError(absl::InternalError("error"));
  b_err.SetError(absl::InternalError("error"));

  EXPECT_TRUE(DynCast<A>(a_err.AsPtr()));
  EXPECT_TRUE(DynCast<B>(b_err.AsPtr()));
  EXPECT_FALSE(DynCast<C>(a_err.AsPtr()));

  // Indirect async value can't be DynCast until it's forwarded unless it's a
  // no-op DynCast to the same type.
  auto indirect = MakeIndirectAsyncValue();
  AsyncValueRef<A> c_indirect(indirect);
  EXPECT_TRUE(DynCast<A>(c_indirect.AsPtr()));
  EXPECT_FALSE(DynCast<C>(c_indirect.AsPtr()));

  // After forwarding indirect async value to a concrete one it can be DynCast
  // to a concrete type.
  indirect->ForwardTo(c_ref.CopyRCRef());
  EXPECT_TRUE(DynCast<A>(c_indirect.AsPtr()));
  EXPECT_TRUE(DynCast<C>(c_indirect.AsPtr()));
}

TEST(AsyncValuePtrTest, Cast) {
  AsyncValueRef<A> a_ref = MakeAvailableAsyncValueRef<A>();
  AsyncValueRef<A> b_ref = MakeAvailableAsyncValueRef<B>();
  AsyncValueRef<A> c_ref = MakeAvailableAsyncValueRef<C>();
  AsyncValueRef<A> d_ref = MakeAvailableAsyncValueRef<D>();

  EXPECT_TRUE(Cast<A>(a_ref.AsPtr()));
  EXPECT_TRUE(Cast<B>(b_ref.AsPtr()));
  EXPECT_TRUE(Cast<C>(c_ref.AsPtr()));
  EXPECT_TRUE(Cast<D>(d_ref.AsPtr()));

  EXPECT_TRUE(Cast<A>(c_ref.AsPtr()));

  // Error async value can be Cast to any type in the hierarchy.
  AsyncValueRef<A> err = MakeErrorAsyncValueRef(absl::InternalError("error"));
  EXPECT_TRUE(Cast<A>(err.AsPtr()));
  EXPECT_TRUE(Cast<B>(err.AsPtr()));
  EXPECT_TRUE(Cast<C>(err.AsPtr()));
  EXPECT_TRUE(Cast<D>(err.AsPtr()));

  // If the value was constructed with a concrete type it should Cast
  // successfully even it it was set to error later but only if types match.
  AsyncValueRef<A> a_err = MakeConstructedAsyncValueRef<A>();
  AsyncValueRef<B> b_err = MakeConstructedAsyncValueRef<B>();
  a_err.SetError(absl::InternalError("error"));
  b_err.SetError(absl::InternalError("error"));

  EXPECT_TRUE(Cast<A>(a_err.AsPtr()));
  EXPECT_TRUE(Cast<B>(b_err.AsPtr()));

  // Indirect async value can't be Cast until it's forwarded unless it's a
  // no-op Cast to the same type.
  auto indirect = MakeIndirectAsyncValue();
  AsyncValueRef<A> c_indirect(indirect);
  EXPECT_TRUE(Cast<A>(c_indirect.AsPtr()));

  // After forwarding indirect async value to a concrete one it can be Cast
  // to a concrete type.
  indirect->ForwardTo(c_ref.CopyRCRef());
  EXPECT_TRUE(Cast<A>(c_indirect.AsPtr()));
  EXPECT_TRUE(Cast<C>(c_indirect.AsPtr()));
}

}  // namespace tsl
