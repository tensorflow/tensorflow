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

namespace {
struct A {
  virtual ~A() = default;
};
struct B : public A {};
struct C : public B {};
struct D : public A {};
}  // namespace

TEST(AsyncValuePtrTest, Isa) {
  // Empty async pointer always returns false for any Isa<T>().
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
}

}  // namespace tsl
