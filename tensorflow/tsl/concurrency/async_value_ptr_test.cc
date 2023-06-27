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

#include "tensorflow/tsl/concurrency/async_value_ref.h"
#include "tensorflow/tsl/platform/test.h"

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

}  // namespace tsl
