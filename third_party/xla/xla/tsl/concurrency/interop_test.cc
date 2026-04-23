/* Copyright 2026 Google LLC. All Rights Reserved.

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

#include "xla/tsl/concurrency/interop.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/test.h"

namespace tsl {

// Use this helper function to test that `MakeFutureWhenReady` correctly extends
// the lifetime of captured async values.
template <typename T>
static void SetStateConcrete(AsyncValueRef<T> ref) {
  ref.SetStateConcrete();
}

TEST(InteropTest, MakeFutureWhenAsyncValuesReady) {
  auto v0 = MakeConstructedAsyncValueRef<int32_t>(1);
  auto v1 = MakeConstructedAsyncValueRef<int32_t>(2);

  auto future = MakeFutureWhenReady({v0.GetAsyncValue(), v1.GetAsyncValue()});
  ASSERT_FALSE(future.IsReady());

  SetStateConcrete(std::move(v0));
  SetStateConcrete(std::move(v1));

  ASSERT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::OkStatus());
}

TEST(InteropTest, MakeFutureWhenRCReferencesReady) {
  auto v0 = MakeConstructedAsyncValueRef<int32_t>(1);
  auto v1 = MakeConstructedAsyncValueRef<int32_t>(2);

  std::vector<RCReference<AsyncValue>> values;
  values.push_back(v0.CopyRCRef());
  values.push_back(v1.CopyRCRef());

  auto future = MakeFutureWhenReady(values);
  ASSERT_FALSE(future.IsReady());

  SetStateConcrete(std::move(v0));
  SetStateConcrete(std::move(v1));

  ASSERT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::OkStatus());
}

TEST(InteropTest, MakeFutureWhenReadyWithError) {
  auto v0 = MakeConstructedAsyncValueRef<int32_t>(1);
  auto v1 = MakeConstructedAsyncValueRef<int32_t>(2);

  auto future = MakeFutureWhenReady({v0.GetAsyncValue(), v1.GetAsyncValue()});
  ASSERT_FALSE(future.IsReady());

  SetStateConcrete(std::move(v0));
  v1.SetError(absl::InternalError("test error"));

  ASSERT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::InternalError("test error"));
}

TEST(InteropTest, MakeFutureWhenAvailable) {
  auto v0 = MakeAvailableAsyncValueRef<int32_t>(1);
  auto v1 = MakeAvailableAsyncValueRef<int32_t>(2);

  auto future = MakeFutureWhenReady({v0.GetAsyncValue(), v1.GetAsyncValue()});
  ASSERT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::OkStatus());
}

TEST(InteropTest, MakeFutureWhenAsyncValueReady) {
  auto v0 = MakeConstructedAsyncValueRef<int32_t>(1);

  auto future = MakeFutureWhenReady(v0.GetAsyncValue());
  ASSERT_FALSE(future.IsReady());

  SetStateConcrete(std::move(v0));

  ASSERT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::OkStatus());
}

TEST(InteropTest, MakeFutureWhenAsyncValueError) {
  auto v0 = MakeConstructedAsyncValueRef<int32_t>(1);

  auto future = MakeFutureWhenReady(v0.GetAsyncValue());
  ASSERT_FALSE(future.IsReady());

  v0.SetError(absl::InternalError("single error"));

  ASSERT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::InternalError("single error"));
}

TEST(InteropTest, MakeFutureWhenRCReferenceReady) {
  auto v0 = MakeConstructedAsyncValueRef<int32_t>(1);

  auto future = MakeFutureWhenReady(v0.CopyRCRef());
  ASSERT_FALSE(future.IsReady());

  SetStateConcrete(std::move(v0));

  ASSERT_TRUE(future.IsReady());
  EXPECT_EQ(future.Await(), absl::OkStatus());
}

}  // namespace tsl
