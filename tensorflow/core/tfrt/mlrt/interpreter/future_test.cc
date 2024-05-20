/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/mlrt/interpreter/future.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/status_matchers.h"

namespace mlrt {
namespace {

TEST(FutureTest, Basic) {
  {
    auto promise = Promise::Allocate<int>();
    auto future = promise.GetFuture();
    std::move(promise).Set<int>(1);
    EXPECT_FALSE(promise);  // NOLINT(bugprone-use-after-move)
    ASSERT_TRUE(future);
    ASSERT_TRUE(future.IsReady());
    EXPECT_EQ(future.Get<int>(), 1);
  }

  {
    auto promise = Promise::Allocate<int>();
    auto future = promise.GetFuture();
    int u = 0;
    ASSERT_TRUE(future);
    std::move(future).Then(
        [&](absl::StatusOr<int> result) { u = result.value(); });
    EXPECT_FALSE(future);  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(u, 0);
    std::move(promise).Set<int>(1);
    EXPECT_EQ(u, 1);
  }

  {
    auto promise = Promise::Allocate<int>();
    auto future = promise.GetFuture();
    int v = 0;
    ASSERT_TRUE(future);
    std::move(future).Then([&](int result) { v = result; });
    EXPECT_FALSE(future);  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(v, 0);
    std::move(promise).Set<int>(1);
    EXPECT_EQ(v, 1);
  }

  {
    auto promise = Promise::Allocate<int>();
    auto future = promise.GetFuture();
    int w = 0;
    ASSERT_TRUE(future);
    std::move(future).Then([&]() { w = 2; });
    EXPECT_FALSE(future);  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(w, 0);
    std::move(promise).Set<int>(1);
    EXPECT_EQ(w, 2);
  }

  {
    auto promise = Promise::Allocate<int>();
    auto future = promise.GetFuture();
    absl::Status s = absl::InternalError("error");
    ASSERT_TRUE(future);
    std::move(future).Then([&](absl::Status status) { s = status; });
    EXPECT_FALSE(future);  // NOLINT(bugprone-use-after-move)
    EXPECT_FALSE(s.ok());
    std::move(promise).Set<int>(1);
    EXPECT_TRUE(s.ok());
  }
}

TEST(FutureTest, CopyAndMove) {
  auto promise = Promise::Allocate<int>();
  auto future = promise.GetFuture();

  EXPECT_EQ(future.UseCount(), 2);

  {
    auto copy = future;
    EXPECT_EQ(copy.UseCount(), 3);
  }

  auto move = std::move(future);
  EXPECT_EQ(move.UseCount(), 2);

  std::move(promise).Set<int>(1);

  EXPECT_EQ(move.UseCount(), 1);
}

TEST(FutureTest, CreateFromAsyncValue) {
  auto promise = tsl::MakeUnconstructedAsyncValueRef<int>();
  mlrt::Future future(promise);

  int v = 0;
  std::move(future).Then([&](int result) { v = result; });
  EXPECT_EQ(v, 0);

  promise.emplace(1);
  EXPECT_EQ(v, 1);
}

TEST(FutureTest, Error) {
  {
    auto promise = Promise::Allocate<int>();
    auto future = promise.GetFuture();

    std::move(promise).SetError(absl::InternalError("test error"));

    ASSERT_TRUE(future.IsError());
    EXPECT_THAT(
        future.GetError(),
        ::tsl::testing::StatusIs(absl::StatusCode::kInternal, "test error"));
  }

  {
    auto promise = Promise::Allocate<int>();
    auto future = promise.GetFuture();
    absl::StatusOr<int> r;
    std::move(future).Then(
        [&](absl::StatusOr<int> result) { r = std::move(result); });

    std::move(promise).SetError(absl::InternalError("test error"));

    EXPECT_THAT(
        r, ::tsl::testing::StatusIs(absl::StatusCode::kInternal, "test error"));
  }

  {
    auto promise = Promise::Allocate<int>();
    auto future = promise.GetFuture();
    absl::Status s;
    std::move(future).Then([&](absl::Status status) { s = std::move(status); });
    std::move(promise).SetError(absl::InternalError("test error"));

    EXPECT_THAT(
        s, ::tsl::testing::StatusIs(absl::StatusCode::kInternal, "test error"));
  }
}

}  // namespace
}  // namespace mlrt
