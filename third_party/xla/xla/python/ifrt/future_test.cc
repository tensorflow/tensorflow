/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/future.h"

#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla::ifrt {
namespace {

using ::testing::HasSubstr;

TEST(FutureTest, JoinZeroFuture) {
  tsl::Future<> future = JoinFutures({});

  TF_EXPECT_OK(future.Await());
}

TEST(FutureTest, JoinOneOkFuture) {
  auto [promise, future] = tsl::Future<>::MakePromise();
  std::vector<tsl::Future<>> futures = {std::move(future)};

  tsl::Future<> joined = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(joined.IsReady());
  promise.Set(absl::OkStatus());
  TF_EXPECT_OK(joined.Await());
}

TEST(FutureTest, JoinOneFailingFuture) {
  auto [promise, future] = tsl::Future<>::MakePromise();
  std::vector<tsl::Future<>> futures = {std::move(future)};

  tsl::Future<> joined = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(joined.IsReady());
  promise.Set(absl::InvalidArgumentError("Some error"));
  EXPECT_THAT(joined.Await(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Some error")));
}

TEST(FutureTest, JoinAllOkFutures) {
  constexpr int kNumFutures = 3;
  std::vector<tsl::Promise<>> promises;
  std::vector<tsl::Future<>> futures;
  promises.reserve(kNumFutures);
  futures.reserve(kNumFutures);
  for (int i = 0; i < kNumFutures; ++i) {
    std::tie(promises.emplace_back(), futures.emplace_back()) =
        tsl::Future<>::MakePromise();
  }

  tsl::Future<> joined = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(joined.IsReady());
  for (tsl::Promise<>& promise : promises) {
    promise.Set(absl::OkStatus());
  }
  TF_EXPECT_OK(joined.Await());
}

TEST(FutureTest, JoinAllFailingFutures) {
  constexpr int kNumFutures = 3;
  std::vector<tsl::Promise<>> promises;
  std::vector<tsl::Future<>> futures;
  promises.reserve(kNumFutures);
  futures.reserve(kNumFutures);
  for (int i = 0; i < kNumFutures; ++i) {
    std::tie(promises.emplace_back(), futures.emplace_back()) =
        tsl::Future<>::MakePromise();
  }

  tsl::Future<> joined = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(joined.IsReady());
  for (tsl::Promise<>& promise : promises) {
    promise.Set(absl::InvalidArgumentError("Some error"));
  }
  EXPECT_THAT(joined.Await(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Some error")));
}

class JoinAllOkFuturesExceptForOneTest : public testing::TestWithParam<int> {};

TEST_P(JoinAllOkFuturesExceptForOneTest, JoinAllOkFuturesExceptForOne) {
  const int kNumFutures = 3;
  const int failing_future_idx = GetParam();
  std::vector<tsl::Promise<>> promises;
  std::vector<tsl::Future<>> futures;
  promises.reserve(kNumFutures);
  futures.reserve(kNumFutures);
  for (int i = 0; i < kNumFutures; ++i) {
    std::tie(promises.emplace_back(), futures.emplace_back()) =
        tsl::Future<>::MakePromise();
  }

  tsl::Future<> joined = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(joined.IsReady());
  for (int i = 0; i < kNumFutures; ++i) {
    if (i == failing_future_idx) {
      promises[i].Set(absl::InvalidArgumentError("Some error"));
    } else {
      promises[i].Set(absl::OkStatus());
    }
  }
  EXPECT_THAT(joined.Await(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Some error")));
}

INSTANTIATE_TEST_SUITE_P(FutureTest, JoinAllOkFuturesExceptForOneTest,
                         testing::Range(0, 3));

}  // namespace
}  // namespace xla::ifrt
