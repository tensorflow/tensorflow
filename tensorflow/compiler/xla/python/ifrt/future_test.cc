/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/future.h"

#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

TEST(FutureTest, JoinZeroFuture) {
  Future<Status> future = JoinFutures({});

  TF_EXPECT_OK(future.Await());
}

TEST(FutureTest, JoinOneOkFuture) {
  Promise<Status> promise = Future<Status>::CreatePromise();
  std::vector<Future<Status>> futures;
  futures.push_back(Future<Status>(promise));

  Future<Status> future = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(future.IsReady());
  promise.Set(OkStatus());
  TF_EXPECT_OK(future.Await());
}

TEST(FutureTest, JoinOneFailingFuture) {
  Promise<Status> promise = Future<Status>::CreatePromise();
  std::vector<Future<Status>> futures;
  futures.push_back(Future<Status>(promise));

  Future<Status> future = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(future.IsReady());
  promise.Set(InvalidArgument("Some error"));
  EXPECT_THAT(future.Await(), StatusIs(tensorflow::error::INVALID_ARGUMENT,
                                       HasSubstr("Some error")));
}

TEST(FutureTest, JoinAllOkFutures) {
  constexpr int kNumFutures = 3;
  std::vector<Promise<Status>> promises;
  std::vector<Future<Status>> futures;
  promises.reserve(kNumFutures);
  futures.reserve(kNumFutures);
  for (int i = 0; i < kNumFutures; ++i) {
    promises.push_back(Future<Status>::CreatePromise());
    futures.push_back(Future<Status>(promises.back()));
  }

  Future<Status> future = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(future.IsReady());
  for (Promise<Status>& promise : promises) {
    promise.Set(OkStatus());
  }
  TF_EXPECT_OK(future.Await());
}

TEST(FutureTest, JoinAllFailingFutures) {
  constexpr int kNumFutures = 3;
  std::vector<Promise<Status>> promises;
  std::vector<Future<Status>> futures;
  promises.reserve(kNumFutures);
  futures.reserve(kNumFutures);
  for (int i = 0; i < kNumFutures; ++i) {
    promises.push_back(Future<Status>::CreatePromise());
    futures.push_back(Future<Status>(promises.back()));
  }

  Future<Status> future = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(future.IsReady());
  for (Promise<Status>& promise : promises) {
    promise.Set(InvalidArgument("Some error"));
  }
  EXPECT_THAT(future.Await(), StatusIs(tensorflow::error::INVALID_ARGUMENT,
                                       HasSubstr("Some error")));
}

class JoinAllOkFuturesExceptForOneTest : public testing::TestWithParam<int> {};

TEST_P(JoinAllOkFuturesExceptForOneTest, JoinAllOkFuturesExceptForOne) {
  const int kNumFutures = 3;
  const int failing_future_idx = GetParam();
  std::vector<Promise<Status>> promises;
  std::vector<Future<Status>> futures;
  promises.reserve(kNumFutures);
  futures.reserve(kNumFutures);
  for (int i = 0; i < kNumFutures; ++i) {
    promises.push_back(Future<Status>::CreatePromise());
    futures.push_back(Future<Status>(promises.back()));
  }

  Future<Status> future = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(future.IsReady());
  for (int i = 0; i < kNumFutures; ++i) {
    if (i == failing_future_idx) {
      promises[i].Set(InvalidArgument("Some error"));
    } else {
      promises[i].Set(OkStatus());
    }
  }
  EXPECT_THAT(future.Await(), StatusIs(tensorflow::error::INVALID_ARGUMENT,
                                       HasSubstr("Some error")));
}

INSTANTIATE_TEST_SUITE_P(FutureTest, JoinAllOkFuturesExceptForOneTest,
                         testing::Range(0, 3));

}  // namespace
}  // namespace ifrt
}  // namespace xla
