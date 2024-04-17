/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_future.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/platform/test.h"

namespace xla {

TEST(PjRtFutureTest, StatelessFuture) {
  auto promise = PjRtFuture<>::CreatePromise();
  PjRtFuture<> future(promise);

  EXPECT_FALSE(future.IsReady());
  promise.Set();
  EXPECT_TRUE(future.IsReady());

  EXPECT_EQ(future.Await(), absl::OkStatus());

  future.OnReady(
      [](absl::Status status) { EXPECT_EQ(status, absl::OkStatus()); });
}

TEST(PjRtFutureTest, StatelessError) {
  auto promise = PjRtFuture<>::CreatePromise();
  PjRtFuture<> future(promise);

  EXPECT_FALSE(future.IsReady());
  promise.Set(absl::InternalError("test"));
  EXPECT_TRUE(future.IsReady());

  absl::Status status = future.Await();
  EXPECT_EQ(status, absl::InternalError("test"));

  future.OnReady([](absl::Status status) {
    EXPECT_EQ(status, absl::InternalError("test"));
  });
}

TEST(PjRtFutureTest, StatelessImmediate) {
  PjRtFuture<> ok_future(absl::OkStatus());
  PjRtFuture<> error_future(absl::InternalError("test"));

  EXPECT_TRUE(ok_future.IsReady());
  EXPECT_TRUE(error_future.IsReady());

  EXPECT_EQ(ok_future.Await(), absl::OkStatus());
  EXPECT_EQ(error_future.Await(), absl::InternalError("test"));

  ok_future.OnReady(
      [](absl::Status status) { EXPECT_EQ(status, absl::OkStatus()); });

  error_future.OnReady([](absl::Status status) {
    EXPECT_EQ(status, absl::InternalError("test"));
  });
}

TEST(PjRtFutureTest, StatefulFuture) {
  auto promise = PjRtFuture<int32_t>::CreatePromise();
  PjRtFuture<int32_t> future(promise);

  EXPECT_FALSE(future.IsReady());
  promise.Set(42);
  EXPECT_TRUE(future.IsReady());

  future.OnReady([](int32_t value) { EXPECT_EQ(value, 42); });
}

TEST(PjRtFutureTest, StatusFuture) {
  auto promise = PjRtFuture<absl::Status>::CreatePromise();
  PjRtFuture<absl::Status> future(promise);

  EXPECT_FALSE(future.IsReady());
  promise.Set(absl::OkStatus());
  EXPECT_TRUE(future.IsReady());

  future.OnReady(
      [](absl::Status status) { EXPECT_EQ(status, absl::OkStatus()); });
}

TEST(PjRtFutureTest, StatusOrFuture) {
  auto promise = PjRtFuture<absl::StatusOr<int32_t>>::CreatePromise();
  PjRtFuture<absl::StatusOr<int32_t>> future(promise);

  EXPECT_FALSE(future.IsReady());
  promise.Set(absl::StatusOr<int32_t>(42));
  EXPECT_TRUE(future.IsReady());

  future.OnReady([](absl::StatusOr<int32_t> value) { EXPECT_EQ(*value, 42); });
}

TEST(PjRtFutureTest, JoinFutures) {
  auto empty_join = JoinFutures({});
  EXPECT_TRUE(empty_join.IsReady());
  EXPECT_EQ(empty_join.Await(), absl::OkStatus());

  auto promise0 = PjRtFuture<>::CreatePromise();
  auto promise1 = PjRtFuture<>::CreatePromise();

  std::vector<PjRtFuture<>> futures0 = {PjRtFuture<>(promise0)};
  std::vector<PjRtFuture<>> futures1 = {PjRtFuture<>(promise0),
                                        PjRtFuture<>(promise1)};

  auto join_one = JoinFutures(futures0);
  EXPECT_FALSE(join_one.IsReady());

  auto join_two = JoinFutures(futures1);
  EXPECT_FALSE(join_two.IsReady());

  promise0.Set();
  EXPECT_TRUE(join_one.IsReady());
  EXPECT_FALSE(join_two.IsReady());
  EXPECT_EQ(join_one.Await(), absl::OkStatus());

  promise1.Set();
  EXPECT_TRUE(join_two.IsReady());
  EXPECT_EQ(join_two.Await(), absl::OkStatus());
}

TEST(PjRtFutureTest, JoinErrors) {
  auto empty_join = JoinFutures({});
  EXPECT_TRUE(empty_join.IsReady());
  EXPECT_EQ(empty_join.Await(), absl::OkStatus());

  auto promise0 = PjRtFuture<>::CreatePromise();
  auto promise1 = PjRtFuture<>::CreatePromise();

  std::vector<PjRtFuture<>> futures0 = {PjRtFuture<>(promise0)};
  std::vector<PjRtFuture<>> futures1 = {PjRtFuture<>(promise0),
                                        PjRtFuture<>(promise1)};

  auto join_one = JoinFutures(futures0);
  EXPECT_FALSE(join_one.IsReady());

  auto join_two = JoinFutures(futures1);
  EXPECT_FALSE(join_two.IsReady());

  promise0.Set(absl::InternalError("error #0"));
  EXPECT_TRUE(join_one.IsReady());
  EXPECT_FALSE(join_two.IsReady());
  EXPECT_EQ(join_one.Await(), absl::InternalError("error #0"));

  promise1.Set(absl::InternalError("error #1"));
  EXPECT_TRUE(join_two.IsReady());
  EXPECT_EQ(join_two.Await(), absl::InternalError("error #0"));
}

}  // namespace xla
