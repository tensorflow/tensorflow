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
#include "tensorflow/core/common_runtime/eager/eager_executor.h"

#include <memory>
#include <utility>

#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace {

class TestState {
 public:
  enum State { kSuccess, kNotRun, kFailure };
  TestState() : state_(kNotRun) {}
  TestState(const TestState&) = delete;
  TestState& operator=(const TestState&) = delete;
  State read_state() { return state_; }
  void update_success_state() { state_ = kSuccess; }
  void update_run_error_state() { state_ = kFailure; }

 private:
  State state_;
};

class TestEagerNode : public EagerNode {
 public:
  explicit TestEagerNode(TestState* state,
                         absl::Status prepare_return_status = absl::OkStatus(),
                         absl::Status run_return_status = absl::OkStatus())
      : state_(state),
        prepare_return_status_(prepare_return_status),
        run_return_status_(run_return_status) {}
  TestEagerNode(const TestEagerNode&) = delete;
  TestEagerNode& operator=(const TestEagerNode&) = delete;
  absl::Status Prepare() override { return prepare_return_status_; }

  absl::Status Run() override {
    if (run_return_status_.ok()) {
      state_->update_success_state();
    } else {
      state_->update_run_error_state();
    }
    return run_return_status_;
  };

  void Abort(absl::Status status) override {}
  string DebugString() const override { return "testEagerNode"; }

 private:
  TestState* state_;
  absl::Status prepare_return_status_;
  absl::Status run_return_status_;
};

class TestAsyncEagerNode : public AsyncEagerNode {
 public:
  explicit TestAsyncEagerNode(
      TestState* state, absl::Status prepare_return_status = absl::OkStatus(),
      absl::Status run_return_status = absl::OkStatus())
      : state_(state),
        prepare_return_status_(prepare_return_status),
        run_return_status_(run_return_status) {}
  TestAsyncEagerNode(const TestAsyncEagerNode&) = delete;
  TestAsyncEagerNode& operator=(const TestAsyncEagerNode&) = delete;

  absl::Status Prepare() override { return prepare_return_status_; }

  void RunAsync(StatusCallback done) override {
    if (run_return_status_.ok()) {
      state_->update_success_state();
    } else {
      state_->update_run_error_state();
    }
    done(run_return_status_);
  };

  void Abort(absl::Status status) override {}
  string DebugString() const override { return "testAsyncEagerNode"; }

 private:
  TestState* state_;
  absl::Status prepare_return_status_;
  absl::Status run_return_status_;
};

TEST(EagerExecutorTest, TestSyncExecutorWithEagerNode) {
  auto sync_executor = std::make_unique<EagerExecutor>(
      /*async=*/false, /*enable_streaming_enqueue=*/true);
  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestEagerNode>(state.get());

  TF_ASSERT_OK(sync_executor->AddOrExecute(std::move(node)));
  ASSERT_EQ(state->read_state(), TestState::State::kSuccess);
}

TEST(EagerExecutorTest, TestSyncExecuteMethodFailureCases) {
  // Async Executor with Eager node fails
  auto async_executor = std::make_unique<EagerExecutor>(
      /*async=*/true, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto sync_node = std::make_unique<TestEagerNode>(state.get());

  EXPECT_THAT(async_executor->SyncExecute(sync_node.get()),
              tensorflow::testing::StatusIs(tensorflow::error::INTERNAL));
  ASSERT_EQ(state->read_state(), TestState::kNotRun);

  // Sync Executor with Async node fails
  auto sync_executor = std::make_unique<EagerExecutor>(
      /*async=*/false, /*enable_streaming_enqueue=*/true);

  state = std::make_unique<TestState>();
  auto async_node = std::make_unique<TestAsyncEagerNode>(state.get());

  EXPECT_THAT(sync_executor->SyncExecute(async_node.get()),
              tensorflow::testing::StatusIs(tensorflow::error::INTERNAL));
  ASSERT_EQ(state->read_state(), TestState::State::kNotRun);
}

TEST(EagerExecutorTest, TestSyncExecuteMethodSuccessCase) {
  auto sync_executor = std::make_unique<EagerExecutor>(
      /*async=*/false, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestEagerNode>(state.get());

  TF_ASSERT_OK(sync_executor->SyncExecute(node.get()));
  ASSERT_EQ(state->read_state(), TestState::State::kSuccess);
}

TEST(EagerExecutorTest, TestSyncExecutorFailPrepare) {
  auto sync_executor = std::make_unique<EagerExecutor>(
      /*async=*/false, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestEagerNode>(state.get(),
                                              errors::InvalidArgument("test"));
  auto status = sync_executor->AddOrExecute(std::move(node));

  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_EQ(state->read_state(), TestState::State::kNotRun);
}

TEST(EagerExecutorTest, TestSyncExecutorFailRun) {
  auto sync_executor = std::make_unique<EagerExecutor>(
      /*async=*/false, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestEagerNode>(state.get(), absl::OkStatus(),
                                              errors::Internal("test"));

  auto status = sync_executor->AddOrExecute(std::move(node));
  ASSERT_EQ(status.code(), tensorflow::error::INTERNAL);
  ASSERT_EQ(state->read_state(), TestState::State::kFailure);
}

TEST(EagerExecutorTest, TestAsyncExecutorWithAsyncEagerNode) {
  auto async_executor = std::make_unique<EagerExecutor>(
      /*async=*/true, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestAsyncEagerNode>(state.get());

  TF_ASSERT_OK(async_executor->AddOrExecute(std::move(node)));
  TF_ASSERT_OK(async_executor->WaitForAllPendingNodes());
  ASSERT_EQ(state->read_state(), TestState::State::kSuccess);
}

TEST(EagerExecutorTest, TestAsyncExecutorWithInFlightRequestLimit) {
  auto async_executor = std::make_unique<EagerExecutor>(
      /*async=*/true, /*enable_streaming_enqueue=*/true,
      /*in_flight_nodes_limit=*/1);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestAsyncEagerNode>(state.get());

  TF_ASSERT_OK(async_executor->AddOrExecute(std::move(node)));

  auto node1 = std::make_unique<TestAsyncEagerNode>(state.get());
  TF_ASSERT_OK(async_executor->AddOrExecute(std::move(node1)));
  TF_ASSERT_OK(async_executor->WaitForAllPendingNodes());
  ASSERT_EQ(state->read_state(), TestState::State::kSuccess);
}

TEST(EagerExecutorTest, TestAsyncExecutorWithEagerNode) {
  auto async_executor = std::make_unique<EagerExecutor>(
      /*async=*/true, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestEagerNode>(state.get());

  TF_ASSERT_OK(async_executor->AddOrExecute(std::move(node)));
  TF_ASSERT_OK(async_executor->WaitForAllPendingNodes());
  ASSERT_EQ(state->read_state(), TestState::State::kSuccess);
}

TEST(EagerExecutorTest, TestAsyncExecutorFailPrepare) {
  auto async_executor = std::make_unique<EagerExecutor>(
      /*async=*/true, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestEagerNode>(state.get(),
                                              errors::InvalidArgument("test"));

  auto status = async_executor->AddOrExecute(std::move(node));

  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_EQ(state->read_state(), TestState::State::kNotRun);
}

TEST(EagerExecutorTest, TestAsyncExecutorFailRun) {
  auto async_executor = std::make_unique<EagerExecutor>(
      /*async=*/true, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestEagerNode>(state.get(), absl::OkStatus(),
                                              errors::Internal("test"));

  TF_ASSERT_OK(async_executor->AddOrExecute(std::move(node)));
  auto status = async_executor->WaitForAllPendingNodes();
  ASSERT_EQ(status.code(), tensorflow::error::INTERNAL);
  ASSERT_EQ(state->read_state(), TestState::State::kFailure);
}

TEST(EagerExecutorTest, TestAsyncExecutorFailPrepareWithAsyncNode) {
  auto async_executor = std::make_unique<EagerExecutor>(
      /*async=*/true, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestAsyncEagerNode>(
      state.get(), errors::InvalidArgument("test"));
  auto status = async_executor->AddOrExecute(std::move(node));

  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_EQ(state->read_state(), TestState::State::kNotRun);
}

TEST(EagerExecutorTest, TestAsyncExecutorFailRunWithAsyncNode) {
  auto async_executor = std::make_unique<EagerExecutor>(
      /*async=*/true, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestAsyncEagerNode>(
      state.get(), absl::OkStatus(), errors::Internal("test"));

  TF_ASSERT_OK(async_executor->AddOrExecute(std::move(node)));

  auto status = async_executor->WaitForAllPendingNodes();
  ASSERT_EQ(status.code(), tensorflow::error::INTERNAL);
  ASSERT_EQ(state->read_state(), TestState::State::kFailure);
}

TEST(EagerExecutorTest, TestAsyncExecutorAddNodesAfterShutdown) {
  auto async_executor = std::make_unique<EagerExecutor>(
      /*async=*/true, /*enable_streaming_enqueue=*/true);

  auto state = std::make_unique<TestState>();
  auto node = std::make_unique<TestAsyncEagerNode>(state.get());

  TF_ASSERT_OK(async_executor->ShutDown());
  EXPECT_THAT(
      async_executor->AddOrExecute(std::move(node)),
      tensorflow::testing::StatusIs(tensorflow::error::FAILED_PRECONDITION));
}
}  // namespace
}  // namespace tensorflow
