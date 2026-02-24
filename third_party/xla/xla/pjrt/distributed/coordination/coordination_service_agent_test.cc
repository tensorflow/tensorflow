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

#include "xla/pjrt/distributed/coordination/coordination_service_agent.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/coordination/coordination_client.h"
#include "xla/pjrt/distributed/coordination/coordination_service_error_util.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"

namespace xla {
namespace {
using tensorflow::CoordinatedTask;
using tensorflow::KeyValueEntry;

using ::testing::_;
using ::testing::DoAll;
using ::testing::InvokeArgument;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::UnorderedPointwise;
using ::testing::WithArgs;

MATCHER(KvEq, "simple KeyValueEntry matcher") {
  const KeyValueEntry& kv0 = std::get<0>(arg);
  const KeyValueEntry& kv1 = std::get<1>(arg);
  return kv0.key() == kv1.key() && kv0.value() == kv1.value();
}

// Note: b/169705709: no protobuf matchers in OSS.
MATCHER_P2(IsBarrierRequest, id, counter, "") {
  return id == arg->barrier_id() && counter == arg->counter();
}

KeyValueEntry CreateKv(const std::string& key, const std::string& value) {
  KeyValueEntry kv;
  kv.set_key(key);
  kv.set_value(value);
  return kv;
}

class TestCoordinationClient : public CoordinationClient {
 public:
  TestCoordinationClient() = default;
  MOCK_METHOD(void, GetKeyValueAsync,
              (tsl::CallOptions * call_opts, const GetKeyValueRequest*,
               GetKeyValueResponse*, tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, TryGetKeyValueAsync,
              (const TryGetKeyValueRequest*, TryGetKeyValueResponse*,
               tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, IncrementKeyValueAsync,
              (const IncrementKeyValueRequest*, IncrementKeyValueResponse*,
               tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, GetKeyValueDirAsync,
              (const GetKeyValueDirRequest*, GetKeyValueDirResponse*,
               tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, InsertKeyValueAsync,
              (const InsertKeyValueRequest*, InsertKeyValueResponse*,
               tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, DeleteKeyValueAsync,
              (const DeleteKeyValueRequest*, DeleteKeyValueResponse*,
               tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, RegisterTaskAsync,
              (tsl::CallOptions*, const RegisterTaskRequest*,
               RegisterTaskResponse*, tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, ShutdownTaskAsync,
              (tsl::CallOptions*, const ShutdownTaskRequest*,
               ShutdownTaskResponse*, tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, ResetTaskAsync,
              (const ResetTaskRequest*, ResetTaskResponse*,
               tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, BarrierAsync,
              (tsl::CallOptions * call_opts, const BarrierRequest*,
               BarrierResponse*, tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, CancelBarrierAsync,
              (const CancelBarrierRequest*, CancelBarrierResponse*,
               tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, GetAliveTasksAsync,
              (const GetAliveTasksRequest*, GetAliveTasksResponse*,
               tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, WatchJobStateAsync,
              (tsl::CallOptions*, const WatchJobStateRequest*,
               WatchJobStateResponse*, tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, HeartbeatAsync,
              (tsl::CallOptions*, const HeartbeatRequest*, HeartbeatResponse*,
               tsl::StatusCallback),
              (override));
  MOCK_METHOD(void, PollForErrorAsync,
              (tsl::CallOptions * call_opts, const PollForErrorRequest*,
               PollForErrorResponse*, tsl::StatusCallback),
              (override));
};

class CoordinationServiceAgentTest : public ::testing::Test {
 public:
  void SetUp() override {
    ON_CALL(*client_, RegisterTaskAsync(_, _, _, _))
        .WillByDefault(InvokeArgument<3>(absl::OkStatus()));
    ON_CALL(*client_, HeartbeatAsync(_, _, _, _))
        .WillByDefault(InvokeArgument<3>(absl::OkStatus()));
    ON_CALL(*client_, ShutdownTaskAsync(_, _, _, _))
        .WillByDefault(InvokeArgument<3>(absl::OkStatus()));
    ON_CALL(*client_, ResetTaskAsync(_, _, _))
        .WillByDefault(InvokeArgument<2>(absl::OkStatus()));
    ON_CALL(*client_, BarrierAsync(_, _, _, _))
        .WillByDefault(InvokeArgument<3>(absl::OkStatus()));
    ON_CALL(*client_, CancelBarrierAsync(_, _, _))
        .WillByDefault(InvokeArgument<2>(absl::OkStatus()));
  }

  // Should be called after mocking service responses, before testing the agent.
  void InitializeAgent(CoordinationServiceAgent::Config config = {}) {
    TF_ASSERT_OK_AND_ASSIGN(
        agent_, CoordinationServiceAgent::Create(
                    tsl::Env::Default(),
                    /*task_id=*/0, config, std::move(client_),
                    /*error_fn=*/[](absl::Status s) {
                      LOG(ERROR) << "Coordination agent is set to error: " << s;
                    }));
  }

  TestCoordinationClient* GetClient() {
    // InitializeAgent() transfers ownership of the coordination client.
    CHECK(client_ != nullptr)
        << "GetClient() was called after InitializeAgent()";
    return client_.get();
  }

 protected:
  std::unique_ptr<CoordinationServiceAgent> agent_;
  std::unique_ptr<TestCoordinationClient> client_ =
      std::make_unique<TestCoordinationClient>();
};

TEST_F(CoordinationServiceAgentTest, GetKeyValue_Simple_Success) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  // Mock server response: set key-value pair and invoke done callback.
  GetKeyValueResponse mocked_response;
  auto kv = mocked_response.mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _, _))
      .WillByDefault(DoAll(SetArgPointee<2>(mocked_response),
                           InvokeArgument<3>(absl::OkStatus())));
  // Initialize coordination agent.
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key);

  TF_ASSERT_OK(result.status());
  EXPECT_EQ(*result, test_value);
}

TEST_F(CoordinationServiceAgentTest, GetKeyValue_WithTimeout_Success) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  // Mock server response: set key-value pair and invoke done callback.
  GetKeyValueResponse mocked_response;
  auto kv = mocked_response.mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _, _))
      .WillByDefault(DoAll(SetArgPointee<2>(mocked_response),
                           InvokeArgument<3>(absl::OkStatus())));
  // Initialize coordination agent.
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(10));

  TF_ASSERT_OK(result.status());
  EXPECT_EQ(*result, test_value);
}

TEST_F(CoordinationServiceAgentTest, GetKeyValue_Timeout_ReturnError) {
  const std::string& test_key = "test_key";
  tsl::StatusCallback owned_done;
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _, _))
      .WillByDefault(WithArgs<3>([&](tsl::StatusCallback done) {
        // Copy method argument to prevent deallocation.
        owned_done = done;
      }));
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(1));

  EXPECT_TRUE(absl::IsDeadlineExceeded(result.status()));
  // Needed to tear down test safely since agent dtor would cancel pending
  // calls, which would reference deallocated call_opts.
  owned_done(absl::CancelledError("error"));
}

TEST_F(CoordinationServiceAgentTest,
       GetKeyValue_DelayedResponse_TimeoutWithoutMemoryError) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  auto client = std::make_unique<TestCoordinationClient>();
  GetKeyValueResponse* owned_response;
  tsl::StatusCallback owned_done;
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _, _))
      .WillByDefault(WithArgs<2, 3>(
          [&](GetKeyValueResponse* response, tsl::StatusCallback done) {
            // Copy method arguments to prevent deallocation before mocking the
            // server callback beyond timeout.
            owned_response = response;
            owned_done = done;
          }));
  // Initialize coordination service agent.
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(1));
  EXPECT_TRUE(absl::IsDeadlineExceeded(result.status()));
  // Delayed server response: set key-value response, and invoke done callback.
  auto kv = owned_response->mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  owned_done(absl::OkStatus());
  // No explicit test, but used to verify there is no stack-use-after-return
  // or other memory-related errors.
}

TEST_F(CoordinationServiceAgentTest,
       GetKeyValue_DelayedResponseBeforeTimeout_Success) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  // Mock delayed server response before timeout: set key-value pair and invoke
  // done callback.
  auto client = std::make_unique<TestCoordinationClient>();
  std::unique_ptr<tsl::Thread> async_thread;
  GetKeyValueResponse* owned_response;
  tsl::StatusCallback owned_done;
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _, _))
      // Setup async callback to insert key-value after a brief delay (5s)
      // before timeout (10s).
      .WillByDefault(WithArgs<2, 3>(
          [&](GetKeyValueResponse* response, tsl::StatusCallback done) {
            // Copy method arguments to prevent deallocation before
            //  triggering this async callback.
            owned_response = response;
            owned_done = done;
            async_thread = absl::WrapUnique(tsl::Env::Default()->StartThread(
                tsl::ThreadOptions(), "async_thread", [&]() {
                  // Set brief delay.
                  absl::SleepFor(absl::Seconds(5));
                  // Set key-value response, and invoke done callback.
                  auto kv = owned_response->mutable_kv();
                  kv->set_key(test_key);
                  kv->set_value(test_value);
                  owned_done(absl::OkStatus());
                }));
          }));
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(10));

  TF_ASSERT_OK(result.status());
  EXPECT_EQ(*result, test_value);
}

TEST_F(CoordinationServiceAgentTest, CancelGetKeyValue_Success) {
  const std::string test_key = "test_key";
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _, _))
      .WillByDefault(WithArgs<0, 3>(
          [](tsl::CallOptions* call_opts, tsl::StatusCallback done) {
            // Mock RPC call cancellation.
            call_opts->SetCancelCallback([callback = std::move(done)]() {
              callback(absl::CancelledError("RPC call cancelled."));
            });
          }));
  InitializeAgent();

  absl::Status status;
  std::shared_ptr<tsl::CallOptions> get_kv_call_opts = agent_->GetKeyValueAsync(
      test_key, [&status](const absl::StatusOr<std::string>& result) {
        status = result.status();
      });
  get_kv_call_opts->StartCancel();

  EXPECT_TRUE(absl::IsCancelled(status)) << status;
  // This is to prevent memory leaks due to how we set this particular cancel
  // callback. In practice, this should not be necessary.
  get_kv_call_opts->ClearCancelCallback();
}

TEST_F(CoordinationServiceAgentTest, TryGetKeyValue_Simple_Success) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  // Mock server response: set key-value pair and invoke done callback.
  TryGetKeyValueResponse mocked_response;
  auto kv = mocked_response.mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  ON_CALL(*GetClient(), TryGetKeyValueAsync(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<1>(mocked_response),
                           InvokeArgument<2>(absl::OkStatus())));

  // Initialize coordination agent.
  InitializeAgent();
  auto result = agent_->TryGetKeyValue(test_key);
  TF_ASSERT_OK(result.status());
  EXPECT_EQ(*result, test_value);
}

TEST_F(CoordinationServiceAgentTest, IncrementKeyValue_Simple_Success) {
  constexpr absl::string_view test_key = "test_key";
  constexpr absl::string_view test_value = "11";
  // Mock server response: set key-value pair and invoke done callback.
  IncrementKeyValueResponse mocked_response;
  auto kv = mocked_response.mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  ON_CALL(*GetClient(), IncrementKeyValueAsync(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<1>(mocked_response),
                           InvokeArgument<2>(absl::OkStatus())));
  // Initialize coordination agent.
  InitializeAgent();
  auto result = agent_->IncrementKeyValue(test_key, 1);
  EXPECT_THAT(result, absl_testing::IsOkAndHolds(11));
}

TEST_F(CoordinationServiceAgentTest, GetKeyValueDir_Simple_Success) {
  const std::string test_key = "test_key_dir";
  std::vector<KeyValueEntry> test_values;
  test_values.push_back(CreateKv("test_key_dir/task_0", "0"));
  test_values.push_back(CreateKv("test_key_dir/task_1", "1"));
  // Mock server response: set key-value pair and invoke done callback.
  GetKeyValueDirResponse mocked_response;
  mocked_response.set_directory_key(test_key);
  *mocked_response.mutable_kv() = {test_values.begin(), test_values.end()};
  ON_CALL(*GetClient(), GetKeyValueDirAsync(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<1>(mocked_response),
                           InvokeArgument<2>(absl::OkStatus())));
  // Initialize coordination agent.
  InitializeAgent();

  auto result = agent_->GetKeyValueDir(test_key);

  TF_ASSERT_OK(result.status());
  EXPECT_THAT(*result, UnorderedPointwise(KvEq(), test_values));
}

TEST_F(CoordinationServiceAgentTest, Reset_ConnectedButNotInError_Fail) {
  // Connect agent.
  InitializeAgent();
  TF_ASSERT_OK(agent_->Connect());

  auto status = agent_->Reset();

  // Fails because agent is not in ERROR state.
  EXPECT_TRUE(absl::IsFailedPrecondition(status));
}

TEST_F(CoordinationServiceAgentTest, ConnectAfterReset_WithErrorPolling) {
  // Connect coordination agent and set it to error.
  PollForErrorResponse mocked_response;
  EXPECT_CALL(*GetClient(), PollForErrorAsync(_, _, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(mocked_response),
                      InvokeArgument<3>(absl::UnavailableError("Test Error."))))
      .WillOnce(DoAll(SetArgPointee<2>(mocked_response),
                      InvokeArgument<3>(absl::InternalError("Test Error."))));

  CoordinationServiceAgent::Config config;
  config.poll_for_error_from_service_at_startup = true;
  InitializeAgent(config);
  // The agent will be in ERROR state after the first call to Connect()
  // because the error polling thread will be created and will immediately
  // return an error.
  TF_ASSERT_OK(agent_->Connect());
  // Wait a bit for the error polling thread to start.
  absl::SleepFor(absl::Seconds(2));
  ASSERT_TRUE(agent_->IsError());

  TF_ASSERT_OK(agent_->Reset());
  // Agent should be able to reconnect to the service after resetting. The
  // error polling thread will be recreated when the agent is connected again.
  TF_EXPECT_OK(agent_->Connect());
  absl::SleepFor(absl::Seconds(2));
  // The agent should again be in ERROR state after Connect().
  EXPECT_TRUE(agent_->IsError());
}

TEST_F(CoordinationServiceAgentTest, CancelledPollForErrorRequest) {
  // Connect coordination agent.
  PollForErrorResponse mocked_response;
  EXPECT_CALL(*GetClient(), PollForErrorAsync(_, _, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(mocked_response),
                      InvokeArgument<3>(absl::CancelledError("Test Error."))));

  CoordinationServiceAgent::Config config;
  config.poll_for_error_from_service_at_startup = true;
  InitializeAgent(config);
  TF_ASSERT_OK(agent_->Connect());
  // Wait a bit for the error polling thread to start.
  absl::SleepFor(absl::Seconds(2));
  // Cancelled error polling request will not set agent to error.
  ASSERT_FALSE(agent_->IsError());
}

TEST_F(CoordinationServiceAgentTest, InvalidPollForErrorRequest) {
  // Connect coordination agent.
  PollForErrorResponse mocked_response;
  EXPECT_CALL(*GetClient(), PollForErrorAsync(_, _, _, _))
      .WillOnce(
          DoAll(SetArgPointee<2>(mocked_response),
                InvokeArgument<3>(absl::InvalidArgumentError("Test Error."))));

  CoordinationServiceAgent::Config config;
  config.poll_for_error_from_service_at_startup = true;
  InitializeAgent(config);
  TF_ASSERT_OK(agent_->Connect());
  // Wait a bit for the error polling thread to start.
  absl::SleepFor(absl::Seconds(2));
  ASSERT_TRUE(agent_->IsError());
}

TEST_F(CoordinationServiceAgentTest,
       PollForErrorRequestWithFailedPrecondition) {
  // Connect coordination agent.
  PollForErrorResponse mocked_response;
  EXPECT_CALL(*GetClient(), PollForErrorAsync(_, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<2>(mocked_response),
          InvokeArgument<3>(absl::FailedPreconditionError("Test Error."))));

  CoordinationServiceAgent::Config config;
  config.poll_for_error_from_service_at_startup = true;
  InitializeAgent(config);
  TF_ASSERT_OK(agent_->Connect());
  // Wait a bit for the error polling thread to start.
  absl::SleepFor(absl::Seconds(2));
  ASSERT_TRUE(agent_->IsError());
}

TEST_F(CoordinationServiceAgentTest, GetOwnTask) {
  InitializeAgent();
  EXPECT_EQ(agent_->task_id(), 0);
}

TEST_F(CoordinationServiceAgentTest, GetEnv_SucceedsAfterInit) {
  InitializeAgent();
  absl::StatusOr<tsl::Env*> result = agent_->GetEnv();
  TF_ASSERT_OK(result.status());
  EXPECT_EQ(*result, tsl::Env::Default());
}

TEST_F(CoordinationServiceAgentTest, Connect_AbortedErrorShouldBeRetried) {
  // Mock connection failing for the first two times.
  EXPECT_CALL(*GetClient(), RegisterTaskAsync(_, _, _, _))
      .WillOnce(
          InvokeArgument<3>(absl::AbortedError("DuplicateTaskRegistration")))
      .WillOnce(
          InvokeArgument<3>(absl::AbortedError("DuplicateTaskRegistration")))
      .WillOnce(InvokeArgument<3>(absl::OkStatus()));
  InitializeAgent();

  TF_EXPECT_OK(agent_->Connect());
}

TEST_F(CoordinationServiceAgentTest, Connect_AbortedErrorShouldFailEventually) {
  // Mock connection failing - old incarnation of coordination service never
  // restarts.
  EXPECT_CALL(*GetClient(), RegisterTaskAsync(_, _, _, _))
      .WillRepeatedly(
          InvokeArgument<3>(absl::AbortedError("DuplicateTaskRegistration")));
  CoordinationServiceAgent::Config config;
  // Connect should only be retried for 3 seconds.
  config.cluster_register_timeout = absl::Seconds(3);
  InitializeAgent(config);

  absl::Status s = agent_->Connect();

  EXPECT_TRUE(absl::IsAborted(s));
}

TEST_F(CoordinationServiceAgentTest, Connect_InternalErrorShouldBeRetried) {
  EXPECT_CALL(*GetClient(), RegisterTaskAsync(_, _, _, _))
      .WillOnce(InvokeArgument<3>(
          absl::InternalError("Coordination service is not enabled.")))
      .WillOnce(InvokeArgument<3>(
          absl::InternalError("Coordination service is not enabled.")))
      .WillOnce(InvokeArgument<3>(absl::OkStatus()));
  InitializeAgent();

  TF_EXPECT_OK(agent_->Connect());
}

TEST_F(CoordinationServiceAgentTest, WaitAtBarrier_Twice_Success) {
  const std::string barrier_id = "barrier_id";
  // tsl::Call expectations need to be set before the agent is initialized.
  EXPECT_CALL(*GetClient(),
              BarrierAsync(_, IsBarrierRequest(barrier_id, 0), _, _))
      .WillOnce(InvokeArgument<3>(absl::OkStatus()));
  EXPECT_CALL(*GetClient(),
              // Check that the counter is incremented.
              BarrierAsync(_, IsBarrierRequest(barrier_id, 1), _, _))
      .WillOnce(InvokeArgument<3>(absl::OkStatus()));

  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());

  TF_EXPECT_OK(
      agent_->WaitAtBarrier(barrier_id, absl::Seconds(1), /*tasks=*/{}));
  TF_EXPECT_OK(
      agent_->WaitAtBarrier(barrier_id, absl::Seconds(1), /*tasks=*/{}));
}

TEST_F(CoordinationServiceAgentTest, WaitAtBarrier_Ongoing_Fails) {
  const std::string barrier_id = "barrier_id";
  // tsl::Call expectations need to be set before the agent is initialized.
  EXPECT_CALL(*GetClient(),
              BarrierAsync(_, IsBarrierRequest(barrier_id, 0), _, _))
      // Let the first call hang by not invoking the done callback.
      .WillOnce(Return());

  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());

  agent_->WaitAtBarrierAsync(barrier_id, absl::Seconds(1), /*tasks=*/{},
                             [](const absl::Status& s) {});

  EXPECT_THAT(agent_->WaitAtBarrier(barrier_id, absl::Seconds(1), /*tasks=*/{}),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(CoordinationServiceAgentTest,
       WaitAtBarrier_FailedWithBarrierError_IncrementCounter) {
  const std::string barrier_id = "barrier_id";
  // tsl::Call expectations need to be set before the agent is initialized.
  // First barrier fails with service error (has coordination payload).
  EXPECT_CALL(*GetClient(),
              BarrierAsync(_, IsBarrierRequest(barrier_id, 0), _, _))
      .WillOnce(InvokeArgument<3>(MakeCoordinationError(MakeBarrierError(
          absl::InternalError("Barrier failed."), barrier_id, 0))));
  EXPECT_CALL(*GetClient(),
              // Second barrier should have incremented counter.
              BarrierAsync(_, IsBarrierRequest(barrier_id, 1), _, _))
      .WillOnce(InvokeArgument<3>(absl::OkStatus()));

  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());
  EXPECT_THAT(agent_->WaitAtBarrier(barrier_id, absl::Seconds(1), /*tasks=*/{}),
              absl_testing::StatusIs(absl::StatusCode::kInternal));

  TF_EXPECT_OK(agent_->WaitAtBarrier(barrier_id, absl::Seconds(1), {}));
}

TEST_F(CoordinationServiceAgentTest,
       WaitAtBarrier_FailedWithRpcError_DoesNotIncrementCounter) {
  const std::string barrier_id = "barrier_id";
  // tsl::Call expectations need to be set before the agent is initialized.
  // First barrier fails with RPC error (no coordination payload).
  EXPECT_CALL(*GetClient(),
              BarrierAsync(_, IsBarrierRequest(barrier_id, 0), _, _))
      .WillOnce(InvokeArgument<3>(absl::UnavailableError("Connection lost.")))
      // Second call will use the same un-incremented counter.
      .WillOnce(InvokeArgument<3>(absl::OkStatus()));

  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());
  EXPECT_THAT(agent_->WaitAtBarrier(barrier_id, absl::Seconds(1), /*tasks=*/{}),
              absl_testing::StatusIs(absl::StatusCode::kUnavailable));

  TF_EXPECT_OK(agent_->WaitAtBarrier(barrier_id, absl::Seconds(1), {}));
}

TEST_F(CoordinationServiceAgentTest, CancelBarrier_OngoingBarrier_Cancelled) {
  const std::string barrier_id = "barrier_id";
  EXPECT_CALL(*GetClient(),
              BarrierAsync(_, IsBarrierRequest(barrier_id, 0), _, _))
      // Let the first call hang by not invoking the done callback.
      .WillOnce(Return());
  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());
  agent_->WaitAtBarrierAsync(barrier_id, absl::Seconds(1), /*tasks=*/{},
                             // Can't test this since this would be invoked on
                             // service after cancel invocation.
                             [](const absl::Status& s) {});

  EXPECT_THAT(agent_->CancelBarrier(barrier_id),
              absl_testing::StatusIs(absl::StatusCode::kOk));
}

TEST_F(CoordinationServiceAgentTest, CancelBarrier_NonExistent_Fails) {
  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());

  EXPECT_THAT(agent_->CancelBarrier("nonexistent_barrier"),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(CoordinationServiceAgentTest, CancelBarrier_CompletedBarrier_Fails) {
  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());
  TF_EXPECT_OK(
      agent_->WaitAtBarrier("barrier_id", absl::Seconds(1), /*tasks=*/{}));

  EXPECT_THAT(agent_->CancelBarrier("barrier_id"),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(CoordinationServiceAgentTest, CancelBarrier_ErroredBarrier_Fails) {
  EXPECT_CALL(*GetClient(), BarrierAsync(_, _, _, _))
      .WillOnce(InvokeArgument<3>(absl::InternalError("Test Error.")));
  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());
  ASSERT_THAT(
      agent_->WaitAtBarrier("barrier_id", absl::Seconds(1), /*tasks=*/{}),
      absl_testing::StatusIs(absl::StatusCode::kInternal));

  EXPECT_THAT(agent_->CancelBarrier("barrier_id"),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace xla
