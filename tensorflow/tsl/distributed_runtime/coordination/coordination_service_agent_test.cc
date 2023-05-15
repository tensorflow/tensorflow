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

#include "tensorflow/tsl/distributed_runtime/coordination/coordination_service_agent.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/tsl/distributed_runtime/call_options.h"
#include "tensorflow/tsl/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/protobuf/coordination_config.pb.h"
#include "tensorflow/tsl/protobuf/coordination_service.pb.h"

namespace tsl {
namespace {
using tensorflow::CoordinatedTask;
using tensorflow::CoordinationServiceConfig;
using tensorflow::KeyValueEntry;

using ::testing::_;
using ::testing::DoAll;
using ::testing::InvokeArgument;
using ::testing::SetArgPointee;
using ::testing::UnorderedPointwise;
using ::testing::WithArgs;

MATCHER(KvEq, "simple KeyValueEntry matcher") {
  const KeyValueEntry& kv0 = std::get<0>(arg);
  const KeyValueEntry& kv1 = std::get<1>(arg);
  return kv0.key() == kv1.key() && kv0.value() == kv1.value();
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
              (CallOptions * call_opts, const GetKeyValueRequest*,
               GetKeyValueResponse*, StatusCallback),
              (override));
  MOCK_METHOD(void, TryGetKeyValueAsync,
              (const TryGetKeyValueRequest*, TryGetKeyValueResponse*,
               StatusCallback),
              (override));
  MOCK_METHOD(void, GetKeyValueDirAsync,
              (const GetKeyValueDirRequest*, GetKeyValueDirResponse*,
               StatusCallback),
              (override));
  MOCK_METHOD(void, RegisterTaskAsync,
              (CallOptions*, const RegisterTaskRequest*, RegisterTaskResponse*,
               StatusCallback),
              (override));
  MOCK_METHOD(void, ShutdownTaskAsync,
              (CallOptions*, const ShutdownTaskRequest*, ShutdownTaskResponse*,
               StatusCallback),
              (override));
  MOCK_METHOD(void, ResetTaskAsync,
              (const ResetTaskRequest*, ResetTaskResponse*, StatusCallback),
              (override));
  MOCK_METHOD(void, ReportErrorToServiceAsync,
              (const ReportErrorToServiceRequest*,
               ReportErrorToServiceResponse*, StatusCallback),
              (override));
  MOCK_METHOD(void, BarrierAsync,
              (const BarrierRequest*, BarrierResponse*, StatusCallback),
              (override));
  MOCK_METHOD(void, GetTaskStateAsync,
              (const GetTaskStateRequest*, GetTaskStateResponse*,
               StatusCallback),
              (override));
  MOCK_METHOD(void, HeartbeatAsync,
              (CallOptions*, const HeartbeatRequest*, HeartbeatResponse*,
               StatusCallback),
              (override));

#define UNIMPLEMENTED(method)                                         \
  void method##Async(const method##Request* request,                  \
                     method##Response* response, StatusCallback done) \
      override {                                                      \
    done(errors::Unimplemented(#method "Async"));                     \
  }

  UNIMPLEMENTED(WaitForAllTasks);
  UNIMPLEMENTED(InsertKeyValue);
  UNIMPLEMENTED(DeleteKeyValue);
  UNIMPLEMENTED(CancelBarrier);
#undef UNIMPLEMENTED
  void ReportErrorToTaskAsync(CallOptions* call_opts,
                              const ReportErrorToTaskRequest* request,
                              ReportErrorToTaskResponse* response,
                              StatusCallback done) override {
    done(errors::Unimplemented("ReportErrorToTaskAsync"));
  }
};

class CoordinationServiceAgentTest : public ::testing::Test {
 public:
  void SetUp() override {
    ON_CALL(*client_, RegisterTaskAsync(_, _, _, _))
        .WillByDefault(InvokeArgument<3>(OkStatus()));
    ON_CALL(*client_, HeartbeatAsync(_, _, _, _))
        .WillByDefault(InvokeArgument<3>(OkStatus()));
    ON_CALL(*client_, ShutdownTaskAsync(_, _, _, _))
        .WillByDefault(InvokeArgument<3>(OkStatus()));
    ON_CALL(*client_, ReportErrorToServiceAsync(_, _, _))
        .WillByDefault(InvokeArgument<2>(OkStatus()));
    ON_CALL(*client_, ResetTaskAsync(_, _, _))
        .WillByDefault(InvokeArgument<2>(OkStatus()));
    ON_CALL(*client_, BarrierAsync(_, _, _))
        .WillByDefault(InvokeArgument<2>(OkStatus()));
    ON_CALL(*client_, GetTaskStateAsync(_, _, _))
        .WillByDefault(InvokeArgument<2>(OkStatus()));
  }

  // Should be called after mocking service responses, before testing the agent.
  void InitializeAgent(CoordinationServiceConfig config = {}) {
    config.set_service_leader("test_leader");
    TF_ASSERT_OK(agent_->Initialize(
        Env::Default(), /*job_name=*/"test_job",
        /*task_id=*/0, config, std::move(client_),
        /*error_fn=*/[](Status s) {
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
  std::unique_ptr<CoordinationServiceAgent> agent_ =
      CreateCoordinationServiceAgent();
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
                           InvokeArgument<3>(OkStatus())));
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
                           InvokeArgument<3>(OkStatus())));
  // Initialize coordination agent.
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(10));

  TF_ASSERT_OK(result.status());
  EXPECT_EQ(*result, test_value);
}

TEST_F(CoordinationServiceAgentTest, GetKeyValue_Timeout_ReturnError) {
  const std::string& test_key = "test_key";
  StatusCallback owned_done;
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _, _))
      .WillByDefault(WithArgs<3>([&](StatusCallback done) {
        // Copy method argument to prevent de-allocation.
        owned_done = done;
      }));
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(1));

  EXPECT_EQ(result.status().code(), error::DEADLINE_EXCEEDED);
  // Needed to tear down test safely since agent dtor would cancel pending
  // calls, which would reference deallocated call_opts.
  owned_done(errors::Cancelled("error"));
}

TEST_F(CoordinationServiceAgentTest,
       GetKeyValue_DelayedResponse_TimeoutWithoutMemoryError) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  auto client = std::make_unique<TestCoordinationClient>();
  GetKeyValueResponse* owned_response;
  StatusCallback owned_done;
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _, _))
      .WillByDefault(WithArgs<2, 3>(
          [&](GetKeyValueResponse* response, StatusCallback done) {
            // Copy method arguments to prevent de-allocation before mocking the
            // server callback beyond timeout.
            owned_response = response;
            owned_done = done;
          }));
  // Initialize coordination service agent.
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(1));
  EXPECT_EQ(result.status().code(), error::DEADLINE_EXCEEDED);

  // Delayed server response: set key-value response, and invoke done callback.
  auto kv = owned_response->mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  owned_done(OkStatus());
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
  std::unique_ptr<Thread> async_thread;
  GetKeyValueResponse* owned_response;
  StatusCallback owned_done;
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _, _))
      // Setup async callback to insert key-value after a brief delay (5s)
      // before timeout (10s).
      .WillByDefault(WithArgs<2, 3>(
          [&](GetKeyValueResponse* response, StatusCallback done) {
            // Copy method arguments to prevent de-allocation before
            //  triggering this async callback.
            owned_response = response;
            owned_done = done;
            async_thread = absl::WrapUnique(Env::Default()->StartThread(
                ThreadOptions(), "async_thread", [&]() {
                  // Set brief delay.
                  absl::SleepFor(absl::Seconds(5));
                  // Set key-value response, and invoke done callback.
                  auto kv = owned_response->mutable_kv();
                  kv->set_key(test_key);
                  kv->set_value(test_value);
                  owned_done(OkStatus());
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
      .WillByDefault(
          WithArgs<0, 3>([](CallOptions* call_opts, StatusCallback done) {
            // Mock RPC call cancellation.
            call_opts->SetCancelCallback([callback = std::move(done)]() {
              callback(errors::Cancelled("RPC call cancelled."));
            });
          }));
  InitializeAgent();

  Status status;
  std::shared_ptr<CallOptions> get_kv_call_opts = agent_->GetKeyValueAsync(
      test_key, [&status](const StatusOr<std::string>& result) {
        status = result.status();
      });
  get_kv_call_opts->StartCancel();

  EXPECT_TRUE(errors::IsCancelled(status)) << status;
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
                           InvokeArgument<2>(OkStatus())));

  // Initialize coordination agent.
  InitializeAgent();
  auto result = agent_->TryGetKeyValue(test_key);
  TF_ASSERT_OK(result.status());
  EXPECT_EQ(*result, test_value);
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
                           InvokeArgument<2>(OkStatus())));
  // Initialize coordination agent.
  InitializeAgent();

  auto result = agent_->GetKeyValueDir(test_key);

  TF_ASSERT_OK(result.status());
  EXPECT_THAT(*result, UnorderedPointwise(KvEq(), test_values));
}

TEST_F(CoordinationServiceAgentTest, ShutdownInErrorShouldReturnError) {
  // Connect coordination agent and set it to error.
  InitializeAgent();
  TF_ASSERT_OK(agent_->Connect());
  TF_ASSERT_OK(agent_->ReportError(errors::Internal("Test Error.")));

  // Shutdown should return error.
  Status s = agent_->Shutdown();

  EXPECT_TRUE(errors::IsFailedPrecondition(s));
}

TEST_F(CoordinationServiceAgentTest, Reset_ConnectedButNotInError_Fail) {
  // Connect agent.
  InitializeAgent();
  TF_ASSERT_OK(agent_->Connect());

  auto status = agent_->Reset();

  // Fails because agent is not in ERROR state.
  EXPECT_TRUE(errors::IsFailedPrecondition(status));
}

TEST_F(CoordinationServiceAgentTest, ConnectAfterResetError) {
  // Connect coordination agent and set it to error.
  InitializeAgent();
  TF_ASSERT_OK(agent_->Connect());
  TF_ASSERT_OK(agent_->ReportError(errors::Internal("Test Error.")));

  // Reset error.
  TF_ASSERT_OK(agent_->Reset());
  // Agent should be able to reconnect to the service after resetting.
  TF_EXPECT_OK(agent_->Connect());
}

TEST_F(CoordinationServiceAgentTest, ResetCanBeRetried) {
  // Mock reset error failing for the first time.
  EXPECT_CALL(*GetClient(), ResetTaskAsync(_, _, _))
      .WillOnce(InvokeArgument<2>(errors::Internal("Reset error")))
      .WillOnce(InvokeArgument<2>(OkStatus()));
  // Connect coordination agent and set it to error.
  InitializeAgent();
  TF_ASSERT_OK(agent_->Connect());
  TF_ASSERT_OK(agent_->ReportError(errors::Internal("Test Error.")));

  // Reset error fails for the first time.
  Status reset_status = agent_->Reset();
  EXPECT_TRUE(errors::IsInternal(reset_status));

  // Agent should be able to attempt resetting again.
  TF_ASSERT_OK(agent_->Reset());
  // Agent should be able to reconnect to the service after resetting.
  TF_EXPECT_OK(agent_->Connect());
}

TEST_F(CoordinationServiceAgentTest, GetOwnTask) {
  InitializeAgent();

  auto result = agent_->GetOwnTask();

  TF_ASSERT_OK(result.status());
  CoordinatedTask actual_task = *result;
  // These fields are from the arguments used in InitializeAgent().
  CoordinatedTask expected_task;
  expected_task.set_job_name("test_job");
  expected_task.set_task_id(0);
  EXPECT_EQ(actual_task.job_name(), expected_task.job_name());
  EXPECT_EQ(actual_task.task_id(), expected_task.task_id());
}

TEST_F(CoordinationServiceAgentTest, GetOwnTask_Uninitialized) {
  auto result = agent_->GetOwnTask();

  EXPECT_TRUE(errors::IsFailedPrecondition(result.status()));
}

TEST_F(CoordinationServiceAgentTest, WaitAtBarrier_SameIdUsedTwice_Fails) {
  InitializeAgent();
  const std::string barrier_id = "only_use_once";
  TF_ASSERT_OK(agent_->Connect());
  // Wait at barrier for the first time should succeed.
  TF_ASSERT_OK(
      agent_->WaitAtBarrier(barrier_id, absl::Seconds(1), /*tasks=*/{}));

  // Subsequent calls should fail.
  auto result =
      agent_->WaitAtBarrier(barrier_id, absl::Seconds(1), /*tasks=*/{});

  EXPECT_TRUE(errors::IsFailedPrecondition(result));
}

TEST_F(CoordinationServiceAgentTest, GetEnv_SucceedsAfterInit) {
  EXPECT_TRUE(errors::IsFailedPrecondition(agent_->GetEnv().status()));
  InitializeAgent();

  StatusOr<Env*> result = agent_->GetEnv();

  TF_ASSERT_OK(result.status());
  EXPECT_EQ(*result, Env::Default());
}

TEST_F(CoordinationServiceAgentTest, Connect_AbortedErrorShouldBeRetried) {
  // Mock connection failing for the first two times.
  EXPECT_CALL(*GetClient(), RegisterTaskAsync(_, _, _, _))
      .WillOnce(InvokeArgument<3>(errors::Aborted("DuplicateTaskRegistration")))
      .WillOnce(InvokeArgument<3>(errors::Aborted("DuplicateTaskRegistration")))
      .WillOnce(InvokeArgument<3>(OkStatus()));
  InitializeAgent();

  TF_EXPECT_OK(agent_->Connect());
}

TEST_F(CoordinationServiceAgentTest, Connect_AbortedErrorShouldFailEventually) {
  // Mock connection failing - old incarnation of coordination service never
  // restarts.
  EXPECT_CALL(*GetClient(), RegisterTaskAsync(_, _, _, _))
      .WillRepeatedly(
          InvokeArgument<3>(errors::Aborted("DuplicateTaskRegistration")));
  CoordinationServiceConfig config;
  // Connect should only be retried for 3 seconds.
  config.set_cluster_register_timeout_in_ms(
      absl::ToInt64Milliseconds(absl::Seconds(3)));
  InitializeAgent(config);

  Status s = agent_->Connect();

  EXPECT_TRUE(errors::IsAborted(s));
}

TEST_F(CoordinationServiceAgentTest, Connect_InternalErrorShouldBeRetried) {
  EXPECT_CALL(*GetClient(), RegisterTaskAsync(_, _, _, _))
      .WillOnce(InvokeArgument<3>(
          errors::Internal("Coordination service is not enabled.")))
      .WillOnce(InvokeArgument<3>(
          errors::Internal("Coordination service is not enabled.")))
      .WillOnce(InvokeArgument<3>(OkStatus()));
  InitializeAgent();

  TF_EXPECT_OK(agent_->Connect());
}

}  // namespace
}  // namespace tsl
