/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/next_pluggable_device/c_plugin_coordination_service_agent.h"

#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace {
using tsl::CoordinationClient;
using tsl::CoordinationServiceAgent;

using tsl::CallOptions;
using tsl::DeleteKeyValueRequest;
using tsl::DeleteKeyValueResponse;
using tsl::GetKeyValueRequest;
using tsl::GetKeyValueResponse;
using tsl::InsertKeyValueRequest;
using tsl::InsertKeyValueResponse;

using ::testing::_;
using ::testing::DoAll;
using ::testing::InvokeArgument;
using ::testing::Pointee;
using ::testing::SetArgPointee;
using ::testing::WithArgs;

// TODO(b/229726259) Switch to OSS version after it's available.
// Simple implementation of a proto matcher comparing string representations.
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tsl::protobuf::Message& expected)
      : expected_(expected.DebugString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p,
                       ::testing::MatchResultListener*) const {
    return p.DebugString() == expected_;
  }

  void DescribeTo(std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const std::string expected_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const tsl::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

MATCHER(KvEq, "simple KeyValueEntry matcher") {
  const KeyValueEntry& kv0 = std::get<0>(arg);
  const KeyValueEntry& kv1 = std::get<1>(arg);
  return kv0.key() == kv1.key() && kv0.value() == kv1.value();
}

class TestCoordinationClient : public CoordinationClient {
 public:
  TestCoordinationClient() = default;
  // Methods under test.
  MOCK_METHOD(void, GetKeyValueAsync,
              (CallOptions * call_opts, const GetKeyValueRequest*,
               GetKeyValueResponse*, StatusCallback),
              (override));
  MOCK_METHOD(void, TryGetKeyValueAsync,
              (const TryGetKeyValueRequest*, TryGetKeyValueResponse*,
               StatusCallback),
              (override));
  MOCK_METHOD(void, IncrementKeyValueAsync,
              (const IncrementKeyValueRequest*, IncrementKeyValueResponse*,
               StatusCallback),
              (override));
  MOCK_METHOD(void, InsertKeyValueAsync,
              (const InsertKeyValueRequest*, InsertKeyValueResponse*,
               StatusCallback),
              (override));
  MOCK_METHOD(void, DeleteKeyValueAsync,
              (const DeleteKeyValueRequest*, DeleteKeyValueResponse*,
               StatusCallback),
              (override));
  // Dummy methods to implement API.
  void GetKeyValueDirAsync(const tsl::GetKeyValueDirRequest* request,
                           tsl::GetKeyValueDirResponse* response,
                           StatusCallback done) override {
    done(absl::UnimplementedError("GetKeyValueDirAsync"));
  }
  void ResetTaskAsync(const tsl::ResetTaskRequest* request,
                      tsl::ResetTaskResponse* response,
                      StatusCallback done) override {
    done(absl::UnimplementedError("ResetTaskAsync"));
  }
  void ReportErrorToServiceAsync(
      const tsl::ReportErrorToServiceRequest* request,
      tsl::ReportErrorToServiceResponse* response,
      StatusCallback done) override {
    done(absl::UnimplementedError("ReportErrorToServiceAsync"));
  }
  void BarrierAsync(CallOptions* call_opts, const tsl::BarrierRequest* request,
                    tsl::BarrierResponse* response,
                    StatusCallback done) override {
    done(absl::UnimplementedError("BarrierAsync"));
  }
  void GetTaskStateAsync(const tsl::GetTaskStateRequest* request,
                         tsl::GetTaskStateResponse* response,
                         StatusCallback done) override {
    done(absl::UnimplementedError("GetTaskStateAsync"));
  }
  void WatchJobStateAsync(tsl::CallOptions*,
                          const tsl::WatchJobStateRequest* request,
                          tsl::WatchJobStateResponse* response,
                          StatusCallback done) override {
    done(absl::UnimplementedError("WatchJobStateAsync"));
  }
  void WaitForAllTasksAsync(const tsl::WaitForAllTasksRequest* request,
                            tsl::WaitForAllTasksResponse* response,
                            StatusCallback done) override {
    done(absl::UnimplementedError("WaitForAllTasksAsync"));
  }
  void CancelBarrierAsync(const tsl::CancelBarrierRequest* request,
                          tsl::CancelBarrierResponse* response,
                          StatusCallback done) override {
    done(absl::UnimplementedError("CancelBarrierAsync"));
  }
  void GetAliveTasksAsync(const tsl::GetAliveTasksRequest* request,
                          tsl::GetAliveTasksResponse* response,
                          StatusCallback done) override {
    done(absl::UnimplementedError("GetAliveTasksAsync"));
  }
  void RegisterTaskAsync(tsl::CallOptions*,
                         const tsl::RegisterTaskRequest* request,
                         tsl::RegisterTaskResponse* response,
                         StatusCallback done) override {
    done(absl::UnimplementedError("RegisterTaskAsync"));
  }
  void ShutdownTaskAsync(tsl::CallOptions*,
                         const tsl::ShutdownTaskRequest* request,
                         tsl::ShutdownTaskResponse* response,
                         StatusCallback done) override {
    done(absl::UnimplementedError("ShutdownTaskAsync"));
  }
  void HeartbeatAsync(tsl::CallOptions*, const tsl::HeartbeatRequest* request,
                      tsl::HeartbeatResponse* response,
                      StatusCallback done) override {
    done(absl::UnimplementedError("HeartbeatAsync"));
  }
  void ReportErrorToTaskAsync(CallOptions* call_opts,
                              const ReportErrorToTaskRequest* request,
                              ReportErrorToTaskResponse* response,
                              StatusCallback done) override {
    done(absl::UnimplementedError("ReportErrorToTaskAsync"));
  }
  void PollForErrorAsync(CallOptions* call_opts,
                         const PollForErrorRequest* request,
                         PollForErrorResponse* response,
                         StatusCallback done) override {
    done(absl::UnimplementedError("PollForErrorAsync"));
  }
};

class CPluginCoordinationServiceAgentTest : public ::testing::Test {
 public:
  // Should be called after mocking service responses, before testing the agent.
  void InitializeAgent(CoordinationServiceConfig config = {}) {
    config.set_service_leader("test_leader");
    TF_ASSERT_OK(impl_->Initialize(
        tsl::Env::Default(), /*job_name=*/"test_job",
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
  std::unique_ptr<CoordinationServiceAgent> impl_ =
      tsl::CreateCoordinationServiceAgent();
  std::unique_ptr<CPluginCoordinationServiceAgent> agent_ =
      std::make_unique<CPluginCoordinationServiceAgent>(impl_.get());
  std::unique_ptr<TestCoordinationClient> client_ =
      std::make_unique<TestCoordinationClient>();
};

TEST_F(CPluginCoordinationServiceAgentTest, GetKeyValue_Simple_Success) {
  const std::string test_key = "test_key";
  const std::string test_value = "test_value";
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

TEST_F(CPluginCoordinationServiceAgentTest, GetKeyValue_WithTimeout_Success) {
  const std::string test_key = "test_key";
  const std::string test_value = "test_value";
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

TEST_F(CPluginCoordinationServiceAgentTest, GetKeyValue_Timeout_ReturnError) {
  const std::string test_key = "test_key";
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
  owned_done(absl::CancelledError("error"));
}

TEST_F(CPluginCoordinationServiceAgentTest,
       GetKeyValue_ZeroTimeout_ReturnError) {
  const std::string test_key = "test_key";
  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::ZeroDuration());

  EXPECT_EQ(result.status().code(), error::INVALID_ARGUMENT);
}

TEST_F(CPluginCoordinationServiceAgentTest,
       GetKeyValue_NegativeTimeout_ReturnError) {
  const std::string test_key = "test_key";
  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(-1));

  EXPECT_EQ(result.status().code(), error::INVALID_ARGUMENT);
}

TEST_F(CPluginCoordinationServiceAgentTest, InsertKeyValue_Success) {
  const std::string test_key = "test_key";
  const std::string test_value = "test_value";
  InsertKeyValueRequest expected_input;
  auto kv = expected_input.mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);

  EXPECT_CALL(*GetClient(),
              InsertKeyValueAsync(Pointee(EqualsProto(expected_input)), _, _))
      .WillOnce(InvokeArgument<2>(absl::OkStatus()));
  InitializeAgent();

  TF_ASSERT_OK(agent_->InsertKeyValue(test_key, test_value));
}

TEST_F(CPluginCoordinationServiceAgentTest, DeleteKeyValue_Success) {
  const std::string test_key = "test_x_key";
  DeleteKeyValueRequest expected_input;
  expected_input.set_key(test_key);
  expected_input.set_is_directory(true);  // This is default.

  EXPECT_CALL(*GetClient(),
              DeleteKeyValueAsync(Pointee(EqualsProto(expected_input)), _, _))
      .WillOnce(InvokeArgument<2>(absl::OkStatus()));
  InitializeAgent();

  TF_ASSERT_OK(agent_->DeleteKeyValue(test_key));
}

TEST_F(CPluginCoordinationServiceAgentTest, TryGetKeyValue_Simple_Success) {
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
}  // namespace
}  // namespace tensorflow
