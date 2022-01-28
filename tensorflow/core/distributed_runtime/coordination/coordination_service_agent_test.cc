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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"

namespace tensorflow {
namespace {
using ::testing::_;
using ::testing::DoAll;
using ::testing::InvokeArgument;
using ::testing::SetArgPointee;
using ::testing::WithArgs;

class TestCoordinationClient : public CoordinationClient {
 public:
  TestCoordinationClient() = default;
  // MOCK_METHOD does not work on Windows build, using deprecated MOCK_METHOD3
  // instead.
  MOCK_METHOD3(GetKeyValueAsync, void(const GetKeyValueRequest*,
                                      GetKeyValueResponse*, StatusCallback));

  void RegisterWorkerAsync(CallOptions* opts,
                           const RegisterWorkerRequest* request,
                           RegisterWorkerResponse* response,
                           StatusCallback done) override {
    done(errors::Unimplemented("RegisterWorkerAsync"));
  }

#define UNIMPLEMENTED(method)                                         \
  void method##Async(const method##Request* request,                  \
                     method##Response* response, StatusCallback done) \
      override {                                                      \
    done(errors::Unimplemented(#method "Async"));                     \
  }

  UNIMPLEMENTED(Heartbeat);
  UNIMPLEMENTED(WaitForAllTasks);
  UNIMPLEMENTED(ReportErrorToAgent);
  UNIMPLEMENTED(ReportErrorToService);
  UNIMPLEMENTED(InsertKeyValue);
  UNIMPLEMENTED(DeleteKeyValue);

#undef UNIMPLEMENTED
};

TEST(CoordinationServiceAgentTest, GetKeyValue_Simple_Success) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  auto client = std::make_unique<TestCoordinationClient>();
  // Mock server response: set key-value pair and invoke done callback.
  GetKeyValueResponse mocked_response;
  auto kv = mocked_response.mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  ON_CALL(*client, GetKeyValueAsync(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<1>(mocked_response),
                           InvokeArgument<2>(Status::OK())));
  // Initialize coordination agent.
  auto agent = CreateCoordinationServiceAgent();
  CoordinationServiceConfig config;
  config.set_service_leader("test_leader");
  TF_EXPECT_OK(agent->Initialize(
      /*env=*/nullptr, "test_job", 0, config, std::move(client),
      /*error_fn=*/[](Status s) {
        LOG(ERROR) << "Coordination agent is set to error: " << s;
      }));

  auto result = agent->GetKeyValue(test_key);

  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.ValueOrDie(), test_value);
}

TEST(CoordinationServiceAgentTest, GetKeyValue_WithTimeout_Success) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  auto client = std::make_unique<TestCoordinationClient>();
  // Mock server response: set key-value pair and invoke done callback.
  GetKeyValueResponse mocked_response;
  auto kv = mocked_response.mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  ON_CALL(*client, GetKeyValueAsync(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<1>(mocked_response),
                           InvokeArgument<2>(Status::OK())));
  // Initialize coordination agent.
  auto agent = CreateCoordinationServiceAgent();
  CoordinationServiceConfig config;
  config.set_service_leader("test_leader");
  TF_EXPECT_OK(agent->Initialize(
      /*env=*/nullptr, "test_job", 0, config, std::move(client),
      /*error_fn=*/[](Status s) {
        LOG(ERROR) << "Coordination agent is set to error: " << s;
      }));

  auto result = agent->GetKeyValue(test_key, /*timeout=*/absl::Seconds(10));

  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.ValueOrDie(), test_value);
}

TEST(CoordinationServiceAgentTest, GetKeyValue_Timeout_ReturnError) {
  const std::string& test_key = "test_key";
  // Initialize coordination service agent.
  auto client = std::make_unique<TestCoordinationClient>();
  auto agent = CreateCoordinationServiceAgent();
  CoordinationServiceConfig config;
  config.set_service_leader("test_leader");
  TF_EXPECT_OK(agent->Initialize(
      /*env=*/nullptr, "test_job", 0, config, std::move(client),
      /*error_fn=*/[](Status s) {
        LOG(ERROR) << "Coordination agent is set to error: " << s;
      }));

  auto result = agent->GetKeyValue(test_key, /*timeout=*/absl::Seconds(1));

  EXPECT_EQ(result.status().code(), error::DEADLINE_EXCEEDED);
}

TEST(CoordinationServiceAgentTest,
     GetKeyValue_DelayedResponse_TimeoutWithoutMemoryError) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  auto client = std::make_unique<TestCoordinationClient>();
  GetKeyValueResponse* owned_response;
  StatusCallback owned_done;
  ON_CALL(*client, GetKeyValueAsync(_, _, _))
      .WillByDefault(WithArgs<1, 2>(
          [&](GetKeyValueResponse* response, StatusCallback done) {
            // Copy method arguments to prevent de-allocation before mocking the
            // server callback beyond timeout.
            owned_response = response;
            owned_done = done;
          }));
  // Initialize coordination service agent.
  auto agent = CreateCoordinationServiceAgent();
  CoordinationServiceConfig config;
  config.set_service_leader("test_leader");
  TF_EXPECT_OK(agent->Initialize(
      /*env=*/nullptr, "test_job", 0, config, std::move(client),
      /*error_fn=*/[](Status s) {
        LOG(ERROR) << "Coordination agent is set to error: " << s;
      }));

  auto result = agent->GetKeyValue(test_key, /*timeout=*/absl::Seconds(3));
  EXPECT_EQ(result.status().code(), error::DEADLINE_EXCEEDED);

  // Delayed server response: set key-value response, and invoke done callback.
  auto kv = owned_response->mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  owned_done(Status::OK());
  // No explicit test, but used to verify there is no stack-use-after-return
  // or other memory-related errors.
}

TEST(CoordinationServiceAgentTest,
     GetKeyValue_DelayedResponseBeforeTimeout_Success) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  // Mock delayed server response before timeout: set key-value pair and invoke
  // done callback.
  auto client = std::make_unique<TestCoordinationClient>();
  std::unique_ptr<Thread> async_thread;
  GetKeyValueResponse* owned_response;
  StatusCallback owned_done;
  ON_CALL(*client, GetKeyValueAsync(_, _, _))
      // Setup async callback to insert key-value after a brief delay (5s)
      // before timeout (10s).
      .WillByDefault(WithArgs<1, 2>(
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
                  owned_done(Status::OK());
                }));
          }));
  auto agent = CreateCoordinationServiceAgent();
  CoordinationServiceConfig config;
  config.set_service_leader("test_leader");
  TF_EXPECT_OK(agent->Initialize(
      /*env=*/nullptr, "test_job", 0, config, std::move(client),
      /*error_fn=*/[](Status s) {
        LOG(ERROR) << "Coordination agent is set to error: " << s;
      }));

  auto result = agent->GetKeyValue(test_key, /*timeout=*/absl::Seconds(10));

  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.ValueOrDie(), test_value);
}
}  // namespace
}  // namespace tensorflow
