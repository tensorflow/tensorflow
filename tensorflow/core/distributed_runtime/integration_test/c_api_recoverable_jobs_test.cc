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

#include <memory>
#include <string>
#include <utility>

#include "absl/time/time.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {
namespace {

// TODO(b/249134783): Put the below into a common util.

constexpr char kCoordinationServiceType[] = "standalone";

void ConfigCoordinationService(tensorflow::ServerDef* server_def,
                               bool agent_destruction_without_shutdown = false,
                               bool enable_health_check = false) {
  // Set the number of threads here since in some environment the default number
  // of threads may be small which could cause RPC to hang.
  server_def->mutable_default_session_config()
      ->set_inter_op_parallelism_threads(10);
  auto coord_config = server_def->mutable_default_session_config()
                          ->mutable_experimental()
                          ->mutable_coordination_config();
  coord_config->set_service_type(kCoordinationServiceType);
  coord_config->set_service_leader("/job:worker/replica:0/task:0");
  coord_config->set_agent_destruction_without_shutdown(
      agent_destruction_without_shutdown);
  coord_config->set_heartbeat_timeout_in_ms(
      absl::ToInt64Milliseconds(absl::Seconds(5)));
  coord_config->set_shutdown_barrier_timeout_in_ms(
      absl::ToInt64Milliseconds(absl::Seconds(5)));
  coord_config->set_enable_health_check(enable_health_check);
  // Allow restarted clients to reconnect.
  coord_config->set_allow_new_incarnation_to_reconnect(true);
}

class SingleClientRecoverableJobsTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<bool> {};

TEST_P(SingleClientRecoverableJobsTest, TestRecoverWorkerFailure) {
  // If the param is true, it means client is restarted.
  const bool client_restart = GetParam();
  tensorflow::ServerDef server_def = GetServerDef("worker", 2);
  const char task0_name[] = "/job:worker/replica:0/task:0/device:CPU:0";
  const char task1_name[] = "/job:worker/replica:0/task:1/device:CPU:0";

  ConfigCoordinationService(&server_def,
                            /*agent_destruction_without_shutdown=*/false,
                            /*enable_health_check=*/true);
  ServerFactory* factory;
  ASSERT_TRUE(ServerFactory::GetFactory(server_def, &factory).ok());
  server_def.set_job_name("worker");
  server_def.set_task_index(0);
  std::unique_ptr<tensorflow::ServerInterface> w0;
  ASSERT_TRUE(
      factory->NewServer(server_def, ServerFactory::Options(), &w0).ok());
  ASSERT_TRUE(w0->Start().ok());
  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::ServerInterface> w1;
  ASSERT_TRUE(
      factory->NewServer(server_def, ServerFactory::Options(), &w1).ok());
  ASSERT_TRUE(w1->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  server_def.set_task_index(0);
  // Add localhost job for the remote client task
  auto cluster = server_def.mutable_cluster();
  auto client_job = cluster->add_job();
  client_job->set_name("localhost");
  int client_port = tensorflow::testing::PickUnusedPortOrDie();
  client_job->mutable_tasks()->insert(
      {0, absl::StrCat("localhost:", client_port)});
  server_def.set_job_name("localhost");
  std::string serialized = server_def.SerializeAsString();

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_Op* report_op = TFE_NewOp(ctx, "TestReportErrorToCluster", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* error_code =
      TestScalarTensorHandle(ctx, error::Code::UNAVAILABLE);
  TFE_OpAddInput(report_op, error_code, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* error_message =
      TestScalarTensorHandle(ctx, tstring("test_error_message"));
  TFE_OpAddInput(report_op, error_message, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Run report op from task1. The error would not propagate.
  TFE_OpSetDevice(report_op, task1_name, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  int num_retvals = 0;
  TFE_Execute(report_op, nullptr, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  if (client_restart) {
    // Run report op from task0. This should succeed since task0 is not in error
    // state.
    TFE_OpSetDevice(report_op, task0_name, status);
    ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
    TFE_Execute(report_op, nullptr, &num_retvals, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }

  TFE_ContextAsyncWait(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_DeleteTensorHandle(error_code);
  TFE_DeleteTensorHandle(error_message);
  TFE_DeleteOp(report_op);

  TFE_ContextAsyncWait(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  if (client_restart) {
    TFE_DeleteContext(ctx);

    // Pick a new address for the client.
    client_port = tensorflow::testing::PickUnusedPortOrDie();
    auto& jobs = *server_def.mutable_cluster()->mutable_job();
    for (auto& job : jobs) {
      if (job.name() == "localhost") {
        job.mutable_tasks()->clear();
        job.mutable_tasks()->insert(
            {0, absl::StrCat("localhost:", client_port)});
        break;
      }
    }

    ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    serialized = server_def.SerializeAsString();
    TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(),
                            status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  } else {
    TF_EXPECT_OK(w1->StopCoordinationService());
    w1.release();

    // Pick a new address for task1 and clear the client job for ServerDef.
    // Otherwise we cannot restart the server.
    auto& jobs = *server_def.mutable_cluster()->mutable_job();
    tensorflow::JobDef saved_client_job;
    for (auto iter = jobs.begin(); iter != jobs.end();) {
      if (iter->name() == "worker") {
        int worker_1_port = tensorflow::testing::PickUnusedPortOrDie();
        auto& tasks = *iter->mutable_tasks();
        tasks[1] = absl::StrCat("localhost:", worker_1_port);
        ++iter;
      } else if (iter->name() == "localhost") {
        saved_client_job = *iter;
        iter = jobs.erase(iter);
      }
    }

    server_def.set_job_name("worker");
    server_def.set_task_index(0);
    ASSERT_TRUE(ServerFactory::GetFactory(server_def, &factory).ok());
    server_def.set_task_index(1);
    ASSERT_TRUE(
        factory->NewServer(server_def, ServerFactory::Options(), &w1).ok());
    ASSERT_TRUE(w1->Start().ok());

    server_def.set_job_name("localhost");
    server_def.set_task_index(0);
    cluster = server_def.mutable_cluster();
    *cluster->add_job() = std::move(saved_client_job);

    TFE_DeleteContextOptions(opts);
    serialized = server_def.SerializeAsString();
    TFE_ContextUpdateServerDef(ctx, 0, serialized.data(), serialized.size(),
                               status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  }

  // Report error from task 1 again. It should succeed since task 1 is
  // healthy.
  report_op = TFE_NewOp(ctx, "TestReportErrorToCluster", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  error_code = TestScalarTensorHandle(ctx, error::Code::UNAVAILABLE);
  TFE_OpAddInput(report_op, error_code, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  error_message = TestScalarTensorHandle(ctx, tstring("test_error_message"));
  TFE_OpAddInput(report_op, error_message, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_OpSetDevice(report_op, task1_name, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_Execute(report_op, nullptr, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_ContextAsyncWait(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_DeleteTensorHandle(error_code);
  TFE_DeleteTensorHandle(error_message);
  TFE_DeleteOp(report_op);

  TFE_ContextAsyncWait(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_DeleteStatus(status);
  TFE_DeleteContext(ctx);

  TF_EXPECT_OK(w0->StopCoordinationService());
  TF_EXPECT_OK(w1->StopCoordinationService());
  w0.release();
  w1.release();
}

INSTANTIATE_TEST_SUITE_P(CAPI, SingleClientRecoverableJobsTest,
                         ::testing::Bool());

}  // namespace
}  // namespace tensorflow
