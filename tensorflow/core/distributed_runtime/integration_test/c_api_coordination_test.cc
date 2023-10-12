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

#include <memory>
#include <string>

#include "absl/time/time.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/protobuf/coordination_config.pb.h"

namespace tensorflow {
namespace {

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
}

string SetConfigKeyValueFn() {
  FunctionDef fdef;
  tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'SetConfigKeyValueFn'"
      "      input_arg {"
      "        name: 'config_key'"
      "        type: DT_STRING"
      "      }"
      "      input_arg {"
      "        name: 'config_value'"
      "        type: DT_STRING"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'set0'"
      "      op: 'TestSetConfigKeyValue'"
      "      input: 'config_key'"
      "      input: 'config_value'"
      "    }"
      "    ret {"
      "    }",
      &fdef);
  return fdef.SerializeAsString();
}

string GetConfigKeyValueFn() {
  FunctionDef fdef;
  tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'GetConfigKeyValueFn'"
      "      input_arg {"
      "        name: 'config_key'"
      "        type: DT_STRING"
      "      }"
      "      output_arg {"
      "        name: 'config_value'"
      "        type: DT_STRING"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'get0'"
      "      op: 'TestGetConfigKeyValue'"
      "      input: 'config_key'"
      "    }"
      "    ret {"
      "      key: 'config_value'"
      "      value: 'get0:value:0'"
      "    }",
      &fdef);
  return fdef.SerializeAsString();
}

TEST(CAPI, MultiClientCoordinationService) {
  const int cluster_size = 3;
  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size);
  // Agent needs to be destroyed without shutdown to simulate network failure,
  // which would trigger stale heartbeat detection on the service-side.
  ConfigCoordinationService(&server_def,
                            /*agent_destruction_without_shutdown=*/true);
  auto worker_thread_fn = [&](int worker_id) {
    tensorflow::ServerDef server_def_copy = server_def;
    // By default, server_def has task index set to 0.
    server_def_copy.set_task_index(worker_id);
    std::string serialized = server_def_copy.SerializeAsString();

    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetAsync(opts,
                               static_cast<unsigned char>(/*enable=*/true));
    TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                               TFE_DEVICE_PLACEMENT_SILENT);
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    // Normal execution: all cluster members are online.
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    // Sleep for 10 seconds and run collective ops on cluster except worker/1.
    // Since worker/1 thread directly exits here, its heartbeat will expire,
    // leading to UnavailableError on leader and then propagate to all other
    // members in cluster.
    if (worker_id != 1) {
      // Wait for 10 seconds, during this period of time worker/1 exits and
      // its heartbeat will expire.
      std::this_thread::sleep_for(std::chrono::seconds(10));
      TFE_TensorHandle* in = TestMatrixTensorHandle(ctx);
      TFE_Op* allreduce = AllReduceOp(ctx, in, cluster_size);
      TFE_TensorHandle* retvals[1];
      int num_retvals = 1;
      TFE_Execute(allreduce, &retvals[0], &num_retvals, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

      TFE_DeleteTensorHandle(in);
      TFE_DeleteTensorHandle(retvals[0]);
      TFE_DeleteOp(allreduce);

      // Since we created async executor, op status is eventually reported at
      // the sync barrier.
      TFE_ExecutorWaitForAllPendingNodes(executor, status);
      ASSERT_EQ(TF_UNAVAILABLE, TF_GetCode(status)) << TF_Message(status);
    }
    TFE_DeleteExecutor(executor);
    TFE_DeleteContext(ctx);
    TF_DeleteStatus(status);
  };
  std::thread thread_worker1([&] { worker_thread_fn(0); });
  std::thread thread_worker2([&] { worker_thread_fn(1); });
  std::thread thread_worker3([&] { worker_thread_fn(2); });
  thread_worker1.join();
  thread_worker2.join();
  thread_worker3.join();
}

TEST(CAPI, MultiClientSetGetConfigInOp) {
  const int cluster_size = 3;
  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size);
  ConfigCoordinationService(&server_def);
  BlockingCounter finish_counter(cluster_size);
  auto worker_thread_fn = [&](int worker_id) {
    tensorflow::ServerDef server_def_copy = server_def;
    // By default, server_def has task index set to 0.
    server_def_copy.set_task_index(worker_id);
    std::string serialized = server_def_copy.SerializeAsString();

    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetAsync(opts,
                               static_cast<unsigned char>(/*enable=*/true));
    TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                               TFE_DEVICE_PLACEMENT_SILENT);
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    TFE_Op* set_op = TFE_NewOp(ctx, "TestSetConfigKeyValue", status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_TensorHandle* my_key = TestScalarTensorHandle(
        ctx, tstring(strings::StrCat("worker_", worker_id)));
    TFE_OpAddInput(set_op, my_key, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_TensorHandle* my_val = TestScalarTensorHandle(
        ctx, tstring(strings::StrCat("value_", worker_id)));
    TFE_OpAddInput(set_op, my_val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    int num_retvals = 0;
    TFE_Execute(set_op, nullptr, &num_retvals, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteTensorHandle(my_key);
    TFE_DeleteTensorHandle(my_val);
    TFE_DeleteOp(set_op);

    TFE_Op* get_op = TFE_NewOp(ctx, "TestGetConfigKeyValue", status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_TensorHandle* next_key = TestScalarTensorHandle(
        ctx,
        tstring(strings::StrCat("worker_", (worker_id + 1) % cluster_size)));
    TFE_OpAddInput(get_op, next_key, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    TFE_TensorHandle* retvals[1];
    num_retvals = 1;
    TFE_Execute(get_op, retvals, &num_retvals, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    const tstring& next_val = *static_cast<tstring*>(TF_TensorData(t));
    const tstring& expected_val =
        tstring(strings::StrCat("value_", (worker_id + 1) % cluster_size));
    EXPECT_EQ(next_val, expected_val) << strings::StrCat(
        "Expecting value ", expected_val, ", but got ", next_val);

    TFE_DeleteTensorHandle(next_key);
    TFE_DeleteTensorHandle(retvals[0]);
    TF_DeleteTensor(t);
    TFE_DeleteOp(get_op);

    // Since we created async executor, op status is eventually reported at
    // the sync barrier.
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TF_DeleteStatus(status);
    finish_counter.DecrementCount();
    finish_counter.Wait();
    TFE_DeleteExecutor(executor);
    TFE_DeleteContext(ctx);
  };
  std::thread thread_worker1([&] { worker_thread_fn(0); });
  std::thread thread_worker2([&] { worker_thread_fn(1); });
  std::thread thread_worker3([&] { worker_thread_fn(2); });
  thread_worker1.join();
  thread_worker2.join();
  thread_worker3.join();
}

TEST(CAPI, MultiClientCoordinationSetGetConfigs) {
  const int cluster_size = 3;
  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size);
  ConfigCoordinationService(&server_def);
  tensorflow::BlockingCounter counter1(cluster_size);
  tensorflow::BlockingCounter counter2(cluster_size);
  tensorflow::BlockingCounter counter3(cluster_size);

  auto worker_thread_fn = [&](int worker_id) {
    tensorflow::ServerDef server_def_copy = server_def;
    // By default, server_def has task index set to 0.
    server_def_copy.set_task_index(worker_id);
    std::string serialized = server_def_copy.SerializeAsString();

    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetAsync(opts,
                               static_cast<unsigned char>(/*enable=*/true));
    TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                               TFE_DEVICE_PLACEMENT_SILENT);
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    // For each worker i, set (keyi, valuei)
    const std::string& key = tensorflow::strings::StrCat("key", worker_id);
    TFE_InsertConfigKeyValue(
        ctx, key.c_str(),
        tensorflow::strings::StrCat("value", worker_id).c_str(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    counter1.DecrementCount();
    counter1.Wait();

    const int next_id = (worker_id + 1) % cluster_size;
    // Setting next_key errors out because it has been set by another worker
    const std::string& next_key = tensorflow::strings::StrCat("key", next_id);
    TFE_InsertConfigKeyValue(ctx, next_key.c_str(), "some_value", status);
    EXPECT_EQ(TF_ALREADY_EXISTS, TF_GetCode(status)) << TF_Message(status);
    // Getting next_key returns the value set by another worker
    TF_Buffer* value_buf = TF_NewBuffer();
    TFE_GetConfigKeyValue(ctx, next_key.c_str(), /*timeout_in_ms=*/5000,
                          value_buf, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    std::string value_str{static_cast<const char*>(value_buf->data),
                          value_buf->length};
    EXPECT_EQ(value_str, tensorflow::strings::StrCat("value", next_id));
    TF_DeleteBuffer(value_buf);
    counter2.DecrementCount();
    counter2.Wait();

    // Delete key
    TFE_DeleteConfigKeyValue(ctx, key.c_str(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    counter3.DecrementCount();
    counter3.Wait();

    TFE_DeleteContext(ctx);
    TF_DeleteStatus(status);
  };
  std::thread thread_worker1([&] { worker_thread_fn(0); });
  std::thread thread_worker2([&] { worker_thread_fn(1); });
  std::thread thread_worker3([&] { worker_thread_fn(2); });
  thread_worker1.join();
  thread_worker2.join();
  thread_worker3.join();
}

TEST(CAPI, MultiClientPropagateError) {
  const int cluster_size = 3;
  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size);
  ConfigCoordinationService(&server_def);
  // Barrier for initializing the cluster.
  tensorflow::BlockingCounter counter1(cluster_size);
  // Barrier for finishing executing operations on all workers.
  tensorflow::BlockingCounter counter2(cluster_size);

  auto worker_thread_fn = [&](int worker_id) {
    tensorflow::ServerDef server_def_copy = server_def;
    // By default, server_def has task index set to 0.
    server_def_copy.set_task_index(worker_id);
    std::string serialized = server_def_copy.SerializeAsString();

    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetAsync(opts,
                               static_cast<unsigned char>(/*enable=*/false));
    TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                               TFE_DEVICE_PLACEMENT_SILENT);
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    counter1.DecrementCount();
    counter1.Wait();

    // Set error from worker/1
    if (worker_id == 1) {
      TFE_ReportErrorToCluster(ctx, TF_INVALID_ARGUMENT, "my_error", status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    }

    // Run collective on all workers. The collective will not finish because
    // worker/1 already in error status. Check that all workers get the same
    // error reported from running the collective ops.
    TFE_TensorHandle* in = TestMatrixTensorHandle(ctx);
    TFE_Op* allreduce = AllReduceOp(ctx, in, cluster_size);
    TFE_TensorHandle* retvals[1];
    int num_retvals = 1;
    TFE_Execute(allreduce, &retvals[0], &num_retvals, status);
    EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status)) << TF_Message(status);

    TFE_DeleteTensorHandle(in);
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_DeleteOp(allreduce);
    counter2.DecrementCount();
    counter2.Wait();

    TFE_DeleteContext(ctx);
    TF_DeleteStatus(status);
  };
  std::thread thread_worker1([&] { worker_thread_fn(0); });
  std::thread thread_worker2([&] { worker_thread_fn(1); });
  std::thread thread_worker3([&] { worker_thread_fn(2); });
  thread_worker1.join();
  thread_worker2.join();
  thread_worker3.join();
}

class SingleClientCoordinationServiceTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<bool> {};

TEST_P(SingleClientCoordinationServiceTest, TestSetGetConfigInOp) {
  const bool use_worker0_as_client = GetParam();
  tensorflow::ServerDef server_def = GetServerDef("worker", 3);
  const char task0_name[] = "/job:worker/replica:0/task:0/device:CPU:0";
  const char task1_name[] = "/job:worker/replica:0/task:1/device:CPU:0";
  const char task2_name[] = "/job:worker/replica:0/task:2/device:CPU:0";

  ConfigCoordinationService(&server_def);
  ServerFactory* factory;
  ASSERT_TRUE(ServerFactory::GetFactory(server_def, &factory).ok());
  server_def.set_job_name("worker");
  server_def.set_task_index(0);
  std::unique_ptr<tensorflow::ServerInterface> w0;
  if (!use_worker0_as_client) {
    // Start a separate server for worker0 if it's not used as the client
    ASSERT_TRUE(
        factory->NewServer(server_def, ServerFactory::Options(), &w0).ok());
    ASSERT_TRUE(w0->Start().ok());
  }
  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::ServerInterface> w1;
  ASSERT_TRUE(
      factory->NewServer(server_def, ServerFactory::Options(), &w1).ok());
  ASSERT_TRUE(w1->Start().ok());
  server_def.set_task_index(2);
  std::unique_ptr<tensorflow::ServerInterface> w2;
  ASSERT_TRUE(
      factory->NewServer(server_def, ServerFactory::Options(), &w2).ok());
  ASSERT_TRUE(w2->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(/*enable=*/true));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  server_def.set_task_index(0);
  if (!use_worker0_as_client) {
    // Add localhost job for the remote client task
    auto cluster = server_def.mutable_cluster();
    auto client_job = cluster->add_job();
    client_job->set_name("localhost");
    const int client_port = tensorflow::testing::PickUnusedPortOrDie();
    client_job->mutable_tasks()->insert(
        {0, strings::StrCat("localhost:", client_port)});
    server_def.set_job_name("localhost");
  }
  server_def.mutable_default_session_config()
      ->mutable_experimental()
      ->mutable_coordination_config()
      ->set_service_leader(task0_name);
  const std::string serialized = server_def.SerializeAsString();

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  // Try to get value of a nonexistent key.
  TFE_Op* get_op = TFE_NewOp(ctx, "TestGetConfigKeyValue", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* get_key = TestScalarTensorHandle(ctx, tstring("test_key"));
  TFE_OpAddInput(get_op, get_key, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  // This get op is non-blocking.
  TFE_OpSetAttrBool(get_op, "blocking", false);
  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  // Run get op from task2.
  TFE_OpSetDevice(get_op, task2_name, status);
  // Since we are using async executor, TFE_Execute only returns the enqueue
  // status, and TFE_ExecutorWaitForAllPendingNodes will return the real error.
  TFE_Execute(get_op, retvals, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  EXPECT_EQ(TF_NOT_FOUND, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteTensorHandle(retvals[0]);
  TFE_DeleteOp(get_op);
  // Reset executor and status.
  TFE_ExecutorClearError(executor);
  TF_SetStatus(status, TF_OK, "");

  TFE_Op* set_op = TFE_NewOp(ctx, "TestSetConfigKeyValue", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* set_key = TestScalarTensorHandle(ctx, tstring("test_key"));
  TFE_OpAddInput(set_op, set_key, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* set_val = TestScalarTensorHandle(ctx, tstring("test_val"));
  TFE_OpAddInput(set_op, set_val, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  // Run set op from task1
  TFE_OpSetDevice(set_op, task1_name, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  num_retvals = 0;
  TFE_Execute(set_op, nullptr, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteTensorHandle(set_key);
  TFE_DeleteTensorHandle(set_val);
  TFE_DeleteOp(set_op);

  TFE_Op* get_op2 = TFE_NewOp(ctx, "TestGetConfigKeyValue", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(get_op2, get_key, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpSetAttrBool(get_op2, "blocking", true);
  num_retvals = 1;
  // Run get op from task2
  TFE_OpSetDevice(get_op2, task2_name, status);
  TFE_Execute(get_op2, retvals, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  const tstring& get_val = *static_cast<tstring*>(TF_TensorData(t));
  EXPECT_EQ(get_val, "test_val")
      << strings::StrCat("Expecting value test_val but got ", get_val);
  TFE_DeleteTensorHandle(get_key);
  TFE_DeleteTensorHandle(retvals[0]);
  TF_DeleteTensor(t);
  TFE_DeleteOp(get_op2);

  const string& set_fdef = SetConfigKeyValueFn();
  TFE_ContextAddFunctionDef(ctx, set_fdef.data(), set_fdef.size(), status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_Op* set_fn = TFE_NewOp(ctx, "SetConfigKeyValueFn", status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  set_key = TestScalarTensorHandle(ctx, tstring("test_fn_key"));
  TFE_OpAddInput(set_fn, set_key, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  set_val = TestScalarTensorHandle(ctx, tstring("test_fn_val"));
  TFE_OpAddInput(set_fn, set_val, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  // Run set fn on task2
  TFE_OpSetDevice(set_fn, task2_name, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  num_retvals = 0;
  TFE_Execute(set_fn, nullptr, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteTensorHandle(set_key);
  TFE_DeleteTensorHandle(set_val);
  TFE_DeleteOp(set_fn);

  const string& get_fdef = GetConfigKeyValueFn();
  TFE_ContextAddFunctionDef(ctx, get_fdef.data(), get_fdef.size(), status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_Op* get_fn = TFE_NewOp(ctx, "GetConfigKeyValueFn", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  get_key = TestScalarTensorHandle(ctx, tstring("test_fn_key"));
  TFE_OpAddInput(get_fn, get_key, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_TensorHandle* fn_retvals[1];
  num_retvals = 1;
  // Run get fn on task1
  TFE_OpSetDevice(get_fn, task2_name, status);
  TFE_Execute(get_fn, fn_retvals, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  t = TFE_TensorHandleResolve(fn_retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  const tstring& get_fn_val = *static_cast<tstring*>(TF_TensorData(t));
  EXPECT_EQ(get_fn_val, "test_fn_val")
      << strings::StrCat("Expecting value test_fn_val but got ", get_fn_val);
  TFE_DeleteTensorHandle(get_key);
  TFE_DeleteTensorHandle(fn_retvals[0]);
  TF_DeleteTensor(t);
  TFE_DeleteOp(get_fn);

  // Since we created async executor, op status is eventually reported at
  // the sync barrier.
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_DeleteExecutor(executor);
  TFE_DeleteContext(ctx);

  // Grpc servers do not support clean down.
  w0.release();
  w1.release();
  w2.release();
}

INSTANTIATE_TEST_SUITE_P(CAPI, SingleClientCoordinationServiceTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> arg) {
                           return arg.param ? "use_worker0_as_client"
                                            : "use_remote_client";
                         });

}  // namespace
}  // namespace tensorflow
