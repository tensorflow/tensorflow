/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace {

using ::tensorflow::string;

tensorflow::ServerDef GetServerDef(const string& job_name, int num_tasks) {
  tensorflow::ServerDef server_def;
  server_def.set_protocol("grpc");
  server_def.set_job_name(job_name);
  server_def.set_task_index(0);
  tensorflow::ClusterDef* cluster_def = server_def.mutable_cluster();
  tensorflow::JobDef* job_def = cluster_def->add_job();
  job_def->set_name(job_name);
  for (int i = 0; i < num_tasks; i++) {
    int port = tensorflow::testing::PickUnusedPortOrDie();
    job_def->mutable_tasks()->insert(
        {i, tensorflow::strings::StrCat("localhost:", port)});
  }
  return server_def;
}

tensorflow::ServerDef GetServerDef(int num_tasks) {
  return GetServerDef("localhost", num_tasks);
}

void TestRemoteExecute(bool async) {
  tensorflow::ServerDef server_def = GetServerDef(2);

  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);

  std::unique_ptr<tensorflow::GrpcServer> worker_server;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server)
                  .ok());
  ASSERT_TRUE(worker_server->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                             TFE_DEVICE_PLACEMENT_EXPLICIT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* h0_task0 = TestMatrixTensorHandle(ctx);
  TFE_TensorHandle* h1_task0 = TestMatrixTensorHandle(ctx);
  const char remote_device_name[] =
      "/job:localhost/replica:0/task:1/device:CPU:0";
  auto* h0_task1 =
      TFE_TensorHandleCopyToDevice(h0_task0, ctx, remote_device_name, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  auto* h1_task1 =
      TFE_TensorHandleCopyToDevice(h1_task0, ctx, remote_device_name, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_Op* matmul = MatMulOp(ctx, h0_task1, h1_task1);
  TFE_OpSetDevice(matmul, remote_device_name, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  float product[4] = {0};
  EXPECT_EQ(sizeof(product), TF_TensorByteSize(t));
  memcpy(&product[0], TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);
  EXPECT_EQ(7, product[0]);
  EXPECT_EQ(10, product[1]);
  EXPECT_EQ(15, product[2]);
  EXPECT_EQ(22, product[3]);

  TFE_DeleteTensorHandle(h0_task0);
  TFE_DeleteTensorHandle(h1_task0);
  TFE_DeleteTensorHandle(h0_task1);
  TFE_DeleteTensorHandle(h1_task1);
  TFE_DeleteTensorHandle(retvals[0]);

  TFE_DeleteOp(matmul);

  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteExecutor(executor);
  TFE_DeleteContext(ctx);

  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
}

TEST(CAPI, RemoteExecute) { TestRemoteExecute(false); }
TEST(CAPI, RemoteExecuteAsync) { TestRemoteExecute(true); }

string MatMulFunction() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'MatMulFunction'"
      "      input_arg {"
      "        name: 'a'"
      "        type: DT_FLOAT"
      "      }"
      "      input_arg {"
      "        name: 'b'"
      "        type: DT_FLOAT"
      "      }"
      "      output_arg {"
      "        name: 'm'"
      "        type: DT_FLOAT"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'matmul'"
      "      op: 'MatMul'"
      "      input: 'a'"
      "      input: 'b'"
      "      attr {"
      "        key: 'T'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    ret {"
      "      key: 'm'"
      "      value: 'matmul:product'"
      "    }",
      &def));
  return def.SerializeAsString();
}

// If heavy_load_on_streaming_rpc is true, send some rpc reqeusts before the one
// which creates a remote remote input, to simulate a scenario that the remote
// input is not ready when we start running an op or a function.
void TestRemoteExecuteSilentCopies(bool async, bool remote, bool func,
                                   bool heavy_load_on_streaming_rpc) {
  tensorflow::ServerDef server_def = GetServerDef(3);

  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::GrpcServer> worker_server1;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server1)
                  .ok());
  ASSERT_TRUE(worker_server1->Start().ok());

  server_def.set_task_index(2);
  std::unique_ptr<tensorflow::GrpcServer> worker_server2;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server2)
                  .ok());
  ASSERT_TRUE(worker_server2->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_TensorHandle* h0_task0 = TestMatrixTensorHandle(ctx);
  TFE_TensorHandle* h1_task0 = TestMatrixTensorHandle(ctx);
  std::vector<TFE_TensorHandle*> handles_task0;
  if (heavy_load_on_streaming_rpc) {
    // Send 50 tensor copy requests to simulate that there have been some RPC
    // requests been enqueued.
    for (int i = 0; i < 50; ++i) {
      handles_task0.push_back(TestMatrixTensorHandle(ctx));
    }
  }
  const char task1_name[] = "/job:localhost/replica:0/task:1/device:CPU:0";
  const char task2_name[] = "/job:localhost/replica:0/task:2/device:CPU:0";

  std::vector<TFE_TensorHandle*> handles_task2;
  for (auto* h_task0 : handles_task0) {
    handles_task2.push_back(
        TFE_TensorHandleCopyToDevice(h_task0, ctx, task2_name, status));
    ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  }

  auto* h1_task2 =
      TFE_TensorHandleCopyToDevice(h1_task0, ctx, task2_name, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_Op* matmul = nullptr;
  if (func) {
    string function_def = MatMulFunction();
    TFE_ContextAddFunctionDef(ctx, function_def.data(), function_def.size(),
                              status);
    CHECK_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

    matmul = TFE_NewOp(ctx, "MatMulFunction", status);
    ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
    TFE_OpAddInput(matmul, h0_task0, status);
    ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
    TFE_OpAddInput(matmul, h1_task2, status);
    ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  } else {
    // Handles are on task0 (local), and task2, but op is on task1.
    matmul = MatMulOp(ctx, h0_task0, h1_task2);
  }
  if (remote) {
    TFE_OpSetDevice(matmul, task1_name, status);
    ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  } else if (!async) {
    // Set the local device to CPU to easily validate mirroring
    string cpu_device_name;
    ASSERT_TRUE(GetDeviceName(ctx, &cpu_device_name, "CPU"));
    TFE_OpSetDevice(matmul, cpu_device_name.c_str(), status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
    auto remote_arg =
        tensorflow::TensorHandleFromInterface(tensorflow::unwrap(h1_task2));
    // The input handles should never change since they have been mirrored.
    ASSERT_FALSE(remote_arg->HasLocalMirror(nullptr));
  }

  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  // TODO(gjn): Add support for waiting on async local mirrors
  if (!remote && !async) {
    auto remote_arg =
        tensorflow::TensorHandleFromInterface(tensorflow::unwrap(h1_task2));
    // The input handles should never change since they have been mirrored.
    ASSERT_TRUE(remote_arg->HasLocalMirror(nullptr));
  }

  auto* retval_task0 = TFE_TensorHandleCopyToDevice(
      retvals[0], ctx, "/job:localhost/replica:0/task:0/device:CPU:0", status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TF_Tensor* t = TFE_TensorHandleResolve(retval_task0, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_DeleteTensorHandle(retval_task0);
  float product[4] = {0};
  EXPECT_EQ(sizeof(product), TF_TensorByteSize(t));
  memcpy(&product[0], TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);
  EXPECT_EQ(7, product[0]);
  EXPECT_EQ(10, product[1]);
  EXPECT_EQ(15, product[2]);
  EXPECT_EQ(22, product[3]);

  TFE_DeleteTensorHandle(h0_task0);
  TFE_DeleteTensorHandle(h1_task0);
  TFE_DeleteTensorHandle(h1_task2);
  TFE_DeleteTensorHandle(retvals[0]);
  for (auto* h : handles_task0) {
    TFE_DeleteTensorHandle(h);
  }
  for (auto* h : handles_task2) {
    TFE_DeleteTensorHandle(h);
  }

  TFE_DeleteOp(matmul);

  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_DeleteExecutor(executor);
  if (func) {
    TFE_ContextRemoveFunction(ctx, "MatMulFunction", status);
  }
  TFE_DeleteContext(ctx);

  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server1.release();
  worker_server2.release();
}

TEST(CAPI, RemoteExecuteSilentCopies) {
  TestRemoteExecuteSilentCopies(/*async=*/false, /*remote=*/true,
                                /*func=*/false,
                                /*heavy_load_on_streaming_rpc=*/false);
}
TEST(CAPI, RemoteExecuteSilentCopiesAsync) {
  TestRemoteExecuteSilentCopies(/*async=*/true, /*remote=*/true, /*func=*/false,
                                /*heavy_load_on_streaming_rpc=*/false);
}
TEST(CAPI, RemoteExecuteSilentCopiesAsyncFunc) {
  TestRemoteExecuteSilentCopies(/*async=*/true, /*remote=*/true, /*func=*/true,
                                /*heavy_load_on_streaming_rpc=*/false);
}
TEST(CAPI, RemoteExecuteSilentCopiesLocal) {
  TestRemoteExecuteSilentCopies(/*async=*/false, /*remote=*/false,
                                /*func=*/false,
                                /*heavy_load_on_streaming_rpc=*/false);
}
TEST(CAPI, RemoteExecuteSilentCopiesLocalAsync) {
  TestRemoteExecuteSilentCopies(/*async=*/true, /*remote=*/false,
                                /*func=*/false,
                                /*heavy_load_on_streaming_rpc=*/false);
}
TEST(CAPI, RemoteExecuteSilentCopiesLocalAsyncFunc) {
  TestRemoteExecuteSilentCopies(/*async=*/true, /*remote=*/false, /*func=*/true,
                                /*heavy_load_on_streaming_rpc=*/false);
}
TEST(CAPI, RemoteExecuteSilentCopiesLocalAsyncFuncOrdering) {
  // A remote input may be not ready when we start running a function. Test that
  // the function execution should wait until the remote input is ready.
  TestRemoteExecuteSilentCopies(/*async=*/true, /*remote=*/false, /*func=*/true,
                                /*heavy_load_on_streaming_rpc=*/true);
}

// Add the values of three variables on three different tasks.
string AddVariablesFunction() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'AddVariablesFunction'"
      "      input_arg {"
      "        name: 'var'"
      "        type: DT_RESOURCE"
      "      }"
      "      output_arg {"
      "        name: 'sum'"
      "        type: DT_FLOAT"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'read0'"
      "      op: 'ReadVariableOp'"
      "      input: 'var'"
      "      device: '/job:localhost/replica:0/task:0/device:CPU:0'"
      "      attr {"
      "        key: 'dtype'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'read1'"
      "      op: 'ReadVariableOp'"
      "      input: 'var'"
      "      device: '/job:localhost/replica:0/task:1/device:CPU:0'"
      "      attr {"
      "        key: 'dtype'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'read2'"
      "      op: 'ReadVariableOp'"
      "      input: 'var'"
      "      device: '/job:localhost/replica:0/task:2/device:CPU:0'"
      "      attr {"
      "        key: 'dtype'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'add1'"
      "      op: 'Add'"
      "      input: 'read0:value:0'"
      "      input: 'read1:value:0'"
      "      attr {"
      "        key: 'T'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'add2'"
      "      op: 'Add'"
      "      input: 'add1:z:0'"
      "      input: 'read2:value:0'"
      "      attr {"
      "        key: 'T'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    ret {"
      "      key: 'sum'"
      "      value: 'add2:z:0'"
      "    }",
      &def));
  return def.SerializeAsString();
}

void VarIsInitialized(TFE_Context* ctx, TFE_TensorHandle* var_handle) {
  TF_Status* status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(ctx, "VarIsInitializedOp", status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_OpAddInput(op, var_handle, status);
  TFE_TensorHandle* is_initialized[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(op, &is_initialized[0], &num_retvals, status);
  CHECK_EQ(1, num_retvals);
  TF_Tensor* t = TFE_TensorHandleResolve(is_initialized[0], status);
  bool initialized = false;
  memcpy(&initialized, TF_TensorData(t), TF_TensorByteSize(t));
  EXPECT_EQ(initialized, true);
  TF_DeleteTensor(t);
  TFE_DeleteTensorHandle(is_initialized[0]);
  TFE_DeleteOp(op);
  delete status;
}

void TestFunctionWithPackedInput(const bool remote) {
  tensorflow::ServerDef server_def = GetServerDef(3);

  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::GrpcServer> worker_server1;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server1)
                  .ok());
  ASSERT_TRUE(worker_server1->Start().ok());

  server_def.set_task_index(2);
  std::unique_ptr<tensorflow::GrpcServer> worker_server2;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server2)
                  .ok());
  ASSERT_TRUE(worker_server2->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(/*enable=*/true));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  const char task0_name[] = "/job:localhost/replica:0/task:0/device:CPU:0";
  const char task1_name[] = "/job:localhost/replica:0/task:1/device:CPU:0";
  const char task2_name[] = "/job:localhost/replica:0/task:2/device:CPU:0";

  // Create one variable per task.
  TFE_TensorHandle* h0 = TestVariable(ctx, 1.0, task0_name);
  TFE_TensorHandle* h1 = TestVariable(ctx, 2.0, task1_name);
  TFE_TensorHandle* h2 = TestVariable(ctx, 3.0, task2_name);

  // Add a sync point in order to make sure that variables have been initialized
  // before the function execution starts.
  // TODO(b/155789951): Remove once b/155789951 is fixed.
  VarIsInitialized(ctx, h1);
  VarIsInitialized(ctx, h2);

  // Pack 3 variable handles into one TFE_TensorHandle.
  int num_replicas = 3;
  std::vector<TFE_TensorHandle*> handles = {h0, h1, h2};
  TFE_TensorHandle* packed_handle =
      TFE_CreatePackedTensorHandle(ctx, handles.data(), &num_replicas, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  EXPECT_EQ(TFE_TensorHandleDataType(packed_handle), TF_RESOURCE);
  EXPECT_EQ(TFE_TensorHandleNumDims(packed_handle, status), 0);
  EXPECT_EQ(TFE_TensorHandleNumElements(packed_handle, status), 1);

  const string composite_device_name =
      "/job:localhost/replica:0/task:0/device:COMPOSITE:0";
  EXPECT_EQ(TFE_TensorHandleDeviceName(packed_handle, status),
            composite_device_name);
  EXPECT_EQ(TFE_TensorHandleBackingDeviceName(packed_handle, status),
            composite_device_name);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  // Register and run a function which returns the sum of 3 variables.
  const string function_def = AddVariablesFunction();
  TFE_ContextAddFunctionDef(ctx, function_def.data(), function_def.size(),
                            status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_Op* func = TFE_NewOp(ctx, "AddVariablesFunction", status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_OpAddInput(func, packed_handle, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  if (remote) {
    TFE_OpSetDevice(func, task1_name, status);
    ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  }

  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(func, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  ASSERT_EQ(1, num_retvals);
  TFE_DeleteOp(func);
  TFE_DeleteTensorHandle(packed_handle);
  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_DeleteTensorHandle(retvals[0]);
  float sum = 0;
  EXPECT_EQ(sizeof(sum), TF_TensorByteSize(t));
  memcpy(&sum, TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);
  EXPECT_EQ(sum, 6.0);

  TFE_DeleteTensorHandle(h0);
  TFE_DeleteTensorHandle(h1);
  TFE_DeleteTensorHandle(h2);

  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_DeleteExecutor(executor);
  TFE_ContextRemoveFunction(ctx, "AddVariablesFunction", status);
  TFE_DeleteContext(ctx);

  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server1.release();
  worker_server2.release();
}

TEST(CAPI, TestLocalFunctionWithPackedInput) {
  TestFunctionWithPackedInput(/*remote=*/false);
}

TEST(CAPI, TestRemoteFunctionWithPackedInput) {
  TestFunctionWithPackedInput(/*remote=*/true);
}

string VariableAddFunction() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'VariableAddFunction'"
      "      input_arg {"
      "        name: 'var0'"
      "        type: DT_RESOURCE"
      "      }"
      "      output_arg {"
      "        name: 'var0_value'"
      "        type: DT_FLOAT"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'read0'"
      "      op: 'ReadVariableOp'"
      "      input: 'var0'"
      "      attr {"
      "        key: 'dtype'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'add'"
      "      op: 'Add'"
      "      input: 'read0:value:0'"
      "      input: 'read0:value:0'"
      "      device: '/job:localhost/task:1/device:CPU:0'"
      "      attr {"
      "        key: 'T'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'identity'"
      "      op: 'Identity'"
      "      input: 'add:z:0'"
      "      device: '/job:localhost/task:0/device:CPU:0'"
      "      attr {"
      "        key: 'T'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    ret {"
      "      key: 'var0_value'"
      "      value: 'identity:output:0'"
      "    }",
      &def));
  return def.SerializeAsString();
}

class FunctionErrorInjectionPass : public tensorflow::FunctionOptimizationPass {
 public:
  FunctionErrorInjectionPass(string error_node, string error_device)
      : error_node_(error_node), error_device_(error_device) {}
  tensorflow::Status Run(const tensorflow::DeviceSet& device_set,
                         const tensorflow::ConfigProto& config_proto,
                         std::unique_ptr<tensorflow::Graph>* graph,
                         tensorflow::FunctionLibraryDefinition* flib_def,
                         std::vector<std::string>* control_ret_node_names,
                         bool* control_rets_updated) override {
    // Inject failure to function instantiation if finding a node that contains
    // the given node name (error_node_) and requested device (error_device_).
    for (const auto node : graph->get()->nodes()) {
      if (node->name().find(error_node_) != string::npos &&
          node->requested_device() == error_device_) {
        return tensorflow::errors::Internal("Injected graph pass error.");
      }
    }
    return tensorflow::Status::OK();
  }

 private:
  const string error_node_;
  const string error_device_;
};

void TestDistributedFunctionCancellation(bool inject_error) {
  tensorflow::ServerDef server_def = GetServerDef(3);
  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::GrpcServer> worker_server1;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server1)
                  .ok());
  ASSERT_TRUE(worker_server1->Start().ok());
  server_def.set_task_index(2);
  std::unique_ptr<tensorflow::GrpcServer> worker_server2;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server2)
                  .ok());
  ASSERT_TRUE(worker_server2->Start().ok());
  const char dev2_name[] = "/job:localhost/replica:0/task:2/device:CPU:0";

  if (inject_error) {
    // Inject a function optimization pass failure when it sees the 'read0' op
    // having a requested device `dev2_name`. During execution:
    //   * task:0 processes the main function `VariableAddFunction` and places
    //     the read0 op on task:2
    //   * task:0 partitions the main function with a subgraph containing read0
    //     sent to task:2
    //   * task:2 graph pass reports an error when it sees read0 with dev2_name
    tensorflow::function_optimization_registration::
        FunctionOptimizationPassRegistration register_test_pass(
            std::make_unique<FunctionErrorInjectionPass>("read0", dev2_name));
  }

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* var_handle = TestVariable(ctx, 2.0, dev2_name);
  EXPECT_NE(var_handle, nullptr);

  const string function_def = VariableAddFunction();
  TFE_ContextAddFunctionDef(ctx, function_def.data(), function_def.size(),
                            status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_Op* func = TFE_NewOp(ctx, "VariableAddFunction", status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_OpAddInput(func, var_handle, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(func, &retvals[0], &num_retvals, status);

  if (inject_error) {
    ASSERT_EQ(TF_INTERNAL, TF_GetCode(status)) << TF_Message(status);
  } else {
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    ASSERT_EQ(1, num_retvals);
    TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteTensorHandle(retvals[0]);
    float sum = 0;
    ASSERT_EQ(sizeof(sum), TF_TensorByteSize(t));
    memcpy(&sum, TF_TensorData(t), TF_TensorByteSize(t));
    TF_DeleteTensor(t);
    ASSERT_EQ(sum, 4.0);
  }

  TFE_DeleteOp(func);
  TFE_DeleteTensorHandle(var_handle);
  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server1.release();
  worker_server2.release();
}

TEST(CAPI, DistributedFunctionNoError) {
  TestDistributedFunctionCancellation(false);
}

TEST(CAPI, DistributedFunctionCancelledOnError) {
  TestDistributedFunctionCancellation(true);
}

void TestRemoteExecuteDeleteContextWithOutstandingRPC(bool async) {
  tensorflow::ServerDef server_def = GetServerDef(2);

  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);

  std::unique_ptr<tensorflow::GrpcServer> worker_server;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server)
                  .ok());
  ASSERT_TRUE(worker_server->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                             TFE_DEVICE_PLACEMENT_EXPLICIT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Use large matrices so that RPCs don't return before we get a chance
  // to call TFE_DeleteContext.
  TFE_TensorHandle* h0_task0 = TestMatrixTensorHandle100x100(ctx);
  TFE_TensorHandle* h1_task0 = TestMatrixTensorHandle100x100(ctx);
  const char remote_device_name[] =
      "/job:localhost/replica:0/task:1/device:CPU:0";
  auto* h0_task1 =
      TFE_TensorHandleCopyToDevice(h0_task0, ctx, remote_device_name, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  auto* h1_task1 =
      TFE_TensorHandleCopyToDevice(h1_task0, ctx, remote_device_name, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_Op* matmul = MatMulOp(ctx, h0_task1, h1_task1);
  TFE_OpSetDevice(matmul, remote_device_name, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);

  TFE_DeleteTensorHandle(h0_task0);
  TFE_DeleteTensorHandle(h1_task0);
  TFE_DeleteTensorHandle(h0_task1);
  TFE_DeleteTensorHandle(h1_task1);
  TFE_DeleteTensorHandle(retvals[0]);

  TFE_DeleteOp(matmul);

  TFE_DeleteContext(ctx);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
}

TEST(CAPI, RemoteExecuteDeleteContextWithOutstandingRPC) {
  TestRemoteExecuteDeleteContextWithOutstandingRPC(false);
}

TEST(CAPI, RemoteExecuteDeleteContextWithOutstandingRPCAsync) {
  TestRemoteExecuteDeleteContextWithOutstandingRPC(true);
}
}  // namespace
