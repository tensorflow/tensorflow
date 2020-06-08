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

}  // namespace
