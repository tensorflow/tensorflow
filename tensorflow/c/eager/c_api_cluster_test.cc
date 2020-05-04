/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
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

void CheckTFE_TensorHandleHasFloats(TFE_TensorHandle* handle,
                                    const std::vector<float>& expected_values) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Tensor* t = TFE_TensorHandleResolve(handle, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  std::unique_ptr<float[]> actual_values(new float[expected_values.size()]);
  EXPECT_EQ(sizeof(float) * expected_values.size(), TF_TensorByteSize(t));
  memcpy(actual_values.get(), TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);

  for (int i = 0; i < expected_values.size(); i++) {
    EXPECT_EQ(expected_values[i], actual_values[i])
        << "Mismatch in expected values at (zero-based) index " << i;
  }
}

void CheckRemoteMatMulExecutesOK(TFE_Context* ctx,
                                 const char* remote_device_name,
                                 const char* local_device_name) {
  TF_Status* status = TF_NewStatus();
  TFE_TensorHandle* h0_task0 = TestMatrixTensorHandle(ctx);

  TFE_Op* matmul = MatMulOp(ctx, h0_task0, h0_task0);
  TFE_OpSetDevice(matmul, remote_device_name, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  auto* retval_task0 =
      TFE_TensorHandleCopyToDevice(retvals[0], ctx, local_device_name, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  CheckTFE_TensorHandleHasFloats(retval_task0, {7, 10, 15, 22});

  TFE_DeleteTensorHandle(retval_task0);
  TFE_DeleteTensorHandle(h0_task0);
  TFE_DeleteTensorHandle(retvals[0]);

  TFE_DeleteOp(matmul);

  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteExecutor(executor);
  TF_DeleteStatus(status);
}

void TestRemoteExecuteChangeServerDef(bool async) {
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
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  const char remote_device_name[] =
      "/job:localhost/replica:0/task:1/device:CPU:0";
  const char local_device_name[] =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  CheckRemoteMatMulExecutesOK(ctx, remote_device_name, local_device_name);

  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();

  // Update the server def with a new set of names (worker instead of
  // localhost).
  tensorflow::ServerDef updated_server_def = GetServerDef("worker", 2);
  serialized = updated_server_def.SerializeAsString();

  updated_server_def.set_task_index(1);
  tensorflow::Status s = tensorflow::GrpcServer::Create(
      updated_server_def, tensorflow::Env::Default(), &worker_server);
  ASSERT_TRUE(s.ok()) << s.error_message();
  ASSERT_TRUE(worker_server->Start().ok());

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Create a new tensor_handle.
  TFE_TensorHandle* h0_task0_new = TestMatrixTensorHandle(ctx);

  // Check that copying it to the old remote device (named localhost) fails.
  TFE_TensorHandleCopyToDevice(h0_task0_new, ctx, remote_device_name, status);
  EXPECT_NE(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Copying and executing on the new remote device works.
  const char new_remote_device_name[] =
      "/job:worker/replica:0/task:1/device:CPU:0";
  const char new_local_device_name[] =
      "/job:worker/replica:0/task:0/device:CPU:0";

  auto* h0_task1_new = TFE_TensorHandleCopyToDevice(
      h0_task0_new, ctx, new_remote_device_name, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_DeleteTensorHandle(h0_task0_new);
  TFE_DeleteTensorHandle(h0_task1_new);

  CheckRemoteMatMulExecutesOK(ctx, new_remote_device_name,
                              new_local_device_name);

  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteExecutor(executor);

  TF_DeleteStatus(status);

  TFE_DeleteContext(ctx);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
}

TEST(CAPI, RemoteExecuteChangeServerDef) {
  TestRemoteExecuteChangeServerDef(false);
}
TEST(CAPI, RemoteExecuteChangeServerDefAsync) {
  TestRemoteExecuteChangeServerDef(true);
}

void TestRemoteExecuteUpdateServerDef(bool async) {
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
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  const char local_device_name[] =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  const char remote_device_name[] =
      "/job:localhost/replica:0/task:1/device:CPU:0";
  CheckRemoteMatMulExecutesOK(ctx, remote_device_name, local_device_name);

  TFE_ContextUpdateServerDef(ctx, 0, serialized.data(), serialized.size(),
                             status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  CheckRemoteMatMulExecutesOK(ctx, remote_device_name, local_device_name);

  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
}

TEST(CAPI, RemoteExecuteUpdateServerDef) {
  TestRemoteExecuteUpdateServerDef(false);
}

TEST(CAPI, RemoteExecuteUpdateServerDefAsync) {
  TestRemoteExecuteUpdateServerDef(true);
}

void TestRemoteExecuteUpdateServerDefWithFailures(bool async) {
  // Fail fast on GetStatus requests so we can get errors instead of timeout
  // when updating cluster with non-exsitent worker
  tensorflow::setenv("GRPC_FAIL_FAST", "TRUE", /*overwrite=*/1);

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
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  const char local_device_name[] =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  const char remote_device_name[] =
      "/job:localhost/replica:0/task:1/device:CPU:0";
  CheckRemoteMatMulExecutesOK(ctx, remote_device_name, local_device_name);

  // Adding a non-existent remote worker to cluster def. This should cause the
  // UpdateServerDef call to fail.
  tensorflow::ClusterDef* cluster_def = server_def.mutable_cluster();
  tensorflow::JobDef* job_def = cluster_def->mutable_job(0);
  int port = tensorflow::testing::PickUnusedPortOrDie();
  job_def->mutable_tasks()->insert(
      {2, tensorflow::strings::StrCat("localhost:", port)});
  string serialized_update = server_def.SerializeAsString();
  TFE_ContextUpdateServerDef(ctx, 0, serialized_update.data(),
                             serialized_update.size(), status);
  EXPECT_NE(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Even after the prevoiusly failed cluster update, another update and op
  // execution should work fine as long as the provided server_def is valid.
  TFE_ContextUpdateServerDef(ctx, 0, serialized.data(), serialized.size(),
                             status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  CheckRemoteMatMulExecutesOK(ctx, remote_device_name, local_device_name);

  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
  tensorflow::unsetenv("GRPC_FAIL_FAST");
}

TEST(CAPI, RemoteExecuteUpdateServerDefWithFailures) {
  TestRemoteExecuteUpdateServerDefWithFailures(false);
}

TEST(CAPI, RemoteExecuteUpdateServerDefWithFailuresAsync) {
  TestRemoteExecuteUpdateServerDefWithFailures(true);
}

}  // namespace
