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

void ReplaceTaskInServerDef(tensorflow::ServerDef* server_def, int task_index) {
  tensorflow::JobDef* job_def = server_def->mutable_cluster()->mutable_job(0);
  int port = tensorflow::testing::PickUnusedPortOrDie();
  job_def->mutable_tasks()->at(task_index) = absl::StrCat("localhost:", port);
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

// Read the value of variable `var` and save it into `out_value`.
void ReadVariable(TFE_Context* ctx, TFE_TensorHandle* var,
                  TFE_TensorHandle** out_value) {
  TF_Status* status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(ctx, "ReadVariableOp", status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpAddInput(op, var, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  int num_retvals = 1;
  TFE_Execute(op, out_value, &num_retvals, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(op);
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
  absl::Status s = tensorflow::GrpcServer::Create(
      updated_server_def, tensorflow::Env::Default(), &worker_server);
  ASSERT_TRUE(s.ok()) << s.message();
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

void TestRemoteExecuteUpdateServerDefResourceAccess(bool async) {
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
  const char dev0_name[] = "/job:localhost/replica:0/task:0/device:CPU:0";
  const char dev1_name[] = "/job:localhost/replica:0/task:1/device:CPU:0";

  TFE_TensorHandle* var_handle0 = TestVariable(ctx, 1.0, dev0_name);
  EXPECT_NE(var_handle0, nullptr);
  TFE_TensorHandle* var_handle1 = TestVariable(ctx, 2.0, dev1_name);
  EXPECT_NE(var_handle1, nullptr);

  TFE_TensorHandle* value_handle = nullptr;
  ReadVariable(ctx, var_handle1, &value_handle);
  CheckTFE_TensorHandleHasFloats(value_handle, {2});
  TFE_DeleteTensorHandle(value_handle);

  // Start a new worker to replace task:1
  ReplaceTaskInServerDef(&server_def, 1);
  server_def.set_task_index(1);
  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server)
                  .ok());
  ASSERT_TRUE(worker_server->Start().ok());

  // Update server def to replace the remote device with the device info on the
  // new worker (different incarnation ID).
  server_def.set_task_index(0);
  string serialized_update = server_def.SerializeAsString();
  TFE_ContextUpdateServerDef(ctx, 0, serialized_update.data(),
                             serialized_update.size(), status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // The device of var_handle0 is local device which is the same before and
  // after cluster update. Remove resource with valid device should succeed.
  TFE_Op* op = TFE_NewOp(ctx, "DestroyResourceOp", status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, var_handle0, status);
  TFE_OpSetDevice(op, dev0_name, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  int num_retvals = 0;
  TFE_Execute(op, nullptr, &num_retvals, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(op);

  // The device of var_handle1 is remote device, which was replaced during
  // cluster update. Removing resource with invalid device should fail
  // gracefully (i.e., with error status) instead of crashing with segfaults.
  op = TFE_NewOp(ctx, "DestroyResourceOp", status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, var_handle1, status);
  TFE_OpSetDevice(op, dev1_name, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  num_retvals = 0;
  TFE_Execute(op, nullptr, &num_retvals, status);
  EXPECT_NE(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(op);

  TFE_DeleteTensorHandle(var_handle0);
  TFE_DeleteTensorHandle(var_handle1);

  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
}

TEST(CAPI, TestRemoteExecuteUpdateServerDefResourceAccess) {
  TestRemoteExecuteUpdateServerDefResourceAccess(false);
}

TEST(CAPI, TestRemoteExecuteUpdateServerDefResourceAccessAsync) {
  TestRemoteExecuteUpdateServerDefResourceAccess(true);
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
  job_def->mutable_tasks()->insert({2, absl::StrCat("localhost:", port)});
  server_def.set_task_index(0);
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

void TestConnectToCluster(bool keep_localhost_for_first_connect) {
  // Fail fast on GetStatus requests so we can get errors instead of timeout
  // when updating cluster with non-exsitent worker
  tensorflow::setenv("GRPC_FAIL_FAST", "TRUE", /*overwrite=*/1);

  const string first_name =
      keep_localhost_for_first_connect ? "localhost" : "abc";
  tensorflow::ServerDef server_def = GetServerDef(first_name, 1);

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  const string dev0_name = "/job:localhost/replica:0/task:0/device:CPU:0";
  TFE_TensorHandle* var_handle0 = TestVariable(ctx, 1.0, dev0_name);
  EXPECT_NE(var_handle0, nullptr);

  absl::Status status2;
  EXPECT_EQ(tensorflow::unwrap(var_handle0)->DeviceName(&status2), dev0_name);

  // Rename local device
  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();
  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  const string dev1_name =
      absl::StrCat("/job:", first_name, "/replica:0/task:0/device:CPU:0");
  TFE_TensorHandle* var_handle1 = TestVariable(ctx, 2.0, dev1_name);
  EXPECT_NE(var_handle1, nullptr);
  EXPECT_EQ(tensorflow::unwrap(var_handle1)->DeviceName(&status2), dev1_name);

  // Another renaming of local device
  const string second_name = "def";
  server_def.set_job_name(second_name);
  server_def.mutable_cluster()->mutable_job(0)->set_name(second_name);
  (*server_def.mutable_cluster()->mutable_job(0)->mutable_tasks())[0] =
      absl::StrCat(second_name, ":",
                   tensorflow::testing::PickUnusedPortOrDie());

  serialized = server_def.SerializeAsString();
  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  const string dev2_name = "/job:def/replica:0/task:0/device:CPU:0";
  TFE_TensorHandle* var_handle2 = TestVariable(ctx, 2.0, dev2_name);
  EXPECT_NE(var_handle2, nullptr);
  EXPECT_EQ(tensorflow::unwrap(var_handle2)->DeviceName(&status2), dev2_name);

  TFE_DeleteTensorHandle(var_handle0);
  TFE_DeleteTensorHandle(var_handle1);
  TFE_DeleteTensorHandle(var_handle2);

  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);

  tensorflow::unsetenv("GRPC_FAIL_FAST");
}

TEST(CAPI, ConnectToClusterLocalhostFirst) { TestConnectToCluster(false); }

TEST(CAPI, ConnectToClusterRenameFirst) { TestConnectToCluster(true); }

}  // namespace
