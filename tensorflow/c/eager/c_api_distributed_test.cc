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

#include <regex>  // NOLINT

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
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
  TFE_TensorHandle* h0 = TestVariable(ctx, 1.0, task1_name);
  TFE_TensorHandle* h1 = TestVariable(ctx, 2.0, task2_name);
  TFE_TensorHandle* h2 = TestVariable(ctx, 3.0, task0_name);

  // Add a sync point in order to make sure that variables have been initialized
  // before the function execution starts.
  TFE_ContextAsyncWait(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Pack 3 variable handles into one TFE_TensorHandle.
  // When remote is false, function device is placed on task0. Handle types are
  // REMOTE, REMOTE, LOCAL on task0. When remote is true, function device is
  // placed on task1, Handle types are LOCAL, REMOTE, LOCAL on task1.
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

string VariableAddFunctionSignature() {
  return "    signature {"
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
         "    }";
}

string VariableAddFunction() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      VariableAddFunctionSignature(), &def));
  return def.SerializeAsString();
}

// A graph optimization pass that would fail when triggered for more than once.
class GraphErrorInjectionPass : public tensorflow::GraphOptimizationPass {
 public:
  static bool enabled_;
  GraphErrorInjectionPass() {}

  tensorflow::Status Run(
      const tensorflow::GraphOptimizationPassOptions& options) override {
    if (!enabled_) {
      return tensorflow::Status::OK();
    }
    if (first_call_) {
      first_call_ = false;
      return tensorflow::Status::OK();
    }
    return tensorflow::errors::Internal("Graph pass runs for more than once!");
  }

 private:
  bool first_call_ = true;
};

// After the graph pass is registered, it takes effect globally and can affect
// other test cases. Define a static variable to switch it on and off.
bool GraphErrorInjectionPass::enabled_ = false;

// Test to ensure that a registered graph optimization pass is only executed
// once (i.e., on the main function side) in running distributed functions.
// This test creates a cluster with two workers, create a variable on the
// second worker, and run a distributed function (VariableAddFunction) whose ops
// span the local and remote workers. If the graph optimization pass is executed
// on both the main function side and the component function side, an error will
// be thrown in the registered graph optimization pass.
TEST(CAPI, DistributedFunctionGraphPassOnlyOnce) {
  // Register graph pass that will raise error if called more than once.
  tensorflow::optimization_registration::OptimizationPassRegistration
      register_test_pass(tensorflow::OptimizationPassRegistry::PRE_PLACEMENT, 0,
                         std::make_unique<GraphErrorInjectionPass>(),
                         "error_injector");
  GraphErrorInjectionPass::enabled_ = true;

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
  TFE_ContextAsyncWait(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

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

  TFE_DeleteOp(func);
  TFE_DeleteTensorHandle(var_handle);
  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server1.release();
  worker_server2.release();

  // Disable the test graph pass so it does not affect other test cases.
  GraphErrorInjectionPass::enabled_ = false;
}

string VariableAddFunctionWithGraphError() {
  string signature = VariableAddFunctionSignature();
  // Replace the node 'read0' with 'read0_maybe_with_graph_error', so that the
  // error injecting pass can identify and introduce graph pass errors.
  signature = std::regex_replace(signature, std::regex("read0"),
                                 "read0_maybe_with_graph_error");
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(signature, &def));
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
    // Inject a function optimization pass failure when it sees the
    // 'read0_maybe_with_graph_error' op having a requested device `dev2_name`.
    // During execution:
    //   * task:0 processes main function `VariableAddFunctionWithGraphError`
    //     and places the 'read0_maybe_with_graph_error' op on task:2
    //   * task:0 partitions the main function with a subgraph containing
    //     'read0_maybe_with_graph_error' sent to task:2
    //   * task:2 graph pass reports an error when it sees
    //     'read0_maybe_with_graph_error' with dev2_name
    tensorflow::function_optimization_registration::
        FunctionOptimizationPassRegistration register_test_pass(
            std::make_unique<FunctionErrorInjectionPass>(
                "read0_maybe_with_graph_error", dev2_name));
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
  TFE_ContextAsyncWait(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  const string function_def = inject_error ? VariableAddFunctionWithGraphError()
                                           : VariableAddFunction();
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

// TODO(b/170399182): Update test once an alternative to using the function
// optimization hook is in place.
TEST(CAPI, DISABLED_DistributedFunctionCancelledOnError) {
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
