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

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_test_util.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/tsl/protobuf/coordination_config.pb.h"

namespace tensorflow {
namespace {

constexpr char kCoordinationServiceType[] = "standalone";

void EnableCoordinationService(tensorflow::ServerDef* server_def) {
  auto coord_config = server_def->mutable_default_session_config()
                          ->mutable_experimental()
                          ->mutable_coordination_config();
  coord_config->set_service_type(kCoordinationServiceType);
  coord_config->set_service_leader("/job:worker/replica:0/task:0");
}

struct SessionParams {
  std::string test_name;
  bool enable_clusterspec_propagation = false;
};

using SingleClientTest = ::testing::TestWithParam<SessionParams>;

TEST_P(SingleClientTest, SetGetConfigInOpTest) {
  bool enable_clusterspec_propagation =
      GetParam().enable_clusterspec_propagation;
  const int num_workers = 3;
  std::string job_name = "worker";
  // NOTE(b/37868888#comment4): Set a different initial name for the job due to
  // the limitation in ClusterSpec propagation.
  if (enable_clusterspec_propagation) {
    job_name = "server_init";
  }
  tensorflow::ServerDef server_def = GetServerDef(job_name, num_workers);
  const char task0_name[] = "/job:worker/replica:0/task:0/device:CPU:0";
  const char task1_name[] = "/job:worker/replica:0/task:1/device:CPU:0";
  const char task2_name[] = "/job:worker/replica:0/task:2/device:CPU:0";
  const std::string& master =
      strings::StrCat("grpc://", server_def.cluster().job(0).tasks().at(0));

  EnableCoordinationService(&server_def);
  server_def.mutable_default_session_config()
      ->mutable_experimental()
      ->mutable_coordination_config()
      ->set_service_leader(task0_name);

  // Start server instances for the workers.
  ServerFactory* factory;
  ASSERT_TRUE(ServerFactory::GetFactory(server_def, &factory).ok());
  std::unique_ptr<tensorflow::ServerInterface> workers[3];
  for (int worker_id = 0; worker_id < num_workers; worker_id++) {
    server_def.set_task_index(worker_id);
    ASSERT_TRUE(factory
                    ->NewServer(server_def, ServerFactory::Options(),
                                &workers[worker_id])
                    .ok());
    ASSERT_TRUE(workers[worker_id]->Start().ok());
  }

  // Build graph with a TestSetConfigKeyValue op on worker/1, and a
  // TestGetConfigKeyValue on worker/2.
  TF_Status* status = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();
  TF_Operation* feed_key = Placeholder(graph, status, "key", TF_STRING, {});
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TF_Operation* feed_val = Placeholder(graph, status, "val", TF_STRING, {});
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TF_OperationDescription* set_desc =
      TF_NewOperation(graph, "TestSetConfigKeyValue", "set");
  TF_AddInput(set_desc, {feed_key, 0});
  TF_AddInput(set_desc, {feed_val, 0});
  TF_SetDevice(set_desc, task1_name);
  TF_Operation* set_op = TF_FinishOperation(set_desc, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TF_OperationDescription* get_desc =
      TF_NewOperation(graph, "TestGetConfigKeyValue", "get");
  TF_Output get_input = {feed_key, 0};
  TF_AddInput(get_desc, get_input);
  // Add control dependency to make sure "get" runs after "set"
  TF_AddControlInput(get_desc, set_op);
  TF_SetDevice(get_desc, task2_name);
  TF_Operation* get_op = TF_FinishOperation(get_desc, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  // Prepare feeds and fetches for running the graph
  const char test_key[] = "test_key";
  const char test_val[] = "test_val";
  TF_TString tstr_key;
  TF_TString_Init(&tstr_key);
  TF_TString_Copy(&tstr_key, test_key, sizeof(test_key) - 1);
  TF_TString tstr_val;
  TF_TString_Init(&tstr_val);
  TF_TString_Copy(&tstr_val, test_val, sizeof(test_val) - 1);
  TF_Output inputs[2] = {{feed_key, 0}, {feed_val, 0}};
  TF_Tensor* input_values[2];
  auto deallocator = [](void* data, size_t len, void* arg) {};
  input_values[0] = TF_NewTensor(TF_STRING, nullptr, 0, &tstr_key,
                                 sizeof(tstr_key), deallocator, nullptr);
  input_values[1] = TF_NewTensor(TF_STRING, nullptr, 0, &tstr_val,
                                 sizeof(tstr_val), deallocator, nullptr);
  TF_Output outputs[1] = {{get_op, 0}};
  TF_Tensor* output_values[1] = {nullptr};

  // Create session to run the graph
  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_SetTarget(opts, master.c_str());
  ConfigProto configs = server_def.default_session_config();
  if (enable_clusterspec_propagation) {
    // NOTE(b/37868888#comment4): Reset name of the job due to the limitation in
    // ClusterSpec propagation.
    server_def.mutable_cluster()->mutable_job(0)->set_name("worker");
    *configs.mutable_cluster_def() = server_def.cluster();
  }
  const std::string& serialized = configs.SerializeAsString();
  TF_SetConfig(opts, serialized.data(), serialized.size(), status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TF_Session* sess = TF_NewSession(graph, opts, status);
  TF_SessionRun(sess, nullptr, inputs, input_values, 2, outputs, output_values,
                1, nullptr, 0, nullptr, status);
  // Verify that the test value was set and extracted from the coordination
  // service correctly.
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  const tstring& output_val =
      *static_cast<tstring*>(TF_TensorData(output_values[0]));
  EXPECT_EQ(output_val, test_val);

  // Clean up.
  TF_CloseSession(sess, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(input_values[0]);
  TF_DeleteTensor(input_values[1]);
  TF_DeleteTensor(output_values[0]);
  TF_DeleteSessionOptions(opts);
  TF_DeleteSession(sess, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(status);

  // Grpc servers do not support clean down.
  for (int worker_id = 0; worker_id < num_workers; worker_id++) {
    workers[worker_id].release();
  }
}

INSTANTIATE_TEST_SUITE_P(
    SessionCoordinationTests, SingleClientTest,
    ::testing::ValuesIn<SessionParams>({
        {"EnableClusterSpecPropagation", true},
        {"DisableClusterSpecPropagation", false},
    }),
    [](const ::testing::TestParamInfo<SingleClientTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace tensorflow
