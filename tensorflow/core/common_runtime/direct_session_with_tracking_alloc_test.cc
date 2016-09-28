/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/direct_session.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

TEST(DirectSessionWithTrackingAllocTest, CostModelTest) {
  EnableCPUAllocatorFullStats(true);

  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  Node* a = test::graph::Constant(&graph, a_tensor);
  a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_tensor, {1, 1});
  Node* x = test::graph::Constant(&graph, x_tensor);
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

  // y = A * x
  Node* y = test::graph::Matmul(&graph, a, x, false, false);
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Node* y_neg = test::graph::Unary(&graph, "Neg", y);
  y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  options.config.mutable_graph_options()->set_build_cost_model(true);
  std::unique_ptr<Session> session(NewSession(options));
  TF_ASSERT_OK(session->Create(def));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y->name() + ":0"};
  std::vector<string> target_nodes = {y_neg->name()};
  std::vector<Tensor> outputs;
  const int64 start_micros = Env::Default()->NowMicros();
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  const int64 run_duration_micros = Env::Default()->NowMicros() - start_micros;
  TF_ASSERT_OK(s);

  DirectSession* ds = static_cast<DirectSession*>(session.get());
  int graph_cnt = 0;
  CostModelManager::CostModelMap cost_models;
  ds->ExportCostModels(&cost_models);
  for (auto& it : cost_models) {
    const Graph* g = (it).first;
    const CostModel* cm = (it).second;
    for (Node* node : g->nodes()) {
      if (node->name() == y->name()) {
        EXPECT_LE(8, cm->MaxMemorySize(node, 0));
        EXPECT_EQ(5, cm->AllocationId(node, 0));
      } else if (node->name() == y_neg->name()) {
        EXPECT_LE(8, cm->MaxMemorySize(node, 0));
        EXPECT_EQ(7, cm->AllocationId(node, 0));
      }
      EXPECT_LE(0, cm->MaxExecutionTime(node));
      EXPECT_GE(run_duration_micros, cm->MaxExecutionTime(node));
    }
    graph_cnt++;
  }
  // We should have 2 cost models since we have 2 cpu devices.
  ASSERT_EQ(2, graph_cnt);
}

static void TestHWAccelerator(bool enableHWTrace) {
  EnableCPUAllocatorFullStats(true);

  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  Node* a = test::graph::Constant(&graph, a_tensor);
  a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_tensor, {1, 1});
  Node* x = test::graph::Constant(&graph, x_tensor);
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/gpu:0");

  // y = A * x
  Node* y = test::graph::Matmul(&graph, a, x, false, false);
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/gpu:0");

  Node* y_neg = test::graph::Unary(&graph, "Neg", y);
  y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 1;
  (*options.config.mutable_device_count())["GPU"] = 1;
  options.config.set_allow_soft_placement(true);
  options.config.mutable_graph_options()->set_build_cost_model(true);
  std::unique_ptr<Session> session(NewSession(options));
  TF_ASSERT_OK(session->Create(def));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y->name() + ":0"};
  std::vector<string> target_nodes = {y_neg->name()};
  std::vector<Tensor> outputs;
  const int64 start_micros = Env::Default()->NowMicros();

  RunOptions run_options;
  if (enableHWTrace) {
    run_options.set_trace_level(RunOptions::FULL_TRACE);
  }
  RunMetadata run_metadata;
  Status s = session->Run(run_options, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  const int64 run_duration_micros = Env::Default()->NowMicros() - start_micros;
  TF_ASSERT_OK(s);

  DirectSession* ds = static_cast<DirectSession*>(session.get());
  int graph_cnt = 0;
  CostModelManager::CostModelMap cost_models;
  ds->ExportCostModels(&cost_models);
  for (auto& it : cost_models) {
    const Graph* g = (it).first;
    const CostModel* cm = (it).second;
    for (Node* node : g->nodes()) {
      if (node->name() == y->name()) {
        EXPECT_LE(8, cm->MaxMemorySize(node, 0));
      } else if (node->name() == y_neg->name()) {
        EXPECT_LE(8, cm->MaxMemorySize(node, 0));
      }
      EXPECT_LE(0, cm->MaxExecutionTime(node));
      EXPECT_GE(run_duration_micros, cm->MaxExecutionTime(node));
    }
    graph_cnt++;
  }
  // We should have 2 cost models since we requested 1 cpu and 1 gpu. However
  // since the placement is soft, we might end up placing everything on cpu.
  ASSERT_GE(2, graph_cnt);
  ASSERT_LE(1, graph_cnt);
}

TEST(DirectSessionWithTrackingAllocTest, CostModelForAccelerator) {
  TestHWAccelerator(false);
}

TEST(DirectSessionWithTrackingAllocTest, CostModelWithHardwareStats) {
  TestHWAccelerator(true);
}

}  // namespace
}  // namespace tensorflow
