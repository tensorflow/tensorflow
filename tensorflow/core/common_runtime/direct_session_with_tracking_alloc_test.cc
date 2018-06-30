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
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

TEST(DirectSessionWithTrackingAllocTest, CostModelTest) {
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
  options.config.mutable_graph_options()->set_build_cost_model(1);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_min_graph_nodes(-1);
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
      if (node->name() == y->name() || node->name() == y_neg->name()) {
        EXPECT_LE(8, cm->MaxMemorySize(node, 0));
        TensorShapeProto shape = cm->MaxMemoryShape(node, 0);
        EXPECT_EQ(2, shape.dim_size());
        EXPECT_EQ(2, shape.dim(0).size());
        EXPECT_EQ(1, shape.dim(1).size());
        if (node->name() == y->name()) {
#ifdef INTEL_MKL
          // if MKL is used, it goes through various additional
          // graph rewrite pass. In TF, everytime a graph pass
          // happens, "constant" nodes are allocated
          // and deallocated. Each allocation calls the
          // (FindChunkPtr of BFCAllocator),
          // which increments the value of AllocationId.
          // Thus AllocationId becomes more than TF if MKL
          // is used. Now IDs for MKL are 8 more than TF.
          EXPECT_EQ(29, cm->AllocationId(node, 0));
#else
          EXPECT_EQ(21, cm->AllocationId(node, 0));
#endif
        } else {
#ifdef INTEL_MKL
          EXPECT_EQ(30, cm->AllocationId(node, 0));
#else
          EXPECT_EQ(22, cm->AllocationId(node, 0));
#endif
        }
      }
      EXPECT_LE(0, cm->MaxExecutionTime(node));
      EXPECT_GE(run_duration_micros, cm->MaxExecutionTime(node));
    }
    graph_cnt++;
  }
  // We should have 2 cost models since we have 2 cpu devices.
  ASSERT_EQ(2, graph_cnt);
}

TEST(DirectSessionWithTrackingAllocTest, CostModelWarmup) {
  Graph g(OpRegistry::Global());
  Tensor vx(DT_FLOAT, TensorShape({}));
  vx.scalar<float>()() = 1.0;
  Node* x = test::graph::Constant(&g, vx);

  int warmup_steps = 10;
  int measure_steps = 15;
  SessionOptions options;
  options.config.mutable_graph_options()->set_build_cost_model(1);
  options.config.mutable_graph_options()->set_build_cost_model_after(
      warmup_steps);
  std::unique_ptr<Session> session(NewSession(options));

  GraphDef def;
  test::graph::ToGraphDef(&g, &def);
  TF_ASSERT_OK(session->Create(def));
  std::vector<Tensor> outputs;

  for (int i = 0; i < warmup_steps + measure_steps; i++) {
    TF_EXPECT_OK(session->Run({}, {x->name() + ":0"}, {}, &outputs));
  }

  DirectSession* ds = static_cast<DirectSession*>(session.get());
  CostModelManager::CostModelMap cost_models;
  ds->ExportCostModels(&cost_models);
  CHECK_GE(cost_models.size(), 1);
  const CostModel* cm = (*cost_models.begin()).second;
  EXPECT_EQ(measure_steps, cm->GetUpdateTimes());
}

static void TestHWAccelerator(bool enableHWTrace) {
  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  Node* a = test::graph::Constant(&graph, a_tensor);
  a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_tensor, {1, 1});
  Node* x = test::graph::Constant(&graph, x_tensor);
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/device:GPU:0");
#ifdef TENSORFLOW_USE_SYCL
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/device:SYCL:0");
#endif  // TENSORFLOW_USE_SYCL

  // y = A * x
  Node* y = test::graph::Matmul(&graph, a, x, false, false);
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/device:GPU:0");
#ifdef TENSORFLOW_USE_SYCL
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/device:SYCL:0");
#endif  // TENSORFLOW_USE_SYCL

  Node* y_neg = test::graph::Unary(&graph, "Neg", y);
  y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 1;
  (*options.config.mutable_device_count())["GPU"] = 1;
#ifdef TENSORFLOW_USE_SYCL
  (*options.config.mutable_device_count())["SYCL"] = 1;
#endif  // TENSORFLOW_USE_SYCL
  options.config.set_allow_soft_placement(true);
  options.config.mutable_graph_options()->set_build_cost_model(1);
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
      if (node->name() == y->name() || node->name() == y_neg->name()) {
        EXPECT_LE(8, cm->MaxMemorySize(node, 0));
        TensorShapeProto shape = cm->MaxMemoryShape(node, 0);
        EXPECT_EQ(2, shape.dim_size());
        EXPECT_EQ(2, shape.dim(0).size());
        EXPECT_EQ(1, shape.dim(1).size());
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

TEST(DirectSessionWithTrackingAllocTest, CostGraph) {
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
  options.config.mutable_graph_options()->set_build_cost_model(1);
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions::L0);
  std::unique_ptr<Session> session(NewSession(options));
  TF_ASSERT_OK(session->Create(def));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  RunOptions run_options;
  std::vector<string> output_names = {y->name() + ":0"};
  std::vector<string> target_nodes = {y_neg->name()};
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  const int64 start_micros = Env::Default()->NowMicros();
  Status s = session->Run(run_options, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  const int64 run_duration_micros = Env::Default()->NowMicros() - start_micros;
  TF_ASSERT_OK(s);

  EXPECT_LE(2, run_metadata.cost_graph().node_size());
  for (const auto& node : run_metadata.cost_graph().node()) {
    if (node.name() == y->name() || node.name() == y_neg->name()) {
      EXPECT_EQ(1, node.output_info_size());
      EXPECT_LE(8, node.output_info(0).size());
      const TensorShapeProto& shape = node.output_info(0).shape();
      EXPECT_EQ(2, shape.dim_size());
      EXPECT_EQ(2, shape.dim(0).size());
      EXPECT_EQ(1, shape.dim(1).size());
      const DataType& dtype = node.output_info(0).dtype();
      EXPECT_EQ(DT_FLOAT, dtype);
    }
    EXPECT_LE(0, node.compute_cost());
    EXPECT_GE(run_duration_micros, node.compute_cost());
  }
}

TEST(DirectSessionWithTrackingAllocTest, TrackMemoryAllocation) {
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

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);
  std::unique_ptr<Session> session(NewSession(options));
  TF_ASSERT_OK(session->Create(def));
  std::vector<std::pair<string, Tensor>> inputs;

  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  std::vector<string> output_names = {y->name() + ":0"};
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  Status s = session->Run(run_options, inputs, output_names, {}, &outputs,
                          &run_metadata);
  TF_ASSERT_OK(s);

  for (const auto& dev_stat : run_metadata.step_stats().dev_stats()) {
    for (const auto& node_stat : dev_stat.node_stats()) {
      if (node_stat.node_name() == y->name()) {
        EXPECT_LT(0, node_stat.memory(0).total_bytes());
        EXPECT_LT(0, node_stat.memory(0).live_bytes());
        EXPECT_LT(0, node_stat.memory(0).peak_bytes());
      }
    }
  }
}
}  // namespace
}  // namespace tensorflow
