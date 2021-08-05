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

#include "tensorflow/core/graph/costmodel.h"

#include <memory>
#include <string>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

static void InitGraph(const string& s, Graph* graph) {
  GraphDef graph_def;

  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(s, &graph_def)) << s;
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
}

static void GenerateStepStats(Graph* graph, StepStats* step_stats,
                              const string& device_name) {
  // Fill RunMetadata's step_stats and partition_graphs fields.
  DeviceStepStats* device_stepstats = step_stats->add_dev_stats();
  device_stepstats->set_device(device_name);
  for (const auto& node_def : graph->nodes()) {
    NodeExecStats* node_stats = device_stepstats->add_node_stats();
    node_stats->set_node_name(node_def->name());
  }
}

REGISTER_OP("Input").Output("o: float").SetIsStateful();

TEST(CostModelTest, GlobalId) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  std::unique_ptr<Graph> graph1 =
      std::unique_ptr<Graph>(new Graph(OpRegistry::Global()));
  std::unique_ptr<Graph> graph2 =
      std::unique_ptr<Graph>(new Graph(OpRegistry::Global()));
  InitGraph(
      "node { name: 'A1' op: 'Input'}"
      "node { name: 'B1' op: 'Input'}"
      "node { name: 'C1' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A1', 'B1'] }"
      "node { name: 'D1' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A1', 'B1'] }",
      graph1.get());
  InitGraph(
      "node { name: 'A2' op: 'Input'}"
      "node { name: 'B2' op: 'Input'}"
      "node { name: 'C2' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A2', 'B2'] }"
      "node { name: 'D2' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A2', 'B2'] }",
      graph2.get());
  StepStats step_stats;
  GenerateStepStats(graph1.get(), &step_stats, "DummyDevice1");
  GenerateStepStats(graph2.get(), &step_stats, "DummyDevice2");
  StepStatsCollector collector(&step_stats);
  std::unordered_map<string, const Graph*> device_map;
  device_map["DummyDevice1"] = graph1.get();
  device_map["DummyDevice2"] = graph2.get();
  CostModelManager cost_model_manager;
  collector.BuildCostModel(&cost_model_manager, device_map);
  CostGraphDef cost_graph_def;
  TF_ASSERT_OK(
      cost_model_manager.AddToCostGraphDef(graph1.get(), &cost_graph_def));
  TF_ASSERT_OK(
      cost_model_manager.AddToCostGraphDef(graph2.get(), &cost_graph_def));
  ASSERT_EQ(cost_graph_def.node_size(), 12);
  absl::flat_hash_map<int32, const CostGraphDef::Node> ids;
  for (auto node : cost_graph_def.node()) {
    int32_t index = node.id();
    auto result = ids.insert({index, node});
    EXPECT_TRUE(result.second);
  }
}

}  // namespace
}  // namespace tensorflow
