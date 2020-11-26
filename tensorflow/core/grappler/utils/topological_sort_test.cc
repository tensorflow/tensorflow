/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/topological_sort.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/benchmark_testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace grappler {

class TopologicalSortTest : public ::testing::Test {
 protected:
  struct NodeConfig {
    NodeConfig(string name, std::vector<string> inputs)
        : name(std::move(name)), inputs(std::move(inputs)) {}
    NodeConfig(string name, string op, std::vector<string> inputs)
        : name(std::move(name)), op(std::move(op)), inputs(std::move(inputs)) {}

    string name;
    string op;
    std::vector<string> inputs;
  };

  static GraphDef CreateGraph(const std::vector<NodeConfig>& nodes) {
    GraphDef graph;

    for (const NodeConfig& node : nodes) {
      NodeDef node_def;
      node_def.set_name(node.name);
      node_def.set_op(node.op);
      for (const string& input : node.inputs) {
        node_def.add_input(input);
      }
      *graph.add_node() = std::move(node_def);
    }

    return graph;
  }
};

TEST_F(TopologicalSortTest, NoLoop) {
  GraphDef graph = CreateGraph({
      {"2", {"5"}},       //
      {"0", {"5", "4"}},  //
      {"1", {"4", "3"}},  //
      {"3", {"2"}},       //
      {"5", {}},          //
      {"4", {}}           //
  });

  std::vector<const NodeDef*> topo_order;
  TF_EXPECT_OK(ComputeTopologicalOrder(graph, &topo_order));

  const std::vector<string> order = {"5", "4", "2", "0", "3", "1"};

  ASSERT_EQ(topo_order.size(), order.size());
  for (int i = 0; i < topo_order.size(); ++i) {
    const NodeDef* node = topo_order[i];
    EXPECT_EQ(node->name(), order[i]);
  }

  TF_EXPECT_OK(TopologicalSort(&graph));
  for (int i = 0; i < topo_order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, WithLoop) {
  GraphDef graph = CreateGraph({
      // Graph with a loop.
      {"2", "Merge", {"1", "5"}},     //
      {"3", "Switch", {"2"}},         //
      {"4", "Identity", {"3"}},       //
      {"5", "NextIteration", {"4"}},  //
      {"1", {}}                       //
  });

  std::vector<const NodeDef*> topo_order;
  TF_EXPECT_OK(ComputeTopologicalOrder(graph, &topo_order));

  const std::vector<string> order = {"1", "2", "3", "4", "5"};

  ASSERT_EQ(topo_order.size(), order.size());
  for (int i = 0; i < topo_order.size(); ++i) {
    const NodeDef* node = topo_order[i];
    EXPECT_EQ(node->name(), order[i]);
  }

  TF_EXPECT_OK(TopologicalSort(&graph));
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, WithIllegalLoop) {
  // A loop without Merge and NextIteration is illegal and the original node
  // order and graph will be preserved.
  GraphDef graph = CreateGraph({
      {"2", {"1", "3"}},  //
      {"3", {"2"}},       //
      {"1", {}}           //
  });

  EXPECT_FALSE(TopologicalSort(&graph).ok());
  std::vector<string> order = {"2", "3", "1"};
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, DuplicatedInputs) {
  GraphDef graph = CreateGraph({
      {"2", {"1", "1"}},  //
      {"1", {}}           //
  });

  TF_EXPECT_OK(TopologicalSort(&graph));
  std::vector<string> order = {"1", "2"};
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, Idempotent) {
  GraphDef graph = CreateGraph({
      {"1", {}},          //
      {"2", {}},          //
      {"3", {"1", "2"}},  //
      {"4", {"1", "3"}},  //
      {"5", {"2", "3"}}   //
  });

  TF_EXPECT_OK(TopologicalSort(&graph));
  std::vector<string> order = {"1", "2", "3", "4", "5"};
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }

  // Run topo sort again to verify that it is idempotent.
  TF_EXPECT_OK(TopologicalSort(&graph));
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, ExtraDependencies) {
  GraphDef graph = CreateGraph({
      {"2", {"5"}},       //
      {"0", {"5", "4"}},  //
      {"1", {"4", "3"}},  //
      {"3", {"2"}},       //
      {"5", {}},          //
      {"4", {}}           //
  });

  // Add an edge from 4 to 5.
  std::vector<TopologicalDependency> extra_dependencies;
  extra_dependencies.push_back({&graph.node(5), &graph.node(4)});

  std::vector<const NodeDef*> topo_order;
  TF_EXPECT_OK(ComputeTopologicalOrder(graph, extra_dependencies, &topo_order));

  const std::vector<string> valid_order_1 = {"4", "5", "2", "0", "3", "1"};
  const std::vector<string> valid_order_2 = {"4", "5", "0", "2", "3", "1"};

  ASSERT_EQ(topo_order.size(), valid_order_1.size());

  std::vector<string> computed_order(6, "");
  for (int i = 0; i < topo_order.size(); ++i) {
    const NodeDef* node = topo_order[i];
    computed_order[i] = node->name();
  }
  EXPECT_TRUE(computed_order == valid_order_1 ||
              computed_order == valid_order_2);

  // Add an edge from `0` to `4`. This will create a loop.
  extra_dependencies.push_back({&graph.node(1), &graph.node(5)});
  EXPECT_FALSE(
      ComputeTopologicalOrder(graph, extra_dependencies, &topo_order).ok());
}

static void BM_ComputeTopologicalOrder(::testing::benchmark::State& state) {
  const int size = state.range(0);

  GraphDef graph = test::CreateRandomGraph(size);

  std::vector<const NodeDef*> topo_order;
  for (auto s : state) {
    topo_order.clear();
    Status st = ComputeTopologicalOrder(graph, &topo_order);
    CHECK(st.ok()) << "Failed to compute topological order";
  }
}
BENCHMARK(BM_ComputeTopologicalOrder)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(25000)
    ->Arg(50000)
    ->Arg(100000);

}  // namespace grappler
}  // namespace tensorflow
