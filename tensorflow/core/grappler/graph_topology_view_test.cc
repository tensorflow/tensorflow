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

#include "tensorflow/core/grappler/graph_topology_view.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class GraphTopologyViewTest : public ::testing::Test {
 protected:
  using NodeConfig = std::pair<string, std::vector<string>>;

  static GraphDef CreateGraph(const std::vector<NodeConfig>& nodes) {
    GraphDef graph;

    for (const NodeConfig& node : nodes) {
      const auto& node_name = node.first;
      const auto& node_inputs = node.second;

      NodeDef node_def;
      node_def.set_name(node_name);
      for (const string& input : node_inputs) {
        node_def.add_input(input);
      }

      *graph.add_node() = std::move(node_def);
    }

    return graph;
  }
};

TEST_F(GraphTopologyViewTest, SimpleGraph) {
  const GraphDef graph = CreateGraph({
      {"a", {}},          // idx: 0
      {"b", {}},          // idx: 1
      {"c", {"a", "b"}},  // idx: 2
      {"d", {"a", "c"}},  // idx: 3
  });

  GraphTopologyView graph_view;
  TF_CHECK_OK(graph_view.InitializeFromGraph(graph));

  EXPECT_TRUE(graph_view.is_initialized());

  const NodeDef* a_by_name = graph_view.GetNode("a");
  const NodeDef* a_by_idx = graph_view.GetNode(0);
  ASSERT_TRUE(a_by_name);
  ASSERT_TRUE(a_by_idx);
  EXPECT_EQ(a_by_name, a_by_idx);

  const NodeDef* b_by_name = graph_view.GetNode("b");
  const NodeDef* b_by_idx = graph_view.GetNode(1);
  ASSERT_TRUE(b_by_name);
  ASSERT_TRUE(b_by_idx);
  EXPECT_EQ(b_by_name, b_by_idx);

  const absl::optional<int> b_idx = graph_view.GetNodeIndex(*b_by_name);
  ASSERT_TRUE(b_idx.has_value());
  EXPECT_EQ(b_idx.value(), 1);

  const absl::optional<int> c_idx = graph_view.GetNodeIndex("c");
  ASSERT_TRUE(c_idx.has_value());
  EXPECT_EQ(c_idx.value(), 2);

  using Fanin = absl::InlinedVector<int, 4>;
  EXPECT_EQ(graph_view.GetFanin(0), Fanin());
  EXPECT_EQ(graph_view.GetFanin(1), Fanin());
  EXPECT_EQ(graph_view.GetFanin(2), Fanin({0, 1}));
  EXPECT_EQ(graph_view.GetFanin(3), Fanin({0, 2}));

  using Fanout = absl::InlinedVector<int, 2>;
  EXPECT_EQ(graph_view.GetFanout(0), Fanout({2, 3}));
  EXPECT_EQ(graph_view.GetFanout(1), Fanout({2}));
  EXPECT_EQ(graph_view.GetFanout(2), Fanout({3}));
  EXPECT_EQ(graph_view.GetFanout(3), Fanout());
}

TEST_F(GraphTopologyViewTest, GraphWithALoop) {
  const GraphDef graph = CreateGraph({
      {"a", {}},               // idx: 0
      {"b", {}},               // idx: 1
      {"c", {"a", "b", "d"}},  // idx: 2 <<<--- 'c' and 'd' have a loop
      {"d", {"a", "c"}},       // idx: 3
  });

  GraphTopologyView graph_view;
  TF_CHECK_OK(graph_view.InitializeFromGraph(graph));
  EXPECT_TRUE(graph_view.is_initialized());

  using Fanin = absl::InlinedVector<int, 4>;
  EXPECT_EQ(graph_view.GetFanin(2), Fanin({0, 1, 3}));
  EXPECT_EQ(graph_view.GetFanin(3), Fanin({0, 2}));

  using Fanout = absl::InlinedVector<int, 2>;
  EXPECT_EQ(graph_view.GetFanout(2), Fanout({3}));
  EXPECT_EQ(graph_view.GetFanout(3), Fanout({2}));
}

}  // namespace grappler
}  // namespace tensorflow
