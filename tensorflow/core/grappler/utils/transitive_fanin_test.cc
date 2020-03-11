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

#include "tensorflow/core/grappler/utils/transitive_fanin.h"

#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class TransitiveFaninTest : public ::testing::Test {
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

TEST_F(TransitiveFaninTest, NoPruning) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}}      //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1"};
  TF_EXPECT_OK(SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes));
  NodeMap node_map(&output_graph);
  ASSERT_TRUE(node_map.NodeExists("1"));
  ASSERT_TRUE(node_map.NodeExists("2"));
  ASSERT_TRUE(node_map.NodeExists("3"));
  ASSERT_TRUE(node_map.NodeExists("4"));
}

TEST_F(TransitiveFaninTest, PruneNodesUnreachableFromSingleTerminalNode) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}},     //
      {"5", {"1"}}   //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1"};
  TF_EXPECT_OK(SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes));
  NodeMap node_map(&output_graph);
  ASSERT_TRUE(node_map.NodeExists("1"));
  ASSERT_TRUE(node_map.NodeExists("2"));
  ASSERT_TRUE(node_map.NodeExists("3"));
  ASSERT_TRUE(node_map.NodeExists("4"));
  ASSERT_FALSE(node_map.NodeExists("5"));
}

TEST_F(TransitiveFaninTest, PruneNodesUnreachableFromMultipleTerminalNodes) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}},     //
      {"5", {"2"}},  //
      {"6", {"1"}}   //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1", "5"};
  TF_EXPECT_OK(SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes));
  NodeMap node_map(&output_graph);
  ASSERT_TRUE(node_map.NodeExists("1"));
  ASSERT_TRUE(node_map.NodeExists("2"));
  ASSERT_TRUE(node_map.NodeExists("3"));
  ASSERT_TRUE(node_map.NodeExists("4"));
  ASSERT_TRUE(node_map.NodeExists("5"));
  ASSERT_FALSE(node_map.NodeExists("6"));
}

TEST_F(TransitiveFaninTest, InvalidGraph) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}},     //
      {"5", {"6"}},  //
      {"7", {"8"}}   //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1", "5"};
  auto s = SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(), "Invalid input graph.");
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
