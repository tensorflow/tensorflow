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
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class TopologicalSortTest : public ::testing::Test {
 protected:
  static NodeDef CreateNode(const string& name,
                            const std::vector<string>& inputs) {
    return CreateNode(name, "", inputs);
  }
  static NodeDef CreateNode(const string& name, const string& op,
                            const std::vector<string>& inputs) {
    NodeDef node;
    node.set_name(name);
    if (!op.empty()) {
      node.set_op(op);
    }
    for (const string& input : inputs) {
      node.add_input(input);
    }
    return node;
  }
};

TEST_F(TopologicalSortTest, NoLoop) {
  GraphDef graph;
  *graph.add_node() = CreateNode("2", {"5"});
  *graph.add_node() = CreateNode("0", {"5", "4"});
  *graph.add_node() = CreateNode("1", {"4", "3"});
  *graph.add_node() = CreateNode("3", {"2"});
  *graph.add_node() = CreateNode("5", {});
  *graph.add_node() = CreateNode("4", {});

  TopologicalSort(&graph);
  std::vector<string> order = {"5", "4", "2", "0", "3", "1"};
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, WithLoop) {
  GraphDef graph;
  // Create a loop
  *graph.add_node() = CreateNode("2", "Merge", {"1", "5"});
  *graph.add_node() = CreateNode("3", "Switch", {"2"});
  *graph.add_node() = CreateNode("4", "Identity", {"3"});
  *graph.add_node() = CreateNode("5", "NextIteration", {"4"});
  *graph.add_node() = CreateNode("1", {});

  TopologicalSort(&graph);
  std::vector<string> order = {"1", "2", "3", "4", "5"};
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, WithIllegalLoop) {
  GraphDef graph;
  // A loop without Merge and NextIteration is illegal and the original node
  // order and graph will be preserved.
  *graph.add_node() = CreateNode("2", {"1", "3"});
  *graph.add_node() = CreateNode("3", {"2"});
  *graph.add_node() = CreateNode("1", {});

  TopologicalSort(&graph);
  std::vector<string> order = {"2", "3", "1"};
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, DuplicatedInputs) {
  GraphDef graph;
  *graph.add_node() = CreateNode("2", {"1", "1"});
  *graph.add_node() = CreateNode("1", {});

  TopologicalSort(&graph);
  std::vector<string> order = {"1", "2"};
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
