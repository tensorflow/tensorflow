/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/traversal.h"

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class TraversalTest : public ::testing::Test {
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

TEST_F(TraversalTest, ReverseDfsNoLoop) {
  GraphDef graph;
  *graph.add_node() = CreateNode("2", {"5"});
  *graph.add_node() = CreateNode("0", {"5", "4"});
  *graph.add_node() = CreateNode("1", {"4", "3"});
  *graph.add_node() = CreateNode("3", {"2"});
  *graph.add_node() = CreateNode("5", {});
  *graph.add_node() = CreateNode("4", {});

  std::vector<const NodeDef*> start_nodes = {&graph.node(1), &graph.node(2)};
  std::vector<string> pre_order;
  std::vector<string> post_order;
  bool found_back_edge = false;
  ReverseDfs(
      GraphView(&graph), start_nodes,
      [&pre_order](const NodeDef* n) { pre_order.push_back(n->name()); },
      [&post_order](const NodeDef* n) { post_order.push_back(n->name()); },
      [&found_back_edge](const NodeDef*, const NodeDef*) {
        found_back_edge = true;
      });

  // Pre/Post order traversals are non deterministic because a node fanin is an
  // absl::flat_hash_set with non deterministic traversal order.
  using ValidTraversal = std::pair<std::vector<string>, std::vector<string>>;

  std::set<ValidTraversal> valid_traversals = {
      // pre_order                     post_order
      {{"1", "4", "3", "2", "5", "0"}, {"4", "5", "2", "3", "1", "0"}},
      {{"1", "3", "2", "5", "4", "0"}, {"5", "2", "3", "4", "1", "0"}}};

  EXPECT_EQ(valid_traversals.count({pre_order, post_order}), 1);
  EXPECT_FALSE(found_back_edge);
}

TEST_F(TraversalTest, ReverseDfsWithLoop) {
  GraphDef graph;
  // Create a loop
  *graph.add_node() = CreateNode("2", "Merge", {"1", "5"});
  *graph.add_node() = CreateNode("3", "Switch", {"2"});
  *graph.add_node() = CreateNode("4", "Identity", {"3"});
  *graph.add_node() = CreateNode("5", "NextIteration", {"4"});
  *graph.add_node() = CreateNode("1", "Enter", {});
  *graph.add_node() = CreateNode("6", "Exit", {"3"});

  std::vector<const NodeDef*> start_nodes = {&graph.node(5)};
  std::vector<string> pre_order;
  std::vector<string> post_order;
  std::vector<string> back_edges;
  ReverseDfs(
      GraphView(&graph), start_nodes,
      [&pre_order](const NodeDef* n) { pre_order.push_back(n->name()); },
      [&post_order](const NodeDef* n) { post_order.push_back(n->name()); },
      [&back_edges](const NodeDef* src, const NodeDef* dst) {
        back_edges.push_back(strings::StrCat(src->name(), "->", dst->name()));
      });

  // Pre/Post order traversals are non deterministic because a node fanin is an
  // absl::flat_hash_set with non deterministic traversal order.
  using ValidTraversal = std::pair<std::vector<string>, std::vector<string>>;

  std::set<ValidTraversal> valid_traversals = {
      // pre_order                     post_order
      {{"6", "3", "2", "4", "5", "1"}, {"5", "4", "1", "2", "3", "6"}},
      {{"6", "3", "2", "1", "5", "4"}, {"1", "4", "5", "2", "3", "6"}},
      {{"6", "3", "2", "5", "4", "1"}, {"4", "5", "1", "2", "3", "6"}}};

  EXPECT_EQ(valid_traversals.count({pre_order, post_order}), 1);
  EXPECT_EQ(std::vector<string>({"4->3"}), back_edges);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
