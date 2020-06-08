/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/canonicalizer.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

NodeDef MakeNode(const string& op) {
  NodeDef node;
  node.set_name("node");
  node.set_op(op);
  *node.add_input() = "b";
  *node.add_input() = "a";
  *node.add_input() = "^z";
  *node.add_input() = "^y";
  *node.add_input() = "^x";
  *node.add_input() = "^z";
  return node;
}

void Verify(const NodeDef& node) {
  EXPECT_EQ(node.name(), "node");
  ASSERT_EQ(node.input_size(), 5);
  if (node.op() == "Div") {
    EXPECT_EQ(node.input(0), "b");
    EXPECT_EQ(node.input(1), "a");
  } else {
    EXPECT_EQ(node.input(0), "a");
    EXPECT_EQ(node.input(1), "b");
  }
  EXPECT_EQ(node.input(2), "^x");
  EXPECT_EQ(node.input(3), "^y");
  EXPECT_EQ(node.input(4), "^z");
}

TEST(CanonicalizeNode, NonCommutative) {
  NodeDef node = MakeNode("Div");
  CanonicalizeNode(&node);
  Verify(node);
}

TEST(CanonicalizeNode, Commutative) {
  NodeDef node = MakeNode("Mul");
  CanonicalizeNode(&node);
  Verify(node);
}

TEST(CanonicalizeGraph, Simple) {
  GraphDef graph;
  *graph.add_node() = MakeNode("Div");
  *graph.add_node() = MakeNode("Mul");
  CanonicalizeGraph(&graph);
  for (auto node : graph.node()) {
    Verify(node);
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
