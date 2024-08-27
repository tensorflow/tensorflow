/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/translate/node_order.h"

#include <string>
#include <utility>
#include <vector>

#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

REGISTER_OP("TestParams").Output("o: float");
REGISTER_OP("TestInput").Output("a: float").Output("b: float");
REGISTER_OP("TestMul").Input("a: float").Input("b: float").Output("o: float");
REGISTER_OP("TestUnary").Input("a: float").Output("o: float");
REGISTER_OP("TestTwoOutputs").Output("a: float").Output("b: float");
REGISTER_OP("TestBinary")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float");

// Compares that the order of nodes in 'inputs' respects the
// pair orders described in 'ordered_pairs'.
bool ExpectBefore(const std::vector<std::pair<string, string>>& ordered_pairs,
                  const std::vector<Node*>& inputs, string* error) {
  for (const std::pair<string, string>& pair : ordered_pairs) {
    const string& before_node = pair.first;
    const string& after_node = pair.second;
    bool seen_before = false;
    bool seen_both = false;
    for (const Node* node : inputs) {
      if (!seen_before && after_node == node->name()) {
        *error = std::string("Saw ") + after_node + std::string(" before ") +
                 before_node;
        return false;
      }

      if (before_node == node->name()) {
        seen_before = true;
      } else if (after_node == node->name()) {
        seen_both = seen_before;
        break;
      }
    }
    if (!seen_both) {
      *error = std::string("didn't see either ") + before_node +
               std::string(" or ") + after_node;
      return false;
    }
  }

  return true;
}

TEST(AlgorithmTest, TopologicalOrdering) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  using namespace ::tensorflow::ops;  // NOLINT
  Node* n1 = SourceOp("TestParams", b.opts().WithName("n1"));
  Node* n2 =
      SourceOp("TestParams", b.opts().WithName("n2").WithControlInput(n1));
  Node* n3 =
      SourceOp("TestParams", b.opts().WithName("n3").WithControlInput(n2));
  Node* n4 = BinaryOp("TestMul", n1, {n3, 0}, b.opts().WithName("n4"));
  Node* n5 = BinaryOp("TestMul", n1, {n3, 0},
                      b.opts().WithName("n5").WithControlInput(n1));
  Node* n6 = BinaryOp("TestMul", n2, {n3, 0}, b.opts().WithName("n6"));
  n3->set_requested_device("a");
  n4->set_requested_device("a");
  n5->set_requested_device("b");
  n6->set_requested_device("b");

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));

  std::vector<Node*> order;

  TopologicalOrdering(g, [&](Node* n) { order.push_back(n); }, GroupByDevice());

  std::vector<std::pair<string, string>> desired_order = {
      {"n1", "n2"},  // because of control dependency
      {"n2", "n3"},  // because of control dependency
      {"n3", "n4"},  // because of NodeScorerDevice
      {"n1", "n4"},  // data dependency
      {"n1", "n5"},  // data dependency
      {"n2", "n6"},  // data dependency
      {"n3", "n4"},  // data dependency
      {"n3", "n5"},  // data dependency
      {"n3", "n6"},  // data dependency
  };
  string error;
  EXPECT_TRUE(ExpectBefore(desired_order, order, &error)) << error;
}

TEST(AlgorithmTest, TopologicalOrderingOnShallowTree) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  using namespace ::tensorflow::ops;  // NOLINT
  Node* n1 = SourceOp("TestParams", b.opts().WithName("n1").WithDevice("a"));
  Node* n2 =
      SourceOp("TestParams",
               b.opts().WithName("n2").WithDevice("b").WithControlInput(n1));
  Node* n3 =
      SourceOp("TestParams",
               b.opts().WithName("n3").WithDevice("c").WithControlInput(n2));
  Node* n4 =
      SourceOp("TestParams",
               b.opts().WithName("n4").WithDevice("a").WithControlInput(n1));
  Node* n5 =
      SourceOp("TestParams",
               b.opts().WithName("n5").WithDevice("b").WithControlInput(n2));
  Node* n6 =
      SourceOp("TestParams",
               b.opts().WithName("n6").WithDevice("c").WithControlInput(n3));
  Node* n7 =
      SourceOp("TestParams",
               b.opts().WithName("n7").WithDevice("a").WithControlInput(n4));
  Node* n8 =
      SourceOp("TestParams",
               b.opts().WithName("n8").WithDevice("b").WithControlInput(n5));
  Node* n9 =
      SourceOp("TestParams",
               b.opts().WithName("n9").WithDevice("c").WithControlInput(n6));

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));

  std::vector<Node*> order;

  TopologicalOrdering(g, [&](Node* n) { order.push_back(n); }, GroupByDevice());

  std::vector<Node*> desired_order = {
      g.source_node(), n1, n4, n7, n2, n5, n8, n3, n6, n9, g.sink_node()};
  for (int i = 0; i < desired_order.size(); i++) {
    desired_order[i] = g.FindNodeId(desired_order[i]->id());
  }
  EXPECT_EQ(order, desired_order);
}

TEST(AlgorithmTest, TopologicalOrderingGivesTheSameResultIfCalledTwice) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  using namespace ::tensorflow::ops;  // NOLINT
  SourceOp("TestParams", b.opts().WithName("n1"));
  SourceOp("TestParams", b.opts().WithName("n2"));
  SourceOp("TestParams", b.opts().WithName("n3"));
  SourceOp("TestParams", b.opts().WithName("n4"));
  SourceOp("TestParams", b.opts().WithName("n5"));
  SourceOp("TestParams", b.opts().WithName("n6"));
  SourceOp("TestParams", b.opts().WithName("n7"));
  SourceOp("TestParams", b.opts().WithName("n8"));
  SourceOp("TestParams", b.opts().WithName("n9"));

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));

  std::vector<Node*> order1;
  std::vector<Node*> order2;

  TopologicalOrdering(
      g, [&](Node* n) { order1.push_back(n); },
      [&](const Node* node) { return std::string("same"); });

  TopologicalOrdering(
      g, [&](Node* n) { order2.push_back(n); },
      [&](const Node* node) { return std::string("same"); });

  EXPECT_EQ(order1, order2);
}

TEST(AlgorithmTest, TopologicalOrderingOnChain) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  using namespace ::tensorflow::ops;  // NOLINT
  Node* n1 = SourceOp("TestParams", b.opts().WithName("n1"));
  Node* n2 = UnaryOp("TestUnary", n1, b.opts().WithName("n2"));
  Node* n3 = UnaryOp("TestUnary", n2, b.opts().WithName("n3"));
  Node* n4 = UnaryOp("TestUnary", n3, b.opts().WithName("n4"));
  Node* n5 = UnaryOp("TestUnary", n4, b.opts().WithName("n5"));
  Node* n6 = UnaryOp("TestUnary", n5, b.opts().WithName("n6"));

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));

  std::vector<Node*> order;
  TopologicalOrdering(g, [&](Node* n) { order.push_back(n); }, GroupByDevice());

  std::vector<Node*> desired_order = {g.source_node(), n1, n2, n3, n4, n5, n6,
                                      g.sink_node()};
  for (int i = 0; i < desired_order.size(); i++) {
    desired_order[i] = g.FindNodeId(desired_order[i]->id());
  }
  EXPECT_EQ(order, desired_order);
}

TEST(AlgorithmTest, TopologicalOrderingOnMultipleOutputs) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  using namespace ::tensorflow::ops;  // NOLINT
  Node* n1 = SourceOp("TestTwoOutputs", b.opts().WithName("n1"));
  UnaryOp("TestUnary", {n1, 0}, b.opts().WithName("n2"));
  UnaryOp("TestUnary", {n1, 1}, b.opts().WithName("n3"));
  UnaryOp("TestUnary", {n1, 0}, b.opts().WithName("n4"));
  UnaryOp("TestUnary", {n1, 1}, b.opts().WithName("n5"));

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));

  std::vector<Node*> order;
  TopologicalOrdering(g, [&](Node* n) { order.push_back(n); }, GroupByDevice());

  std::vector<std::pair<string, string>> desired_order = {
      {"n1", "n2"},
      {"n1", "n3"},
      {"n1", "n4"},
      {"n1", "n5"},
  };
  string error;
  EXPECT_TRUE(ExpectBefore(desired_order, order, &error)) << error;
}

TEST(AlgorithmTest, TopologicalOrderingSameAsReversePostOrder) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  using namespace ::tensorflow::ops;  // NOLINT
  Node* n = SourceOp("TestTwoOutputs", b.opts().WithName("n"));
  Node* n0 = UnaryOp("TestUnary", {n, 0}, b.opts().WithName("n2"));
  Node* n1 = UnaryOp("TestUnary", {n, 1}, b.opts().WithName("n1"));
  UnaryOp("TestUnary", n0, b.opts().WithName("n1a"));
  UnaryOp("TestUnary", n0, b.opts().WithName("n8a"));
  UnaryOp("TestUnary", n0, b.opts().WithName("n2a"));
  UnaryOp("TestUnary", n0, b.opts().WithName("n7a"));
  UnaryOp("TestUnary", n1, b.opts().WithName("n1b"));
  UnaryOp("TestUnary", n1, b.opts().WithName("n8b"));
  UnaryOp("TestUnary", n1, b.opts().WithName("n2b"));
  UnaryOp("TestUnary", n1, b.opts().WithName("n7b"));
  UnaryOp("TestUnary", n0, b.opts().WithName("n3a"));
  UnaryOp("TestUnary", n0, b.opts().WithName("n6a"));
  UnaryOp("TestUnary", n0, b.opts().WithName("n4a"));
  UnaryOp("TestUnary", n0, b.opts().WithName("n5a"));
  UnaryOp("TestUnary", n1, b.opts().WithName("n3b"));
  UnaryOp("TestUnary", n1, b.opts().WithName("n6b"));
  UnaryOp("TestUnary", n1, b.opts().WithName("n4b"));
  UnaryOp("TestUnary", n1, b.opts().WithName("n5b"));

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));

  std::vector<Node*> order;
  TopologicalOrdering(g, [&](Node* n) { order.push_back(n); }, GroupByDevice());

  std::vector<Node*> desired_order;
  GetReversePostOrder(g, &desired_order, [](const Node* n1, const Node* n2) {
    return n1->name() < n2->name();
  });

  EXPECT_EQ(desired_order, order);
}

TEST(AlgorithmTest, TopologicalOrderingWithEachDeviceUsedOnce) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  using namespace ::tensorflow::ops;  // NOLINT
  SourceOp("TestParams", b.opts().WithName("n1").WithDevice("a"));
  SourceOp("TestParams", b.opts().WithName("n2").WithDevice("b"));
  SourceOp("TestParams", b.opts().WithName("n3").WithDevice("c"));
  SourceOp("TestParams", b.opts().WithName("n4").WithDevice("d"));

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));

  int count = 0;
  TopologicalOrdering(g, [&](Node* n) { count++; }, GroupByDevice());
  EXPECT_EQ(count, 6);
}

}  // namespace
}  // namespace tensorflow
