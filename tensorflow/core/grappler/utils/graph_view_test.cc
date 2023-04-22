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

#include "tensorflow/core/grappler/utils/graph_view.h"

#include <type_traits>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/benchmark_testlib.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace grappler {
namespace utils {
namespace {

using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;

constexpr char kNoOp[] = "NoOp";

GraphDef SimpleTestGraph() {
  return GDef({NDef("a", kNoOp, {"b:2", "d:3", "b:2", "d:3", "^c"}),
               NDef("b", kNoOp, {"d:2", "c:5", "^c"}),
               NDef("c", kNoOp, {"^d", "^d"}), NDef("d", kNoOp, {})},
              /*funcs=*/{});
}

template <typename T>
const string GetGraphViewTypeAsString() {
  return std::is_same<T, class GraphView>::value ? "GraphView"
                                                 : "MutableGraphView";
}

using GraphViewTypes = ::testing::Types<GraphView, MutableGraphView>;

template <typename T>
class TypedGraphViewTest : public ::testing::Test {};
TYPED_TEST_SUITE(TypedGraphViewTest, GraphViewTypes);

TYPED_TEST(TypedGraphViewTest, GraphWithDuplicateNodeNames) {
  GraphDef graph =
      GDef({NDef("a", kNoOp, {}), NDef("a", kNoOp, {})}, /*funcs=*/{});

  Status s;
  TypeParam graph_view(&graph, &s);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            absl::Substitute(
                "$0::$0 error: graph has multiple nodes with the name 'a'.",
                GetGraphViewTypeAsString<TypeParam>()));
}

TYPED_TEST(TypedGraphViewTest, GraphWithMissingFanins) {
  GraphDef graph = GDef({NDef("a", kNoOp, {"b:3"})}, /*funcs=*/{});

  Status s;
  TypeParam graph_view(&graph, &s);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            absl::Substitute("$0::$0 error: node 'a' has missing fanin 'b:3'.",
                             GetGraphViewTypeAsString<TypeParam>()));
}

TYPED_TEST(TypedGraphViewTest, GraphWithSelfCycles) {
  GraphDef graph = GDef({NDef("a", kNoOp, {"a:4"})}, /*funcs=*/{});

  Status s;
  TypeParam graph_view(&graph, &s);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(
      s.error_message(),
      absl::Substitute("$0::$0 error: node 'a' has self cycle fanin 'a:4'.",
                       GetGraphViewTypeAsString<TypeParam>()));
}

TYPED_TEST(TypedGraphViewTest, GraphWithMisorderedFanins) {
  GraphDef graph = GDef({NDef("a", kNoOp, {"^b", "b:4"}), NDef("b", kNoOp, {})},
                        /*funcs=*/{});

  Status s;
  TypeParam graph_view(&graph, &s);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            absl::Substitute("$0::$0 error: node 'a' has regular fanin 'b:4' "
                             "after controlling fanins.",
                             GetGraphViewTypeAsString<TypeParam>()));
}

TYPED_TEST(TypedGraphViewTest, GetNodeWithIndex) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  const int num_nodes = graph_view.NumNodes();
  ASSERT_EQ(graph_view.NumNodes(), graph.node_size());
  for (int i = 0; i < num_nodes; ++i) {
    const auto* node = graph_view.GetNode(i);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->node(), graph.mutable_node(i));
  }

  const auto* bad_node = graph_view.GetNode(-1);
  ASSERT_EQ(bad_node, nullptr);
  bad_node = graph_view.GetNode(num_nodes);
  ASSERT_EQ(bad_node, nullptr);
}

TYPED_TEST(TypedGraphViewTest, GetNodeWithName) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  std::vector<string> node_names = {"a", "b", "c", "d"};
  for (int i = 0; i < node_names.size(); ++i) {
    const string& node_name = node_names[i];
    const auto* node = graph_view.GetNode(node_name);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->node(), graph.mutable_node(i));
  }

  // Missing node.
  const auto* bad_node = graph_view.GetNode("e");
  ASSERT_EQ(bad_node, nullptr);
}

TYPED_TEST(TypedGraphViewTest, GetNodes) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  const auto& nodes = graph_view.GetNodes();
  const int num_nodes = nodes.size();
  EXPECT_EQ(num_nodes, 4);

  ASSERT_EQ(num_nodes, graph.node_size());
  for (int i = 0; i < num_nodes; ++i) {
    EXPECT_EQ(nodes[i].node(), graph.mutable_node(i));
  }
}

TYPED_TEST(TypedGraphViewTest, HasNode) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  for (const string& node_name : {"a", "b", "c", "d"}) {
    EXPECT_TRUE(graph_view.HasNode(node_name));
  }

  // Missing node.
  EXPECT_FALSE(graph_view.HasNode("e"));
}

TYPED_TEST(TypedGraphViewTest, NumNodes) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  EXPECT_EQ(graph_view.NumNodes(), 4);
}

TYPED_TEST(TypedGraphViewTest, NumNodesEmptyGraph) {
  GraphDef graph;

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  EXPECT_EQ(graph_view.NumNodes(), 0);
}

TEST(MutableGraphViewTest, DedupControlDependencies) {
  GraphDef graph = GDef(
      {NDef("a", kNoOp, {}), NDef("b", kNoOp, {}), NDef("c", kNoOp, {}),
       NDef("d", kNoOp, {"a:2", "b:1", "^c", "^c", "^a", "^a", "^b", "^c"})},
      /*funcs=*/{});

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  EXPECT_EQ(graph_view.NumNodes(), 4);

  const auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  const auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  const auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  const auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  EXPECT_EQ(d_node->NumRegularFanins(), 2);
  ASSERT_NE(d_node->node(), nullptr);
  ASSERT_EQ(d_node->node()->input_size(), 5);
  EXPECT_EQ(d_node->node()->input(0), "a:2");
  EXPECT_EQ(d_node->node()->input(1), "b:1");
  EXPECT_EQ(d_node->node()->input(2), "^c");
  EXPECT_EQ(d_node->node()->input(3), "^b");
  EXPECT_EQ(d_node->node()->input(4), "^a");
  ASSERT_EQ(d_node->NumControllingFanins(), 3);
  const auto& d_control_fanins = d_node->GetControllingFanins();
  ASSERT_EQ(d_control_fanins.size(), 3);
  ASSERT_NE(d_control_fanins[0].node_view(), nullptr);
  EXPECT_EQ(d_control_fanins[0].node_view()->GetName(), "c");
  ASSERT_NE(d_control_fanins[1].node_view(), nullptr);
  EXPECT_EQ(d_control_fanins[1].node_view()->GetName(), "b");
  ASSERT_NE(d_control_fanins[2].node_view(), nullptr);
  EXPECT_EQ(d_control_fanins[2].node_view()->GetName(), "a");
}

template <typename T>
class TypedNodeViewTest : public ::testing::Test {};
TYPED_TEST_SUITE(TypedNodeViewTest, GraphViewTypes);

TYPED_TEST(TypedNodeViewTest, GetName) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  for (const NodeDef& node : graph.node()) {
    const auto* node_view = graph_view.GetNode(node.name());
    ASSERT_NE(node_view, nullptr);
    EXPECT_EQ(node_view->GetName(), node.name());
    EXPECT_EQ(node_view->GetName(), node_view->node()->name());
  }
}

TYPED_TEST(TypedNodeViewTest, GetOp) {
  GraphDef graph = GDef({NDef("a", "op_a", {}), NDef("b", "op_b", {}),
                         NDef("c", "op_c", {}), NDef("d", "op_d", {})},
                        /*funcs=*/{});

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  const auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  EXPECT_EQ(a_node->GetOp(), "op_a");
  EXPECT_EQ(a_node->node()->op(), "op_a");
  const auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  EXPECT_EQ(b_node->GetOp(), "op_b");
  EXPECT_EQ(b_node->node()->op(), "op_b");
  const auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  EXPECT_EQ(c_node->GetOp(), "op_c");
  EXPECT_EQ(c_node->node()->op(), "op_c");
  const auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  EXPECT_EQ(d_node->GetOp(), "op_d");
  EXPECT_EQ(d_node->node()->op(), "op_d");
}

TYPED_TEST(TypedNodeViewTest, GetDevice) {
  GraphDef graph = GDef(
      {NDef("a", "", {}, {}, "device_a"), NDef("b", "", {}, {}, "device_b"),
       NDef("c", "", {}, {}, "device_c"), NDef("d", "", {}, {})},
      /*funcs=*/{});

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  const auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  EXPECT_EQ(a_node->GetDevice(), "device_a");
  EXPECT_EQ(a_node->node()->device(), "device_a");
  const auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  EXPECT_EQ(b_node->GetDevice(), "device_b");
  EXPECT_EQ(b_node->node()->device(), "device_b");
  const auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  EXPECT_EQ(c_node->GetDevice(), "device_c");
  EXPECT_EQ(c_node->node()->device(), "device_c");
  const auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  EXPECT_EQ(d_node->GetDevice(), "");
  EXPECT_EQ(d_node->node()->device(), "");
}

template <typename T>
class TypedFaninTest : public ::testing::Test {};
using FaninTypes =
    ::testing::Types<std::pair<FanoutView, GraphView>,
                     std::pair<MutableFanoutView, MutableGraphView>>;
TYPED_TEST_SUITE(TypedFaninTest, FaninTypes);

TYPED_TEST(TypedFaninTest, GetRegularFanins) {
  using FanoutViewType = typename TypeParam::first_type;
  using GraphViewType = typename TypeParam::second_type;

  GraphDef graph = SimpleTestGraph();

  Status s;
  GraphViewType graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  const auto& a_fanins = a_node->GetRegularFanins();
  ASSERT_EQ(a_fanins.size(), 4);
  EXPECT_EQ(a_fanins[0], FanoutViewType(&graph_view, b_node->node_index(), 2));
  EXPECT_EQ(a_fanins[1], FanoutViewType(&graph_view, d_node->node_index(), 3));
  EXPECT_EQ(a_fanins[2], FanoutViewType(&graph_view, b_node->node_index(), 2));
  EXPECT_EQ(a_fanins[3], FanoutViewType(&graph_view, d_node->node_index(), 3));

  const auto& d_fanins = d_node->GetRegularFanins();
  EXPECT_EQ(d_fanins.size(), 0);
}

TYPED_TEST(TypedFaninTest, GetRegularFanin) {
  using FanoutViewType = typename TypeParam::first_type;
  using GraphViewType = typename TypeParam::second_type;

  GraphDef graph = SimpleTestGraph();

  Status s;
  GraphViewType graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  const auto& a_fanin_0 = a_node->GetRegularFanin(0);
  EXPECT_EQ(a_fanin_0, FanoutViewType(&graph_view, b_node->node_index(), 2));
  const auto& a_fanin_1 = a_node->GetRegularFanin(1);
  EXPECT_EQ(a_fanin_1, FanoutViewType(&graph_view, d_node->node_index(), 3));
  const auto& a_fanin_2 = a_node->GetRegularFanin(2);
  EXPECT_EQ(a_fanin_2, FanoutViewType(&graph_view, b_node->node_index(), 2));
  const auto& a_fanin_3 = a_node->GetRegularFanin(3);
  EXPECT_EQ(a_fanin_3, FanoutViewType(&graph_view, d_node->node_index(), 3));

  // Out of bounds.
  const FanoutViewType missing_fanin;
  EXPECT_EQ(missing_fanin, FanoutViewType(nullptr, -1, -2));
  EXPECT_EQ(missing_fanin.node_view(), nullptr);
  const auto& a_fanin_4 = a_node->GetRegularFanin(4);
  EXPECT_EQ(a_fanin_4, missing_fanin);
  const auto& a_fanin_5 = a_node->GetRegularFanin(5);
  EXPECT_EQ(a_fanin_5, missing_fanin);
  const auto& a_fanin_control = a_node->GetRegularFanin(Graph::kControlSlot);
  EXPECT_EQ(a_fanin_control, missing_fanin);
  const auto& a_fanin_bad = a_node->GetRegularFanin(-2);
  EXPECT_EQ(a_fanin_bad, missing_fanin);
}

TYPED_TEST(TypedFaninTest, GetControllingFanins) {
  using FanoutViewType = typename TypeParam::first_type;
  using GraphViewType = typename TypeParam::second_type;

  GraphDef graph = SimpleTestGraph();

  Status s;
  GraphViewType graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  const auto& a_fanins = a_node->GetControllingFanins();
  ASSERT_EQ(a_fanins.size(), 1);
  EXPECT_EQ(a_fanins[0], FanoutViewType(&graph_view, c_node->node_index(),
                                        Graph::kControlSlot));

  const auto& c_fanins = c_node->GetControllingFanins();
  FanoutViewType d_control_fanin(&graph_view, d_node->node_index(),
                                 Graph::kControlSlot);
  if (std::is_same<GraphViewType, GraphView>::value) {
    ASSERT_EQ(c_fanins.size(), 2);
    EXPECT_EQ(c_fanins[0], d_control_fanin);
    EXPECT_EQ(c_fanins[1], d_control_fanin);
  } else {  // MutableGraphView will dedup control dependency.
    ASSERT_EQ(c_fanins.size(), 1);
    EXPECT_EQ(c_fanins[0], d_control_fanin);
  }

  const auto& d_fanins = d_node->GetControllingFanins();
  EXPECT_EQ(d_fanins.size(), 0);
}

template <typename T>
class TypedFanoutTest : public ::testing::Test {};
using FanoutTypes =
    ::testing::Types<std::pair<FaninView, GraphView>,
                     std::pair<MutableFaninView, MutableGraphView>>;
TYPED_TEST_SUITE(TypedFanoutTest, FanoutTypes);

TYPED_TEST(TypedFanoutTest, GetRegularFanouts) {
  using FaninViewType = typename TypeParam::first_type;
  using GraphViewType = typename TypeParam::second_type;

  GraphDef graph = SimpleTestGraph();

  Status s;
  GraphViewType graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  const auto& d_fanouts = d_node->GetRegularFanouts();
  ASSERT_EQ(d_fanouts.size(), 4);
  for (int i = 0; i < d_fanouts.size(); ++i) {
    if (i == 2) {
      ASSERT_EQ(d_fanouts[i].size(), 1);
      EXPECT_EQ(d_fanouts[i][0],
                FaninViewType(&graph_view, b_node->node_index(), 0));
    } else if (i == 3) {
      ASSERT_EQ(d_fanouts[i].size(), 2);
      absl::flat_hash_set<FaninViewType> fanouts(d_fanouts[i].begin(),
                                                 d_fanouts[i].end());
      EXPECT_TRUE(fanouts.contains(
          FaninViewType(&graph_view, a_node->node_index(), 1)));
      EXPECT_TRUE(fanouts.contains(
          FaninViewType(&graph_view, a_node->node_index(), 3)));
    } else {
      EXPECT_EQ(d_fanouts[i].size(), 0);
    }
  }

  const auto& a_fanouts = a_node->GetRegularFanouts();
  EXPECT_EQ(a_fanouts.size(), 0);
}

TYPED_TEST(TypedFanoutTest, GetRegularFanout) {
  using FaninViewType = typename TypeParam::first_type;
  using GraphViewType = typename TypeParam::second_type;

  GraphDef graph = SimpleTestGraph();

  Status s;
  GraphViewType graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  const auto& d_fanouts_2 = d_node->GetRegularFanout(2);
  ASSERT_EQ(d_fanouts_2.size(), 1);
  EXPECT_EQ(d_fanouts_2.at(0),
            FaninViewType(&graph_view, b_node->node_index(), 0));

  const auto& d_fanouts_3 = d_node->GetRegularFanout(3);
  EXPECT_EQ(d_fanouts_3.size(), 2);
  absl::flat_hash_set<FaninViewType> d_fanouts_3_set(d_fanouts_3.begin(),
                                                     d_fanouts_3.end());
  EXPECT_TRUE(d_fanouts_3_set.contains(
      FaninViewType(&graph_view, a_node->node_index(), 1)));
  EXPECT_TRUE(d_fanouts_3_set.contains(
      FaninViewType(&graph_view, a_node->node_index(), 3)));

  // Invalid or empty.
  const std::vector<FaninViewType> no_fanouts;
  EXPECT_EQ(d_node->GetRegularFanout(-2), no_fanouts);
  EXPECT_EQ(d_node->GetRegularFanout(Graph::kControlSlot), no_fanouts);
  EXPECT_EQ(d_node->GetRegularFanout(0), no_fanouts);
  EXPECT_EQ(d_node->GetRegularFanout(1), no_fanouts);
  EXPECT_EQ(d_node->GetRegularFanout(4), no_fanouts);
  EXPECT_EQ(d_node->GetRegularFanout(5), no_fanouts);
}

TYPED_TEST(TypedFanoutTest, GetControlledFanouts) {
  using FaninViewType = typename TypeParam::first_type;
  using GraphViewType = typename TypeParam::second_type;

  GraphDef graph = SimpleTestGraph();

  Status s;
  GraphViewType graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  const auto& c_fanouts = c_node->GetControlledFanouts();
  EXPECT_EQ(c_fanouts.size(), 2);
  absl::flat_hash_set<FaninViewType> c_fanouts_set(c_fanouts.begin(),
                                                   c_fanouts.end());
  EXPECT_TRUE(c_fanouts_set.contains(
      FaninViewType(&graph_view, b_node->node_index(), Graph::kControlSlot)));
  EXPECT_TRUE(c_fanouts_set.contains(
      FaninViewType(&graph_view, a_node->node_index(), Graph::kControlSlot)));

  const auto& d_fanouts = d_node->GetControlledFanouts();
  FaninViewType c_control_fanout(&graph_view, c_node->node_index(),
                                 Graph::kControlSlot);
  if (std::is_same<GraphViewType, GraphView>::value) {
    ASSERT_EQ(d_fanouts.size(), 2);
    EXPECT_EQ(d_fanouts[0], c_control_fanout);
    EXPECT_EQ(d_fanouts[1], c_control_fanout);
  } else {  // MutableGraphView will dedup control dependency.
    ASSERT_EQ(d_fanouts.size(), 1);
    EXPECT_EQ(d_fanouts[0], c_control_fanout);
  }

  const auto& a_fanouts = a_node->GetControlledFanouts();
  EXPECT_EQ(a_fanouts.size(), 0);
}

TYPED_TEST(TypedNodeViewTest, NumRegularFanins) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  EXPECT_EQ(a_node->NumRegularFanins(), 4);
  EXPECT_EQ(b_node->NumRegularFanins(), 2);
  EXPECT_EQ(c_node->NumRegularFanins(), 0);
  EXPECT_EQ(d_node->NumRegularFanins(), 0);
}

TYPED_TEST(TypedNodeViewTest, NumControllingFanins) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  EXPECT_EQ(a_node->NumControllingFanins(), 1);
  EXPECT_EQ(b_node->NumControllingFanins(), 1);
  if (std::is_same<TypeParam, GraphView>::value) {
    EXPECT_EQ(c_node->NumControllingFanins(), 2);
  } else {
    EXPECT_EQ(c_node->NumControllingFanins(), 1);
  }
  EXPECT_EQ(d_node->NumControllingFanins(), 0);
}

TYPED_TEST(TypedNodeViewTest, NumRegularFanouts) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  EXPECT_EQ(a_node->NumRegularFanouts(), 0);
  EXPECT_EQ(b_node->NumRegularFanouts(), 2);
  EXPECT_EQ(c_node->NumRegularFanouts(), 1);
  EXPECT_EQ(d_node->NumRegularFanouts(), 3);
}

TYPED_TEST(TypedNodeViewTest, NumControlledFanouts) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  EXPECT_EQ(a_node->NumControlledFanouts(), 0);
  EXPECT_EQ(b_node->NumControlledFanouts(), 0);
  EXPECT_EQ(c_node->NumControlledFanouts(), 2);
  if (std::is_same<TypeParam, GraphView>::value) {
    EXPECT_EQ(d_node->NumControlledFanouts(), 2);
  } else {
    EXPECT_EQ(d_node->NumControlledFanouts(), 1);
  }
}

TYPED_TEST(TypedNodeViewTest, HasFanin) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);

  // Existing regular fanin.
  EXPECT_TRUE(a_node->HasFanin({&graph_view, b_node->node_index(), 2}));
  // Missing regular fanin.
  EXPECT_FALSE(a_node->HasFanin({&graph_view, c_node->node_index(), 4}));
  // Existing controlling fanin.
  EXPECT_TRUE(a_node->HasFanin(
      {&graph_view, c_node->node_index(), Graph::kControlSlot}));
  // Missing controlling fanin.
  EXPECT_FALSE(a_node->HasFanin(
      {&graph_view, b_node->node_index(), Graph::kControlSlot}));
  // Bad fanins.
  EXPECT_FALSE(a_node->HasFanin({&graph_view, a_node->node_index(), 0}));
  EXPECT_FALSE(a_node->HasFanin(
      {&graph_view, b_node->node_index(), internal::kMissingSlot}));
}

TYPED_TEST(TypedNodeViewTest, HasFanout) {
  GraphDef graph = SimpleTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  auto* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  // Existing regular fanout.
  EXPECT_TRUE(b_node->HasFanout({&graph_view, a_node->node_index(), 2}));
  // Missing regular fanout.
  EXPECT_FALSE(b_node->HasFanout({&graph_view, a_node->node_index(), 1}));
  // Existing controlled fanout.
  EXPECT_TRUE(d_node->HasFanout(
      {&graph_view, c_node->node_index(), Graph::kControlSlot}));
  // Missing controlled fanout.
  EXPECT_FALSE(d_node->HasFanout(
      {&graph_view, a_node->node_index(), Graph::kControlSlot}));
  // Bad fanouts.
  EXPECT_FALSE(d_node->HasFanout({&graph_view, d_node->node_index(), 0}));
  EXPECT_FALSE(a_node->HasFanout({&graph_view, b_node->node_index(), 0}));
  EXPECT_FALSE(a_node->HasFanout({&graph_view, 4, 0}));
  EXPECT_FALSE(d_node->HasFanout(
      {&graph_view, b_node->node_index(), internal::kMissingSlot}));
}

GraphDef SimpleAttrTestGraph() {
  return GDef({NDef("a", kNoOp, {}), NDef("b", kNoOp, {}, {{"attr", 1}}),
               NDef("c", kNoOp, {}, {{"attr_1", "a"}, {"attr_2", 2.0f}})},
              /*funcs=*/{});
}

TYPED_TEST(TypedNodeViewTest, GetAttr) {
  GraphDef graph = SimpleAttrTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);

  EXPECT_EQ(c_node->GetAttr("attr_1")->s(), "a");
}

TYPED_TEST(TypedNodeViewTest, GetAttrs) {
  GraphDef graph = SimpleAttrTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);

  const auto& actual_attrs = c_node->GetAttrs();
  EXPECT_EQ(actual_attrs.size(), 2);
  const auto* attr_1 = actual_attrs.Find("attr_1");
  EXPECT_NE(attr_1, nullptr);
  EXPECT_EQ(attr_1->s(), "a");
  const auto* attr_2 = actual_attrs.Find("attr_2");
  EXPECT_NE(attr_2, nullptr);
  EXPECT_EQ(attr_2->f(), 2.0f);
}

TYPED_TEST(TypedNodeViewTest, NumAttrs) {
  GraphDef graph = SimpleAttrTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  auto* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);

  EXPECT_EQ(a_node->NumAttrs(), 0);
  EXPECT_EQ(b_node->NumAttrs(), 1);
  EXPECT_EQ(c_node->NumAttrs(), 2);
}

TYPED_TEST(TypedNodeViewTest, HasAttr) {
  GraphDef graph = SimpleAttrTestGraph();

  Status s;
  TypeParam graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  auto* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);

  EXPECT_TRUE(c_node->HasAttr("attr_1"));
  EXPECT_FALSE(c_node->HasAttr("attr"));
}

class CompareGraphTest : public GrapplerTest {
 public:
  void CompareGraphViewWithGraph(MutableGraphView* graph_view,
                                 const GraphDef& expected_graph) {
    Status s;
    GraphView expected_graph_view(&expected_graph, &s);
    TF_ASSERT_OK(s);

    EXPECT_EQ(graph_view->NumNodes(), expected_graph_view.NumNodes());

    for (const NodeView& expected_node_view : expected_graph_view.GetNodes()) {
      const string& node_name = expected_node_view.GetName();
      MutableNodeView* node_view = graph_view->GetNode(node_name);
      ASSERT_NE(node_view, nullptr);

      EXPECT_EQ(node_view->GetName(), expected_node_view.GetName());

      EXPECT_EQ(node_view->GetOp(), expected_node_view.GetOp());

      EXPECT_EQ(node_view->GetDevice(), expected_node_view.GetDevice());

      const int actual_num_fanins = node_view->node()->input_size();
      EXPECT_EQ(actual_num_fanins, expected_node_view.node()->input_size());

      const int expected_num_regular_fanins =
          expected_node_view.NumRegularFanins();
      bool same_num_regular_fanins =
          node_view->NumRegularFanins() == expected_num_regular_fanins;
      EXPECT_TRUE(same_num_regular_fanins);
      for (int i = 0; i < expected_num_regular_fanins; ++i) {
        const auto& expected_fanin = expected_node_view.GetRegularFanin(i);

        auto* actual_fanin_node =
            graph_view->GetNode(expected_fanin.node_view()->GetName());
        ASSERT_NE(actual_fanin_node, nullptr);
        EXPECT_TRUE(
            node_view->HasFanin({actual_fanin_node, expected_fanin.index()}));
        if (i < node_view->NumRegularFanins()) {
          auto& actual_fanin = node_view->GetRegularFanin(i);
          EXPECT_EQ(actual_fanin, MutableFanoutView(actual_fanin_node,
                                                    expected_fanin.index()));
          EXPECT_EQ(actual_fanin.node_index(),
                    actual_fanin.node_view()->node_index());
        }
      }

      if (same_num_regular_fanins) {
        for (int i = 0; i < expected_num_regular_fanins; ++i) {
          const auto& fanin = node_view->GetRegularFanin(i);
          EXPECT_EQ(ParseTensorName(node_view->node()->input(i)),
                    TensorId(fanin.node_view()->GetName(), fanin.index()));
        }
      }

      const int expected_num_controlling_fanins =
          expected_node_view.NumControllingFanins();
      bool same_num_controlling_fanins =
          node_view->NumControllingFanins() == expected_num_controlling_fanins;
      EXPECT_TRUE(same_num_controlling_fanins);
      for (int i = 0; i < expected_num_controlling_fanins; ++i) {
        auto& expected_fanin = expected_node_view.GetControllingFanins()[i];

        auto* actual_fanin_node =
            graph_view->GetNode(expected_fanin.node_view()->GetName());
        ASSERT_NE(actual_fanin_node, nullptr);
        MutableFanoutView actual_fanin(actual_fanin_node,
                                       expected_fanin.index());
        EXPECT_TRUE(node_view->HasFanin(actual_fanin));

        int found = 0;
        for (const auto& actual_fanin : node_view->GetControllingFanins()) {
          if (actual_fanin.index() == expected_fanin.index() &&
              actual_fanin.node_view()->GetName() ==
                  expected_fanin.node_view()->GetName()) {
            EXPECT_EQ(actual_fanin.node_index(),
                      actual_fanin.node_view()->node_index());
            ++found;
          }
        }
        EXPECT_EQ(found, 1);
      }

      if (same_num_controlling_fanins && same_num_regular_fanins) {
        for (int i = 0; i < expected_num_controlling_fanins; ++i) {
          const auto& fanin = node_view->GetControllingFanins()[i];
          EXPECT_EQ(ParseTensorName(node_view->node()->input(
                        i + expected_num_regular_fanins)),
                    TensorId(fanin.node_view()->GetName(), fanin.index()));
        }
      }

      EXPECT_EQ(node_view->NumRegularFanouts(),
                expected_node_view.NumRegularFanouts());
      const int num_output_ports =
          expected_node_view.GetRegularFanouts().size();
      ASSERT_EQ(node_view->GetRegularFanouts().size(), num_output_ports);
      for (int i = 0; i < num_output_ports; ++i) {
        auto& expected_fanouts_at_port_i = node_view->GetRegularFanouts()[i];
        const int num_fanouts_at_port = expected_fanouts_at_port_i.size();

        auto& actual_fanouts_at_port_i = node_view->GetRegularFanouts()[i];
        EXPECT_EQ(actual_fanouts_at_port_i.size(), num_fanouts_at_port);

        for (int j = 0; j < num_fanouts_at_port; ++j) {
          auto& expected_fanout = expected_fanouts_at_port_i[j];

          auto* actual_fanout_node =
              graph_view->GetNode(expected_fanout.node_view()->GetName());

          ASSERT_NE(actual_fanout_node, nullptr);
          MutableFaninView actual_fanout(actual_fanout_node,
                                         expected_fanout.index());
          EXPECT_TRUE(node_view->HasFanout(actual_fanout));

          int found = 0;
          for (const auto& fanout : actual_fanouts_at_port_i) {
            if (fanout.index() == expected_fanout.index() &&
                fanout.node_view()->GetName() ==
                    expected_fanout.node_view()->GetName()) {
              EXPECT_EQ(fanout.node_index(), fanout.node_view()->node_index());
              ++found;
            }
          }
          EXPECT_EQ(found, 1);
        }
      }

      const int num_controlled_fanouts =
          expected_node_view.NumControlledFanouts();
      EXPECT_EQ(node_view->NumControlledFanouts(), num_controlled_fanouts);
      for (int i = 0; i < num_controlled_fanouts; ++i) {
        const auto& expected_fanout =
            expected_node_view.GetControlledFanouts()[i];

        auto* actual_fanout_node =
            graph_view->GetNode(expected_fanout.node_view()->GetName());
        ASSERT_NE(actual_fanout_node, nullptr);
        MutableFaninView actual_fanout(actual_fanout_node,
                                       expected_fanout.index());
        EXPECT_TRUE(node_view->HasFanout(actual_fanout));

        int found = 0;
        for (const auto& fanout : node_view->GetControlledFanouts()) {
          if (fanout.index() == expected_fanout.index() &&
              fanout.node_view()->GetName() ==
                  expected_fanout.node_view()->GetName()) {
            EXPECT_EQ(fanout.node_index(), fanout.node_view()->node_index());
            ++found;
          }
        }
        EXPECT_EQ(found, 1);
      }

      EXPECT_EQ(node_view->NumAttrs(), expected_node_view.NumAttrs());
      for (const auto& expected_attr : expected_node_view.GetAttrs()) {
        auto* attr = node_view->GetAttr(expected_attr.first);
        EXPECT_TRUE(AreAttrValuesEqual(*attr, expected_attr.second));
      }
    }
    CompareGraphs(*graph_view->graph(), expected_graph);
  }
};

class MutationTest : public CompareGraphTest {};

constexpr char kDeviceCPU0[] = "/device:CPU:0";
constexpr char kDeviceGPU0[] = "/device:GPU:0";

GraphDef SimpleTestGraphForMutation() {
  return GDef({NDef("a", kNoOp, {}, {}, kDeviceCPU0),
               NDef("b", kNoOp, {}, {}, kDeviceCPU0),
               NDef("c", kNoOp, {}, {}, kDeviceCPU0),
               NDef("d", kNoOp, {"a:2", "b:3", "a:4", "^c", "^b"},
                    {{"attr_1", "a"}, {"attr_2", 2.0f}}, kDeviceCPU0)},
              /*funcs=*/{});
}

TEST_F(MutationTest, AddNewNode) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  NodeDef empty_node;
  mutation->AddNode(std::move(empty_node), &s);
  TF_EXPECT_OK(s);
  s = errors::Internal("error");

  NodeDef valid_node =
      NDef("valid", "IdentityN", {"a:1", "^b"}, {{"N", 1}}, "foo");
  mutation->AddNode(std::move(valid_node), &s);
  TF_EXPECT_OK(s);

  NodeDef bad_node_1 =
      NDef("bad", "IdentityN", {"^b", "a:1"}, {{"N", 1}}, "foo");
  mutation->AddNode(std::move(bad_node_1), &s);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "Mutation::AddNode error: node 'bad' has regular fanin 'a:1' after "
            "controlling fanins.");

  NodeDef bad_node_2 = NDef("bad", "IdentityN", {"bad:1"}, {}, "foo");
  mutation->AddNode(std::move(bad_node_2), &s);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "Mutation::AddNode error: node 'bad' has self cycle fanin "
            "'bad:1'.");

  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());
}

TEST_F(MutationTest, NewNodeBadFaninsAfterAdd) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  NodeDef valid_node =
      NDef("valid", "IdentityN", {"a:1", "^b"}, {{"N", 1}}, "foo");
  MutationNewNode new_node = mutation->AddNode(std::move(valid_node), &s);

  mutation->AddOrUpdateRegularFanin(new_node, 1, {"valid", 2});
  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: new node 'valid' is ill-formed.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());
}

TEST_F(MutationTest, NewNodesConflictingNames) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  NodeDef new_node_1 = NDef("a", "", {});
  mutation->AddNode(std::move(new_node_1), &s);
  TF_EXPECT_OK(s);

  NodeDef new_node_2 = NDef("a", "", {});
  mutation->AddNode(std::move(new_node_2), &s);
  TF_EXPECT_OK(s);

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: multiple nodes with the name: 'a' exists in "
      "Mutation.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());
}

TEST_F(MutationTest, UpdateNodeAndAddSelfLoop) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  mutation->AddControllingFanin(d_node, "d");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: inplace updated node 'd' is ill-formed.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());
}

TEST_F(MutationTest, RenameNodeAndAddSelfLoop) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  mutation->UpdateNodeName(d_node, "e");
  mutation->AddControllingFanin(d_node, "e");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: renamed updated node 'e' ('d') is ill-formed.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());
}

TEST_F(MutationTest, ExistingNodesConflictingNames) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  mutation->UpdateNodeName(a_node, "b");

  MutableNodeView* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  mutation->UpdateNodeOp(b_node, "Identity");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: multiple nodes with the name: 'b' exists in "
      "Mutation.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());
}

TEST_F(MutationTest, NewAndExistingNodesConflictingNames) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  NodeDef new_node = NDef("a", "", {});
  mutation->AddNode(std::move(new_node), &s);
  TF_EXPECT_OK(s);

  MutableNodeView* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  mutation->UpdateNodeDevice(a_node, "foo");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: multiple nodes with the name: 'a' exists in "
      "Mutation.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());
}

TEST_F(MutationTest, NewAndExistingRenamedNodesConflictingNames) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  NodeDef new_node = NDef("e", "", {});
  mutation->AddNode(std::move(new_node), &s);
  TF_EXPECT_OK(s);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  mutation->UpdateNodeName(d_node, "e");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: multiple nodes with the name: 'e' exists in "
      "Mutation.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());
}

TEST_F(MutationTest, RemoveNodesWithFanouts) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  mutation->RemoveNode(b_node);

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: fanout 'd' exist for missing node 'b'.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  mutation->RemoveNode(d_node);

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph = GDef({NDef("a", kNoOp, {}, {}, kDeviceCPU0),
                                  NDef("c", kNoOp, {}, {}, kDeviceCPU0)},
                                 /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, SwapNodeNamesWithCycle) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  mutation->UpdateNodeName(d_node, "b");
  MutableNodeView* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  mutation->UpdateNodeName(b_node, "d");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: renamed updated node 'b' ('d') is ill-formed.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());

  mutation->AddOrUpdateRegularFanin(d_node, 1, {"d", 3});
  mutation->RemoveControllingFanin(d_node, "b");

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph =
      GDef({NDef("a", kNoOp, {}, {}, kDeviceCPU0),
            NDef("d", kNoOp, {}, {}, kDeviceCPU0),
            NDef("c", kNoOp, {}, {}, kDeviceCPU0),
            NDef("b", kNoOp, {"a:2", "d:3", "a:4", "^c"},
                 {{"attr_1", "a"}, {"attr_2", 2.0f}}, kDeviceCPU0)},
           /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, RenamedNodeWithFanouts) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  mutation->UpdateNodeName(a_node, "b");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: fanout 'd' exist for missing node 'a'.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());

  mutation->UpdateNodeName(a_node, "a");

  MutableNodeView* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  mutation->UpdateNodeName(b_node, "e");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  expected_error_msg =
      "Mutation::Apply error: fanout 'd' exist for missing "
      "node 'b'.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());
}

TEST_F(MutationTest, RemoveExistingNodeAndReplaceWithNewNode) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  mutation->RemoveNode(d_node);

  NodeDef new_node = NDef("d", kNoOp, {"c:8", "^a"}, {}, kDeviceCPU0);
  mutation->AddNode(std::move(new_node), &s);
  TF_EXPECT_OK(s);

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph =
      GDef({NDef("a", kNoOp, {}, {}, kDeviceCPU0),
            NDef("b", kNoOp, {}, {}, kDeviceCPU0),
            NDef("c", kNoOp, {}, {}, kDeviceCPU0),
            NDef("d", kNoOp, {"c:8", "^a"}, {}, kDeviceCPU0)},
           /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, UpdateNodeNameAndRemoveFanins) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  mutation->UpdateNodeName(d_node, "e");
  mutation->RemoveRegularFanin(d_node, 1);
  mutation->RemoveRegularFanin(d_node, 2);

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph =
      GDef({NDef("a", kNoOp, {}, {}, kDeviceCPU0),
            NDef("b", kNoOp, {}, {}, kDeviceCPU0),
            NDef("c", kNoOp, {}, {}, kDeviceCPU0),
            NDef("e", kNoOp, {"a:2", "^c", "^b"},
                 {{"attr_1", "a"}, {"attr_2", 2.0f}}, kDeviceCPU0)},
           /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, UpdateNodeNameAndRemoveRegularFanout) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);
  mutation->UpdateNodeName(a_node, "e");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: fanout 'd' exist for missing node 'a'.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  mutation->RemoveRegularFanin(d_node, 2);

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  expected_error_msg =
      "Mutation::Apply error: fanout 'd' exist for missing node 'a'.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());

  mutation->AddOrUpdateRegularFanin(d_node, 0, {"b", 1});

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph =
      GDef({NDef("e", kNoOp, {}, {}, kDeviceCPU0),
            NDef("b", kNoOp, {}, {}, kDeviceCPU0),
            NDef("c", kNoOp, {}, {}, kDeviceCPU0),
            NDef("d", kNoOp, {"b:1", "b:3", "^c", "^b"},
                 {{"attr_1", "a"}, {"attr_2", 2.0f}}, kDeviceCPU0)},
           /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, UpdateNodeNameAndRemoveControlledFanout) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  mutation->UpdateNodeName(c_node, "e");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: fanout 'd' exist for missing node 'c'.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  mutation->UpdateNodeDevice(d_node, kDeviceGPU0);

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  expected_error_msg =
      "Mutation::Apply error: fanout 'd' exist for missing node 'c'.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());

  mutation->RemoveControllingFanin(d_node, "c");

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph =
      GDef({NDef("a", kNoOp, {}, {}, kDeviceCPU0),
            NDef("b", kNoOp, {}, {}, kDeviceCPU0),
            NDef("e", kNoOp, {}, {}, kDeviceCPU0),
            NDef("d", kNoOp, {"a:2", "b:3", "a:4", "^b"},
                 {{"attr_1", "a"}, {"attr_2", 2.0f}}, kDeviceGPU0)},
           /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, EmptyMutation) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  TF_EXPECT_OK(mutation->Apply());
  CompareGraphViewWithGraph(&graph_view, SimpleTestGraphForMutation());
}

constexpr char kIdentity[] = "Identity";
constexpr char kDeviceCPU1[] = "/device:CPU:1";
constexpr char kDeviceGPU1[] = "/device:GPU:1";

GraphDef TestGraphForMutation() {
  return GDef(
      {NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0),
       NDef("b", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU0),
       NDef("c", kNoOp, {"^a"}, {{"attr_c", "test"}}, kDeviceCPU1),
       NDef("d", kNoOp, {"a:2", "b:3", "a:4", "^c", "^b"},
            {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU1)},
      /*funcs=*/{});
}

TEST_F(MutationTest, SwapNodeNamesWithNoCycle) {
  GraphDef graph = TestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  MutableNodeView* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);

  mutation->UpdateNodeName(b_node, "c");
  mutation->UpdateNodeName(c_node, "b");

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph = GDef(
      {NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0),
       NDef("c", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU0),
       NDef("b", kNoOp, {"^a"}, {{"attr_c", "test"}}, kDeviceCPU1),
       NDef("d", kNoOp, {"a:2", "b:3", "a:4", "^c", "^b"},
            {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU1)},
      /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, RemoveMultipleDependentNodes) {
  GraphDef graph = TestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  mutation->RemoveNode(c_node);
  mutation->RemoveNode(d_node);

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph = GDef(
      {NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0),
       NDef("b", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU0)},
      /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

constexpr char kDeviceGPU2[] = "/device:GPU:2";

TEST_F(MutationTest, AddSimpleNewNode) {
  GraphDef graph = TestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  NodeDef new_node =
      NDef("new_node", kIdentity, {}, {{"T", DT_INT64}}, kDeviceGPU2);
  mutation->AddNode(std::move(new_node), &s);
  TF_EXPECT_OK(s);

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph = GDef(
      {NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0),
       NDef("b", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU0),
       NDef("c", kNoOp, {"^a"}, {{"attr_c", "test"}}, kDeviceCPU1),
       NDef("d", kNoOp, {"a:2", "b:3", "a:4", "^c", "^b"},
            {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU1),
       NDef("new_node", kIdentity, {}, {{"T", DT_INT64}}, kDeviceGPU2)},
      /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

constexpr char kDeviceGPU3[] = "/device:GPU:3";

TEST_F(MutationTest, AddAndUpdateNodesWithFanins) {
  GraphDef graph = TestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  NodeDef new_node_1 = NDef("new_node_1", kNoOp, {"a:2", "d:5", "^b", "^c"},
                            {{"new_node_1_attr_1", 5.0f}}, kDeviceGPU2);
  mutation->AddNode(std::move(new_node_1), &s);
  TF_EXPECT_OK(s);

  NodeDef new_node_2 =
      NDef("new_node_2", kNoOp, {"a:3", "new_node_1:5", "^d", "^new_node_1"},
           {{"new_node_2_attr_1", 9}}, kDeviceGPU3);
  mutation->AddNode(std::move(new_node_2), &s);
  TF_EXPECT_OK(s);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);
  mutation->AddOrUpdateRegularFanin(d_node, 3, {"c", 6});
  mutation->AddOrUpdateRegularFanin(d_node, 1, {"new_node_1", 5});
  mutation->AddControllingFanin(d_node, "new_node_2");

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph = GDef(
      {NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0),
       NDef("b", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU0),
       NDef("c", kNoOp, {"^a"}, {{"attr_c", "test"}}, kDeviceCPU1),
       NDef("d", kNoOp,
            {"a:2", "new_node_1:5", "a:4", "c:6", "^c", "^b", "^new_node_2"},
            {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU1),
       NDef("new_node_1", kNoOp, {"a:2", "d:5", "^b", "^c"},
            {{"new_node_1_attr_1", 5.0f}}, kDeviceGPU2),
       NDef("new_node_2", kNoOp, {"a:3", "new_node_1:5", "^d", "^new_node_1"},
            {{"new_node_2_attr_1", 9}}, kDeviceGPU3)},
      /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, UpdateNodeNameToReplaceExistingNode) {
  auto test_graph = []() {
    return GDef(
        {NDef("a", kNoOp, {}, {{"attr_a", 8}}, kDeviceCPU0),
         NDef("b", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU1),
         NDef("c", kNoOp, {"b:4", "^a"}, {{"attr_c", "test"}}, kDeviceGPU2),
         NDef("d", kNoOp, {"a:2", "c:5", "a:4", "^a", "^c"},
              {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU3)},
        /*funcs=*/{});
  };

  GraphDef graph = test_graph();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  MutableNodeView* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);

  mutation->UpdateNodeName(b_node, "c");

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph =
      GDef({NDef("a", kNoOp, {}, {{"attr_a", 8}}, kDeviceCPU0),
            NDef("c", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU1),
            NDef("d", kNoOp, {"a:2", "c:5", "a:4", "^a", "^c"},
                 {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU3)},
           /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, NewNodeWithMutations) {
  GraphDef graph = TestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  Mutation* mutation = graph_view.GetMutationBuilder();

  NodeDef new_node_def = NDef("node", kNoOp, {"a:2", "b:3", "^c"},
                              {{"attr_1", 1}, {"attr_2", 2.0f}}, kDeviceGPU3);
  MutationNewNode new_node = mutation->AddNode(std::move(new_node_def), &s);
  TF_EXPECT_OK(s);

  mutation->AddControllingFanin(new_node, "a");
  mutation->RemoveControllingFanin(new_node, "c");
  mutation->AddOrUpdateRegularFanin(new_node, 0, {"b", 6});
  mutation->RemoveRegularFanin(new_node, 1);
  mutation->UpdateNodeName(new_node, "new_node");
  mutation->UpdateNodeOp(new_node, kIdentity);
  mutation->UpdateNodeDevice(new_node, kDeviceGPU2);
  AttrValue attr_3;
  attr_3.set_s("new_node_attr");
  mutation->AddOrUpdateNodeAttr(new_node, "attr_3", attr_3);
  AttrValue attr_1;
  attr_1.set_b(true);
  mutation->AddOrUpdateNodeAttr(new_node, "attr_1", attr_1);
  mutation->RemoveNodeAttr(new_node, "attr_2");
  AttrValue attr_4;
  attr_4.set_type(DT_FLOAT);
  mutation->AddOrUpdateNodeAttr(new_node, "T", attr_4);

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph = GDef(
      {NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0),
       NDef("b", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU0),
       NDef("c", kNoOp, {"^a"}, {{"attr_c", "test"}}, kDeviceCPU1),
       NDef("d", kNoOp, {"a:2", "b:3", "a:4", "^c", "^b"},
            {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU1),
       NDef("new_node", kIdentity, {"b:6", "^a"},
            {{"attr_1", true}, {"attr_3", "new_node_attr"}, {"T", DT_FLOAT}},
            kDeviceGPU2)},
      /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, UpdatedNodeWithNonFaninMutations) {
  GraphDef graph = TestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  Mutation* mutation = graph_view.GetMutationBuilder();

  mutation->UpdateNodeName(d_node, "e");
  mutation->UpdateNodeOp(d_node, kIdentity);
  mutation->UpdateNodeDevice(d_node, kDeviceGPU2);
  AttrValue attr_d_1;
  attr_d_1.set_b(false);
  mutation->AddOrUpdateNodeAttr(d_node, "attr_d_1", attr_d_1);
  AttrValue attr_e_3;
  attr_e_3.set_s("test_string");
  mutation->AddOrUpdateNodeAttr(d_node, "attr_e_3", attr_e_3);
  mutation->RemoveNodeAttr(d_node, "attr_d_2");
  AttrValue attr_e_4;
  attr_e_4.set_type(DT_INT64);
  mutation->AddOrUpdateNodeAttr(d_node, "T", attr_e_4);

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph = GDef(
      {NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0),
       NDef("b", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU0),
       NDef("c", kNoOp, {"^a"}, {{"attr_c", "test"}}, kDeviceCPU1),
       NDef("e", kIdentity, {"a:2", "b:3", "a:4", "^c", "^b"},
            {{"attr_d_1", false}, {"attr_e_3", "test_string"}, {"T", DT_INT64}},
            kDeviceGPU2)},
      /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, Reset) {
  GraphDef graph = TestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  MutableNodeView* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);

  Mutation* mutation = graph_view.GetMutationBuilder();

  mutation->UpdateNodeName(a_node, "e");
  mutation->AddNode({}, &s);
  TF_EXPECT_OK(s);

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  string expected_error_msg =
      "Mutation::Apply error: fanout 'b' exist for missing node 'a'.";
  EXPECT_EQ(s.error_message(), expected_error_msg);
  CompareGraphViewWithGraph(&graph_view, TestGraphForMutation());

  mutation->Reset();
  TF_EXPECT_OK(mutation->Apply());
  CompareGraphViewWithGraph(&graph_view, TestGraphForMutation());
}

TEST_F(MutationTest, RenameNodeAndAddNewNodeWithRenamedNodeOldName) {
  GraphDef graph = TestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  MutableNodeView* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);

  Mutation* mutation = graph_view.GetMutationBuilder();

  mutation->UpdateNodeName(b_node, "e");

  NodeDef new_node =
      NDef("b", kIdentity, {"c:2"}, {{"T", DT_INT64}}, kDeviceGPU3);
  mutation->AddNode(std::move(new_node), &s);
  TF_EXPECT_OK(s);

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph = GDef(
      {NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0),
       NDef("e", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU0),
       NDef("c", kNoOp, {"^a"}, {{"attr_c", "test"}}, kDeviceCPU1),
       NDef("d", kNoOp, {"a:2", "b:3", "a:4", "^c", "^b"},
            {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU1),
       NDef("b", kIdentity, {"c:2"}, {{"T", DT_INT64}}, kDeviceGPU3)},
      /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, ShiftNodesWithFanouts) {
  auto test_graph = []() {
    return GDef({NDef("d", kNoOp, {"a:2", "b:3", "a:4", "^a", "^c", "^b"},
                      {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU1),
                 NDef("c", kNoOp, {"^a"}, {{"attr_c", "test"}}, kDeviceCPU1),
                 NDef("b", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU0),
                 NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}},
                      kDeviceGPU0)},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  MutableNodeView* c_node = graph_view.GetNode("c");
  ASSERT_NE(c_node, nullptr);
  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  Mutation* mutation = graph_view.GetMutationBuilder();

  mutation->RemoveControllingFanin(d_node, "c");
  mutation->RemoveNode(c_node);

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph = GDef(
      {NDef("d", kNoOp, {"a:2", "b:3", "a:4", "^a", "^b"},
            {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU1),
       NDef("b", kNoOp, {"a:2"}, {{"attr_b", 3.0f}}, kDeviceCPU0),
       NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0)},
      /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, RemoveFaninFanoutAndShiftFanout) {
  auto test_graph = []() {
    return GDef({NDef("a", kNoOp, {}, {}, kDeviceGPU0),
                 NDef("b", kNoOp, {"a:2", "a:1"}, {}, kDeviceGPU1),
                 NDef("c", kNoOp, {"a:1", "a:2"}, {}, kDeviceGPU2)},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  MutableNodeView* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);

  Mutation* mutation = graph_view.GetMutationBuilder();

  mutation->RemoveRegularFanin(b_node, 1);

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph =
      GDef({NDef("a", kNoOp, {}, {}, kDeviceGPU0),
            NDef("b", kNoOp, {"a:2"}, {}, kDeviceGPU1),
            NDef("c", kNoOp, {"a:1", "a:2"}, {}, kDeviceGPU2)},
           /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

TEST_F(MutationTest, ConsecutiveMutations) {
  GraphDef graph = TestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  MutableNodeView* b_node = graph_view.GetNode("b");
  ASSERT_NE(b_node, nullptr);
  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  Mutation* mutation = graph_view.GetMutationBuilder();

  mutation->RemoveNode(b_node);
  mutation->AddOrUpdateRegularFanin(d_node, 1, {"c", 5});
  mutation->RemoveControllingFanin(d_node, "b");

  NodeDef new_node_1 = NDef("new_node_1", kIdentity, {"a:3", "d:5", "^d"},
                            {{"T", DT_FLOAT}}, kDeviceGPU2);
  MutationNewNode new_node_1_node =
      mutation->AddNode(std::move(new_node_1), &s);
  TF_EXPECT_OK(s);

  mutation->AddOrUpdateRegularFanin(new_node_1_node, 0, {"c", 5});
  mutation->RemoveRegularFanin(new_node_1_node, 1);
  mutation->AddOrUpdateRegularFanin(new_node_1_node, 1, {"a", 6});
  mutation->AddControllingFanin(new_node_1_node, "a");
  mutation->RemoveControllingFanin(new_node_1_node, "d");

  TF_EXPECT_OK(mutation->Apply());
  GraphDef expected_graph = GDef(
      {NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0),
       NDef("c", kNoOp, {"^a"}, {{"attr_c", "test"}}, kDeviceCPU1),
       NDef("d", kNoOp, {"a:2", "c:5", "a:4", "^c"},
            {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU1),
       NDef("new_node_1", kIdentity, {"c:5", "a:6", "^a"}, {{"T", DT_FLOAT}},
            kDeviceGPU2)},
      /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);

  d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  mutation->AddOrUpdateRegularFanin(d_node, 3, {"new_node_2", 6});
  mutation->AddOrUpdateRegularFanin(d_node, 1, {"new_node_1", 8});
  mutation->AddControllingFanin(d_node, "new_node_2");
  mutation->AddControllingFanin(d_node, "a");
  mutation->RemoveControllingFanin(d_node, "c");

  NodeDef new_node_2 =
      NDef("new_node_2", kNoOp, {"c:4", "new_node_1:5", "^d", "^c"});
  MutationNewNode new_node_2_node =
      mutation->AddNode(std::move(new_node_2), &s);
  TF_EXPECT_OK(s);

  mutation->UpdateNodeDevice(new_node_2_node, kDeviceGPU3);
  mutation->AddOrUpdateRegularFanin(new_node_2_node, 0, {"new_node_1", 4});
  mutation->RemoveRegularFanin(new_node_2_node, 1);
  mutation->RemoveControllingFanin(new_node_2_node, "c");
  mutation->AddControllingFanin(new_node_2_node, "a");
  mutation->AddControllingFanin(new_node_2_node, "new_node_1");

  TF_EXPECT_OK(mutation->Apply());
  expected_graph = GDef(
      {NDef("a", kIdentity, {}, {{"attr_a", 8}, {"T", DT_FLOAT}}, kDeviceGPU0),
       NDef("c", kNoOp, {"^a"}, {{"attr_c", "test"}}, kDeviceCPU1),
       NDef("d", kNoOp,
            {"a:2", "new_node_1:8", "a:4", "new_node_2:6", "^new_node_2", "^a"},
            {{"attr_d_1", "a"}, {"attr_d_2", 2.0f}}, kDeviceGPU1),
       NDef("new_node_1", kIdentity, {"c:5", "a:6", "^a"}, {{"T", DT_FLOAT}},
            kDeviceGPU2),
       NDef("new_node_2", kNoOp, {"new_node_1:4", "^d", "^a", "^new_node_1"},
            {}, kDeviceGPU3)},
      /*funcs=*/{});
  CompareGraphViewWithGraph(&graph_view, expected_graph);
}

constexpr char kMatchingFiles[] = "MatchingFiles";

TEST_F(MutationTest, OpWithUnsupportedDevice) {
  GTEST_SKIP() << "Reenable once offline optimization tests enable CUDA.";
  auto test_graph = []() {
    return GDef({NDef("a", kMatchingFiles, {}, {}, kDeviceCPU0)},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  MutableNodeView* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);

  Mutation* mutation = graph_view.GetMutationBuilder();

  // Unsupported device.
  mutation->UpdateNodeDevice(a_node, kDeviceGPU1);

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  CompareGraphViewWithGraph(&graph_view, test_graph());

  mutation->Reset();

  // New node with unsupported device.
  NodeDef new_node = NDef("new_node", kMatchingFiles, {}, {}, kDeviceGPU2);
  mutation->AddNode(std::move(new_node), &s);
  TF_EXPECT_OK(s);

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  CompareGraphViewWithGraph(&graph_view, test_graph());
}

TEST_F(MutationTest, OpMissingAttribute) {
  GTEST_SKIP() << "Reenable once offline optimization tests enable CUDA.";
  auto test_graph = []() {
    return GDef({NDef("a", kIdentity, {}, {{"T", DT_FLOAT}}, kDeviceGPU0)},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  MutableNodeView* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);

  Mutation* mutation = graph_view.GetMutationBuilder();

  // Remove necessary attribute.
  mutation->RemoveNodeAttr(a_node, "T");

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  CompareGraphViewWithGraph(&graph_view, test_graph());

  mutation->Reset();

  // New node without necessary attribute.
  NodeDef new_node = NDef("new_node", kIdentity, {}, {}, kDeviceGPU2);
  mutation->AddNode(std::move(new_node), &s);
  TF_EXPECT_OK(s);

  s = mutation->Apply();
  EXPECT_FALSE(s.ok());
  CompareGraphViewWithGraph(&graph_view, test_graph());
}

TEST_F(MutationTest, EmptyMutationUpdateIndexPersisting) {
  auto test_graph = []() {
    return GDef({NDef("a", kIdentity, {}, {{"T", DT_FLOAT}}, kDeviceGPU0)},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);

  MutableNodeView* a_node = graph_view.GetNode("a");
  ASSERT_NE(a_node, nullptr);

  Mutation* mutation = graph_view.GetMutationBuilder();

  // Empty MutableNodeViewDiff.
  mutation->UpdateNodeName(a_node, "a");

  TF_EXPECT_OK(mutation->Apply());
  CompareGraphViewWithGraph(&graph_view, test_graph());

  mutation->Reset();

  // Empty MutableNodeViewDiff, `update_index_` should not persist.
  mutation->UpdateNodeName(a_node, "a");

  TF_EXPECT_OK(mutation->Apply());
  CompareGraphViewWithGraph(&graph_view, test_graph());
}

class TopologicalSortTest : public CompareGraphTest {
 protected:
  void CompareGraphOrder(const MutableGraphView& graph_view,
                         absl::Span<const string> node_names) {
    const int num_nodes = graph_view.NumNodes();
    ASSERT_EQ(num_nodes, node_names.size());
    for (int i = 0; i < num_nodes; ++i) {
      EXPECT_EQ(graph_view.GetNode(i)->GetName(), node_names[i]);
    }
  }

  void CompareGraphNodePrecedences(
      const MutableGraphView& graph_view,
      absl::Span<const std::pair<string, string>> node_precedences) {
    for (const auto& node_precedence : node_precedences) {
      auto* parent_node = graph_view.GetNode(node_precedence.first);
      ASSERT_NE(parent_node, nullptr);
      auto* child_node = graph_view.GetNode(node_precedence.second);
      ASSERT_NE(child_node, nullptr);
      EXPECT_TRUE(parent_node->node_index() < child_node->node_index());
    }
  }
};

TEST_F(TopologicalSortTest, ActiveMutationSort) {
  auto test_graph = []() {
    return GDef({NDef("a", kIdentity, {}, {{"T", DT_FLOAT}}, kDeviceGPU0),
                 NDef("b", kIdentity, {"a"}, {{"T", DT_FLOAT}}, kDeviceGPU1)},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  Mutation* mutation = graph_view.GetMutationBuilder();
  mutation->AddNode({}, &status);
  TF_ASSERT_OK(status);

  for (bool ignore_cycles : {false, true}) {
    status = graph_view.SortTopologically(ignore_cycles, {});
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(
        status.error_message(),
        "MutableGraphView::SortTopologically error: active mutation exists.");
    CompareGraphViewWithGraph(&graph_view, test_graph());
    CompareGraphOrder(graph_view, {"a", "b"});
  }
}

TEST_F(TopologicalSortTest, BadExtraDependenciesSort) {
  auto test_graph = []() {
    return GDef({NDef("a", kIdentity, {}, {{"T", DT_FLOAT}}, kDeviceGPU0),
                 NDef("b", kIdentity, {}, {{"T", DT_FLOAT}}, kDeviceGPU1)},
                /*funcs=*/{});
  };

  GraphDef graph_1 = test_graph();
  Status status;
  MutableGraphView graph_view_1(&graph_1, &status);
  TF_ASSERT_OK(status);
  MutableNodeView* a_node_1 = graph_view_1.GetNode("a");

  GraphDef graph_2 = test_graph();
  MutableGraphView graph_view_2(&graph_2, &status);
  TF_ASSERT_OK(status);
  MutableNodeView* b_node_2 = graph_view_2.GetNode("b");

  for (bool ignore_cycles : {false, true}) {
    status =
        graph_view_2.SortTopologically(ignore_cycles, {{a_node_1, b_node_2}});
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.error_message(),
              "MutableGraphView::SortTopologically error: invalid extra "
              "dependencies.");
    CompareGraphViewWithGraph(&graph_view_2, test_graph());
    CompareGraphOrder(graph_view_2, {"a", "b"});
  }
}

TEST_F(TopologicalSortTest, NoCyclesAllowed) {
  auto test_graph = []() {
    return GDef(
        {NDef("a", kIdentity, {}, {{"T", DT_FLOAT}}, kDeviceGPU0),
         NDef("b", kIdentity, {"a", "c"}, {{"T", DT_FLOAT}}, kDeviceGPU1),
         NDef("c", kIdentity, {"b"}, {{"T", DT_FLOAT}}, kDeviceGPU1)},
        /*funcs=*/{});
  };

  GraphDef graph = test_graph();
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  status = graph_view.SortTopologically(/*ignore_cycles=*/false, {});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_message(),
            "MutableGraphView::SortTopologically error: detected edge(s) "
            "creating cycle(s) {'c' -> 'b'}.");
  CompareGraphViewWithGraph(&graph_view, test_graph());
  CompareGraphOrder(graph_view, {"a", "b", "c"});

  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/true, {}));
  CompareGraphViewWithGraph(&graph_view, test_graph());
  CompareGraphNodePrecedences(graph_view, {{"a", "b"}, {"a", "c"}});
}

TEST_F(TopologicalSortTest, NoNodesWithZeroFanins) {
  auto test_graph = []() {
    return GDef({NDef("a", kIdentity, {"b"}, {{"T", DT_FLOAT}}, kDeviceGPU0),
                 NDef("b", kIdentity, {"a"}, {{"T", DT_FLOAT}}, kDeviceGPU1)},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  status = graph_view.SortTopologically(/*ignore_cycles=*/false, {});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_message(),
            "MutableGraphView::SortTopologically error: was not able to sort "
            "all nodes topologically.");
  CompareGraphViewWithGraph(&graph_view, test_graph());
  CompareGraphOrder(graph_view, {"a", "b"});

  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/true, {}));
  CompareGraphViewWithGraph(&graph_view, test_graph());
}

TEST_F(TopologicalSortTest, DidNotReachAllNodes) {
  auto test_graph = []() {
    return GDef({NDef("c", kIdentity, {}, {{"T", DT_FLOAT}}, kDeviceGPU2),
                 NDef("a", kIdentity, {"b"}, {{"T", DT_FLOAT}}, kDeviceGPU0),
                 NDef("b", kIdentity, {"a"}, {{"T", DT_FLOAT}}, kDeviceGPU1)},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  status = graph_view.SortTopologically(/*ignore_cycles=*/false, {});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_message(),
            "MutableGraphView::SortTopologically error: was not able to sort "
            "all nodes topologically.");
  CompareGraphViewWithGraph(&graph_view, test_graph());
  CompareGraphOrder(graph_view, {"c", "a", "b"});

  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/true, {}));
  CompareGraphViewWithGraph(&graph_view, test_graph());
  CompareGraphOrder(graph_view, {"a", "b", "c"});
}

TEST_F(TopologicalSortTest, NoLoopGraph) {
  auto test_graph = []() {
    return GDef({NDef("c", kIdentity, {"f"}), NDef("a", kIdentity, {"f", "e"}),
                 NDef("b", kIdentity, {"e", "d"}), NDef("d", kIdentity, {"c"}),
                 NDef("f", kIdentity, {}), NDef("e", kIdentity, {})},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  CompareGraphViewWithGraph(&graph_view, test_graph());
  CompareGraphNodePrecedences(
      graph_view,
      {{"f", "a"}, {"f", "c"}, {"e", "a"}, {"e", "b"}, {"c", "d"}, {"d", "b"}});
}

TEST_F(TopologicalSortTest, ValidLoopGraph) {
  // Control flow loop.
  auto test_graph = []() {
    return GDef(
        {NDef("while/Const_1", "Const", {}),
         NDef("while/Enter_2", "Enter", {"while/Const_1"},
              {{"frame_name", "while/while_context"}}),
         NDef("while/Const", "Const", {}),
         NDef("while/Enter_1", "Enter", {"while/Const"},
              {{"frame_name", "while/while_context"}}),
         NDef("while/iteration_counter", "Const", {}),
         NDef("while/Enter", "Enter", {"while/iteration_counter"},
              {{"frame_name", "while/while_context"}}),
         NDef("while/maximum_iterations", "Const", {}),
         NDef("while/Less/Enter", "Enter", {"while/maximum_iterations"},
              {{"frame_name", "while/while_context"}}),
         NDef("while/Less", "Less", {"while/Merge", "while/Less/Enter"}),
         NDef("while/LogicalAnd", "LogicalAnd",
              {"while/Less", "while/cond/Merge"}),
         NDef("while/LoopCond", "LoopCond", {"while/LogicalAnd"}),
         NDef("while/Switch", "Switch", {"while/Merge", "while/LoopCond"},
              {{"_class", "loc:@while/Merge"}}),
         NDef("while/Identity", "Identity", {"while/Switch:1"}),
         NDef("while/add", "Add", {"while/Identity", "while/add/y"}),
         NDef("while/NextIteration", "NextIteration", {"while/add"}),
         NDef("while/Merge", "Merge", {"while/Enter", "while/NextIteration"}),
         NDef("while/Less_1/y", "Const", {"^while/Merge"}),
         NDef("while/add/y", "Const", {"^while/Identity"}),
         NDef("while/mul/y", "Const", {"^while/Identity"}),
         NDef("while/add_2/y", "Const", {"^while/Identity"}),
         NDef("while/Switch_1", "Switch", {"while/Merge_1", "while/LoopCond"},
              {{"_class", "loc:@while/Merge_1"}}),
         NDef("while/Identity_1", "Identity", {"while/Switch_1:1"}),
         NDef("while/add_2", "Add", {"while/Identity_1", "while/add_2/y"}),
         NDef("while/NextIteration_1", "NextIteration", {"while/add_2"}),
         NDef("while/Merge_1", "Merge",
              {"while/Enter_1", "while/NextIteration_1"}),
         NDef("while/Less_1", "Less", {"while/Merge_1", "while/Less_1/y"}),
         NDef("while/cond/Switch", "Switch", {"while/Less_1", "while/Less_1"}),
         NDef("while/cond/switch_f", "Identity", {"while/cond/Switch"}),
         NDef("while/cond/Const_1", "Const", {"^while/cond/switch_f"}),
         NDef("while/cond/switch_t", "Identity", {"while/cond/Switch:1"}),
         NDef("while/cond/Const", "Const", {"^while/cond/switch_t"}),
         NDef("while/cond/Merge", "Merge",
              {"while/cond/Const_1", "while/cond/Const"}),
         NDef("TensorArrayUnstack/range/delta", "Const", {}),
         NDef("TensorArrayUnstack/range/start", "Const", {}),
         NDef("TensorArrayUnstack/strided_slice/stack_2", "Const", {}),
         NDef("TensorArrayUnstack/strided_slice/stack_1", "Const", {}),
         NDef("TensorArrayUnstack/strided_slice/stack", "Const", {}),
         NDef("TensorArrayUnstack/Shape", "Const", {}),
         NDef("TensorArrayUnstack/strided_slice", "StridedSlice",
              {"TensorArrayUnstack/Shape",
               "TensorArrayUnstack/strided_slice/stack",
               "TensorArrayUnstack/strided_slice/stack_1",
               "TensorArrayUnstack/strided_slice/stack_2"}),
         NDef("TensorArrayUnstack/range", "Range",
              {"TensorArrayUnstack/range/start",
               "TensorArrayUnstack/strided_slice",
               "TensorArrayUnstack/range/delta"}),
         NDef("TensorArray/size", "Const", {}),
         NDef("TensorArray", "TensorArrayV3", {"TensorArray/size"}),
         NDef("while/TensorArrayReadV3/Enter", "Enter", {"TensorArray"},
              {{"frame_name", "while/while_context"}}),
         NDef("Const", "Const", {}),
         NDef("TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3",
              "TensorArrayScatterV3",
              {"TensorArray", "TensorArrayUnstack/range", "Const",
               "TensorArray:1"},
              {{"_class", "loc@Const"}}),
         NDef("while/TensorArrayReadV3/Enter_1", "Enter",
              {"TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3"},
              {{"frame_name", "while/while_context"}}),
         NDef("while/TensorArrayReadV3", "TensorArrayReadV3",
              {"while/TensorArrayReadV3/Enter", "while/Identity_1",
               "while/TensorArrayReadV3/Enter_1"}),
         NDef("while/add_1", "Add", {"while/mul", "while/TensorArrayReadV3"}),
         NDef("while/NextIteration_2", "NextIteration", {"while/add_1"}),
         NDef("while/Merge_2", "Merge",
              {"while/Enter_2", "while/NextIteration_2"}),
         NDef("while/Switch_2", "Switch", {"while/Merge_2", "while/LoopCond"},
              {{"_class", "loc@while/Merge_2"}}),
         NDef("while/Exit_2", "Exit", {"while/Switch_2"}),
         NDef("while/Identity_2", "Identity", {"while/Switch_2:1"}),
         NDef("while/mul", "Mul", {"while/Identity_2", "while/mul/y"})},
        /*funcs=*/{});
  };

  GraphDef graph = test_graph();
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  CompareGraphViewWithGraph(&graph_view, test_graph());
}

TEST_F(TopologicalSortTest, DuplicateFanins) {
  auto test_graph = []() {
    return GDef(
        {NDef("b", kIdentity, {"a", "a", "^a"}), NDef("a", "Const", {})},
        /*funcs=*/{});
  };

  GraphDef graph = test_graph();
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  CompareGraphViewWithGraph(&graph_view, test_graph());
  CompareGraphOrder(graph_view, {"a", "b"});
}

TEST_F(TopologicalSortTest, DiamondDependencyNotACycle) {
  auto test_graph = []() {
    return GDef({NDef("e", kIdentity, {"b", "c", "d"}),
                 NDef("b", kIdentity, {"a"}), NDef("a", "Const", {}),
                 NDef("d", kIdentity, {"a"}), NDef("c", kIdentity, {"a"})},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  CompareGraphViewWithGraph(&graph_view, test_graph());
  CompareGraphNodePrecedences(
      graph_view,
      {{"a", "b"}, {"a", "c"}, {"a", "d"}, {"b", "e"}, {"c", "e"}, {"d", "e"}});
}

TEST_F(TopologicalSortTest, ExtraDependencies) {
  auto test_graph = []() {
    return GDef({NDef("c", kIdentity, {"f"}), NDef("a", kIdentity, {"f", "e"}),
                 NDef("b", kIdentity, {"e", "d"}), NDef("d", kIdentity, {"c"}),
                 NDef("f", kIdentity, {}), NDef("e", kIdentity, {})},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  auto* e_node = graph_view.GetNode("e");
  ASSERT_NE(e_node, nullptr);
  auto* f_node = graph_view.GetNode("f");
  ASSERT_NE(f_node, nullptr);

  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false,
                                            {{e_node, f_node}}));
  CompareGraphViewWithGraph(&graph_view, test_graph());
  CompareGraphNodePrecedences(graph_view, {{"f", "a"},
                                           {"f", "c"},
                                           {"e", "a"},
                                           {"e", "b"},
                                           {"c", "d"},
                                           {"d", "b"},
                                           {"e", "f"}});
}

TEST_F(TopologicalSortTest, PushVisitedNodes) {
  auto test_graph = []() {
    return GDef({NDef("d", kIdentity, {"c"}), NDef("c", kIdentity, {"b", "a"}),
                 NDef("b", kIdentity, {"a"}), NDef("a", kIdentity, {})},
                /*funcs=*/{});
  };

  GraphDef graph = test_graph();
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  CompareGraphViewWithGraph(&graph_view, test_graph());
  CompareGraphNodePrecedences(graph_view,
                              {{"a", "b"}, {"a", "c"}, {"b", "c"}, {"c", "d"}});
}

#define RUN_NUM_NODE_NUM_EDGE_BENCHMARK(name) \
  BENCHMARK(name)                             \
      ->ArgPair(10, 2)                        \
      ->ArgPair(100, 2)                       \
      ->ArgPair(1000, 2)                      \
      ->ArgPair(10000, 2)                     \
      ->ArgPair(25000, 2)                     \
      ->ArgPair(50000, 2)                     \
      ->ArgPair(100000, 2)                    \
      ->ArgPair(10, 4)                        \
      ->ArgPair(100, 4)                       \
      ->ArgPair(1000, 4)                      \
      ->ArgPair(10000, 4)                     \
      ->ArgPair(25000, 4)                     \
      ->ArgPair(50000, 4)                     \
      ->ArgPair(100000, 4)                    \
      ->ArgPair(10, 8)                        \
      ->ArgPair(100, 8)                       \
      ->ArgPair(1000, 8)                      \
      ->ArgPair(10000, 8)                     \
      ->ArgPair(25000, 8)                     \
      ->ArgPair(50000, 8)                     \
      ->ArgPair(100000, 8)                    \
      ->ArgPair(10, 16)                       \
      ->ArgPair(100, 16)                      \
      ->ArgPair(1000, 16)                     \
      ->ArgPair(10000, 16)                    \
      ->ArgPair(25000, 16)                    \
      ->ArgPair(50000, 16)                    \
      ->ArgPair(100000, 16);

template <typename GraphViewT>
void BM_GraphViewTConstruction(::testing::benchmark::State& state) {
  const int num_nodes = state.range(0);
  const int num_edges_per_node = state.range(1);

  GraphDef graph_def = test::CreateGraphDef(num_nodes, num_edges_per_node);

  for (auto i : state) {
    Status s;
    GraphViewT graph_view(&graph_def, &s);
  }
}

void BM_GraphViewConstruction(::testing::benchmark::State& state) {
  BM_GraphViewTConstruction<GraphView>(state);
}

void BM_MutableGraphViewConstruction(::testing::benchmark::State& state) {
  BM_GraphViewTConstruction<MutableGraphView>(state);
}

void BM_MutableGraphViewClearAttrs(::testing::benchmark::State& state) {
  const int num_nodes = state.range(0);
  const int num_edges_per_node = state.range(1);

  GraphDef graph_def = test::CreateGraphDef(num_nodes, num_edges_per_node);

  Status s;
  MutableGraphView graph_view(&graph_def, &s);

  for (auto i : state) {
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    for (int j = 0; j < num_nodes; ++j) {
      mutation->RemoveNodeAttr(graph_view.GetNode(j), "_some_random_attr");
    }
    s = mutation->Apply();
  }
}

RUN_NUM_NODE_NUM_EDGE_BENCHMARK(BM_GraphViewConstruction);
RUN_NUM_NODE_NUM_EDGE_BENCHMARK(BM_MutableGraphViewConstruction);
RUN_NUM_NODE_NUM_EDGE_BENCHMARK(BM_MutableGraphViewClearAttrs);

#define RUN_NUM_NODE_BENCHMARK(name) \
  BENCHMARK(name)                    \
      ->Arg(10)                      \
      ->Arg(100)                     \
      ->Arg(1000)                    \
      ->Arg(10000)                   \
      ->Arg(25000)                   \
      ->Arg(50000)                   \
      ->Arg(100000);

template <typename GraphViewT>
void BM_GraphViewTConstructionWithControlDependencies(
    ::testing::benchmark::State& state) {
  const int num_fanins_fanouts = state.range(0);

  GraphDef graph_def =
      test::CreateFaninFanoutNodeGraph(num_fanins_fanouts, num_fanins_fanouts,
                                       num_fanins_fanouts, num_fanins_fanouts,
                                       /*fanout_unique_index=*/true);

  for (auto i : state) {
    Status s;
    GraphViewT graph_view(&graph_def, &s);
  }
}

void BM_GraphViewConstructionWithControlDependencies(
    ::testing::benchmark::State& state) {
  BM_GraphViewTConstructionWithControlDependencies<GraphView>(state);
}

void BM_MutableGraphViewConstructionWithControlDependencies(
    ::testing::benchmark::State& state) {
  BM_GraphViewTConstructionWithControlDependencies<MutableGraphView>(state);
}

RUN_NUM_NODE_BENCHMARK(BM_GraphViewConstructionWithControlDependencies);
RUN_NUM_NODE_BENCHMARK(BM_MutableGraphViewConstructionWithControlDependencies);

template <typename GraphViewT>
void BM_GraphViewTGetNode(::testing::benchmark::State& state) {
  const int num_nodes = state.range(0);

  GraphDef graph_def =
      test::CreateGraphDef(num_nodes, /*num_edges_per_node=*/16);
  Status s;
  GraphViewT graph_view(&graph_def, &s);

  for (auto i : state) {
    graph_view.GetNode("out");
  }
}

void BM_GraphViewGetNode(::testing::benchmark::State& state) {
  BM_GraphViewTGetNode<GraphView>(state);
}

void BM_MutableGraphViewGetNode(::testing::benchmark::State& state) {
  BM_GraphViewTGetNode<MutableGraphView>(state);
}

RUN_NUM_NODE_BENCHMARK(BM_GraphViewGetNode);
RUN_NUM_NODE_BENCHMARK(BM_MutableGraphViewGetNode);

#define RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(name) \
  BENCHMARK(name)                                \
      ->ArgPair(10, 10)                          \
      ->ArgPair(10, 100)                         \
      ->ArgPair(10, 1000)                        \
      ->ArgPair(10, 10000)                       \
      ->ArgPair(10, 100000)                      \
      ->ArgPair(100, 10)                         \
      ->ArgPair(100, 100)                        \
      ->ArgPair(100, 1000)                       \
      ->ArgPair(100, 10000)                      \
      ->ArgPair(100, 100000)                     \
      ->ArgPair(1000, 10)                        \
      ->ArgPair(1000, 100)                       \
      ->ArgPair(1000, 1000)                      \
      ->ArgPair(1000, 10000)                     \
      ->ArgPair(1000, 100000)                    \
      ->ArgPair(10000, 10)                       \
      ->ArgPair(10000, 100)                      \
      ->ArgPair(10000, 1000)                     \
      ->ArgPair(10000, 10000)                    \
      ->ArgPair(10000, 100000)                   \
      ->ArgPair(100000, 10)                      \
      ->ArgPair(100000, 100)                     \
      ->ArgPair(100000, 1000)                    \
      ->ArgPair(100000, 10000)                   \
      ->ArgPair(100000, 100000);

template <typename GraphViewT>
void BM_GraphViewTGetRegularFanin(::testing::benchmark::State& state) {
  const int num_fanins = state.range(0);
  const int num_fanouts = state.range(1);

  GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  Status s;
  GraphViewT graph_view(&graph_def, &s);

  for (auto i : state) {
    auto* node = graph_view.GetNode("node");
    node->GetRegularFanin(0);
  }
}

void BM_GraphViewGetRegularFanin(::testing::benchmark::State& state) {
  BM_GraphViewTGetRegularFanin<GraphView>(state);
}

void BM_MutableGraphViewGetRegularFanin(::testing::benchmark::State& state) {
  BM_GraphViewTGetRegularFanin<MutableGraphView>(state);
}

RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewGetRegularFanin);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewGetRegularFanin);

template <typename GraphViewT>
void BM_GraphViewTGetRegularFanout(::testing::benchmark::State& state) {
  const int num_fanins = state.range(0);
  const int num_fanouts = state.range(1);

  GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  Status s;
  GraphViewT graph_view(&graph_def, &s);

  for (auto i : state) {
    auto* node = graph_view.GetNode("node");
    node->GetRegularFanout(0);
  }
}

void BM_GraphViewGetRegularFanout(::testing::benchmark::State& state) {
  BM_GraphViewTGetRegularFanout<GraphView>(state);
}

void BM_MutableGraphViewGetRegularFanout(::testing::benchmark::State& state) {
  BM_GraphViewTGetRegularFanout<MutableGraphView>(state);
}

RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewGetRegularFanout);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewGetRegularFanout);

template <typename GraphViewT>
void BM_GraphViewTGetRegularFanins(::testing::benchmark::State& state) {
  const int num_fanins = state.range(0);
  const int num_fanouts = state.range(1);

  GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  Status s;
  GraphViewT graph_view(&graph_def, &s);

  for (auto i : state) {
    auto* node = graph_view.GetNode("node");
    node->GetRegularFanins();
  }
}

void BM_GraphViewGetRegularFanins(::testing::benchmark::State& state) {
  BM_GraphViewTGetRegularFanins<GraphView>(state);
}

void BM_MutableGraphViewGetRegularFanins(::testing::benchmark::State& state) {
  BM_GraphViewTGetRegularFanins<MutableGraphView>(state);
}

RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewGetRegularFanins);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewGetRegularFanins);

template <typename GraphViewT>
void BM_GraphViewTGetRegularFanouts(::testing::benchmark::State& state) {
  const int num_fanins = state.range(0);
  const int num_fanouts = state.range(1);

  GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  Status s;
  GraphViewT graph_view(&graph_def, &s);

  for (auto i : state) {
    auto* node = graph_view.GetNode("node");
    node->GetRegularFanouts();
  }
}

void BM_GraphViewGetRegularFanouts(::testing::benchmark::State& state) {
  BM_GraphViewTGetRegularFanouts<GraphView>(state);
}

void BM_MutableGraphViewGetRegularFanouts(::testing::benchmark::State& state) {
  BM_GraphViewTGetRegularFanouts<MutableGraphView>(state);
}

RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewGetRegularFanouts);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewGetRegularFanouts);

template <typename GraphViewT>
void BM_GraphViewTGetControllingFanins(::testing::benchmark::State& state) {
  const int num_fanins = state.range(0);
  const int num_fanouts = state.range(1);

  GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  Status s;
  GraphViewT graph_view(&graph_def, &s);

  for (auto i : state) {
    auto* node = graph_view.GetNode("node");
    node->GetControllingFanins();
  }
}

void BM_GraphViewGetControllingFanins(::testing::benchmark::State& state) {
  BM_GraphViewTGetControllingFanins<GraphView>(state);
}

void BM_MutableGraphViewGetControllingFanins(
    ::testing::benchmark::State& state) {
  BM_GraphViewTGetControllingFanins<MutableGraphView>(state);
}

RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewGetControllingFanins);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewGetControllingFanins);

template <typename GraphViewT>
void BM_GraphViewTGetControlledFanouts(::testing::benchmark::State& state) {
  const int num_fanins = state.range(0);
  const int num_fanouts = state.range(1);

  GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  Status s;
  GraphViewT graph_view(&graph_def, &s);

  for (auto i : state) {
    auto* node = graph_view.GetNode("node");
    node->GetControlledFanouts();
  }
}

void BM_GraphViewGetControlledFanouts(::testing::benchmark::State& state) {
  BM_GraphViewTGetControlledFanouts<GraphView>(state);
}

void BM_MutableGraphViewGetControlledFanouts(
    ::testing::benchmark::State& state) {
  BM_GraphViewTGetControlledFanouts<MutableGraphView>(state);
}

RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewGetControlledFanouts);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewGetControlledFanouts);

template <typename GraphViewT, bool IsLast>
inline void BM_GraphViewTHasRegularFanin(::testing::benchmark::State& state) {
  const int num_fanins = state.range(0);
  const int num_fanouts = state.range(1);

  GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, /*num_controlling_fanins=*/0,
      /*num_controlled_fanouts=*/0, /*fanout_unique_index=*/false);
  Status s;
  GraphViewT graph_view(&graph_def, &s);
  const int index = IsLast ? num_fanouts - 1 : 0;
  auto* node = graph_view.GetNode(absl::StrFormat("out%05d", index));
  auto* fanin = graph_view.GetNode("node");

  for (auto i : state) {
    node->HasFanin({&graph_view, fanin->node_index(), 0});
  }
}

void BM_GraphViewHasRegularFaninFirst(::testing::benchmark::State& state) {
  BM_GraphViewTHasRegularFanin<GraphView, false>(state);
}

void BM_GraphViewHasRegularFaninLast(::testing::benchmark::State& state) {
  BM_GraphViewTHasRegularFanin<GraphView, true>(state);
}

void BM_MutableGraphViewHasRegularFaninFirst(
    ::testing::benchmark::State& state) {
  BM_GraphViewTHasRegularFanin<MutableGraphView, false>(state);
}

void BM_MutableGraphViewHasRegularFaninLast(
    ::testing::benchmark::State& state) {
  BM_GraphViewTHasRegularFanin<MutableGraphView, true>(state);
}

RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewHasRegularFaninFirst);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewHasRegularFaninLast);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewHasRegularFaninFirst);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewHasRegularFaninLast);

template <typename GraphViewT, bool IsLast>
inline void BM_GraphViewTHasControllingFanin(
    ::testing::benchmark::State& state) {
  const int num_fanins = state.range(0);
  const int num_fanouts = state.range(1);

  GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  Status s;
  GraphViewT graph_view(&graph_def, &s);
  const int index = IsLast ? num_fanouts - 1 : 0;
  auto* node = graph_view.GetNode(absl::StrFormat("control_out%05d", index));
  auto* fanin = graph_view.GetNode("node");

  for (auto i : state) {
    node->HasFanin({&graph_view, fanin->node_index(), Graph::kControlSlot});
  }
}

void BM_GraphViewHasControllingFaninFirst(::testing::benchmark::State& state) {
  BM_GraphViewTHasControllingFanin<GraphView, false>(state);
}

void BM_GraphViewHasControllingFaninLast(::testing::benchmark::State& state) {
  BM_GraphViewTHasControllingFanin<GraphView, true>(state);
}

void BM_MutableGraphViewHasControllingFaninFirst(
    ::testing::benchmark::State& state) {
  BM_GraphViewTHasControllingFanin<MutableGraphView, false>(state);
}

void BM_MutableGraphViewHasControllingFaninLast(
    ::testing::benchmark::State& state) {
  BM_GraphViewTHasControllingFanin<MutableGraphView, true>(state);
}

RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewHasControllingFaninFirst);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewHasControllingFaninLast);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewHasControllingFaninFirst);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewHasControllingFaninLast);

template <typename GraphViewT, bool IsLast>
inline void BM_GraphViewTHasRegularFanout(::testing::benchmark::State& state) {
  const int num_fanins = state.range(0);
  const int num_fanouts = state.range(1);

  GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, /*num_controlling_fanins=*/0,
      /*num_controlled_fanouts=*/0, /*fanout_unique_index=*/false);
  Status s;
  GraphViewT graph_view(&graph_def, &s);
  const int index = IsLast ? num_fanins - 1 : 0;
  auto* node = graph_view.GetNode(absl::StrFormat("in%05d", index));
  auto* fanout = graph_view.GetNode("node");

  for (auto i : state) {
    node->HasFanout({&graph_view, fanout->node_index(), index});
  }
}

void BM_GraphViewHasRegularFanoutFirst(::testing::benchmark::State& state) {
  BM_GraphViewTHasRegularFanout<GraphView, false>(state);
}

void BM_GraphViewHasRegularFanoutLast(::testing::benchmark::State& state) {
  BM_GraphViewTHasRegularFanout<GraphView, true>(state);
}

void BM_MutableGraphViewHasRegularFanoutFirst(
    ::testing::benchmark::State& state) {
  BM_GraphViewTHasRegularFanout<MutableGraphView, false>(state);
}

void BM_MutableGraphViewHasRegularFanoutLast(
    ::testing::benchmark::State& state) {
  BM_GraphViewTHasRegularFanout<MutableGraphView, true>(state);
}

RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewHasRegularFanoutFirst);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewHasRegularFanoutLast);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewHasRegularFanoutFirst);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewHasRegularFanoutLast);

template <typename GraphViewT, bool IsLast>
inline void BM_GraphViewTHasControlledFanout(
    ::testing::benchmark::State& state) {
  const int num_fanins = state.range(0);
  const int num_fanouts = state.range(1);

  GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/false);
  Status s;
  GraphViewT graph_view(&graph_def, &s);
  const int index = IsLast ? num_fanins - 1 : 0;
  auto* node = graph_view.GetNode(absl::StrFormat("control_in%05d", index));
  auto* fanout = graph_view.GetNode("node");

  for (auto i : state) {
    node->HasFanout({&graph_view, fanout->node_index(), Graph::kControlSlot});
  }
}

void BM_GraphViewHasControlledFanoutFirst(::testing::benchmark::State& state) {
  BM_GraphViewTHasControlledFanout<GraphView, false>(state);
}

void BM_GraphViewHasControlledFanoutLast(::testing::benchmark::State& state) {
  BM_GraphViewTHasControlledFanout<GraphView, true>(state);
}

void BM_MutableGraphViewHasControlledFanoutFirst(
    ::testing::benchmark::State& state) {
  BM_GraphViewTHasControlledFanout<MutableGraphView, false>(state);
}

void BM_MutableGraphViewHasControlledFanoutLast(
    ::testing::benchmark::State& state) {
  BM_GraphViewTHasControlledFanout<MutableGraphView, true>(state);
}

RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewHasControlledFanoutFirst);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_GraphViewHasControlledFanoutLast);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewHasControlledFanoutFirst);
RUN_NUM_FANIN_NUM_FANOUT_BENCHMARK(BM_MutableGraphViewHasControlledFanoutLast);

void BM_SortTopologically(::testing::benchmark::State& state) {
  const int size = state.range(0);

  GraphDef graph = test::CreateRandomGraph(size);
  Status status;
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);

  for (auto i : state) {
    TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  }
}

RUN_NUM_NODE_BENCHMARK(BM_SortTopologically);

}  // namespace
}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow
