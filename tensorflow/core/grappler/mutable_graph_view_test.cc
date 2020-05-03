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

#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using ::tensorflow::test::function::NDef;
using FDH = FunctionDefHelper;

void CompareNodeFanins(const MutableGraphView& graph, NodeDef* node,
                       absl::Span<const string> fanins) {
  ASSERT_EQ(node->input_size(), fanins.size());
  for (int i = 0; i < node->input_size(); ++i) {
    TensorId tensor_id = ParseTensorName(fanins[i]);
    EXPECT_EQ(ParseTensorName(node->input(i)), tensor_id);
    int port;
    if (tensor_id.index() == Graph::kControlSlot) {
      port = Graph::kControlSlot;
    } else {
      port = i;
    }
    MutableGraphView::InputPort input_port(node, port);
    MutableGraphView::OutputPort output_port =
        graph.GetOutputPort(tensor_id.node(), tensor_id.index());
    EXPECT_TRUE(graph.GetFanin(input_port).contains(output_port));
    EXPECT_TRUE(graph.GetFanout(output_port).contains(input_port));
  }
}

void CompareNodeFanouts(const MutableGraphView& graph, NodeDef* node,
                        absl::Span<const string> fanouts) {
  auto node_fanouts =
      graph.GetFanouts(*node, /*include_controlled_nodes=*/true);
  EXPECT_EQ(node_fanouts.size(), fanouts.size());
  for (const string& fanout : fanouts) {
    TensorId tensor_id = ParseTensorName(fanout);
    MutableGraphView::InputPort input_port(graph.GetNode(tensor_id.node()),
                                           tensor_id.index());
    EXPECT_TRUE(node_fanouts.contains(input_port));
  }
}

void CheckNode(const MutableGraphView& graph, absl::string_view node_name,
               absl::string_view op, absl::string_view device,
               absl::Span<const std::pair<string, FDH::AttrValueWrapper>> attrs,
               absl::Span<const string> fanins,
               absl::Span<const string> fanouts) {
  NodeDef* node = graph.GetNode(node_name);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->op(), op);
  EXPECT_EQ(node->device(), device);
  EXPECT_EQ(node->attr_size(), attrs.size());
  for (const auto& attr : attrs) {
    auto it = node->attr().find(attr.first);
    ASSERT_NE(it, node->attr().end());
    EXPECT_TRUE(AreAttrValuesEqual(it->second, attr.second.proto));
  }
  CompareNodeFanins(graph, node, fanins);
  CompareNodeFanouts(graph, node, fanouts);
}

void CheckGraph(const MutableGraphView& mutable_graph) {
  GraphView immutable_graph(mutable_graph.graph());
  EXPECT_EQ(mutable_graph.graph()->node_size(),
            immutable_graph.graph()->node_size());
  EXPECT_EQ(mutable_graph.graph(), immutable_graph.graph());

  auto check_edges =
      [](const absl::flat_hash_set<MutableGraphView::Edge>& mutable_edges,
         const absl::flat_hash_set<GraphView::Edge>& immutable_edges) {
        EXPECT_EQ(mutable_edges.size(), immutable_edges.size());
        for (const auto& fanin_edge : mutable_edges) {
          GraphView::Edge immutable_edge(
              {fanin_edge.src.node, fanin_edge.src.port_id},
              {fanin_edge.dst.node, fanin_edge.dst.port_id});
          EXPECT_TRUE(immutable_edges.contains(immutable_edge));
        }
      };

  // Check graph connectivity.
  for (auto& node : *mutable_graph.graph()->mutable_node()) {
    EXPECT_EQ(&node, immutable_graph.GetNode(node.name()));

    auto mutable_fanins =
        mutable_graph.GetFanins(node, /*include_controlling_nodes=*/true);
    auto immutable_fanins =
        immutable_graph.GetFanins(node, /*include_controlling_nodes=*/true);
    EXPECT_EQ(mutable_fanins.size(), immutable_fanins.size());
    for (const auto& fanin : mutable_fanins) {
      GraphView::OutputPort immutable_fanin(fanin.node, fanin.port_id);
      EXPECT_TRUE(immutable_fanins.contains(immutable_fanin));
    }

    auto mutable_fanouts =
        mutable_graph.GetFanouts(node, /*include_controlled_nodes=*/true);
    auto immutable_fanouts =
        immutable_graph.GetFanouts(node, /*include_controlled_nodes=*/true);
    EXPECT_EQ(mutable_fanouts.size(), immutable_fanouts.size());
    for (const auto& fanout : mutable_fanouts) {
      GraphView::InputPort immutable_fanout(fanout.node, fanout.port_id);
      EXPECT_TRUE(immutable_fanouts.contains(immutable_fanout));
    }

    auto mutable_fanin_edges =
        mutable_graph.GetFaninEdges(node, /*include_controlling_edges=*/true);
    auto immutable_fanin_edges =
        immutable_graph.GetFaninEdges(node, /*include_controlling_edges=*/true);
    check_edges(mutable_fanin_edges, immutable_fanin_edges);

    auto mutable_fanout_edges =
        mutable_graph.GetFanoutEdges(node, /*include_controlled_edges=*/true);
    auto immutable_fanout_edges =
        immutable_graph.GetFanoutEdges(node, /*include_controlled_edges=*/true);
    check_edges(mutable_fanout_edges, immutable_fanout_edges);
  }
}

TEST(MutableGraphViewTest, AddSubgraph) {
  GraphDef graph_def = test::function::GDef(
      {
          NDef("foo", "NotImportant", {}, {}),
          NDef("bar", "NotImportant", {}, {}),
          NDef("baz", "NotImportant", {"foo", "bar"}),
      },
      /*funcs=*/{});
  MutableGraphView graph(&graph_def);

  // `s/bar` node has inputs that are valid only if we add subgraph into the
  // original graph.
  GraphDef subgraph = test::function::GDef(
      {
          NDef("s/n0", "NotImportant", {}, {}),
          NDef("s/n1", "NotImportant", {"bar", "s/n0"}, {}),
      },
      /*funcs=*/{});

  TF_EXPECT_OK(graph.AddSubgraph(std::move(subgraph)));

  // Fanins and fanouts must be updated for the nodes of the original graph, and
  // added subgraph.
  CheckNode(graph, "bar", "NotImportant", "", {}, {}, {"baz:1", "s/n1"});
  CheckNode(graph, "s/n1", "NotImportant", "", {}, {"bar", "s/n0"}, {});
  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddSubgraphAndAddFunction) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);

  FunctionDef x_times_two = test::function::XTimesTwo();
  GraphDef subgraph = test::function::GDef({}, {x_times_two});

  TF_EXPECT_OK(graph.AddSubgraph(std::move(subgraph)));
  EXPECT_EQ(graph_def.library().function_size(), 1);
}

TEST(MutableGraphViewTest, AddSubgraphAndSkipSameFunction) {
  FunctionDef x_times_two = test::function::XTimesTwo();

  GraphDef graph_def = test::function::GDef({}, {x_times_two});
  MutableGraphView graph(&graph_def);

  GraphDef subgraph = test::function::GDef({}, {x_times_two});

  TF_EXPECT_OK(graph.AddSubgraph(std::move(subgraph)));
  EXPECT_EQ(graph_def.library().function_size(), 1);
}

TEST(MutableGraphViewTest, AddSubgraphAndFailIfFunctionDifferent) {
  FunctionDef x_times_four = test::function::XTimesFour();
  x_times_four.mutable_signature()->set_name("XTimesTwo");

  GraphDef graph_def = test::function::GDef({}, {x_times_four});
  MutableGraphView graph(&graph_def);

  FunctionDef x_times_two = test::function::XTimesTwo();
  GraphDef subgraph = test::function::GDef({}, {x_times_two});

  Status status = graph.AddSubgraph(std::move(subgraph));
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_message(),
            "MutableGraphView::AddSubgraph(function_size=1) error: Found "
            "different function definition with the same name: XTimesTwo.");
}

TEST(MutableGraphViewTest, UpdateNodeNoDedupControlDependency) {
  constexpr char kDevice[] = "/device:foo:0";
  GraphDef graph_def = test::function::GDef(
      {NDef("bar_1", "Switch", {}, {}), NDef("bar_2", "Identity", {"bar_1:1"}),
       NDef("other", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"bar_2", "other", "bar_2:1", "^bar_2"}),
       NDef("foo_2", "NotImportant", {"other:1", "bar_2:2", "^bar_2"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  AttrValue list_value;
  list_value.mutable_list()->add_type(DT_FLOAT);
  TF_EXPECT_OK(
      graph.UpdateNode("bar_2", "IdentityN", kDevice, {{"T", list_value}}));

  CheckNode(graph, "bar_1", "Switch", "", {}, {}, {"bar_2"});
  CheckNode(graph, "bar_2", "IdentityN", kDevice, {{"T", list_value}},
            {"bar_1:1"}, {"foo_1", "foo_1:2", "^foo_1", "foo_2:1", "^foo_2"});
  CheckNode(graph, "other", "NotImportant", "", {}, {}, {"foo_1:1", "foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"bar_2", "other", "bar_2:1", "^bar_2"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {},
            {"other:1", "bar_2:2", "^bar_2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateNodeDedupControlDependency) {
  constexpr char kDevice[] = "/device:foo:0";
  GraphDef graph_def = test::function::GDef(
      {NDef("bar_1", "Switch", {}, {}), NDef("bar_2", "Identity", {"bar_1:1"}),
       NDef("other", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"bar_2", "other", "bar_2:1", "^bar_2"}),
       NDef("foo_2", "NotImportant", {"other:1", "bar_2:2", "^bar_2"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.UpdateNode("bar_2", "NotImportant", kDevice, {}));

  CheckNode(graph, "bar_1", "Switch", "", {}, {}, {"bar_2"});
  CheckNode(graph, "bar_2", "NotImportant", kDevice, {}, {"bar_1:1"},
            {"foo_1", "foo_1:2", "foo_2:1"});
  CheckNode(graph, "other", "NotImportant", "", {}, {}, {"foo_1:1", "foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"bar_2", "other", "bar_2:1"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {}, {"other:1", "bar_2:2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateNodeSwitchNoControlDependency) {
  constexpr char kDevice[] = "/device:foo:0";
  GraphDef graph_def =
      test::function::GDef({NDef("foo", "NotImportant", {}, {}),
                            NDef("bar", "NotImportant", {"foo:1"})},
                           /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.UpdateNode("foo", "Switch", kDevice, {}));

  CheckNode(graph, "foo", "Switch", kDevice, {}, {}, {"bar"});
  CheckNode(graph, "bar", "NotImportant", "", {}, {"foo:1"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateNodeSwitchControlDependency) {
  constexpr char kDevice[] = "/device:foo:0";
  GraphDef graph_def =
      test::function::GDef({NDef("foo", "NotImportant", {}, {}),
                            NDef("bar", "NotImportant", {"^foo"})},
                           /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  AttrValue attr;
  attr.set_type(DT_FLOAT);
  Status s = graph.UpdateNode("foo", "Switch", kDevice, {{"T", attr}});
  EXPECT_FALSE(s.ok());
  string expected_msg =
      "MutableGraphView::UpdateNodeOp(node_name='foo', op='Switch', "
      "device='/device:foo:0', attrs={('T', type: DT_FLOAT)}) error: can't "
      "change node op to Switch when node drives a control dependency "
      "(alternatively, we could add the identity node needed, but it seems "
      "like an unlikely event and probably a mistake).";
  EXPECT_EQ(s.error_message(), expected_msg);

  CheckNode(graph, "foo", "NotImportant", "", {}, {}, {"^bar"});
  CheckNode(graph, "bar", "NotImportant", "", {}, {"^foo"}, {});

  CheckGraph(graph);
}

absl::flat_hash_map<string, std::vector<string>> GetNodeInputsFromGraph(
    const GraphDef& graph, absl::string_view node_to_exclude) {
  absl::flat_hash_map<string, std::vector<string>> node_inputs;
  for (const auto& node : graph.node()) {
    if (node.name() == node_to_exclude) {
      continue;
    }
    node_inputs[node.name()] =
        std::vector<string>(node.input().begin(), node.input().end());
  }
  return node_inputs;
}

void CheckUnmodifiedNodeFanins(
    const GraphDef& graph, absl::string_view node_to_exclude,
    const absl::flat_hash_map<string, std::vector<string>>&
        unmodified_node_inputs) {
  for (const auto& node : graph.node()) {
    if (node.name() == node_to_exclude) {
      continue;
    }
    auto it = unmodified_node_inputs.find(node.name());
    ASSERT_NE(it, unmodified_node_inputs.end());
    ASSERT_EQ(it->second.size(), node.input_size());
    for (int i = 0; i < node.input_size(); ++i) {
      EXPECT_EQ(node.input(i), it->second[i]);
    }
  }
}

void TestUpdateNodeName(absl::string_view from_node_name, bool node_exists,
                        absl::string_view to_node_name, bool update_fanouts,
                        bool success, const string& error_msg,
                        absl::Span<const string> expected_fanins) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a"}),
       NDef("c", "NotImportant", {}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  NodeDef* node = graph.GetNode(from_node_name);
  if (node_exists) {
    EXPECT_NE(node, nullptr);
  } else {
    EXPECT_EQ(node, nullptr);
  }

  absl::flat_hash_map<string, std::vector<string>> unmodified_node_inputs =
      GetNodeInputsFromGraph(graph_def, from_node_name);

  Status s = graph.UpdateNodeName(from_node_name, to_node_name, update_fanouts);
  EXPECT_EQ(s.ok(), success);
  string updated_node_name;
  if (success) {
    updated_node_name = string(to_node_name);
  } else {
    updated_node_name = string(from_node_name);
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (node_exists) {
    EXPECT_EQ(node->name(), updated_node_name);
    CompareNodeFanins(graph, node, expected_fanins);
  }

  CheckUnmodifiedNodeFanins(graph_def, updated_node_name,
                            unmodified_node_inputs);

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateNodeName) {
  string error_msg;
  // Node has no fanouts.
  TestUpdateNodeName("b", /*node_exists=*/true, "d", /*update_fanouts=*/false,
                     /*success=*/true, error_msg, {"a"});
  // Node has fanouts and rename to self.
  TestUpdateNodeName("b", /*node_exists=*/true, "b", /*update_fanouts=*/false,
                     /*success=*/true, error_msg, {"a"});
  // Node has no fanouts and rename to self.
  TestUpdateNodeName("a", /*node_exists=*/true, "a", /*update_fanouts=*/false,
                     /*success=*/true, error_msg, {});

  // New node name is in use.
  error_msg =
      "MutableGraphView::UpdateNodeName(from_node_name='c', to_node_name='b', "
      "update_fanouts=false) error: can't update node name because new node "
      "name is in use.";
  TestUpdateNodeName("c", /*node_exists=*/true, "b", /*update_fanouts=*/false,
                     /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::UpdateNodeName(from_node_name='a', to_node_name='b', "
      "update_fanouts=true) error: can't update node name because new node "
      "name is in use.";
  TestUpdateNodeName("a", /*node_exists=*/true, "b", /*update_fanouts=*/true,
                     /*success=*/false, error_msg, {});
  // Node has fanouts.
  error_msg =
      "MutableGraphView::UpdateNodeName(from_node_name='a', to_node_name='d', "
      "update_fanouts=false) error: can't update node name because node has "
      "fanouts.";
  TestUpdateNodeName("a", /*node_exists=*/true, "d", /*update_fanouts=*/false,
                     /*success=*/false, error_msg, {});
  // Node does not exist.
  error_msg =
      "MutableGraphView::UpdateNodeName(from_node_name='d', to_node_name='e', "
      "update_fanouts=false) error: node 'd' was not found.";
  TestUpdateNodeName("d", /*node_exists=*/false, "e", /*update_fanouts=*/false,
                     /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::UpdateNodeName(from_node_name='d', to_node_name='e', "
      "update_fanouts=true) error: node 'd' was not found.";
  TestUpdateNodeName("d", /*node_exists=*/false, "e", /*update_fanouts=*/true,
                     /*success=*/false, error_msg, {});
}

TEST(MutableGraphViewTest, UpdateNodeNameWithFanouts) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:2"}),
       NDef("c", "NotImportant", {"b", "^a"}),
       NDef("d", "NotImportant", {"^b", "^a"}),
       NDef("e", "NotImportant", {"b:2", "c:4", "b:1", "^a"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.UpdateNodeName("b", "f", /*update_fanouts=*/true));

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"f", "^c", "^d", "^e"});
  CheckNode(graph, "f", "NotImportant", "", {}, {"a:2"},
            {"c", "^d", "e", "e:2"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"f", "^a"}, {"e:1"});
  CheckNode(graph, "d", "NotImportant", "", {}, {"^f", "^a"}, {});
  CheckNode(graph, "e", "NotImportant", "", {}, {"f:2", "c:4", "f:1", "^a"},
            {});

  CheckGraph(graph);
}

GraphDef SimpleSwapNodeNamesMutationGraph() {
  return test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("switch_1", "Switch", {"a"}),
       NDef("identity_1", "Identity", {"switch_1:1"}),
       NDef("b", "NotImportant", {}, {}), NDef("switch_2", "Switch", {"b"}),
       NDef("identity_2", "Identity", {"switch_2:0"}),
       NDef("foo_1", "NotImportant", {"identity_1", "^identity_1"}),
       NDef("foo_2", "NotImportant", {"identity_2", "^identity_2"})},
      /*funcs=*/{});
}

void TestSwapNodeNames(bool update_fanouts) {
  GraphDef graph_def = SimpleSwapNodeNamesMutationGraph();

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.SwapNodeNames("foo_1", "foo_2", update_fanouts));

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"switch_1"});
  CheckNode(graph, "switch_1", "Switch", "", {}, {"a"}, {"identity_1"});
  CheckNode(graph, "identity_1", "Identity", "", {}, {"switch_1:1"},
            {"foo_2", "^foo_2"});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"switch_2"});
  CheckNode(graph, "switch_2", "Switch", "", {}, {"b"}, {"identity_2"});
  CheckNode(graph, "identity_2", "Identity", "", {}, {"switch_2:0"},
            {"foo_1", "^foo_1"});
  CheckNode(graph, "foo_2", "NotImportant", "", {},
            {"identity_1", "^identity_1"}, {});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"identity_2", "^identity_2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphView, SwapNodeNames) {
  TestSwapNodeNames(/*update_fanouts=*/false);
  TestSwapNodeNames(/*update_fanouts=*/true);
}

void TestSwapNodeNamesWithSameNames(bool update_fanouts) {
  GraphDef graph_def = SimpleSwapNodeNamesMutationGraph();

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.SwapNodeNames("identity_1", "identity_1", update_fanouts));

  // No changes to graph.
  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"switch_1"});
  CheckNode(graph, "switch_1", "Switch", "", {}, {"a"}, {"identity_1"});
  CheckNode(graph, "identity_1", "Identity", "", {}, {"switch_1:1"},
            {"foo_1", "^foo_1"});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"switch_2"});
  CheckNode(graph, "switch_2", "Switch", "", {}, {"b"}, {"identity_2"});
  CheckNode(graph, "identity_2", "Identity", "", {}, {"switch_2:0"},
            {"foo_2", "^foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"identity_1", "^identity_1"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {},
            {"identity_2", "^identity_2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphView, SwapNodeNamesSameName) {
  TestSwapNodeNamesWithSameNames(/*update_fanouts=*/false);
  TestSwapNodeNamesWithSameNames(/*update_fanouts=*/true);
}

TEST(MutableGraphView, SwapNodeNamesBetweenSwitches) {
  GraphDef graph_def = SimpleSwapNodeNamesMutationGraph();

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(
      graph.SwapNodeNames("switch_1", "switch_2", /*update_fanouts=*/false));

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"switch_2"});
  CheckNode(graph, "switch_2", "Switch", "", {}, {"a"}, {"identity_2"});
  CheckNode(graph, "identity_1", "Identity", "", {}, {"switch_1:1"},
            {"foo_1", "^foo_1"});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"switch_1"});
  CheckNode(graph, "switch_1", "Switch", "", {}, {"b"}, {"identity_1"});
  CheckNode(graph, "identity_2", "Identity", "", {}, {"switch_2:0"},
            {"foo_2", "^foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"identity_1", "^identity_1"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {},
            {"identity_2", "^identity_2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphView, SwapNodeNamesBetweenSwitchesAndUpdateFanouts) {
  GraphDef graph_def = SimpleSwapNodeNamesMutationGraph();

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(
      graph.SwapNodeNames("switch_1", "switch_2", /*update_fanouts=*/true));

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"switch_2"});
  CheckNode(graph, "switch_2", "Switch", "", {}, {"a"}, {"identity_1"});
  CheckNode(graph, "identity_1", "Identity", "", {}, {"switch_2:1"},
            {"foo_1", "^foo_1"});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"switch_1"});
  CheckNode(graph, "switch_1", "Switch", "", {}, {"b"}, {"identity_2"});
  CheckNode(graph, "identity_2", "Identity", "", {}, {"switch_1:0"},
            {"foo_2", "^foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"identity_1", "^identity_1"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {},
            {"identity_2", "^identity_2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphView, SwapNodeNamesSwitchAndNonSwitch) {
  GraphDef graph_def = SimpleSwapNodeNamesMutationGraph();

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.SwapNodeNames("a", "switch_1", /*update_fanouts=*/false));

  // Dedup controls and fix self loop.
  CheckNode(graph, "switch_1", "NotImportant", "", {}, {}, {"a", "identity_1"});
  CheckNode(graph, "a", "Switch", "", {}, {"switch_1"}, {});
  CheckNode(graph, "identity_1", "Identity", "", {}, {"switch_1:1"}, {"foo_1"});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"switch_2"});
  CheckNode(graph, "switch_2", "Switch", "", {}, {"b"}, {"identity_2"});
  CheckNode(graph, "identity_2", "Identity", "", {}, {"switch_2:0"},
            {"foo_2", "^foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {}, {"identity_1"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {},
            {"identity_2", "^identity_2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphView, SwapNodeNamesSwitchAndNonSwitchAndUpdateFanouts) {
  GraphDef graph_def = SimpleSwapNodeNamesMutationGraph();

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.SwapNodeNames("a", "switch_1", /*update_fanouts=*/true));

  CheckNode(graph, "switch_1", "NotImportant", "", {}, {}, {"a"});
  CheckNode(graph, "a", "Switch", "", {}, {"switch_1"}, {"identity_1"});
  CheckNode(graph, "identity_1", "Identity", "", {}, {"a:1"},
            {"foo_1", "^foo_1"});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"switch_2"});
  CheckNode(graph, "switch_2", "Switch", "", {}, {"b"}, {"identity_2"});
  CheckNode(graph, "identity_2", "Identity", "", {}, {"switch_2:0"},
            {"foo_2", "^foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"identity_1", "^identity_1"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {},
            {"identity_2", "^identity_2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphView, SwapNodeNamesNonSwitchAndSwitch) {
  GraphDef graph_def = SimpleSwapNodeNamesMutationGraph();

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.SwapNodeNames("switch_2", "b", /*update_fanouts=*/false));

  // Dedup controls and fix self loop.
  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"switch_1"});
  CheckNode(graph, "switch_1", "Switch", "", {}, {"a"}, {"identity_1"});
  CheckNode(graph, "identity_1", "Identity", "", {}, {"switch_1:1"},
            {"foo_1", "^foo_1"});
  CheckNode(graph, "switch_2", "NotImportant", "", {}, {}, {"b", "identity_2"});
  CheckNode(graph, "b", "Switch", "", {}, {"switch_2"}, {});
  CheckNode(graph, "identity_2", "Identity", "", {}, {"switch_2:0"}, {"foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"identity_1", "^identity_1"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {}, {"identity_2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphView, SwapNodeNamesNonSwitchAndSwitchAndUpdateFanouts) {
  GraphDef graph_def = SimpleSwapNodeNamesMutationGraph();

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.SwapNodeNames("switch_2", "b", /*update_fanouts=*/true));

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"switch_1"});
  CheckNode(graph, "switch_1", "Switch", "", {}, {"a"}, {"identity_1"});
  CheckNode(graph, "identity_1", "Identity", "", {}, {"switch_1:1"},
            {"foo_1", "^foo_1"});
  CheckNode(graph, "switch_2", "NotImportant", "", {}, {}, {"b"});
  CheckNode(graph, "b", "Switch", "", {}, {"switch_2"}, {"identity_2"});
  CheckNode(graph, "identity_2", "Identity", "", {}, {"b:0"},
            {"foo_2", "^foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"identity_1", "^identity_1"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {},
            {"identity_2", "^identity_2"}, {});

  CheckGraph(graph);
}

void TestSwapNodeNamesSimpleSelfLoop(bool update_fanouts) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {"b:7"}), NDef("b", "NotImportant", {"a:10"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.SwapNodeNames("a", "b", update_fanouts));

  // No self loops.
  CheckNode(graph, "a", "NotImportant", "", {}, {"b:10"}, {"b:0"});
  CheckNode(graph, "b", "NotImportant", "", {}, {"a:7"}, {"a:0"});

  CheckGraph(graph);
}

TEST(MutableGraphView, SwapNodeNamesSelfLoops) {
  TestSwapNodeNamesSimpleSelfLoop(/*update_fanouts=*/false);
  TestSwapNodeNamesSimpleSelfLoop(/*update_fanouts=*/true);
}

void TestSwapNodeNamesError(absl::string_view from_node_name,
                            absl::string_view to_node_name, bool update_fanouts,
                            const string& error_msg) {
  GraphDef graph_def = SimpleSwapNodeNamesMutationGraph();

  MutableGraphView graph(&graph_def);

  Status s = graph.SwapNodeNames(from_node_name, to_node_name, update_fanouts);
  EXPECT_EQ(s.ok(), false);
  EXPECT_EQ(s.error_message(), error_msg);

  // No changes to graph.
  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"switch_1"});
  CheckNode(graph, "switch_1", "Switch", "", {}, {"a"}, {"identity_1"});
  CheckNode(graph, "identity_1", "Identity", "", {}, {"switch_1:1"},
            {"foo_1", "^foo_1"});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"switch_2"});
  CheckNode(graph, "switch_2", "Switch", "", {}, {"b"}, {"identity_2"});
  CheckNode(graph, "identity_2", "Identity", "", {}, {"switch_2:0"},
            {"foo_2", "^foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"identity_1", "^identity_1"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {},
            {"identity_2", "^identity_2"}, {});

  CheckGraph(graph);
}

// TODO(lyandy): add tests with update_fanouts == true.
TEST(MutableGraphView, SwapNodeNamesError) {
  string error_msg;
  // Missing nodes.
  error_msg =
      "MutableGraphView::SwapNodeNames(from_node_name='foo_3', "
      "to_node_name='foo_2', update_fanouts=false) error: node 'foo_3' was not "
      "found.";
  TestSwapNodeNamesError("foo_3", "foo_2", /*update_fanouts=*/false, error_msg);
  error_msg =
      "MutableGraphView::SwapNodeNames(from_node_name='foo_3', "
      "to_node_name='foo_2', update_fanouts=true) error: node 'foo_3' was not "
      "found.";
  TestSwapNodeNamesError("foo_3", "foo_2", /*update_fanouts=*/true, error_msg);
  error_msg =
      "MutableGraphView::SwapNodeNames(from_node_name='foo_1', "
      "to_node_name='foo_4', update_fanouts=false) error: node 'foo_4' was not "
      "found.";
  TestSwapNodeNamesError("foo_1", "foo_4", /*update_fanouts=*/false, error_msg);
  error_msg =
      "MutableGraphView::SwapNodeNames(from_node_name='foo_1', "
      "to_node_name='foo_4', update_fanouts=true) error: node 'foo_4' was not "
      "found.";
  TestSwapNodeNamesError("foo_1", "foo_4", /*update_fanouts=*/true, error_msg);
  error_msg =
      "MutableGraphView::SwapNodeNames(from_node_name='foo_5', "
      "to_node_name='foo_6', update_fanouts=false) error: node 'foo_5' was not "
      "found.";
  TestSwapNodeNamesError("foo_5", "foo_6", /*update_fanouts=*/false, error_msg);
  error_msg =
      "MutableGraphView::SwapNodeNames(from_node_name='foo_5', "
      "to_node_name='foo_6', update_fanouts=true) error: node 'foo_5' was not "
      "found.";
  TestSwapNodeNamesError("foo_5", "foo_6", /*update_fanouts=*/true, error_msg);

  // Switch control dependencies.
  error_msg =
      "MutableGraphView::SwapNodeNames(from_node_name='switch_2', "
      "to_node_name='identity_1', update_fanouts=false) error: can't swap node "
      "name 'switch_2' as it will become a Switch control dependency.";
  TestSwapNodeNamesError("switch_2", "identity_1", /*update_fanouts=*/false,
                         error_msg);
  error_msg =
      "MutableGraphView::SwapNodeNames(from_node_name='identity_2', "
      "to_node_name='switch_1', update_fanouts=false) error: can't swap node "
      "name 'switch_1' as it will become a Switch control dependency.";
  TestSwapNodeNamesError("identity_2", "switch_1", /*update_fanouts=*/false,
                         error_msg);
}

TEST(MutableGraphViewTest, AddAndUpdateFanouts) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("bar", "NotImportant", {}, {}),
       NDef("other", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"bar", "other", "bar:1", "^bar"}),
       NDef("foo_2", "NotImportant", {"other:1", "bar:2", "^bar"}),
       NDef("foo_3", "NotImportant", {"other:2", "^bar"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  NodeDef* new_bar = graph.AddNode(NDef("new_bar", "NotImportant", {}, {}));

  TF_EXPECT_OK(graph.UpdateFanouts("bar", new_bar->name()));

  // Fanins and fanouts must be updated.
  CheckNode(graph, "bar", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "other", "NotImportant", "", {}, {},
            {"foo_1:1", "foo_2", "foo_3"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"new_bar", "other", "new_bar:1"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {}, {"other:1", "new_bar:2"},
            {});
  CheckNode(graph, "foo_3", "NotImportant", "", {}, {"other:2", "^new_bar"},
            {});
  CheckNode(graph, "new_bar", "NotImportant", "", {}, {},
            {"foo_1:0", "foo_1:2", "foo_2:1", "^foo_3"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddAndUpdateFanoutsKeepControls) {
  GraphDef graph_def = test::function::GDef(
      {NDef("bar_1", "Switch", {}, {}), NDef("bar_2", "Identity", {"bar_1:1"}),
       NDef("other", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"bar_2", "other", "bar_2:1", "^bar_2"}),
       NDef("foo_2", "NotImportant", {"other:1", "bar_2:2", "^bar_2"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  NodeDef* new_bar = graph.AddNode(NDef("new_bar", "Identity", {"bar_1:2"}));

  TF_EXPECT_OK(graph.UpdateFanouts("bar_2", new_bar->name()));

  // Fanins and fanouts must be updated.
  CheckNode(graph, "bar_1", "Switch", "", {}, {}, {"bar_2", "new_bar"});
  CheckNode(graph, "bar_2", "Identity", "", {}, {"bar_1:1"}, {});
  CheckNode(graph, "other", "NotImportant", "", {}, {}, {"foo_1:1", "foo_2"});
  CheckNode(graph, "foo_1", "NotImportant", "", {},
            {"new_bar", "other", "new_bar:1", "^new_bar"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {},
            {"other:1", "new_bar:2", "^new_bar"}, {});
  CheckNode(graph, "new_bar", "Identity", "", {}, {"bar_1:2"},
            {"foo_1", "foo_1:2", "^foo_1", "foo_2:1", "^foo_2"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddAndUpdateFanoutsWithoutSelfLoops) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def =
      test::function::GDef({NDef("bar", "NotImportant", {}, {}),
                            NDef("foo_1", "NotImportant", {"bar", "^bar"}),
                            NDef("foo_2", "NotImportant", {"^bar"})},
                           /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  // `new_bar` reads the output of an original `bar` node.
  NodeDef* new_bar = graph.AddNode(NDef("new_bar", "NewBar", {"bar"}, {}));

  TF_EXPECT_OK(graph.UpdateFanouts("bar", new_bar->name()));

  // Fanins and fanouts must be updated.
  CheckNode(graph, "bar", "NotImportant", "", {}, {}, {"new_bar"});
  CheckNode(graph, "foo_1", "NotImportant", "", {}, {"new_bar"}, {});
  CheckNode(graph, "foo_2", "NotImportant", "", {}, {"^new_bar"}, {});
  CheckNode(graph, "new_bar", "NewBar", "", {}, {"bar"}, {"foo_1", "^foo_2"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateFanoutsToSwitchWithControlFromSwitch) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "Switch", {}, {}),
       NDef("c", "NotImportant", {}, {}), NDef("d", "NotImportant", {}, {}),
       NDef("e", "NotImportant", {"c", "b", "^a", "^d"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  Status s = graph.UpdateFanouts("a", "b");
  EXPECT_FALSE(s.ok());
  string expected_msg =
      "MutableGraphView::UpdateFanouts(from_node_name='a', to_node_name='b') "
      "error: can't update fanouts to node 'b' as it will become a Switch "
      "control dependency.";
  EXPECT_EQ(s.error_message(), expected_msg);
  s = graph.UpdateFanouts("d", "b");
  EXPECT_FALSE(s.ok());
  expected_msg =
      "MutableGraphView::UpdateFanouts(from_node_name='d', to_node_name='b') "
      "error: can't update fanouts to node 'b' as it will become a Switch "
      "control dependency.";
  EXPECT_EQ(s.error_message(), expected_msg);

  EXPECT_EQ(graph.graph()->node_size(), 5);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"^e"});
  CheckNode(graph, "b", "Switch", "", {}, {}, {"e:1"});
  CheckNode(graph, "c", "NotImportant", "", {}, {}, {"e:0"});
  CheckNode(graph, "d", "NotImportant", "", {}, {}, {"^e"});
  CheckNode(graph, "e", "NotImportant", "", {}, {"c", "b", "^a", "^d"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateFanoutsToSwitchWithNoControlFromSwitch) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "Switch", {}, {}),
       NDef("c", "NotImportant", {}, {}), NDef("d", "NotImportant", {}, {}),
       NDef("e", "NotImportant", {"c", "b", "^a", "^d"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.UpdateFanouts("c", "b"));

  EXPECT_EQ(graph.graph()->node_size(), 5);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"^e"});
  CheckNode(graph, "b", "Switch", "", {}, {}, {"e:0", "e:1"});
  CheckNode(graph, "c", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "d", "NotImportant", "", {}, {}, {"^e"});
  CheckNode(graph, "e", "NotImportant", "", {}, {"b", "b", "^a", "^d"}, {});

  CheckGraph(graph);
}

GraphDef SimpleMutateFaninGraph() {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "NotImportant", {}, {}), NDef("d", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"a"}),
       NDef("foo_2", "NotImportant", {"b", "^a", "^c"}),
       NDef("foo_3", "NotImportant", {"b", "a:1", "a:1"}),
       NDef("foo_4", "NotImportant", {"a", "b:2", "b:2", "^c", "^d"}),
       NDef("foo_5", "NotImportant", {}),
       NDef("foo_6", "NotImportant", {"^a", "^b"})},
      /*funcs=*/{});
  return graph_def;
}

void TestAddRegularFanin(absl::string_view node_name, bool node_exists,
                         const TensorId& fanin_to_add, bool success,
                         const string& error_msg,
                         absl::Span<const string> expected_fanins) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  NodeDef* node = graph.GetNode(node_name);
  if (node_exists) {
    EXPECT_NE(node, nullptr);
  } else {
    EXPECT_EQ(node, nullptr);
  }

  absl::flat_hash_map<string, std::vector<string>> unmodified_node_inputs =
      GetNodeInputsFromGraph(graph_def, node_name);

  Status s = graph.AddRegularFanin(node_name, fanin_to_add);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (node_exists) {
    CompareNodeFanins(graph, node, expected_fanins);
  }

  CheckUnmodifiedNodeFanins(graph_def, node_name, unmodified_node_inputs);

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddRegularFanin) {
  string error_msg;
  // Add input to node with 1 input 0 controls.
  TestAddRegularFanin("foo_1", /*node_exists=*/true, {"b", 1}, /*success=*/true,
                      error_msg, {"a", "b:1"});
  // Add input to node with multiple inputs and 0 controls.
  TestAddRegularFanin("foo_3", /*node_exists=*/true, {"b", 2}, /*success=*/true,
                      error_msg, {"b", "a:1", "a:1", "b:2"});
  // Add input to node with 1 input multiple controls.
  TestAddRegularFanin("foo_2", /*node_exists=*/true, {"a", 0}, /*success=*/true,
                      error_msg, {"b", "a", "^c"});
  // Add input to node with multiple inputs and controls.
  TestAddRegularFanin("foo_4", /*node_exists=*/true, {"a", 1}, /*success=*/true,
                      error_msg, {"a", "b:2", "b:2", "a:1", "^d", "^c"});
  // Add input to node with 0 inputs 0 controls.
  TestAddRegularFanin("foo_5", /*node_exists=*/true, {"a", 1}, /*success=*/true,
                      error_msg, {"a:1"});
  // Add input to node with 0 inputs multiple controls.
  TestAddRegularFanin("foo_6", /*node_exists=*/true, {"c", 1}, /*success=*/true,
                      error_msg, {"c:1", "^b", "^a"});

  // Add control to node with 1 input 0 controls.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_1', fanin='^b') error: "
      "fanin '^b' must be a regular tensor id.";
  TestAddRegularFanin("foo_1", /*node_exists=*/true, {"b", Graph::kControlSlot},
                      /*success=*/false, error_msg, {"a"});
  // Add control to node with multiple inputs and 0 controls.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_3', fanin='^c') error: "
      "fanin '^c' must be a regular tensor id.";
  TestAddRegularFanin("foo_3", /*node_exists=*/true, {"c", Graph::kControlSlot},
                      /*success=*/false, error_msg, {"b", "a:1", "a:1"});
  // Add control to node with 1 input multiple controls.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_2', fanin='^d') error: "
      "fanin '^d' must be a regular tensor id.";
  TestAddRegularFanin("foo_2", /*node_exists=*/true, {"d", Graph::kControlSlot},
                      /*success=*/false, error_msg, {"b", "^a", "^c"});
  // Add control to node with multiple input multiple controls.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_4', fanin='^a') error: "
      "fanin '^a' must be a regular tensor id.";
  TestAddRegularFanin("foo_4", /*node_exists=*/true, {"a", Graph::kControlSlot},
                      /*success=*/false, error_msg,
                      {"a", "b:2", "b:2", "^c", "^d"});
  // Add control to node with 0 inputs 0 controls.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_5', fanin='^a') error: "
      "fanin '^a' must be a regular tensor id.";
  TestAddRegularFanin("foo_5", /*node_exists=*/true, {"a", Graph::kControlSlot},
                      /*success=*/false, error_msg, {});
  // Add control to node with 0 inputs multiple controls.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_6', fanin='^c') error: "
      "fanin '^c' must be a regular tensor id.";
  TestAddRegularFanin("foo_6", /*node_exists=*/true, {"c", Graph::kControlSlot},
                      /*success=*/false, error_msg, {"^a", "^b"});
  // Add control to node with control that already exists.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_2', fanin='^a') error: "
      "fanin '^a' must be a regular tensor id.";
  TestAddRegularFanin("foo_2", /*node_exists=*/true, {"a", Graph::kControlSlot},
                      /*success=*/false, error_msg, {"b", "^a", "^c"});

  // Add fanin to node where node is missing.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_missing', fanin='a:0') "
      "error: node 'foo_missing' was not found.";
  TestAddRegularFanin("foo_missing", /*node_exists=*/false, {"a", 0},
                      /*success=*/false, error_msg, {});
  // Add fanin to node where fanin is missing.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_1', "
      "fanin='bar_missing:0') error: node 'bar_missing' was not found.";
  TestAddRegularFanin("foo_1", /*node_exists=*/true, {"bar_missing", 0},
                      /*success=*/false, error_msg, {"a"});
  // Add fanin to node where node and fanin are missing.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_missing', "
      "fanin='bar_missing:0') error: node 'foo_missing' was not found.";
  TestAddRegularFanin("foo_missing", /*node_exists=*/false, {"bar_missing", 0},
                      /*success=*/false, error_msg, {});
  // Add control fanin to node where node and fanin are missing.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_missing', "
      "fanin='^bar_missing') error: fanin '^bar_missing' must be a regular "
      "tensor id.";
  TestAddRegularFanin("foo_missing", /*node_exists=*/false,
                      {"bar_missing", Graph::kControlSlot},
                      /*success=*/false, error_msg, {});

  // Add self to create cycle.
  error_msg =
      "MutableGraphView::AddRegularFanin(node_name='foo_6', fanin='foo_6:2') "
      "error: can't add fanin 'foo_6:2' to self.";
  TestAddRegularFanin("foo_6", /*node_exists=*/true, {"foo_6", 2},
                      /*success=*/false, error_msg, {"^a", "^b"});
}

void TestAddRegularFaninByPort(absl::string_view node_name, bool node_exists,
                               int port, const TensorId& fanin_to_add,
                               bool success, const string& error_msg,
                               absl::Span<const string> expected_fanins) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  NodeDef* node = graph.GetNode(node_name);
  if (node_exists) {
    EXPECT_NE(node, nullptr);
  } else {
    EXPECT_EQ(node, nullptr);
  }

  absl::flat_hash_map<string, std::vector<string>> unmodified_node_inputs =
      GetNodeInputsFromGraph(graph_def, node_name);

  Status s = graph.AddRegularFaninByPort(node_name, port, fanin_to_add);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (node_exists) {
    CompareNodeFanins(graph, node, expected_fanins);
  }

  CheckUnmodifiedNodeFanins(graph_def, node_name, unmodified_node_inputs);

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddRegularFaninByPort) {
  string error_msg;
  // Add input at start to node with some inputs and no controls.
  TestAddRegularFaninByPort("foo_3", /*node_exists=*/true, /*port=*/0, {"d", 2},
                            /*success=*/true, error_msg,
                            {"d:2", "b", "a:1", "a:1"});
  // Add input at end to node with some inputs and no controls.
  TestAddRegularFaninByPort("foo_3", /*node_exists=*/true, /*port=*/3, {"d", 2},
                            /*success=*/true, error_msg,
                            {"b", "a:1", "a:1", "d:2"});
  // Add input in middle to node with some inputs and no controls.
  TestAddRegularFaninByPort("foo_3", /*node_exists=*/true, /*port=*/2, {"d", 2},
                            /*success=*/true, error_msg,
                            {"b", "a:1", "d:2", "a:1"});
  // Add input at start to node with some inputs and some controls.
  TestAddRegularFaninByPort("foo_2", /*node_exists=*/true, /*port=*/0, {"d", 2},
                            /*success=*/true, error_msg,
                            {"d:2", "b", "^c", "^a"});
  // Add input at end to node with some inputs and some controls.
  TestAddRegularFaninByPort("foo_2", /*node_exists=*/true, /*port=*/1, {"d", 2},
                            /*success=*/true, error_msg,
                            {"b", "d:2", "^c", "^a"});
  // Add input in middle to node with some inputs and some controls, and dedup
  // controls.
  TestAddRegularFaninByPort("foo_4", /*node_exists=*/true, /*port=*/2, {"d", 2},
                            /*success=*/true, error_msg,
                            {"a", "b:2", "d:2", "b:2", "^c"});
  // Add input to node with no inputs and no controls.
  TestAddRegularFaninByPort("foo_5", /*node_exists=*/true, /*port=*/0, {"d", 2},
                            /*success=*/true, error_msg, {"d:2"});
  // Add input to node with no inputs and some controls.
  TestAddRegularFaninByPort("foo_6", /*node_exists=*/true, /*port=*/0, {"d", 2},
                            /*success=*/true, error_msg, {"d:2", "^b", "^a"});
  // Add fanin should dedup control.
  TestAddRegularFaninByPort("foo_6", /*node_exists=*/true, /*port=*/0, {"b", 2},
                            /*success=*/true, error_msg, {"b:2", "^a"});

  // Add controlling fanin.
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_4', port=2, "
      "fanin='^d') error: fanin '^d' must be a regular tensor id.";
  TestAddRegularFaninByPort(
      "foo_4", /*node_exists=*/true, /*port=*/2, {"d", Graph::kControlSlot},
      /*success=*/false, error_msg, {"a", "b:2", "b:2", "^c", "^d"});

  // Add fanin at out of bounds port.
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_5', port=-1, "
      "fanin='d:2') error: port must be in range [0, 0].";
  TestAddRegularFaninByPort("foo_5", /*node_exists=*/true, /*port=*/-1,
                            {"d", 2},
                            /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_5', port=1, "
      "fanin='d:2') error: port must be in range [0, 0].";
  TestAddRegularFaninByPort("foo_5", /*node_exists=*/true, /*port=*/1, {"d", 2},
                            /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_6', port=-1, "
      "fanin='d:2') error: port must be in range [0, 0].";
  TestAddRegularFaninByPort("foo_6", /*node_exists=*/true, /*port=*/-1,
                            {"d", 2},
                            /*success=*/false, error_msg, {"^a", "^b"});
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_6', port=1, "
      "fanin='d:2') error: port must be in range [0, 0].";
  TestAddRegularFaninByPort("foo_6", /*node_exists=*/true, /*port=*/1, {"d", 2},
                            /*success=*/false, error_msg, {"^a", "^b"});
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_4', port=-1, "
      "fanin='d:2') error: port must be in range [0, 3].";
  TestAddRegularFaninByPort(
      "foo_4", /*node_exists=*/true, /*port=*/-1, {"d", 2},
      /*success=*/false, error_msg, {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_4', port=4, "
      "fanin='d:2') error: port must be in range [0, 3].";
  TestAddRegularFaninByPort("foo_4", /*node_exists=*/true, /*port=*/4, {"d", 2},
                            /*success=*/false, error_msg,
                            {"a", "b:2", "b:2", "^c", "^d"});

  // Add fanin to node where node is missing.
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_missing', "
      "port=0, fanin='a:0') error: node 'foo_missing' was not found.";
  TestAddRegularFaninByPort("foo_missing", /*node_exists=*/false, /*port=*/0,
                            {"a", 0},
                            /*success=*/false, error_msg, {});
  // Add fanin to node where fanin is missing.
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_1', port=0, "
      "fanin='bar_missing:0') error: node 'bar_missing' was not found.";
  TestAddRegularFaninByPort("foo_1", /*node_exists=*/true, /*port=*/0,
                            {"bar_missing", 0},
                            /*success=*/false, error_msg, {"a"});
  // Add fanin to node where node and fanin are missing.
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_missing', "
      "port=0, fanin='bar_missing:0') error: node 'foo_missing' was not found.";
  TestAddRegularFaninByPort("foo_missing", /*node_exists=*/false, /*port=*/0,
                            {"bar_missing", 0},
                            /*success=*/false, error_msg, {});

  // Add self to create cycle.
  error_msg =
      "MutableGraphView::AddRegularFaninByPort(node_name='foo_6', port=0, "
      "fanin='foo_6:2') error: can't add fanin 'foo_6:2' to self.";
  TestAddRegularFaninByPort("foo_6", /*node_exists=*/true, /*port=*/0,
                            {"foo_6", 2},
                            /*success=*/false, error_msg, {"^a", "^b"});
}

void CheckFanoutRemoved(const MutableGraphView& graph, const TensorId& fanin,
                        absl::string_view node_name) {
  MutableGraphView::OutputPort output_port =
      graph.GetOutputPort(fanin.node(), fanin.index());
  auto fanouts = graph.GetFanout(output_port);
  for (auto fanout : fanouts) {
    EXPECT_NE(fanout.node->name(), fanin.node());
  }
}

void TestRemoveRegularFanin(absl::string_view node_name, bool node_exists,
                            const TensorId& fanin_to_remove, bool success,
                            const string& error_msg,
                            absl::Span<const string> expected_fanins) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  NodeDef* node = graph.GetNode(node_name);
  if (node_exists) {
    EXPECT_NE(nullptr, node);
  } else {
    EXPECT_EQ(nullptr, node);
  }

  absl::flat_hash_map<string, std::vector<string>> unmodified_node_inputs =
      GetNodeInputsFromGraph(graph_def, node_name);

  Status s = graph.RemoveRegularFanin(node_name, fanin_to_remove);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (node_exists) {
    CompareNodeFanins(graph, node, expected_fanins);
    if (success) {
      CheckFanoutRemoved(graph, fanin_to_remove, node_name);
    }
  }

  CheckUnmodifiedNodeFanins(graph_def, node_name, unmodified_node_inputs);

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, RemoveRegularFanin) {
  string error_msg;
  // Remove input from node with 1 input 0 controls.
  TestRemoveRegularFanin("foo_1", /*node_exists=*/true, {"a", 0},
                         /*success=*/true, error_msg, {});
  // Remove input from node with multiple inputs and 0 controls.
  TestRemoveRegularFanin("foo_3", /*node_exists=*/true, {"a", 1},
                         /*success=*/true, error_msg, {"b"});
  // Remove input from node with 1 input multiple controls.
  TestRemoveRegularFanin("foo_2", /*node_exists=*/true, {"b", 0},
                         /*success=*/true, error_msg, {"^a", "^c"});
  // Remove input from node with multiple inputs and controls.
  TestRemoveRegularFanin("foo_4", /*node_exists=*/true, {"b", 2},
                         /*success=*/true, error_msg, {"a", "^c", "^d"});
  // Remove input from node with multiple inputs and controls, and results in
  // shifting of ports.
  TestRemoveRegularFanin("foo_4", /*node_exists=*/true, {"a", 0},
                         /*success=*/true, error_msg,
                         {"b:2", "b:2", "^c", "^d"});

  // Remove control from node with 1 input multiple controls.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_2', fanin='^a') "
      "error: fanin '^a' must be a regular tensor id.";
  TestRemoveRegularFanin("foo_2", /*node_exists=*/true,
                         {"a", Graph::kControlSlot},
                         /*success=*/false, error_msg, {"b", "^a", "^c"});
  // Remove control from node with multiple input multiple controls.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_4', fanin='^d') "
      "error: fanin '^d' must be a regular tensor id.";
  TestRemoveRegularFanin(
      "foo_4", /*node_exists=*/true, {"d", Graph::kControlSlot},
      /*success=*/false, error_msg, {"a", "b:2", "b:2", "^c", "^d"});
  // Remove control from node with 0 inputs multiple controls.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_6', fanin='^a') "
      "error: fanin '^a' must be a regular tensor id.";
  TestRemoveRegularFanin("foo_6", /*node_exists=*/true,
                         {"a", Graph::kControlSlot},
                         /*success=*/false, error_msg, {"^a", "^b"});

  // Remove input from node with 0 inputs 0 controls.
  error_msg = "";
  TestRemoveRegularFanin("foo_5", /*node_exists=*/true, {"a", 1},
                         /*success=*/true, error_msg, {});
  // Remove input from node with 0 inputs multiple controls.
  TestRemoveRegularFanin("foo_6", /*node_exists=*/true, {"a", 1},
                         /*success=*/true, error_msg, {"^a", "^b"});

  // Remove control from node with 1 input 0 controls.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_1', fanin='^b') "
      "error: fanin '^b' must be a regular tensor id.";
  TestRemoveRegularFanin("foo_1", /*node_exists=*/true,
                         {"b", Graph::kControlSlot},
                         /*success=*/false, error_msg, {"a"});
  // Remove control from node with multiple inputs and 0 controls.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_3', fanin='^c') "
      "error: fanin '^c' must be a regular tensor id.";
  TestRemoveRegularFanin("foo_3", /*node_exists=*/true,
                         {"c", Graph::kControlSlot},
                         /*success=*/false, error_msg, {"b", "a:1", "a:1"});
  // Remove control from node with 0 inputs 0 controls.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_5', fanin='^a') "
      "error: fanin '^a' must be a regular tensor id.";
  TestRemoveRegularFanin("foo_5", /*node_exists=*/true,
                         {"a", Graph::kControlSlot},
                         /*success=*/false, error_msg, {});

  // Remove fanin from node where node is missing.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_missing', "
      "fanin='a:0') error: node 'foo_missing' was not found.";
  TestRemoveRegularFanin("foo_missing", /*node_exists=*/false, {"a", 0},
                         /*success=*/false, error_msg, {});
  // Remove fanin from node where fanin is missing.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_1', "
      "fanin='bar_missing:0') error: node 'bar_missing' was not found.";
  TestRemoveRegularFanin("foo_1", /*node_exists=*/true, {"bar_missing", 0},
                         /*success=*/false, error_msg, {"a"});
  // Remove fanin from node where node and fanin are missing.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_missing', "
      "fanin='bar_missing:0') error: node 'foo_missing' was not found.";
  TestRemoveRegularFanin("foo_missing", /*node_exists=*/false,
                         {"bar_missing", 0}, /*success=*/false, error_msg, {});
  // Remove control from node where node and fanin are missing.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_missing', "
      "fanin='^bar_missing') error: fanin '^bar_missing' must be a regular "
      "tensor id.";
  TestRemoveRegularFanin("foo_missing", /*node_exists=*/false,
                         {"bar_missing", Graph::kControlSlot},
                         /*success=*/false, error_msg, {});

  // Remove self.
  error_msg =
      "MutableGraphView::RemoveRegularFanin(node_name='foo_6', "
      "fanin='foo_6:2') error: can't remove fanin 'foo_6:2' from self.";
  TestRemoveRegularFanin("foo_6", /*node_exists=*/true, {"foo_6", 2},
                         /*success=*/false, error_msg, {"^a", "^b"});
}

void TestRemoveRegularFaninByPort(absl::string_view node_name, bool node_exists,
                                  int port, bool success,
                                  const string& error_msg,
                                  absl::Span<const string> expected_fanins) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  NodeDef* node = graph.GetNode(node_name);
  if (node_exists) {
    EXPECT_NE(nullptr, node);
  } else {
    EXPECT_EQ(nullptr, node);
  }

  absl::flat_hash_map<string, std::vector<string>> unmodified_node_inputs =
      GetNodeInputsFromGraph(graph_def, node_name);

  Status s = graph.RemoveRegularFaninByPort(node_name, port);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (node_exists) {
    CompareNodeFanins(graph, node, expected_fanins);
  }

  CheckUnmodifiedNodeFanins(graph_def, node_name, unmodified_node_inputs);

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, RemoveRegularFaninByPort) {
  string error_msg;
  // Remove input at start of node with some inputs and no controls.
  TestRemoveRegularFaninByPort("foo_3", /*node_exists=*/true, /*port=*/0,
                               /*success=*/true, error_msg, {"a:1", "a:1"});
  // Remove input at end of node with some inputs and no controls.
  TestRemoveRegularFaninByPort("foo_3", /*node_exists=*/true, /*port=*/2,
                               /*success=*/true, error_msg, {"b", "a:1"});
  // Remove input in middle of node with some inputs and no controls.
  TestRemoveRegularFaninByPort("foo_3", /*node_exists=*/true, /*port=*/1,
                               /*success=*/true, error_msg, {"b", "a:1"});
  // Remove input at start of node with some inputs and some controls.
  TestRemoveRegularFaninByPort("foo_4", /*node_exists=*/true, /*port=*/0,
                               /*success=*/true, error_msg,
                               {"b:2", "b:2", "^d", "^c"});
  // Remove input at end of node with some inputs and some controls.
  TestRemoveRegularFaninByPort("foo_4", /*node_exists=*/true, /*port=*/2,
                               /*success=*/true, error_msg,
                               {"a", "b:2", "^d", "^c"});
  // Remove input in middle of node with some inputs and some controls.
  TestRemoveRegularFaninByPort("foo_4", /*node_exists=*/true, /*port=*/1,
                               /*success=*/true, error_msg,
                               {"a", "b:2", "^d", "^c"});

  // Remove input from node with no inputs and no controls.
  error_msg =
      "MutableGraphView::RemoveRegularFaninByPort(node_name='foo_5', port=0) "
      "error: no available ports as node has no regular fanins.";
  TestRemoveRegularFaninByPort("foo_5", /*node_exists=*/true, /*port=*/0,
                               /*success=*/false, error_msg, {});
  // Remove input from node with no inputs and some controls.
  error_msg =
      "MutableGraphView::RemoveRegularFaninByPort(node_name='foo_6', port=1) "
      "error: no available ports as node has no regular fanins.";
  TestRemoveRegularFaninByPort("foo_6", /*node_exists=*/true, /*port=*/1,
                               /*success=*/false, error_msg, {"^a", "^b"});

  // Remove fanin at out of bounds port.
  error_msg =
      "MutableGraphView::RemoveRegularFaninByPort(node_name='foo_3', port=-1) "
      "error: port must be in range [0, 2].";
  TestRemoveRegularFaninByPort("foo_3", /*node_exists=*/true, /*port=*/-1,
                               /*success=*/false, error_msg,
                               {"b", "a:1", "a:1"});
  error_msg =
      "MutableGraphView::RemoveRegularFaninByPort(node_name='foo_3', port=3) "
      "error: port must be in range [0, 2].";
  TestRemoveRegularFaninByPort("foo_3", /*node_exists=*/true, /*port=*/3,
                               /*success=*/false, error_msg,
                               {"b", "a:1", "a:1"});
  error_msg =
      "MutableGraphView::RemoveRegularFaninByPort(node_name='foo_4', port=-1) "
      "error: port must be in range [0, 2].";
  TestRemoveRegularFaninByPort("foo_4", /*node_exists=*/true, /*port=*/-1,
                               /*success=*/false, error_msg,
                               {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::RemoveRegularFaninByPort(node_name='foo_4', port=3) "
      "error: port must be in range [0, 2].";
  TestRemoveRegularFaninByPort("foo_4", /*node_exists=*/true, /*port=*/3,
                               /*success=*/false, error_msg,
                               {"a", "b:2", "b:2", "^c", "^d"});

  // Remove fanin from node where node is missing.
  error_msg =
      "MutableGraphView::RemoveRegularFaninByPort(node_name='foo_missing', "
      "port=0) error: node 'foo_missing' was not found.";
  TestRemoveRegularFaninByPort("foo_missing", /*node_exists=*/false, /*port=*/0,
                               /*success=*/false, error_msg, {});
}

void TestRemoveAllFanins(absl::string_view node_name, bool node_exists,
                         bool keep_controlling_nodes, bool success,
                         const string& error_msg,
                         absl::Span<const string> expected_fanins) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  NodeDef* node = graph.GetNode(node_name);
  absl::flat_hash_set<string> fanin_strings;
  if (node_exists) {
    EXPECT_NE(node, nullptr);
    fanin_strings.insert(node->input().begin(), node->input().end());
  } else {
    EXPECT_EQ(node, nullptr);
  }

  absl::flat_hash_map<string, std::vector<string>> unmodified_node_inputs =
      GetNodeInputsFromGraph(graph_def, node_name);

  Status s = graph.RemoveAllFanins(node_name, keep_controlling_nodes);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (node_exists) {
    CompareNodeFanins(graph, node, expected_fanins);
    if (success) {
      TensorId tensor_id;
      auto retained_inputs = absl::flat_hash_set<string>(node->input().begin(),
                                                         node->input().end());
      for (const string& fanin : fanin_strings) {
        if (!retained_inputs.contains(fanin)) {
          tensor_id = ParseTensorName(fanin);
          CheckFanoutRemoved(graph, tensor_id, node_name);
        }
      }
    }
  }

  CheckUnmodifiedNodeFanins(graph_def, node_name, unmodified_node_inputs);

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, RemoveAllFanins) {
  string error_msg;
  // Remove all fanins from node with no control dependencies.
  TestRemoveAllFanins("foo_3", /*node_exists=*/true,
                      /*keep_controlling_nodes=*/false,
                      /*success=*/true, error_msg, {});
  // Remove all fanins from node with control dependencies.
  TestRemoveAllFanins("foo_4", /*node_exists=*/true,
                      /*keep_controlling_nodes=*/false,
                      /*success=*/true, error_msg, {});

  // Remove all fanins from node with no control dependencies and preserve
  // control dependencies.
  TestRemoveAllFanins("foo_3", /*node_exists=*/true,
                      /*keep_controlling_nodes=*/true,
                      /*success=*/true, error_msg, {});
  // Remove all fanins from node with control dependencies and preserve control
  // dependencies.
  TestRemoveAllFanins("foo_4", /*node_exists=*/true,
                      /*keep_controlling_nodes=*/true,
                      /*success=*/true, error_msg, {"^c", "^d"});

  // Remove all fanins from node with no fanins.
  TestRemoveAllFanins("foo_5", /*node_exists=*/true,
                      /*keep_controlling_nodes=*/false,
                      /*success=*/true, error_msg, {});
  TestRemoveAllFanins("foo_5", /*node_exists=*/true,
                      /*keep_controlling_nodes=*/true,
                      /*success=*/true, error_msg, {});

  // Remove all fanins from node with only control dependencies.
  TestRemoveAllFanins("foo_6", /*node_exists=*/true,
                      /*keep_controlling_nodes=*/false,
                      /*success=*/true, error_msg, {});
  TestRemoveAllFanins("foo_6", /*node_exists=*/true,
                      /*keep_controlling_nodes=*/true,
                      /*success=*/true, error_msg, {"^a", "^b"});

  // Remove all fanins from node where node is missing.
  error_msg =
      "MutableGraphView::RemoveAllFanins(node_name='foo_missing', "
      "keep_controlling_fanins=false) error: node 'foo_missing' was not found.";
  TestRemoveAllFanins("foo_missing", /*node_exists=*/false,
                      /*keep_controlling_nodes=*/false,
                      /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::RemoveAllFanins(node_name='foo_missing', "
      "keep_controlling_fanins=true) error: node 'foo_missing' was not found.";
  TestRemoveAllFanins("foo_missing", /*node_exists=*/false,
                      /*keep_controlling_nodes=*/true,
                      /*success=*/false, error_msg, {});
}

void TestUpdateFanin(absl::string_view node_name, bool node_exists,
                     const TensorId& from_fanin, const TensorId& to_fanin,
                     bool success, const string& error_msg,
                     absl::Span<const string> expected_fanins) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  NodeDef* node = graph.GetNode(node_name);
  if (node_exists) {
    EXPECT_NE(node, nullptr);
  } else {
    EXPECT_EQ(node, nullptr);
  }

  absl::flat_hash_map<string, std::vector<string>> unmodified_node_inputs =
      GetNodeInputsFromGraph(graph_def, node_name);

  Status s = graph.UpdateFanin(node_name, from_fanin, to_fanin);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (node_exists) {
    CompareNodeFanins(graph, node, expected_fanins);
    if (success) {
      CheckFanoutRemoved(graph, from_fanin, node_name);
    }
  }

  CheckUnmodifiedNodeFanins(graph_def, node_name, unmodified_node_inputs);

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateFanin) {
  string error_msg;
  // Update fanin from non control to non control.
  TestUpdateFanin("foo_4", /*node_exists=*/true, {"b", 2}, {"b", 3},
                  /*success=*/true, error_msg, {"a", "b:3", "b:3", "^c", "^d"});
  // Update fanin from non control to control.
  TestUpdateFanin("foo_4", /*node_exists=*/true, {"b", 2},
                  {"b", Graph::kControlSlot},
                  /*success=*/true, error_msg, {"a", "^c", "^d", "^b"});
  // Update fanin from control to non control.
  TestUpdateFanin(
      "foo_4", /*node_exists=*/true, {"d", Graph::kControlSlot}, {"d", 1},
      /*success=*/true, error_msg, {"a", "b:2", "b:2", "d:1", "^c"});
  // Update fanin from control to control.
  TestUpdateFanin("foo_4", /*node_exists=*/true, {"c", Graph::kControlSlot},
                  {"b", Graph::kControlSlot}, /*success=*/true, error_msg,
                  {"a", "b:2", "b:2", "^d"});
  // Update fanin from control to existing control.
  TestUpdateFanin("foo_4", /*node_exists=*/true, {"c", Graph::kControlSlot},
                  {"d", Graph::kControlSlot}, /*success=*/true, error_msg,
                  {"a", "b:2", "b:2", "^d"});

  // Update fanin of node where from and to fanins are the same.
  TestUpdateFanin("foo_1", /*node_exists=*/true, {"a", -1}, {"a", -1},
                  /*success=*/true, error_msg, {"a"});
  TestUpdateFanin("foo_1", /*node_exists=*/true, {"a", 0}, {"a", 0},
                  /*success=*/true, error_msg, {"a"});
  TestUpdateFanin("foo_1", /*node_exists=*/true, {"a", 1}, {"a", 1},
                  /*success=*/true, error_msg, {"a"});

  // Update fanin of node where node is missing.
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_missing', "
      "from_fanin='a:0', to_fanin='a:1') error: node 'foo_missing' was not "
      "found.";
  TestUpdateFanin("foo_missing", /*node_exists=*/false, {"a", 0}, {"a", 1},
                  /*success=*/false, error_msg, {});
  // Update fanin of node where from fanin is missing.
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_1', "
      "from_fanin='from_bar_missing:0', to_fanin='a:1') error: node "
      "'from_bar_missing' was not found.";
  TestUpdateFanin("foo_1", /*node_exists=*/true, {"from_bar_missing", 0},
                  {"a", 1},
                  /*success=*/false, error_msg, {"a"});
  // Update fanin of node where to fanin is missing.
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_1', from_fanin='a:0', "
      "to_fanin='to_bar_missing:1') error: node 'to_bar_missing' was not "
      "found.";
  TestUpdateFanin("foo_1", /*node_exists=*/true, {"a", 0},
                  {"to_bar_missing", 1}, /*success=*/false, error_msg, {"a"});
  // Update fanin of node where from/to fanins and node are missing.
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_missing', "
      "from_fanin='from_bar_missing:0', to_fanin='to_bar_missing:1') error: "
      "node 'foo_missing' was not found.";
  TestUpdateFanin("foo_missing", /*node_exists=*/false, {"from_bar_missing", 0},
                  {"to_bar_missing", 1},
                  /*success=*/false, error_msg, {});
  // Update fanin of node where from fanin is invalid.
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_1', from_fanin='a:-2', "
      "to_fanin='a:0') error: fanin 'a:-2' must be a valid tensor id.";
  TestUpdateFanin("foo_1", /*node_exists=*/true, {"a", -2}, {"a", 0},
                  /*success=*/false, error_msg, {"a"});
  // Update fanin of node where to fanin is invalid.
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_1', from_fanin='a:0', "
      "to_fanin='a:-2') error: fanin 'a:-2' must be a valid tensor id.";
  TestUpdateFanin("foo_1", /*node_exists=*/true, {"a", 0}, {"a", -2},
                  /*success=*/false, error_msg, {"a"});
  // Update fanin of node where from/to fanins are invalid and missing and node
  // is missing.
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_missing', "
      "from_fanin='from_bar_missing:-2', to_fanin='to_bar_missing:-3') error: "
      "fanin 'from_bar_missing:-2' must be a valid tensor id.";
  TestUpdateFanin("foo_missing", /*node_exists=*/false,
                  {"from_bar_missing", -2}, {"to_bar_missing", -3},
                  /*success=*/false, error_msg, {});

  // Update to self to create cycle.
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_4', from_fanin='b:2', "
      "to_fanin='foo_4:3') error: can't update fanin to or from self.";
  TestUpdateFanin("foo_4", /*node_exists=*/true, {"b", 2}, {"foo_4", 3},
                  /*success=*/false, error_msg,
                  {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_4', from_fanin='b:2', "
      "to_fanin='^foo_4') error: can't update fanin to or from self.";
  TestUpdateFanin(
      "foo_4", /*node_exists=*/true, {"b", 2}, {"foo_4", Graph::kControlSlot},
      /*success=*/false, error_msg, {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_4', from_fanin='^c', "
      "to_fanin='foo_4:4') error: can't update fanin to or from self.";
  TestUpdateFanin(
      "foo_4", /*node_exists=*/true, {"c", Graph::kControlSlot}, {"foo_4", 4},
      /*success=*/false, error_msg, {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::UpdateFanin(node_name='foo_4', from_fanin='^c', "
      "to_fanin='^foo_4') error: can't update fanin to or from self.";
  TestUpdateFanin("foo_4", /*node_exists=*/true, {"c", Graph::kControlSlot},
                  {"foo_4", Graph::kControlSlot}, /*success=*/false, error_msg,
                  {"a", "b:2", "b:2", "^c", "^d"});
}

void TestUpdateFaninFromFaninToNodeAsSwitchControl(const TensorId& fanin) {
  string tensor_id_str = TensorIdToString(fanin);
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "Switch", {}, {}),
       NDef("c", "NotImportant", {tensor_id_str})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  Status s = graph.UpdateFanin("c", fanin, {"b", Graph::kControlSlot});
  EXPECT_FALSE(s.ok());
  string expected_msg = absl::Substitute(
      "MutableGraphView::UpdateFanin(node_name='c', from_fanin='$0', "
      "to_fanin='^b') error: can't update to fanin '^b' as it will become a "
      "Switch control dependency.",
      fanin.ToString());
  EXPECT_EQ(s.error_message(), expected_msg);

  EXPECT_EQ(graph.graph()->node_size(), 3);

  string fanout = IsControlInput(fanin) ? AsControlDependency("c") : "c";
  CheckNode(graph, "a", "NotImportant", "", {}, {}, {fanout});
  CheckNode(graph, "b", "Switch", "", {}, {}, {});
  CheckNode(graph, "c", "NotImportant", "", {}, {tensor_id_str}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateFaninToNodeAsSwitchControl) {
  TestUpdateFaninFromFaninToNodeAsSwitchControl({"a", 0});
  TestUpdateFaninFromFaninToNodeAsSwitchControl({"a", 1});
  TestUpdateFaninFromFaninToNodeAsSwitchControl({"a", Graph::kControlSlot});
}

void TestUpdateRegularFaninByPort(absl::string_view node_name, bool node_exists,
                                  int port, const TensorId& fanin, bool success,
                                  const string& error_msg,
                                  absl::Span<const string> expected_fanins) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  NodeDef* node = graph.GetNode(node_name);
  if (node_exists) {
    EXPECT_NE(node, nullptr);
  } else {
    EXPECT_EQ(node, nullptr);
  }

  absl::flat_hash_map<string, std::vector<string>> unmodified_node_inputs =
      GetNodeInputsFromGraph(graph_def, node_name);

  Status s = graph.UpdateRegularFaninByPort(node_name, port, fanin);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (node_exists) {
    CompareNodeFanins(graph, node, expected_fanins);
  }

  CheckUnmodifiedNodeFanins(graph_def, node_name, unmodified_node_inputs);

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateRegularFaninByPort) {
  string error_msg;
  // Update input at start to node with some inputs and no controls.
  TestUpdateRegularFaninByPort(
      "foo_3", /*node_exists=*/true, /*port=*/0, {"d", 2},
      /*success=*/true, error_msg, {"d:2", "a:1", "a:1"});
  // Update input at end to node with some inputs and no controls.
  TestUpdateRegularFaninByPort(
      "foo_3", /*node_exists=*/true, /*port=*/2, {"d", 2},
      /*success=*/true, error_msg, {"b", "a:1", "d:2"});
  // Update input in middle to node with some inputs and no controls.
  TestUpdateRegularFaninByPort(
      "foo_3", /*node_exists=*/true, /*port=*/1, {"d", 2},
      /*success=*/true, error_msg, {"b", "d:2", "a:1"});
  // Update input at start to node with some inputs and some controls, and dedup
  // controls.
  TestUpdateRegularFaninByPort(
      "foo_4", /*node_exists=*/true, /*port=*/0, {"d", 2},
      /*success=*/true, error_msg, {"d:2", "b:2", "b:2", "^c"});
  // Update input at end to node with some inputs and some controls, and dedup
  // controls.
  TestUpdateRegularFaninByPort(
      "foo_4", /*node_exists=*/true, /*port=*/2, {"d", 2},
      /*success=*/true, error_msg, {"a", "b:2", "d:2", "^c"});
  // Update input in middle to node with some inputs and some controls and
  // dedup controls.
  TestUpdateRegularFaninByPort(
      "foo_4", /*node_exists=*/true, /*port=*/1, {"d", 2},
      /*success=*/true, error_msg, {"a", "d:2", "b:2", "^c"});

  // Update input to controlling fanin.
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_4', port=1, "
      "fanin='^d') error: fanin '^d' must be a regular tensor id.";
  TestUpdateRegularFaninByPort(
      "foo_4", /*node_exists=*/true, /*port=*/1, {"d", Graph::kControlSlot},
      /*success=*/false, error_msg, {"a", "b:2", "b:2", "^c", "^d"});

  // Update fanin at out of bounds port.
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_5', port=-1, "
      "fanin='d:2') error: no available ports as node has no regular fanins.";
  TestUpdateRegularFaninByPort("foo_5", /*node_exists=*/true, /*port=*/-1,
                               {"d", 2},
                               /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_5', port=0, "
      "fanin='d:2') error: no available ports as node has no regular fanins.";
  TestUpdateRegularFaninByPort("foo_5", /*node_exists=*/true, /*port=*/0,
                               {"d", 2},
                               /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_5', port=1, "
      "fanin='d:2') error: no available ports as node has no regular fanins.";
  TestUpdateRegularFaninByPort("foo_5", /*node_exists=*/true, /*port=*/1,
                               {"d", 2},
                               /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_6', port=-1, "
      "fanin='d:2') error: no available ports as node has no regular fanins.";
  TestUpdateRegularFaninByPort("foo_6", /*node_exists=*/true, /*port=*/-1,
                               {"d", 2},
                               /*success=*/false, error_msg, {"^a", "^b"});
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_6', port=0, "
      "fanin='d:2') error: no available ports as node has no regular fanins.";
  TestUpdateRegularFaninByPort("foo_6", /*node_exists=*/true, /*port=*/0,
                               {"d", 2},
                               /*success=*/false, error_msg, {"^a", "^b"});
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_6', port=1, "
      "fanin='d:2') error: no available ports as node has no regular fanins.";
  TestUpdateRegularFaninByPort("foo_6", /*node_exists=*/true, /*port=*/1,
                               {"d", 2},
                               /*success=*/false, error_msg, {"^a", "^b"});
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_3', port=-1, "
      "fanin='d:2') error: port must be in range [0, 2].";
  TestUpdateRegularFaninByPort(
      "foo_3", /*node_exists=*/true, /*port=*/-1, {"d", 2},
      /*success=*/false, error_msg, {"b", "a:1", "a:1"});
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_3', port=3, "
      "fanin='d:2') error: port must be in range [0, 2].";
  TestUpdateRegularFaninByPort(
      "foo_3", /*node_exists=*/true, /*port=*/3, {"d", 2},
      /*success=*/false, error_msg, {"b", "a:1", "a:1"});
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_4', port=-1, "
      "fanin='d:2') error: port must be in range [0, 2].";
  TestUpdateRegularFaninByPort(
      "foo_4", /*node_exists=*/true, /*port=*/-1, {"d", 2},
      /*success=*/false, error_msg, {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_4', port=3, "
      "fanin='d:2') error: port must be in range [0, 2].";
  TestUpdateRegularFaninByPort(
      "foo_4", /*node_exists=*/true, /*port=*/3, {"d", 2},
      /*success=*/false, error_msg, {"a", "b:2", "b:2", "^c", "^d"});

  // Update fanin to node where node is missing.
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_missing', "
      "port=0, fanin='a:0') error: node 'foo_missing' was not found.";
  TestUpdateRegularFaninByPort("foo_missing", /*node_exists=*/false,
                               /*port=*/0, {"a", 0},
                               /*success=*/false, error_msg, {});
  // Update fanin to node where fanin is missing.
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_1', port=0, "
      "fanin='bar_missing:0') error: node 'bar_missing' was not "
      "found.";
  TestUpdateRegularFaninByPort("foo_1", /*node_exists=*/true, /*port=*/0,
                               {"bar_missing", 0},
                               /*success=*/false, error_msg, {"a"});
  // Update fanin to node where node and fanin are missing.
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_missing', "
      "port=0, fanin='bar_missing:0') error: node 'foo_missing' was not found.";
  TestUpdateRegularFaninByPort("foo_missing", /*node_exists=*/false,
                               /*port=*/0, {"bar_missing", 0},
                               /*success=*/false, error_msg, {});

  // Update self to create cycle.
  error_msg =
      "MutableGraphView::UpdateRegularFaninByPort(node_name='foo_6', port=0, "
      "fanin='foo_6:2') error: can't add fanin 'foo_6:2' to self.";
  TestUpdateRegularFaninByPort("foo_6", /*node_exists=*/true, /*port=*/0,
                               {"foo_6", 2},
                               /*success=*/false, error_msg, {"^a", "^b"});
}

void TestSwapRegularFaninsByPorts(absl::string_view node_name, bool node_exists,
                                  int from_port, int to_port, bool success,
                                  const string& error_msg,
                                  absl::Span<const string> expected_fanins) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  NodeDef* node = graph.GetNode(node_name);
  if (node_exists) {
    EXPECT_NE(node, nullptr);
  } else {
    EXPECT_EQ(node, nullptr);
  }

  absl::flat_hash_map<string, std::vector<string>> unmodified_node_inputs =
      GetNodeInputsFromGraph(graph_def, node_name);

  Status s = graph.SwapRegularFaninsByPorts(node_name, from_port, to_port);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (node_exists) {
    CompareNodeFanins(graph, node, expected_fanins);
  }

  CheckUnmodifiedNodeFanins(graph_def, node_name, unmodified_node_inputs);

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, SwapRegularFaninsByPorts) {
  string error_msg;
  // Swapping first and last regular fanins
  TestSwapRegularFaninsByPorts("foo_3", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/2, /*success=*/true, error_msg,
                               {"a:1", "a:1", "b"});
  TestSwapRegularFaninsByPorts("foo_3", /*node_exists=*/true, /*from_port=*/2,
                               /*to_port=*/0, /*success=*/true, error_msg,
                               {"a:1", "a:1", "b"});
  // Swapping first and last regular fanins, in node with controls.
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/2, /*success=*/true, error_msg,
                               {"b:2", "b:2", "a", "^c", "^d"});
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/2,
                               /*to_port=*/0, /*success=*/true, error_msg,
                               {"b:2", "b:2", "a", "^c", "^d"});
  // Swapping middle regular fanin.
  TestSwapRegularFaninsByPorts("foo_3", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/1, /*success=*/true, error_msg,
                               {"a:1", "b", "a:1"});
  TestSwapRegularFaninsByPorts("foo_3", /*node_exists=*/true, /*from_port=*/1,
                               /*to_port=*/0, /*success=*/true, error_msg,
                               {"a:1", "b", "a:1"});
  // Swapping middle regular fanin, in node with controls.
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/1, /*success=*/true, error_msg,
                               {"b:2", "a", "b:2", "^c", "^d"});
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/1,
                               /*to_port=*/0, /*success=*/true, error_msg,
                               {"b:2", "a", "b:2", "^c", "^d"});
  // Swapping same port.
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/1,
                               /*to_port=*/1, /*success=*/true, error_msg,
                               {"a", "b:2", "b:2", "^c", "^d"});
  // Swapping same fanin but different port.
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/1,
                               /*to_port=*/2, /*success=*/true, error_msg,
                               {"a", "b:2", "b:2", "^c", "^d"});

  // Swapping fanins at out of bounds ports.
  // Node with no regular fanins and no controls.
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_5', "
      "from_port=-1, to_port=0) error: no available ports as node has no "
      "regular fanins.";
  TestSwapRegularFaninsByPorts("foo_5", /*node_exists=*/true, /*from_port=*/-1,
                               /*to_port=*/0, /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_5', "
      "from_port=0, to_port=-1) error: no available ports as node has no "
      "regular fanins.";
  TestSwapRegularFaninsByPorts("foo_5", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/-1, /*success=*/false, error_msg,
                               {});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_5', "
      "from_port=0, to_port=0) error: no available ports as node has no "
      "regular fanins.";
  TestSwapRegularFaninsByPorts("foo_5", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/0, /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_5', "
      "from_port=0, to_port=1) error: no available ports as node has no "
      "regular fanins.";
  TestSwapRegularFaninsByPorts("foo_5", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/1, /*success=*/false, error_msg, {});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_5', "
      "from_port=1, to_port=0) error: no available ports as node has no "
      "regular fanins.";
  TestSwapRegularFaninsByPorts("foo_5", /*node_exists=*/true, /*from_port=*/1,
                               /*to_port=*/0, /*success=*/false, error_msg, {});
  // Node with no regular fanins and some controls.
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_6', "
      "from_port=-1, to_port=0) error: no available ports as node has no "
      "regular fanins.";
  TestSwapRegularFaninsByPorts("foo_6", /*node_exists=*/true, /*from_port=*/-1,
                               /*to_port=*/0, /*success=*/false, error_msg,
                               {"^a", "^b"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_6', "
      "from_port=0, to_port=-1) error: no available ports as node has no "
      "regular fanins.";
  TestSwapRegularFaninsByPorts("foo_6", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/-1, /*success=*/false, error_msg,
                               {"^a", "^b"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_6', "
      "from_port=0, to_port=0) error: no available ports as node has no "
      "regular fanins.";
  TestSwapRegularFaninsByPorts("foo_6", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/0, /*success=*/false, error_msg,
                               {"^a", "^b"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_6', "
      "from_port=0, to_port=1) error: no available ports as node has no "
      "regular fanins.";
  TestSwapRegularFaninsByPorts("foo_6", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/1, /*success=*/false, error_msg,
                               {"^a", "^b"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_6', "
      "from_port=1, to_port=0) error: no available ports as node has no "
      "regular fanins.";
  TestSwapRegularFaninsByPorts("foo_6", /*node_exists=*/true, /*from_port=*/1,
                               /*to_port=*/0, /*success=*/false, error_msg,
                               {"^a", "^b"});
  // Node with regular fanins and no controls.
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_3', "
      "from_port=-1, to_port=0) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_3", /*node_exists=*/true, /*from_port=*/-1,
                               /*to_port=*/0, /*success=*/false, error_msg,
                               {"b", "a:1", "a:1"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_3', "
      "from_port=0, to_port=-1) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_3", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/-1, /*success=*/false, error_msg,
                               {"b", "a:1", "a:1"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_3', "
      "from_port=0, to_port=3) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_3", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/3, /*success=*/false, error_msg,
                               {"b", "a:1", "a:1"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_3', "
      "from_port=3, to_port=0) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_3", /*node_exists=*/true, /*from_port=*/3,
                               /*to_port=*/0, /*success=*/false, error_msg,
                               {"b", "a:1", "a:1"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_3', "
      "from_port=-1, to_port=3) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_3", /*node_exists=*/true, /*from_port=*/-1,
                               /*to_port=*/3, /*success=*/false, error_msg,
                               {"b", "a:1", "a:1"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_3', "
      "from_port=3, to_port=-1) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_3", /*node_exists=*/true, /*from_port=*/3,
                               /*to_port=*/-1, /*success=*/false, error_msg,
                               {"b", "a:1", "a:1"});
  // Node with regular fanins and controls.
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_4', "
      "from_port=-1, to_port=0) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/-1,
                               /*to_port=*/0, /*success=*/false, error_msg,
                               {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_4', "
      "from_port=0, to_port=-1) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/-1, /*success=*/false, error_msg,
                               {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_4', "
      "from_port=0, to_port=3) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/0,
                               /*to_port=*/3, /*success=*/false, error_msg,
                               {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_4', "
      "from_port=3, to_port=0) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/3,
                               /*to_port=*/0, /*success=*/false, error_msg,
                               {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_4', "
      "from_port=-1, to_port=3) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/-1,
                               /*to_port=*/3, /*success=*/false, error_msg,
                               {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_4', "
      "from_port=3, to_port=-1) error: port must be in range [0, 2].";
  TestSwapRegularFaninsByPorts("foo_4", /*node_exists=*/true, /*from_port=*/3,
                               /*to_port=*/-1, /*success=*/false, error_msg,
                               {"a", "b:2", "b:2", "^c", "^d"});

  // Swapping fanin to node where node is missing.
  error_msg =
      "MutableGraphView::SwapRegularFaninsByPorts(node_name='foo_missing', "
      "from_port=0, to_port=1) error: node 'foo_missing' was not found.";
  TestSwapRegularFaninsByPorts("foo_missing", /*node_exists=*/false,
                               /*from_port=*/0, /*to_port=*/1,
                               /*success=*/false, error_msg, {});
}

TEST(MutableGraphViewTest, DedupControllingFaninsOnGraphInit) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "Switch", {}, {}), NDef("d", "Identity", {"c:1"}),
       NDef("foo_1", "IdentityN", {"a", "b:1", "^b"}),
       NDef("foo_2", "IdentityN", {"a", "^b", "^b"}),
       NDef("foo_3", "IdentityN", {"a", "b:1", "^b", "^b"}),
       NDef("foo_4", "IdentityN", {"a:2", "b:1", "^b", "^b", "^a", "^a"}),
       NDef("foo_5", "NotImportant", {"a:2", "b:1", "^b", "^b", "^a", "^a"}),
       NDef("foo_6", "Identity", {"d", "^d"}),
       NDef("foo_7", "NotImportant",
            {"a:3", "b:2", "d", "^d", "^d", "^a", "^b", "^a", "^b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_EQ(graph.graph()->node_size(), 11);

  CheckNode(graph, "a", "NotImportant", "", {}, {},
            {"foo_1", "foo_2", "foo_3", "foo_4", "foo_5", "foo_7"});
  CheckNode(graph, "b", "NotImportant", "", {}, {},
            {"foo_1:1", "^foo_2", "foo_3:1", "foo_4:1", "foo_5:1", "foo_7:1"});
  CheckNode(graph, "c", "Switch", "", {}, {}, {"d"});
  CheckNode(graph, "d", "Identity", "", {}, {"c:1"},
            {"foo_6", "^foo_6", "foo_7:2", "^foo_7"});
  CheckNode(graph, "foo_1", "IdentityN", "", {}, {"a", "b:1"}, {});
  CheckNode(graph, "foo_2", "IdentityN", "", {}, {"a", "^b"}, {});
  CheckNode(graph, "foo_3", "IdentityN", "", {}, {"a", "b:1"}, {});
  CheckNode(graph, "foo_4", "IdentityN", "", {}, {"a:2", "b:1"}, {});
  CheckNode(graph, "foo_5", "NotImportant", "", {}, {"a:2", "b:1"}, {});
  CheckNode(graph, "foo_6", "Identity", "", {}, {"d", "^d"}, {});
  CheckNode(graph, "foo_7", "NotImportant", "", {}, {"a:3", "b:2", "d", "^d"},
            {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, DedupControllingFaninsOnAddFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"^a"}),
       NDef("c", "NotImportant", {"a:1"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.AddRegularFanin("b", {"a", 2}));
  CheckNode(graph, "b", "NotImportant", "", {}, {"a:2"}, {});

  TF_EXPECT_OK(graph.AddControllingFanin("c", {"a", Graph::kControlSlot}));
  CheckNode(graph, "c", "NotImportant", "", {}, {"a:1"}, {});

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b:0", "c:0"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, NoDedupControllingFaninsOnAddFanin) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "Switch", {}, {}), NDef("b", "Identity", {"a:1"}),
       NDef("c", "", {}, {}), NDef("d", "", {}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.AddRegularFanin("c", {"b", 2}));
  CheckNode(graph, "c", "", "", {}, {"b:2"}, {});
  TF_EXPECT_OK(graph.AddControllingFanin("c", {"b", Graph::kControlSlot}));
  CheckNode(graph, "c", "", "", {}, {"b:2", "^b"}, {});
  TF_EXPECT_OK(graph.AddControllingFanin("c", {"b", Graph::kControlSlot}));
  CheckNode(graph, "c", "", "", {}, {"b:2", "^b"}, {});
  TF_EXPECT_OK(graph.AddRegularFanin("c", {"b", 2}));
  CheckNode(graph, "c", "", "", {}, {"b:2", "b:2", "^b"}, {});

  TF_EXPECT_OK(graph.AddControllingFanin("d", {"b", Graph::kControlSlot}));
  CheckNode(graph, "d", "", "", {}, {"^b"}, {});
  TF_EXPECT_OK(graph.AddControllingFanin("d", {"b", Graph::kControlSlot}));
  CheckNode(graph, "d", "", "", {}, {"^b"}, {});

  CheckNode(graph, "a", "Switch", "", {}, {}, {"b"});
  CheckNode(graph, "b", "Identity", "", {}, {"a:1"},
            {"c:0", "c:1", "^c", "^d"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, DedupControllingFaninsOnAddFaninByPort) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def =
      test::function::GDef({NDef("a", "NotImportant", {}, {}),
                            NDef("b", "NotImportant", {"c", "^a"}),
                            NDef("c", "NotImportant", {"a:1"})},
                           /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.AddRegularFaninByPort("b", 0, {"a", 2}));
  CheckNode(graph, "b", "NotImportant", "", {}, {"a:2", "c"}, {});

  TF_EXPECT_OK(graph.AddControllingFanin("c", {"a", Graph::kControlSlot}));
  CheckNode(graph, "c", "NotImportant", "", {}, {"a:1"}, {"b:1"});

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b:0", "c:0"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, NoDedupControllingFaninsOnAddFaninByPort) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "Switch", {}, {}), NDef("b", "Identity", {"a:1"}),
       NDef("c", "", {}, {}), NDef("d", "", {"c:2"}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.AddRegularFaninByPort("d", 1, {"b", 2}));
  CheckNode(graph, "d", "", "", {}, {"c:2", "b:2"}, {});
  TF_EXPECT_OK(graph.AddControllingFanin("d", {"b", Graph::kControlSlot}));
  CheckNode(graph, "d", "", "", {}, {"c:2", "b:2", "^b"}, {});
  TF_EXPECT_OK(graph.AddRegularFaninByPort("d", 0, {"b", 2}));
  CheckNode(graph, "d", "", "", {}, {"b:2", "c:2", "b:2", "^b"}, {});

  CheckNode(graph, "a", "Switch", "", {}, {}, {"b:0"});
  CheckNode(graph, "b", "Identity", "", {}, {"a:1"}, {"d:0", "d:2", "^d"});
  CheckNode(graph, "c", "", "", {}, {}, {"d:1"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, DedupControllingFaninsOnUpdateFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "NotImportant", {"a:1", "^b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.UpdateFanin("c", {"a", 1}, {"b", 2}));

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"c"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"b:2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, NoDedupControllingFaninsOnUpdateFanin) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "Switch", {}, {}), NDef("b", "Identity", {"a:1"}),
       NDef("c", "Identity", {"a:2"}), NDef("d", "NotImportant", {"c", "^b"}),
       NDef("e", "NotImportant", {"b", "^c"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.UpdateFanin("d", {"b", Graph::kControlSlot},
                                 {"c", Graph::kControlSlot}));
  CheckNode(graph, "d", "NotImportant", "", {}, {"c", "^c"}, {});

  TF_EXPECT_OK(graph.UpdateFanin("e", {"b", 0}, {"c", 3}));
  CheckNode(graph, "e", "NotImportant", "", {}, {"c:3", "^c"}, {});

  TF_EXPECT_OK(graph.UpdateFanin("e", {"c", 3}, {"c", Graph::kControlSlot}));
  CheckNode(graph, "e", "NotImportant", "", {}, {"^c"}, {});

  CheckNode(graph, "a", "Switch", "", {}, {}, {"b:0", "c:0"});
  CheckNode(graph, "b", "Identity", "", {}, {"a:1"}, {});
  CheckNode(graph, "c", "Identity", "", {}, {"a:2"}, {"d:0", "^d", "^e"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, DedupControllingFaninsOnUpdateFaninByPort) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "NotImportant", {"a:1", "^b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.UpdateRegularFaninByPort("c", 0, {"b", 2}));

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"c"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"b:2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, NoDedupControllingFaninsOnUpdateFaninByPort) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "Switch", {}, {}), NDef("b", "Identity", {"a:1"}),
       NDef("c", "Identity", {"a:2"}), NDef("d", "NotImportant", {"c", "^b"}),
       NDef("e", "NotImportant", {"b", "^c"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.UpdateRegularFaninByPort("d", 0, {"b", 1}));
  CheckNode(graph, "d", "NotImportant", "", {}, {"b:1", "^b"}, {});

  TF_EXPECT_OK(graph.UpdateRegularFaninByPort("e", 0, {"c", 2}));
  CheckNode(graph, "e", "NotImportant", "", {}, {"c:2", "^c"}, {});

  CheckNode(graph, "a", "Switch", "", {}, {}, {"b:0", "c:0"});
  CheckNode(graph, "b", "Identity", "", {}, {"a:1"}, {"d:0", "^d"});
  CheckNode(graph, "c", "Identity", "", {}, {"a:2"}, {"e:0", "^e"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateMaxRegularOutputPortOnAddFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:1"}),
       NDef("c", "NotImportant", {"^b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.AddRegularFanin("c", {"a", 3}));

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b", "c"});
  CheckNode(graph, "b", "NotImportant", "", {}, {"a:1"}, {"^c"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"a:3", "^b"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateMaxRegularOutputPortOnRemoveFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:1"}),
       NDef("c", "NotImportant", {"a:2"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.RemoveRegularFanin("c", {"a", 2}));
  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b"});
  CheckNode(graph, "b", "NotImportant", "", {}, {"a:1"}, {});
  CheckNode(graph, "c", "NotImportant", "", {}, {}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, KeepMaxRegularOutputPortOnRemoveFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:1"}),
       NDef("c", "NotImportant", {"a:2"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.RemoveRegularFanin("b", {"a", 1}));

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"c"});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "c", "NotImportant", "", {}, {"a:2"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateMaxRegularOutputPortOnUpdateFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:1"}),
       NDef("c", "NotImportant", {"a:2"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.UpdateFanin("c", {"a", 2}, {"b", 3}));

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b"});
  CheckNode(graph, "b", "NotImportant", "", {}, {"a:1"}, {"c"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"b:3"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddControllingFaninMissing) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);
  // Missing fanin.
  Status s = graph.AddControllingFanin("a", {"c", Graph::kControlSlot});
  EXPECT_FALSE(s.ok());
  string expected_msg =
      "MutableGraphView::AddControllingFanin(node_name='a', fanin='^c') error: "
      "node 'c' was not found.";
  EXPECT_EQ(s.error_message(), expected_msg);
  // Missing node.
  s = graph.AddControllingFanin("d", {"a", Graph::kControlSlot});
  EXPECT_FALSE(s.ok());
  expected_msg =
      "MutableGraphView::AddControllingFanin(node_name='d', fanin='^a') error: "
      "node 'd' was not found.";
  EXPECT_EQ(s.error_message(), expected_msg);
  // Missing node and fanin.
  s = graph.AddControllingFanin("c", {"d", Graph::kControlSlot});
  EXPECT_FALSE(s.ok());
  expected_msg =
      "MutableGraphView::AddControllingFanin(node_name='c', fanin='^d') error: "
      "node 'c' was not found.";
  EXPECT_EQ(s.error_message(), expected_msg);

  ASSERT_EQ(graph.graph()->node_size(), 2);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddControllingFaninExistingControl) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);
  TF_EXPECT_OK(graph.AddControllingFanin("a", {"b", Graph::kControlSlot}));
  TF_EXPECT_OK(graph.AddControllingFanin("a", {"b", Graph::kControlSlot}));

  ASSERT_EQ(graph.graph()->node_size(), 2);

  CheckNode(graph, "a", "NotImportant", "", {}, {"^b"}, {});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"^a"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddControllingFaninNotSwitch) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);
  TF_EXPECT_OK(graph.AddControllingFanin("a", {"b", 2}));
  TF_EXPECT_OK(graph.AddControllingFanin("a", {"b", 2}));

  ASSERT_EQ(graph.graph()->node_size(), 2);

  CheckNode(graph, "a", "NotImportant", "", {}, {"^b"}, {});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"^a"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddControllingFaninSwitch) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "Switch", {}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  Status s = graph.AddControllingFanin("a", {"b", Graph::kControlSlot});
  EXPECT_FALSE(s.ok());
  string expected_msg =
      "MutableGraphView::AddControllingFanin(node_name='a', fanin='^b') error: "
      "can't add fanin '^b' as it will become a Switch control dependency.";
  EXPECT_EQ(s.error_message(), expected_msg);

  ASSERT_EQ(graph.graph()->node_size(), 2);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "b", "Switch", "", {}, {}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddControllingFaninSwitchWithIdentity) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("switch", "Switch", {}, {}),
       NDef("identity", "Identity", {"switch"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.AddControllingFanin("a", {"switch", 0}));
  TF_EXPECT_OK(graph.AddControllingFanin("a", {"switch", 0}));

  ASSERT_EQ(graph.graph()->node_size(), 3);

  CheckNode(graph, "a", "NotImportant", "", {}, {"^identity"}, {});
  CheckNode(graph, "switch", "Switch", "", {}, {}, {"identity"});
  CheckNode(graph, "identity", "Identity", "", {}, {"switch"}, {"^a"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddControllingFaninSwitchWithNoExistingIdentity) {
  constexpr char kDevice[] = "/device:foo:0";
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}),
       NDef("switch", "Switch", {}, {{"T", DT_FLOAT}}, kDevice)},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.AddControllingFanin("a", {"switch", 0}));
  TF_EXPECT_OK(graph.AddControllingFanin("a", {"switch", 0}));

  ASSERT_EQ(graph.graph()->node_size(), 3);

  CheckNode(graph, "a", "NotImportant", "", {},
            {"^ConstantFoldingCtrl/switch_0"}, {});
  CheckNode(graph, "switch", "Switch", kDevice, {{"T", DT_FLOAT}}, {},
            {"ConstantFoldingCtrl/switch_0"});
  CheckNode(graph, "ConstantFoldingCtrl/switch_0", "Identity", kDevice,
            {{"T", DT_FLOAT}}, {"switch"}, {"^a"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddControllingFaninSwitchWithExistingAddedIdentity) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("switch", "Switch", {}, {}),
       NDef("ConstantFoldingCtrl/switch_0", "Identity", {"switch"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.AddControllingFanin("a", {"switch", 0}));
  TF_EXPECT_OK(graph.AddControllingFanin("a", {"switch", 0}));

  ASSERT_EQ(graph.graph()->node_size(), 3);

  CheckNode(graph, "a", "NotImportant", "", {},
            {"^ConstantFoldingCtrl/switch_0"}, {});
  CheckNode(graph, "switch", "Switch", "", {}, {},
            {"ConstantFoldingCtrl/switch_0"});
  CheckNode(graph, "ConstantFoldingCtrl/switch_0", "Identity", "", {},
            {"switch"}, {"^a"});

  CheckGraph(graph);
}

void TestAddControllingFaninSelfLoops(absl::string_view node_name,
                                      const TensorId& fanin,
                                      const string& error_msg) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}),
       NDef("b", "Switch", {}, {{"T", DT_FLOAT}}),
       NDef("c", "Identity", {"b:0"}), NDef("d", "Identity", {"b:1"}),
       NDef("e", "NotImportant", {"^a"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  Status s = graph.AddControllingFanin(node_name, fanin);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(), error_msg);

  EXPECT_EQ(graph.graph()->node_size(), 5);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"^e"});
  CheckNode(graph, "b", "Switch", "", {{"T", DT_FLOAT}}, {}, {"c", "d"});
  CheckNode(graph, "c", "Identity", "", {}, {"b"}, {});
  CheckNode(graph, "d", "Identity", "", {}, {"b:1"}, {});
  CheckNode(graph, "e", "NotImportant", "", {}, {"^a"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, AddControllingFaninSelfLoops) {
  string error_msg =
      "MutableGraphView::AddControllingFanin(node_name='a', fanin='^a') error: "
      "can't add fanin '^a' to self.";
  TestAddControllingFaninSelfLoops("a", {"a", Graph::kControlSlot}, error_msg);

  // Adding Switch control dependency to Identity consumer. Node `c` is
  // consuming `b:0`, so adding `b:0` as a control dependency, because it is a
  // Switch, should trigger a lookup of outputs. As `c` is a consumer and an
  // Identity, this will introduce a self loop, so no control dependency should
  // be added.
  error_msg =
      "MutableGraphView::AddControllingFanin(node_name='c', fanin='b:0') "
      "error: can't add found fanin '^c' to self.";
  TestAddControllingFaninSelfLoops("c", {"b", 0}, error_msg);

  // Adding Switch control dependency to Identity consumer. Node `d` is
  // consuming `b:1`, so adding `b:1` as a control dependency, because it is a
  // Switch, should trigger a lookup of outputs. As `d` is a consumer and an
  // Identity, this will introduce a self loop, so no control dependency should
  // be added.
  error_msg =
      "MutableGraphView::AddControllingFanin(node_name='d', fanin='b:1') "
      "error: can't add found fanin '^d' to self.";
  TestAddControllingFaninSelfLoops("d", {"b", 1}, error_msg);
}

TEST(MutableGraphViewTest, AddControllingFaninSelfLoopsGeneratedIdentity) {
  GraphDef graph_def =
      test::function::GDef({NDef("a", "NotImportant", {}, {}),
                            NDef("b", "Switch", {}, {{"T", DT_FLOAT}}),
                            NDef("c", "NotImportant", {}),
                            NDef("ConstantFoldingCtrl/b_1", "Identity", {})},
                           /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  // Adding Switch control dependency to Identity node of the same name as a
  // generated Identity node for pinning the control dependency. Because there
  // are no consumers of `b:1`, there will be an attempt to generate an Identity
  // node, with name `ConstantFoldingCtrl/b_1`. As the input node is of the same
  // name, we will introduce a self loop, so no control dependency should be
  // added.
  Status s = graph.AddControllingFanin("ConstantFoldingCtrl/b_1", {"b", 1});
  EXPECT_FALSE(s.ok());
  string expected_msg =
      "MutableGraphView::AddControllingFanin(node_name='ConstantFoldingCtrl/"
      "b_1', fanin='b:1') error: can't add generated fanin "
      "'^ConstantFoldingCtrl/b_1' to self.";
  EXPECT_EQ(s.error_message(), expected_msg);

  EXPECT_EQ(graph.graph()->node_size(), 4);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "b", "Switch", "", {{"T", DT_FLOAT}}, {}, {});
  CheckNode(graph, "c", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "ConstantFoldingCtrl/b_1", "Identity", "", {}, {}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, RemoveControllingFaninMissing) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "NotImportant", {}, {}),
       NDef("d", "NotImportant", {"^a", "^b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.RemoveControllingFanin("d", "c"));

  ASSERT_EQ(graph.graph()->node_size(), 4);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"^d"});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"^d"});
  CheckNode(graph, "c", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "d", "NotImportant", "", {}, {"^a", "^b"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, RemoveControllingFaninExisting) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "NotImportant", {}, {}),
       NDef("d", "NotImportant", {"^a", "^b", "^c"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.RemoveControllingFanin("d", "a"));
  TF_EXPECT_OK(graph.RemoveControllingFanin("d", "a"));

  ASSERT_EQ(graph.graph()->node_size(), 4);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "b", "NotImportant", "", {}, {}, {"^d"});
  CheckNode(graph, "c", "NotImportant", "", {}, {}, {"^d"});
  CheckNode(graph, "d", "NotImportant", "", {}, {"^c", "^b"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, RemoveControllingFaninOnRegularFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a"}),
       NDef("c", "NotImportant", {"a", "b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.RemoveControllingFanin("c", "a"));
  TF_EXPECT_OK(graph.RemoveControllingFanin("c", "b"));

  ASSERT_EQ(graph.graph()->node_size(), 3);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b", "c"});
  CheckNode(graph, "b", "NotImportant", "", {}, {"a"}, {"c:1"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"a", "b"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, RemoveControllingFaninSelfLoop) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a"}),
       NDef("c", "NotImportant", {"a", "b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  Status s = graph.RemoveControllingFanin("c", "c");
  EXPECT_FALSE(s.ok());
  string expected_msg =
      "MutableGraphView::RemoveControllingFanin(node_name='c', "
      "fanin_node_name='c') error: can't remove fanin '^c' from "
      "self.";
  EXPECT_EQ(s.error_message(), expected_msg);

  ASSERT_EQ(graph.graph()->node_size(), 3);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b", "c"});
  CheckNode(graph, "b", "NotImportant", "", {}, {"a"}, {"c:1"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"a", "b"}, {});

  CheckGraph(graph);
}

void TestUpdateAllRegularFaninsToControlling(
    absl::string_view node_name, bool node_exists, bool success,
    const string& error_msg, absl::Span<const string> expected_fanins) {
  constexpr char kDevice[] = "/device:foo:0";
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}),
       NDef("switch", "Switch", {}, {{"T", DT_FLOAT}}, kDevice),
       NDef("b", "NotImportant", {"switch:1"}, {}),
       NDef("ConstantFoldingCtrl/switch_1", "Identity", {"switch:1"},
            {{"T", DT_FLOAT}}, kDevice),
       NDef("c", "NotImportant", {"a", "^b"}, {}),
       NDef("d", "NotImportant", {"b", "c"}, {}),
       NDef("e", "NotImportant", {"^d"}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  NodeDef* node = graph.GetNode(node_name);
  if (node_exists) {
    EXPECT_NE(node, nullptr);
  } else {
    EXPECT_EQ(node, nullptr);
  }

  absl::flat_hash_map<string, std::vector<string>> unmodified_node_inputs =
      GetNodeInputsFromGraph(graph_def, node_name);

  Status s = graph.UpdateAllRegularFaninsToControlling(node_name);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (node_exists) {
    CompareNodeFanins(graph, node, expected_fanins);
  }

  CheckUnmodifiedNodeFanins(graph_def, node_name, unmodified_node_inputs);

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, UpdateAllRegularFaninsToControlling) {
  string error_msg;
  // Nodes with some regular fanins and some controls.
  TestUpdateAllRegularFaninsToControlling("a", /*node_exists=*/true,
                                          /*success=*/true, error_msg, {});
  TestUpdateAllRegularFaninsToControlling("c", /*node_exists=*/true,
                                          /*success=*/true, error_msg,
                                          {"^a", "^b"});
  TestUpdateAllRegularFaninsToControlling("d", /*node_exists=*/true,
                                          /*success=*/true, error_msg,
                                          {"^b", "^c"});
  TestUpdateAllRegularFaninsToControlling("e", /*node_exists=*/true,
                                          /*success=*/true, error_msg, {"^d"});

  // Use existing Identity to pin control dependency of Switch.
  TestUpdateAllRegularFaninsToControlling("b", /*node_exists=*/true,
                                          /*success=*/true, error_msg,
                                          {"^ConstantFoldingCtrl/switch_1"});

  // Missing node.
  error_msg =
      "MutableGraphView::UpdateAllRegularFaninsToControlling(node_name='f') "
      "error: node 'f' was not found.";
  TestUpdateAllRegularFaninsToControlling("f", /*node_exists=*/false,
                                          /*success=*/false, error_msg, {});

  // Error in getting controlling fanin.
  error_msg =
      "MutableGraphView::UpdateAllRegularFaninsToControlling(node_name='"
      "ConstantFoldingCtrl/switch_1') error: can't add found fanin "
      "'^ConstantFoldingCtrl/switch_1' to self.";
  TestUpdateAllRegularFaninsToControlling("ConstantFoldingCtrl/switch_1",
                                          /*node_exists=*/true,
                                          /*success=*/false, error_msg,
                                          {"switch:1"});
}

TEST(MutableGraphViewTest, UpdateAllRegularFaninsToControllingConsumingSwitch) {
  constexpr char kDevice[] = "/device:foo:0";
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}),
       NDef("switch", "Switch", {}, {{"T", DT_FLOAT}}, kDevice),
       NDef("b", "NotImportant", {"switch:1"}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  TF_EXPECT_OK(graph.UpdateAllRegularFaninsToControlling("b"));

  EXPECT_EQ(graph.graph()->node_size(), 4);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {});
  CheckNode(graph, "switch", "Switch", kDevice, {{"T", DT_FLOAT}}, {},
            {"ConstantFoldingCtrl/switch_1"});
  CheckNode(graph, "b", "NotImportant", "", {},
            {"^ConstantFoldingCtrl/switch_1"}, {});
  CheckNode(graph, "ConstantFoldingCtrl/switch_1", "Identity", kDevice,
            {{"T", DT_FLOAT}}, {"switch:1"}, {"^b"});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, DeleteNodes) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("bar", "NotImportant", {}, {}),
       NDef("other", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"bar", "other", "bar:1", "^bar"}),
       NDef("foo_2", "NotImportant", {"other:1", "bar:2", "^bar"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_NE(graph.GetNode("foo_1"), nullptr);
  TF_EXPECT_OK(graph.DeleteNodes({"foo_1"}));

  EXPECT_EQ(graph.graph()->node_size(), 3);
  EXPECT_EQ(graph.GetNode("foo_1"), nullptr);

  CheckNode(graph, "bar", "NotImportant", "", {}, {}, {"foo_2:1"});
  CheckNode(graph, "other", "NotImportant", "", {}, {}, {"foo_2"});
  CheckNode(graph, "foo_2", "NotImportant", "", {}, {"other:1", "bar:2"}, {});

  CheckGraph(graph);
}

GraphDef SimpleDeleteNodeGraph() {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:2"}),
       NDef("c", "NotImportant", {"a:5", "^b"}), NDef("d", "NotImportant", {}),
       NDef("e", "NotImportant", {"d:2"}),
       NDef("f", "NotImportant", {"d:3", "^e"})},
      /*funcs=*/{});
  return graph_def;
}

TEST(MutableGraphViewTest, DeleteNodesWithFanoutsBeingDeleted) {
  GraphDef graph_def = SimpleDeleteNodeGraph();

  MutableGraphView graph(&graph_def);
  EXPECT_NE(graph.GetNode("a"), nullptr);
  EXPECT_NE(graph.GetNode("b"), nullptr);
  EXPECT_NE(graph.GetNode("c"), nullptr);
  TF_EXPECT_OK(graph.DeleteNodes({"c", "a", "b"}));

  EXPECT_EQ(graph.graph()->node_size(), 3);
  EXPECT_EQ(graph.GetNode("a"), nullptr);
  EXPECT_EQ(graph.GetNode("b"), nullptr);
  EXPECT_EQ(graph.GetNode("c"), nullptr);

  CheckNode(graph, "d", "NotImportant", "", {}, {}, {"e", "f"});
  CheckNode(graph, "e", "NotImportant", "", {}, {"d:2"}, {"^f"});
  CheckNode(graph, "f", "NotImportant", "", {}, {"d:3", "^e"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, DeleteMissingNodes) {
  GraphDef graph_def = SimpleDeleteNodeGraph();

  MutableGraphView graph(&graph_def);

  EXPECT_EQ(graph.GetNode("g"), nullptr);
  EXPECT_EQ(graph.GetNode("h"), nullptr);
  TF_EXPECT_OK(graph.DeleteNodes({"g", "h"}));

  EXPECT_EQ(graph.graph()->node_size(), 6);
  EXPECT_EQ(graph.GetNode("g"), nullptr);
  EXPECT_EQ(graph.GetNode("h"), nullptr);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b", "c"});
  CheckNode(graph, "b", "NotImportant", "", {}, {"a:2"}, {"^c"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"a:5", "^b"}, {});
  CheckNode(graph, "d", "NotImportant", "", {}, {}, {"e", "f"});
  CheckNode(graph, "e", "NotImportant", "", {}, {"d:2"}, {"^f"});
  CheckNode(graph, "f", "NotImportant", "", {}, {"d:3", "^e"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, DeleteMissingNodesAndNodesWithFanoutsBeingDeleted) {
  GraphDef graph_def = SimpleDeleteNodeGraph();

  MutableGraphView graph(&graph_def);

  EXPECT_NE(graph.GetNode("d"), nullptr);
  EXPECT_NE(graph.GetNode("e"), nullptr);
  EXPECT_NE(graph.GetNode("f"), nullptr);
  TF_EXPECT_OK(graph.DeleteNodes({"d", "e", "f", "g", "h"}));

  EXPECT_EQ(graph.graph()->node_size(), 3);
  EXPECT_EQ(graph.GetNode("d"), nullptr);
  EXPECT_EQ(graph.GetNode("e"), nullptr);
  EXPECT_EQ(graph.GetNode("f"), nullptr);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b", "c"});
  CheckNode(graph, "b", "NotImportant", "", {}, {"a:2"}, {"^c"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"a:5", "^b"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, DeleteNodesWithError) {
  GraphDef graph_def = SimpleDeleteNodeGraph();

  MutableGraphView graph(&graph_def);

  Status s = graph.DeleteNodes({"b", "a"});
  EXPECT_FALSE(s.ok());
  string error_msg =
      "MutableGraphView::DeleteNodes(nodes_to_delete={a, b}) error: can't "
      "delete node(s) with retained fanouts(s) [a, b].";
  EXPECT_EQ(s.error_message(), error_msg);

  EXPECT_EQ(graph.graph()->node_size(), 6);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b", "c"});
  CheckNode(graph, "b", "NotImportant", "", {}, {"a:2"}, {"^c"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"a:5", "^b"}, {});
  CheckNode(graph, "d", "NotImportant", "", {}, {}, {"e", "f"});
  CheckNode(graph, "e", "NotImportant", "", {}, {"d:2"}, {"^f"});
  CheckNode(graph, "f", "NotImportant", "", {}, {"d:3", "^e"}, {});

  CheckGraph(graph);
}

TEST(MutableGraphViewTest, DeleteNodesWithLargeError) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:2"}),
       NDef("c", "NotImportant", {"^b"}), NDef("d", "NotImportant", {"c:6"}),
       NDef("e", "NotImportant", {"d:2"}),
       NDef("f", "NotImportant", {"d:3", "^e"}),
       NDef("g", "NotImportant", {"f"}), NDef("h", "NotImportant", {"a"}),
       NDef("i", "NotImportant", {"b"}), NDef("j", "NotImportant", {"c"}),
       NDef("k", "NotImportant", {"d"}), NDef("l", "NotImportant", {"e"}),
       NDef("m", "NotImportant", {"f"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  Status s = graph.DeleteNodes({"a", "b", "c", "d", "e", "f"});
  EXPECT_FALSE(s.ok());
  string error_msg =
      "MutableGraphView::DeleteNodes(nodes_to_delete={a, b, c, d, e, ...}) "
      "error: can't delete node(s) with retained fanouts(s) [a, b, c, d, e, "
      "...].";
  EXPECT_EQ(s.error_message(), error_msg);

  EXPECT_EQ(graph.graph()->node_size(), 13);

  CheckNode(graph, "a", "NotImportant", "", {}, {}, {"b", "h"});
  CheckNode(graph, "b", "NotImportant", "", {}, {"a:2"}, {"^c", "i"});
  CheckNode(graph, "c", "NotImportant", "", {}, {"^b"}, {"d", "j"});
  CheckNode(graph, "d", "NotImportant", "", {}, {"c:6"}, {"e", "f", "k"});
  CheckNode(graph, "e", "NotImportant", "", {}, {"d:2"}, {"^f", "l"});
  CheckNode(graph, "f", "NotImportant", "", {}, {"d:3", "^e"}, {"g", "m"});
  CheckNode(graph, "g", "NotImportant", "", {}, {"f"}, {});
  CheckNode(graph, "h", "NotImportant", "", {}, {"a"}, {});
  CheckNode(graph, "i", "NotImportant", "", {}, {"b"}, {});
  CheckNode(graph, "j", "NotImportant", "", {}, {"c"}, {});
  CheckNode(graph, "k", "NotImportant", "", {}, {"d"}, {});
  CheckNode(graph, "l", "NotImportant", "", {}, {"e"}, {});
  CheckNode(graph, "m", "NotImportant", "", {}, {"f"}, {});

  CheckGraph(graph);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
