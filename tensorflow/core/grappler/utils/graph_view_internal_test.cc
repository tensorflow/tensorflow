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

#include "tensorflow/core/grappler/utils/graph_view_internal.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace utils {
namespace internal {
namespace {

using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;

constexpr char kNodeOp[] = "NotImportant";

GraphDef SimpleTestGraphForMutation() {
  return GDef(
      {NDef("a", kNodeOp, {}), NDef("b", kNodeOp, {}), NDef("c", kNodeOp, {}),
       NDef("d", kNodeOp, {"a:2", "b:3", "a:4", "^c", "^b"},
            {{"attr_1", "a"}, {"attr_2", 2.0f}}, "device_d")},
      /*funcs=*/{});
}

absl::flat_hash_map<absl::string_view, int> GetUpdatedNodeNames(
    const MutableGraphView* graph_view) {
  absl::flat_hash_map<absl::string_view, int> updated_node_names;
  updated_node_names.reserve(graph_view->NumNodes());
  for (const auto& node_view : graph_view->GetNodes()) {
    updated_node_names.emplace(node_view.GetName(), -1);
  }
  return updated_node_names;
}

using MutableNodeViewDiff = NodeViewDiff<MutableGraphView>;

TEST(MutableNodeViewDiffTest, UpdateName) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  UpdateName(&diff, "e");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  UpdateName(&diff, "d");
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, UpdateOp) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  UpdateOp(&diff, "RandomOp");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  UpdateOp(&diff, kNodeOp);
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, UpdateDevice) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  UpdateDevice(&diff, "random_device");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  UpdateDevice(&diff, "device_d");
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, AddOrUpdateRegularFanin) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Bad index.
  AddOrUpdateRegularFanin(&diff, -1, {"a", 0});
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Set fanin to same existing fanin.
  AddOrUpdateRegularFanin(&diff, 0, {"a", 2});
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Update existing fanin.
  AddOrUpdateRegularFanin(&diff, 0, {"a", 3});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Add new fanin at index 4 resulting in missing fanin at index 3.
  AddOrUpdateRegularFanin(&diff, 4, {"b", 4});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));

  // Add new fanin at index 3.
  AddOrUpdateRegularFanin(&diff, 3, {"c", 4});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Add new fanin at index 5.
  AddOrUpdateRegularFanin(&diff, 5, {"c", 5});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, AddOrUpdateRegularFaninBetweenRemovedFanins) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  RemoveRegularFanin(&diff, 0);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
  RemoveRegularFanin(&diff, 2);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 1, {"c", 1});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 0, {"c", 0});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  RemoveRegularFanin(&diff, 0);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 2, {"c", 2});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, RemoveRegularFanin) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Bad index.
  RemoveRegularFanin(&diff, -1);
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
  RemoveRegularFanin(&diff, 3);
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Add new fanin at index 4 resulting in missing fanin at index 3.
  AddOrUpdateRegularFanin(&diff, 4, {"b", 4});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
  // Remove fanin at index 4.
  RemoveRegularFanin(&diff, 4);
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Add new fanin at index 4 resulting in missing fanin at index 3.
  AddOrUpdateRegularFanin(&diff, 4, {"b", 4});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
  // Add new fanin at index 3.
  AddOrUpdateRegularFanin(&diff, 3, {"c", 4});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
  // Remove fanin at index 3.
  RemoveRegularFanin(&diff, 3);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
  // Remove fanin at index 4.
  RemoveRegularFanin(&diff, 4);
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Add new fanin at index 5 resulting in missing fanin at indices 3 and 4.
  AddOrUpdateRegularFanin(&diff, 5, {"b", 6});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
  // Add new fanin at index 3 resulting in missing fanin at index 4.
  AddOrUpdateRegularFanin(&diff, 3, {"c", 4});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
  // Remove missing fanin at index 4.
  RemoveRegularFanin(&diff, 4);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
  // Remove fanin at index 3.
  RemoveRegularFanin(&diff, 3);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
  // Remove fanin at index 5.
  RemoveRegularFanin(&diff, 5);
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Update existing fanin.
  AddOrUpdateRegularFanin(&diff, 1, {"a", 3});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
  // Remove fanin at index 1.
  RemoveRegularFanin(&diff, 1);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
  // Add original fanin at index 1.
  AddOrUpdateRegularFanin(&diff, 1, {"b", 3});
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, RemoveRegularFaninResize) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 3, {"c", 5});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
  AddOrUpdateRegularFanin(&diff, 4, {"c", 6});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
  AddOrUpdateRegularFanin(&diff, 5, {"c", 7});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  // Remove fanin in middle of appended regular fanins.
  RemoveRegularFanin(&diff, 4);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
  // Remove last fanin in appended regular fanins.
  RemoveRegularFanin(&diff, 5);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, AddControllingFanin) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddControllingFanin(&diff, 0, "c");
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddControllingFanin(&diff, kMissingIndex, "a");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, RemoveControllingFanin) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddControllingFanin(&diff, kMissingIndex, "a");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  RemoveControllingFanin(&diff, 0, "c");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  RemoveControllingFanin(&diff, kMissingIndex, "a");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddControllingFanin(&diff, 0, "c");
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, AddOrUpdateAttribute) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AttrValue attr_1;
  attr_1.set_b(true);
  AddOrUpdateAttribute(&diff, "attr_1", attr_1);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AttrValue attr_3;
  attr_3.set_i(4);
  AddOrUpdateAttribute(&diff, "attr_1", attr_3);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, RemoveAttribute) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AttrValue attr_1;
  attr_1.set_b(true);
  AddOrUpdateAttribute(&diff, "attr_1", attr_1);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  RemoveAttribute(&diff, "attr_1");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  RemoveAttribute(&diff, "attr_3");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, Reset) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  RemoveRegularFanin(&diff, 2);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddControllingFanin(&diff, kMissingIndex, "a");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AttrValue attr_1;
  attr_1.set_b(true);
  AddOrUpdateAttribute(&diff, "attr_1", attr_1);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  Reset(&diff);
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedWithRemovedAndAppendedFanins) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  RemoveRegularFanin(&diff, 2);
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 3, {"a", 8});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedSelfLoopRegularUpdate) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 0, {"d", 1});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedSelfLoopRegularNew) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 3, {"d", 1});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedSelfLoopControl) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddControllingFanin(&diff, kMissingIndex, "d");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedMissingFaninRegularUpdate) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 0, {"e", 1});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedMissingFaninRegularNew) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 3, {"e", 1});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedMissingControl) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddControllingFanin(&diff, kMissingIndex, "e");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedRenamedSelfLoopRegularUpdate) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  string old_node_name = "d";
  string new_node_name = "e";
  updated_node_names.erase(old_node_name);
  updated_node_names.emplace(old_node_name, 3);
  updated_node_names.emplace(new_node_name, -1);

  UpdateName(&diff, "e");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 0, {"e", 1});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedRenamedSelfLoopRegularNew) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  string old_node_name = "d";
  string new_node_name = "e";
  updated_node_names.erase(old_node_name);
  updated_node_names.emplace(old_node_name, 3);
  updated_node_names.emplace(new_node_name, -1);

  UpdateName(&diff, "e");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 3, {"e", 1});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedRenamedSelfLoopControl) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  string old_node_name = "d";
  string new_node_name = "e";
  updated_node_names.erase(old_node_name);
  updated_node_names.emplace(old_node_name, 3);
  updated_node_names.emplace(new_node_name, -1);

  UpdateName(&diff, "e");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddControllingFanin(&diff, kMissingIndex, "e");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedRenamedMissingFaninRegularUpdate) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  string old_node_name = "d";
  string new_node_name = "e";
  updated_node_names.erase(old_node_name);
  updated_node_names.emplace(old_node_name, 3);
  updated_node_names.emplace(new_node_name, -1);

  UpdateName(&diff, "e");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 0, {"f", 1});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedRenamedMissingFaninRegularNew) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  string old_node_name = "d";
  string new_node_name = "e";
  updated_node_names.erase(old_node_name);
  updated_node_names.emplace(old_node_name, 3);
  updated_node_names.emplace(new_node_name, -1);

  UpdateName(&diff, "e");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddOrUpdateRegularFanin(&diff, 3, {"f", 1});
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, IsWellFormedRenamedMissingFaninControl) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  string old_node_name = "d";
  string new_node_name = "e";
  updated_node_names.erase(old_node_name);
  updated_node_names.emplace(old_node_name, 3);
  updated_node_names.emplace(new_node_name, -1);

  UpdateName(&diff, "e");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  AddControllingFanin(&diff, kMissingIndex, "f");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, RenamedAndRemovedFanins) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  string old_node_name = "d";
  string new_node_name = "e";
  updated_node_names.erase(old_node_name);
  updated_node_names.emplace(old_node_name, 3);
  updated_node_names.emplace(new_node_name, -1);

  UpdateName(&diff, "e");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  for (int i = 0; i < 3; ++i) {
    RemoveRegularFanin(&diff, i);
  }
  RemoveControllingFanin(&diff, 0, "c");
  RemoveControllingFanin(&diff, 0, "b");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));
}

TEST(MutableNodeViewDiffTest, RenamedWithSelfLoopControl) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutableNodeView* d_node = graph_view.GetNode("d");
  ASSERT_NE(d_node, nullptr);

  MutableNodeViewDiff diff(&graph_view, d_node->node_index());
  EXPECT_TRUE(IsEmpty(&diff));
  EXPECT_TRUE(IsWellFormed(&diff, updated_node_names));

  updated_node_names.erase("d");

  UpdateName(&diff, "c");
  EXPECT_FALSE(IsEmpty(&diff));
  EXPECT_FALSE(IsWellFormed(&diff, updated_node_names));
}

using MutationNewNodeForTest = NewNode<MutableGraphView>;

TEST(MutationNewNodeTest, UpdateName) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutationNewNodeForTest new_node(&graph_view, {});

  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  UpdateName(&new_node, "new");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  UpdateName(&new_node, "");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
}

TEST(MutationNewNodeTest, UpdateOp) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutationNewNodeForTest new_node(&graph_view, {});

  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  UpdateOp(&new_node, "Identity");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  UpdateOp(&new_node, "");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
}

TEST(MutationNewNodeTest, UpdateDevice) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutationNewNodeForTest new_node(&graph_view, {});

  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  UpdateDevice(&new_node, "foo_device");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  UpdateDevice(&new_node, "");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
}

TEST(MutationNewNodeTest, AddOrUpdateRegularFanin) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutationNewNodeForTest new_node(&graph_view, {});

  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  UpdateName(&new_node, "new");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));

  // Bad index.
  AddOrUpdateRegularFanin(&new_node, -1, {"a", 1});
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));

  // Fanin at index 0 is missing.
  AddOrUpdateRegularFanin(&new_node, 1, {"a", 1});
  EXPECT_FALSE(IsWellFormed(&new_node, updated_node_names));
  AddOrUpdateRegularFanin(&new_node, 0, {"b", 2});
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  AddOrUpdateRegularFanin(&new_node, 2, {"c", 3});
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));

  // Update inplace.
  AddOrUpdateRegularFanin(&new_node, 1, {"d", 4});
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));

  // Missing fanin.
  AddOrUpdateRegularFanin(&new_node, 1, {"e", 5});
  EXPECT_FALSE(IsWellFormed(&new_node, updated_node_names));

  // Self loop.
  AddOrUpdateRegularFanin(&new_node, 1, {"new", 6});
  EXPECT_FALSE(IsWellFormed(&new_node, updated_node_names));

  AddOrUpdateRegularFanin(&new_node, 1, {"d", 4});
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
}

TEST(MutationNewNodeTest, RemoveRegularFanin) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutationNewNodeForTest new_node(&graph_view, {});

  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  UpdateName(&new_node, "new");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  AddOrUpdateRegularFanin(&new_node, 0, {"a", 1});
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  AddOrUpdateRegularFanin(&new_node, 1, {"b", 2});
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  AddOrUpdateRegularFanin(&new_node, 2, {"c", 3});
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  RemoveRegularFanin(&new_node, 3);
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  RemoveRegularFanin(&new_node, 2);
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  RemoveRegularFanin(&new_node, 0);
  EXPECT_FALSE(IsWellFormed(&new_node, updated_node_names));
  RemoveRegularFanin(&new_node, 1);
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
}

TEST(MutationNewNodeTest, AddControllingFanin) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutationNewNodeForTest new_node(&graph_view, {});

  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  UpdateName(&new_node, "new");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));

  AddControllingFanin(&new_node, "a");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));

  // Missing fanin.
  AddControllingFanin(&new_node, "e");
  EXPECT_FALSE(IsWellFormed(&new_node, updated_node_names));

  // Self loop.
  AddControllingFanin(&new_node, "new");
  EXPECT_FALSE(IsWellFormed(&new_node, updated_node_names));

  RemoveControllingFanin(&new_node, "e");
  RemoveControllingFanin(&new_node, "new");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
}

TEST(MutationNewNodeTest, RemoveControllingFanin) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutationNewNodeForTest new_node(&graph_view, {});

  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  UpdateName(&new_node, "new");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));

  AddControllingFanin(&new_node, "a");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));

  RemoveControllingFanin(&new_node, "e");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));

  RemoveControllingFanin(&new_node, "new");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));

  RemoveControllingFanin(&new_node, "a");
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
}

TEST(MutationNewNodeTest, AddOrUpdateAttribute) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutationNewNodeForTest new_node(&graph_view, {});

  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  string attr_name = "attr_name";
  AttrValue attr_1;
  attr_1.set_i(8);
  AddOrUpdateAttribute(&new_node, attr_name, attr_1);
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  AttrValue attr_2;
  attr_2.set_f(2.0f);
  AddOrUpdateAttribute(&new_node, attr_name, attr_2);
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
}

TEST(MutationNewNodeTest, RemoveAttribute) {
  GraphDef graph = SimpleTestGraphForMutation();

  Status s;
  MutableGraphView graph_view(&graph, &s);
  TF_ASSERT_OK(s);
  auto updated_node_names = GetUpdatedNodeNames(&graph_view);

  MutationNewNodeForTest new_node(&graph_view, {});

  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  string attr_name = "attr_name";
  AttrValue attr_1;
  attr_1.set_i(8);
  AddOrUpdateAttribute(&new_node, attr_name, attr_1);
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  RemoveAttribute(&new_node, attr_name);
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
  RemoveAttribute(&new_node, attr_name);
  EXPECT_TRUE(IsWellFormed(&new_node, updated_node_names));
}

}  // namespace
}  // namespace internal
}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow
