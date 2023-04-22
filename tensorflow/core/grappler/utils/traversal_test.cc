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

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

namespace {
using ::tensorflow::test::function::NDef;

DfsCallbacks MkCallbacks(std::vector<string>* pre_order,
                         std::vector<string>* post_order,
                         std::vector<string>* back_edges) {
  return {[pre_order](const NodeDef* n) { pre_order->push_back(n->name()); },
          [post_order](const NodeDef* n) { post_order->push_back(n->name()); },
          [back_edges](const NodeDef* src, const NodeDef* dst) {
            back_edges->push_back(absl::StrCat(src->name(), "->", dst->name()));
          }};
}

TEST(TraversalTest, OutputsDfsNoLoop) {
  const string op = "OpIsNotImportantInThisTest";

  GraphDef graph = ::tensorflow::test::function::GDef(  //
      {NDef("2", op, {"5"}, {}),                        //
       NDef("0", op, {"5", "4"}, {}),                   //
       NDef("1", op, {"4", "3"}, {}),                   //
       NDef("3", op, {"2"}, {}),                        //
       NDef("5", op, {}, {}),                           //
       NDef("4", op, {}, {})},                          //
      /*funcs=*/{});

  std::vector<const NodeDef*> start_nodes = {&graph.node(4), &graph.node(5)};

  std::vector<string> pre_order;
  std::vector<string> post_order;
  std::vector<string> back_edges;

  GraphTopologyView graph_view;
  TF_CHECK_OK(graph_view.InitializeFromGraph(graph));
  DfsTraversal(graph_view, start_nodes, TraversalDirection::kFollowOutputs,
               MkCallbacks(&pre_order, &post_order, &back_edges));

  const std::vector<string> expected_pre = {"4", "1", "0", "5", "2", "3"};
  const std::vector<string> expected_post = {"1", "0", "4", "3", "2", "5"};

  EXPECT_EQ(pre_order, expected_pre);
  EXPECT_EQ(post_order, expected_post);
  EXPECT_TRUE(back_edges.empty());
}

TEST(TraversalTest, InputsDfsNoLoop) {
  const string op = "OpIsNotImportantInThisTest";

  GraphDef graph = ::tensorflow::test::function::GDef(  //
      {NDef("2", op, {"5"}, {}),                        //
       NDef("0", op, {"5", "4"}, {}),                   //
       NDef("1", op, {"4", "3"}, {}),                   //
       NDef("3", op, {"2"}, {}),                        //
       NDef("5", op, {}, {}),                           //
       NDef("4", op, {}, {})},                          //
      /*funcs=*/{});

  std::vector<const NodeDef*> start_nodes = {&graph.node(1), &graph.node(2)};

  std::vector<string> pre_order;
  std::vector<string> post_order;
  std::vector<string> back_edges;

  GraphTopologyView graph_view;
  TF_CHECK_OK(graph_view.InitializeFromGraph(graph));
  DfsTraversal(graph_view, start_nodes, TraversalDirection::kFollowInputs,
               MkCallbacks(&pre_order, &post_order, &back_edges));

  const std::vector<string> expected_pre = {"1", "4", "3", "2", "5", "0"};
  const std::vector<string> expected_post = {"4", "5", "2", "3", "1", "0"};

  EXPECT_EQ(pre_order, expected_pre);
  EXPECT_EQ(post_order, expected_post);
  EXPECT_TRUE(back_edges.empty());
}

TEST(TraversalTest, InputsDfsWithLoop) {
  // Graph with a loop.
  GraphDef graph = ::tensorflow::test::function::GDef(  //
      {NDef("2", "Merge", {"1", "5"}, {}),              //
       NDef("3", "Switch", {"2"}, {}),                  //
       NDef("4", "Identity", {"3"}, {}),                //
       NDef("5", "NextIteration", {"4"}, {}),           //
       NDef("1", "Enter", {}, {}),                      //
       NDef("6", "Exit", {"3"}, {})},                   //
      /*funcs=*/{});

  std::vector<const NodeDef*> start_nodes = {&graph.node(5)};

  std::vector<string> pre_order;
  std::vector<string> post_order;
  std::vector<string> back_edges;

  GraphTopologyView graph_view;
  TF_CHECK_OK(graph_view.InitializeFromGraph(graph));
  DfsTraversal(graph_view, start_nodes, TraversalDirection::kFollowInputs,
               MkCallbacks(&pre_order, &post_order, &back_edges));

  const std::vector<string> expected_pre = {"6", "3", "2", "1", "5", "4"};
  const std::vector<string> expected_post = {"1", "4", "5", "2", "3", "6"};
  const std::vector<string> expected_edges = {"4->3"};

  EXPECT_EQ(pre_order, expected_pre);
  EXPECT_EQ(post_order, expected_post);
  EXPECT_EQ(back_edges, expected_edges);
}

TEST(TraversalTest, OutputDfsWithLoop) {
  // Graph with a loop.
  GraphDef graph = ::tensorflow::test::function::GDef(  //
      {NDef("2", "Merge", {"1", "5"}, {}),              //
       NDef("3", "Switch", {"2"}, {}),                  //
       NDef("4", "Identity", {"3"}, {}),                //
       NDef("5", "NextIteration", {"4"}, {}),           //
       NDef("1", "Enter", {}, {}),                      //
       NDef("6", "Exit", {"3"}, {})},                   //
      /*funcs=*/{});

  std::vector<const NodeDef*> start_nodes = {&graph.node(0)};

  std::vector<string> pre_order;
  std::vector<string> post_order;
  std::vector<string> back_edges;

  GraphTopologyView graph_view;
  TF_CHECK_OK(graph_view.InitializeFromGraph(graph));
  DfsTraversal(graph_view, start_nodes, TraversalDirection::kFollowOutputs,
               MkCallbacks(&pre_order, &post_order, &back_edges));

  const std::vector<string> expected_pre = {"2", "3", "6", "4", "5"};
  const std::vector<string> expected_post = {"6", "5", "4", "3", "2"};
  const std::vector<string> expected_edges = {"5->2"};

  EXPECT_EQ(pre_order, expected_pre);
  EXPECT_EQ(post_order, expected_post);
  EXPECT_EQ(back_edges, expected_edges);
}

TEST(TraversalTest, DfsWithEnterPredicate) {
  const string op = "OpIsNotImportantInThisTest";

  GraphDef graph = ::tensorflow::test::function::GDef(  //
      {NDef("1", op, {}, {}),                           //       2 -> 3
       NDef("2", op, {"1"}, {}),                        // 1 -> /      \ -> 6
       NDef("3", op, {"2"}, {}),                        //      \      /
       NDef("4", op, {"1"}, {}),                        //       4 -> 5
       NDef("5", op, {"4"}, {}),                        //
       NDef("6", op, {"3", "5"}, {})},                  //
      /*funcs=*/{});

  // Do not enter the nodes '2' and '3'.
  const auto enter = [](const NodeDef* node) {
    return node->name() != "2" && node->name() != "3";
  };

  std::vector<const NodeDef*> start_nodes = {&graph.node(0)};

  std::vector<string> pre_order;
  std::vector<string> post_order;
  std::vector<string> back_edges;

  GraphTopologyView graph_view;
  TF_CHECK_OK(graph_view.InitializeFromGraph(graph));
  DfsTraversal(graph_view, start_nodes, TraversalDirection::kFollowOutputs,
               DfsPredicates::Enter(enter),
               MkCallbacks(&pre_order, &post_order, &back_edges));

  const std::vector<string> expected_pre = {"1", "4", "5", "6"};
  const std::vector<string> expected_post = {"6", "5", "4", "1"};

  EXPECT_EQ(pre_order, expected_pre);
  EXPECT_EQ(post_order, expected_post);
  EXPECT_TRUE(back_edges.empty());
}

TEST(TraversalTest, DfsWithAdvancePredicate) {
  const string op = "OpIsNotImportantInThisTest";

  GraphDef graph = ::tensorflow::test::function::GDef(  //
      {NDef("1", op, {}, {}),                           //       2 -> 3
       NDef("2", op, {"1"}, {}),                        // 1 -> /      \ -> 6
       NDef("3", op, {"2"}, {}),                        //      \      /
       NDef("4", op, {"1"}, {}),                        //       4 -> 5
       NDef("5", op, {"4"}, {}),                        //
       NDef("6", op, {"3", "5"}, {})},                  //
      {} /* empty function library*/);

  // Do not advance from the nodes '2' and '3'.
  const auto advance = [](const NodeDef* node) {
    return node->name() != "2" && node->name() != "3";
  };

  std::vector<const NodeDef*> start_nodes = {&graph.node(0)};

  std::vector<string> pre_order;
  std::vector<string> post_order;
  std::vector<string> back_edges;

  GraphTopologyView graph_view;
  TF_CHECK_OK(graph_view.InitializeFromGraph(graph));
  DfsTraversal(graph_view, start_nodes, TraversalDirection::kFollowOutputs,
               DfsPredicates::Advance(advance),
               MkCallbacks(&pre_order, &post_order, &back_edges));

  const std::vector<string> expected_pre = {"1", "4", "5", "6", "2"};
  const std::vector<string> expected_post = {"6", "5", "4", "2", "1"};

  EXPECT_EQ(pre_order, expected_pre);
  EXPECT_EQ(post_order, expected_post);
  EXPECT_TRUE(back_edges.empty());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
