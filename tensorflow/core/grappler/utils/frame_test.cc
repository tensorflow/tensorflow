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

#include "tensorflow/core/grappler/utils/frame.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using GraphTypes =
    ::testing::Types<GraphDef, utils::GraphView, utils::MutableGraphView>;

template <typename T>
class FrameViewTest : public ::testing::Test {
 protected:
  NodeDef CreateNode(const string& name, const std::vector<string>& inputs) {
    return CreateNode(name, "", "", inputs);
  }

  NodeDef CreateNode(const string& name, const string& op,
                     const std::vector<string>& inputs) {
    return CreateNode(name, op, "", inputs);
  }

  NodeDef CreateNode(const string& name, const string& op, const string& frame,
                     const std::vector<string>& inputs) {
    NodeDef node;
    node.set_name(name);
    if (!op.empty()) {
      node.set_op(op);
    }
    if (!frame.empty()) {
      AttrValue frame_name;
      frame_name.set_s(frame);
      node.mutable_attr()->insert({"frame_name", frame_name});
    }
    for (const string& input : inputs) {
      node.add_input(input);
    }
    return node;
  }
};

TYPED_TEST_SUITE(FrameViewTest, GraphTypes);

template <typename T>
void InferFromGraph(FrameView* frame_view, GraphDef* graph, bool valid) {
  absl::Status status;
  T graph_view(graph, &status);
  TF_ASSERT_OK(status);
  status = frame_view->InferFromGraphView(graph_view);
  if (valid) {
    TF_ASSERT_OK(status);
  } else {
    ASSERT_FALSE(status.ok());
  }
}

template <>
void InferFromGraph<GraphDef>(FrameView* frame_view, GraphDef* graph,
                              bool valid) {
  absl::Status status = frame_view->InferFromGraph(*graph);
  if (valid) {
    TF_ASSERT_OK(status);
  } else {
    ASSERT_FALSE(status.ok());
  }
}

TYPED_TEST(FrameViewTest, NestedLoop) {
  GraphDef graph;
  // Create a two-level nested loop
  *graph.add_node() = this->CreateNode("0", {});
  *graph.add_node() = this->CreateNode("1", "Enter", "while/context1", {"0"});
  *graph.add_node() = this->CreateNode("2", {"1"});
  *graph.add_node() = this->CreateNode("3", "Merge", {"2", "14"});
  *graph.add_node() = this->CreateNode("4", {"3"});
  *graph.add_node() = this->CreateNode("5", "Switch", {"4"});
  *graph.add_node() = this->CreateNode("6", {"5"});
  *graph.add_node() = this->CreateNode("7", "Enter", "while/context2", {"6"});
  *graph.add_node() = this->CreateNode("8", {"7"});
  *graph.add_node() = this->CreateNode("9", "Merge", {"8", "12"});
  *graph.add_node() = this->CreateNode("10", {"9"});
  *graph.add_node() = this->CreateNode("11", "Switch", {"10"});
  *graph.add_node() = this->CreateNode("12", "NextIteration", {"11"});
  *graph.add_node() = this->CreateNode("13", "Exit", {"11"});
  *graph.add_node() = this->CreateNode("14", "NextIteration", {"13"});
  *graph.add_node() = this->CreateNode("15", {"5"});
  *graph.add_node() = this->CreateNode("16", "Exit", {"15"});
  *graph.add_node() = this->CreateNode("17", {"16"});

  FrameView frame_view;
  InferFromGraph<TypeParam>(&frame_view, &graph, /*valid=*/true);

  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}},      {"1", {0}},     {"2", {0}},     {"3", {0}},
      {"4", {0}},     {"5", {0}},     {"6", {0}},     {"7", {0, 1}},
      {"8", {0, 1}},  {"9", {0, 1}},  {"10", {0, 1}}, {"11", {0, 1}},
      {"12", {0, 1}}, {"13", {0, 1}}, {"14", {0}},    {"15", {0}},
      {"16", {0}},    {"17", {}}};

  EXPECT_EQ(frame_view.num_frames(), 2);
  for (const NodeDef& node : graph.node()) {
    std::vector<int> expected_frames = expected[node.name()];
    std::vector<int> node_frames = frame_view.Frames(node);
    EXPECT_EQ(expected_frames, node_frames);
  }
}

TYPED_TEST(FrameViewTest, MultipleInputsToEnter) {
  GraphDef graph;
  *graph.add_node() = this->CreateNode("0", {});
  *graph.add_node() = this->CreateNode("1", {});
  *graph.add_node() =
      this->CreateNode("2", "Enter", "while/context", {"0", "1"});
  *graph.add_node() = this->CreateNode("3", "Exit", {"2"});

  FrameView frame_view;
  InferFromGraph<TypeParam>(&frame_view, &graph, /*valid=*/true);

  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}}, {"1", {}}, {"2", {0}}, {"3", {0}}};

  EXPECT_EQ(frame_view.num_frames(), 1);
  for (const NodeDef& node : graph.node()) {
    std::vector<int> expected_frames = expected[node.name()];
    std::vector<int> node_frames = frame_view.Frames(node);
    EXPECT_EQ(expected_frames, node_frames);
  }
}

TYPED_TEST(FrameViewTest, ExitOutput) {
  GraphDef graph;
  *graph.add_node() = this->CreateNode("0", {});
  *graph.add_node() = this->CreateNode("1", "Enter", "while/context", {"0"});
  *graph.add_node() = this->CreateNode("2", "Exit", {"1"});
  *graph.add_node() = this->CreateNode("3", {});
  *graph.add_node() = this->CreateNode("4", {"2", "3"});

  FrameView frame_view;
  InferFromGraph<TypeParam>(&frame_view, &graph, /*valid=*/true);

  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}}, {"1", {0}}, {"2", {0}}, {"3", {}}, {"4", {}}};

  EXPECT_EQ(frame_view.num_frames(), 1);
  for (const NodeDef& node : graph.node()) {
    std::vector<int> expected_frames = expected[node.name()];
    std::vector<int> node_frames = frame_view.Frames(node);
    EXPECT_EQ(expected_frames, node_frames);
  }
}

TYPED_TEST(FrameViewTest, MultipleEnterNodes) {
  GraphDef graph;
  *graph.add_node() = this->CreateNode("0", {});
  *graph.add_node() = this->CreateNode("1", "Enter", "while/context", {"0"});
  *graph.add_node() = this->CreateNode("2", {"1"});
  *graph.add_node() = this->CreateNode("5", {});
  *graph.add_node() = this->CreateNode("4", "Enter", "while/context", {"5"});
  *graph.add_node() = this->CreateNode("3", {"4", "2"});
  *graph.add_node() = this->CreateNode("6", "Merge", {"3", "8"});
  *graph.add_node() = this->CreateNode("7", "Switch", {"6"});
  *graph.add_node() = this->CreateNode("8", "NextIteration", {"7"});
  *graph.add_node() = this->CreateNode("9", "Exit", {"7"});

  FrameView frame_view;
  InferFromGraph<TypeParam>(&frame_view, &graph, /*valid=*/true);

  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}}, {"1", {0}}, {"2", {0}}, {"3", {0}}, {"4", {0}},
      {"5", {}}, {"6", {0}}, {"7", {0}}, {"8", {0}}, {"9", {0}}};

  EXPECT_EQ(frame_view.num_frames(), 1);
  for (const NodeDef& node : graph.node()) {
    std::vector<int> expected_frames = expected[node.name()];
    std::vector<int> node_frames = frame_view.Frames(node);
    EXPECT_EQ(expected_frames, node_frames);
  }
}

TYPED_TEST(FrameViewTest, ConflictingFrames) {
  GraphDef graph;
  *graph.add_node() = this->CreateNode("0", {});
  *graph.add_node() = this->CreateNode("1", "Enter", "while/context1", {"0"});
  *graph.add_node() = this->CreateNode("2", "Enter", "while/context2", {"1"});
  *graph.add_node() = this->CreateNode("3", {"1", "2"});

  FrameView frame_view;
  InferFromGraph<TypeParam>(&frame_view, &graph, /*valid=*/false);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
