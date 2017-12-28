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
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class IdentifyFramesTest : public ::testing::Test {
 protected:
  static NodeDef CreateNode(const string& name,
                            const std::vector<string>& inputs) {
    return CreateNode(name, "", "", inputs);
  }
  static NodeDef CreateNode(const string& name, const string& op,
                            const std::vector<string>& inputs) {
    return CreateNode(name, op, "", inputs);
  }
  static NodeDef CreateNode(const string& name, const string& op,
                            const string& frame,
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

TEST_F(IdentifyFramesTest, NestedLoop) {
  GraphDef graph;
  // Create a two-level nested loop
  *graph.add_node() = CreateNode("0", {});
  *graph.add_node() =
      CreateNode("1", "Enter", "map/while/while_context1", {"0"});
  *graph.add_node() = CreateNode("2", {"1"});
  *graph.add_node() = CreateNode("3", "Merge", {"2", "14"});
  *graph.add_node() = CreateNode("4", {"3"});
  *graph.add_node() = CreateNode("5", "Switch", {"4"});
  *graph.add_node() = CreateNode("6", {"5"});
  *graph.add_node() =
      CreateNode("7", "Enter", "map/while/while_context2", {"6"});
  *graph.add_node() = CreateNode("8", {"7"});
  *graph.add_node() = CreateNode("9", "Merge", {"8", "12"});
  *graph.add_node() = CreateNode("10", {"9"});
  *graph.add_node() = CreateNode("11", "Switch", {"10"});
  *graph.add_node() = CreateNode("12", "NextIteration", {"11"});
  *graph.add_node() = CreateNode("13", "Exit", {"11"});
  *graph.add_node() = CreateNode("14", "NextIteration", {"13"});
  *graph.add_node() = CreateNode("15", {"5"});
  *graph.add_node() = CreateNode("16", "Exit", {"15"});
  *graph.add_node() = CreateNode("17", {"16"});

  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}},      {"1", {0}},     {"2", {0}},     {"3", {0}},
      {"4", {0}},     {"5", {0}},     {"6", {0}},     {"7", {0, 1}},
      {"8", {0, 1}},  {"9", {0, 1}},  {"10", {0, 1}}, {"11", {0, 1}},
      {"12", {0, 1}}, {"13", {0, 1}}, {"14", {0}},    {"15", {0}},
      {"16", {0}},    {"17", {}}};
  EXPECT_EQ(num_frames, 2);
  EXPECT_EQ(frames.size(), expected.size());
  std::cout << "Number of frame: " << num_frames << std::endl;
  for (const auto& node : frames) {
    std::cout << node.first->name() << ": ";
    EXPECT_EQ(node.second.size(), expected[node.first->name()].size());
    for (int i = 0; i < node.second.size(); i++) {
      EXPECT_EQ(expected[node.first->name()][i], node.second[i]);
      std::cout << node.second[i] << " ";
    }
    std::cout << std::endl;
  }
}

TEST_F(IdentifyFramesTest, MultipleInputsToEnter) {
  GraphDef graph;
  *graph.add_node() = CreateNode("0", {});
  *graph.add_node() = CreateNode("1", {});
  *graph.add_node() =
      CreateNode("2", "Enter", "map/while/while_context", {"0", "1"});
  *graph.add_node() = CreateNode("3", "Exit", {"2"});

  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}}, {"1", {}}, {"2", {0}}, {"3", {0}}};
  EXPECT_EQ(num_frames, 1);
  EXPECT_EQ(frames.size(), expected.size());
  std::cout << "Number of frame: " << num_frames << std::endl;
  for (const auto& node : frames) {
    std::cout << node.first->name() << ": ";
    EXPECT_EQ(node.second.size(), expected[node.first->name()].size());
    for (int i = 0; i < node.second.size(); i++) {
      EXPECT_EQ(expected[node.first->name()][i], node.second[i]);
      std::cout << node.second[i] << " ";
    }
    std::cout << std::endl;
  }
}

TEST_F(IdentifyFramesTest, ExitOutput) {
  GraphDef graph;
  *graph.add_node() = CreateNode("0", {});
  *graph.add_node() =
      CreateNode("1", "Enter", "map/while/while_context", {"0"});
  *graph.add_node() = CreateNode("2", "Exit", {"1"});
  *graph.add_node() = CreateNode("3", {});
  *graph.add_node() = CreateNode("4", {"2", "3"});

  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}}, {"1", {0}}, {"2", {0}}, {"3", {}}, {"4", {}}};
  EXPECT_EQ(num_frames, 1);
  EXPECT_EQ(frames.size(), expected.size());
  std::cout << "Number of frame: " << num_frames << std::endl;
  for (const auto& node : frames) {
    std::cout << node.first->name() << ": ";
    EXPECT_EQ(node.second.size(), expected[node.first->name()].size());
    for (int i = 0; i < node.second.size(); i++) {
      EXPECT_EQ(expected[node.first->name()][i], node.second[i]);
      std::cout << node.second[i] << " ";
    }
    std::cout << std::endl;
  }
}

TEST_F(IdentifyFramesTest, MultipleEnterNodes) {
  GraphDef graph;
  *graph.add_node() = CreateNode("0", {});
  string frame = "map/while/while_context";
  *graph.add_node() = CreateNode("1", "Enter", frame, {"0"});
  *graph.add_node() = CreateNode("2", {"1"});
  *graph.add_node() = CreateNode("5", {});
  *graph.add_node() = CreateNode("4", "Enter", frame, {"5"});
  *graph.add_node() = CreateNode("3", {"4", "2"});
  *graph.add_node() = CreateNode("6", "Merge", {"3", "8"});
  *graph.add_node() = CreateNode("7", "Switch", {"6"});
  *graph.add_node() = CreateNode("8", "NextIteration", {"7"});
  *graph.add_node() = CreateNode("9", "Exit", {"7"});

  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}}, {"1", {0}}, {"2", {0}}, {"3", {0}}, {"4", {0}},
      {"5", {}}, {"6", {0}}, {"7", {0}}, {"8", {0}}, {"9", {0}}};
  EXPECT_EQ(num_frames, 1);
  EXPECT_EQ(frames.size(), expected.size());
  std::cout << "Number of frame: " << num_frames << std::endl;
  for (const auto& node : frames) {
    std::cout << node.first->name() << ": ";
    EXPECT_EQ(node.second.size(), expected[node.first->name()].size());
    for (int i = 0; i < node.second.size(); i++) {
      EXPECT_EQ(expected[node.first->name()][i], node.second[i]);
      std::cout << node.second[i] << " ";
    }
    std::cout << std::endl;
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
