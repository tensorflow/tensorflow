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

#include "tensorflow/core/grappler/optimizers/loop_optimizer.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class LoopOptimizerTest : public ::testing::Test {
 protected:
  static NodeDef CreateNode(const string& name,
                            const std::vector<string>& inputs) {
    return CreateNode(name, "Identity", "", false, 0, inputs);
  }
  static NodeDef CreateNode(const string& name, const string& op,
                            const std::vector<string>& inputs) {
    return CreateNode(name, op, "", false, 0, inputs);
  }
  static NodeDef CreateNode(const string& name, const string& op,
                            const string& frame,
                            const bool is_constant,
                            const int piterations,
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
    if (op == "Enter") {
      AttrValue is_const;
      is_const.set_b(is_constant);
      node.mutable_attr()->insert({"is_constant", is_const});
      AttrValue parallel_iterations;
      parallel_iterations.set_i(piterations);
      node.mutable_attr()->insert(
          {"parallel_iterations", parallel_iterations});
    }
    AttrValue type;
    type.set_type(DT_FLOAT);
    node.mutable_attr()->insert({"T", type});
    for (const string& input : inputs) {
      node.add_input(input);
    }
    return node;
  }
};

TEST_F(LoopOptimizerTest, Basic) {
  GraphDef graph;
  *graph.add_node() = CreateNode("0", {});
  *graph.add_node() = CreateNode(
      "InvariantEnter", "Enter", "while/while_context", true, 1, {"0"});
  *graph.add_node() = CreateNode(
      "InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"});
  *graph.add_node() = CreateNode(
      "VariantAdd", "Add", {"InvariantAdd", "Identity"});
  *graph.add_node() = CreateNode(
      "VariantEnter", "Enter", "while/while_context", false, 1, {"0"});
  *graph.add_node() = CreateNode(
      "Merge", "Merge", {"VariantEnter", "NextIteration"});
  *graph.add_node() = CreateNode("Less/y", "Const", {"^Identity"});
  *graph.add_node() = CreateNode("Less", "Less", {"VariantAdd", "less/y"});
  *graph.add_node() = CreateNode("LoopCond", "LoopCond", {"Less"});
  *graph.add_node() = CreateNode("Switch", "Switch", {"Merge", "LoopCond"});
  *graph.add_node() = CreateNode("Identity", {"Switch:1"});
  *graph.add_node() = CreateNode(
      "NextIteration", "NextIteration", {"VariantAdd"});
  *graph.add_node() = CreateNode("Exit", "Exit", {"Switch"});
  *graph.add_node() = CreateNode("1", {"Exit"});

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::unique_ptr<NodeMap> node_map;
  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;

  node_map.reset(new NodeMap(&graph));
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).back(), 0);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd")).back(), 0);

  node_map.reset(new NodeMap(&output));
  EXPECT_TRUE(IdentifyFrames(output, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).size(), 0);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd")).back(), 0);
}

TEST_F(LoopOptimizerTest, Const) {
  GraphDef graph;
  *graph.add_node() = CreateNode("0", {});
  *graph.add_node() = CreateNode(
      "InvariantEnter", "Enter", "while/while_context", true, 1, {"0"});
  *graph.add_node() = CreateNode("Const", "Const", {"^Identity"});
  *graph.add_node() = CreateNode(
      "InvariantAdd", "Add", {"InvariantEnter", "Const"});
  *graph.add_node() = CreateNode(
      "VariantAdd", "Add", {"InvariantAdd", "Identity"});
  *graph.add_node() = CreateNode(
      "VariantEnter", "Enter", "while/while_context", false, 1, {"0"});
  *graph.add_node() = CreateNode(
      "Merge", "Merge", {"VariantEnter", "NextIteration"});
  *graph.add_node() = CreateNode("Less/y", "Const", {"^Identity"});
  *graph.add_node() = CreateNode("Less", "Less", {"VariantAdd", "less/y"});
  *graph.add_node() = CreateNode("LoopCond", "LoopCond", {"Less"});
  *graph.add_node() = CreateNode("Switch", "Switch", {"Merge", "LoopCond"});
  *graph.add_node() = CreateNode("Identity", {"Switch:1"});
  *graph.add_node() = CreateNode(
      "NextIteration", "NextIteration", {"VariantAdd"});
  *graph.add_node() = CreateNode("Exit", "Exit", {"Switch"});
  *graph.add_node() = CreateNode("1", {"Exit"});

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::unique_ptr<NodeMap> node_map;
  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;

  node_map.reset(new NodeMap(&graph));
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).back(), 0);
  EXPECT_EQ(frames.at(node_map->GetNode("Const")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("Const")).back(), 0);

  node_map.reset(new NodeMap(&output));
  EXPECT_TRUE(IdentifyFrames(output, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).size(), 0);
  EXPECT_EQ(frames.at(node_map->GetNode("Const")).size(), 0);
}

TEST_F(LoopOptimizerTest, ControlOutput) {
  GraphDef graph;
  *graph.add_node() = CreateNode("0", {});
  *graph.add_node() = CreateNode(
      "InvariantEnter", "Enter", "while/while_context", true, 1, {"0"});
  *graph.add_node() = CreateNode(
      "InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"});
  *graph.add_node() = CreateNode(
      "VariantAdd", "Add", {"InvariantAdd", "Identity"});
  *graph.add_node() = CreateNode(
      "VariantEnter", "Enter", "while/while_context", false, 1, {"0"});
  *graph.add_node() = CreateNode(
      "Merge", "Merge", {"VariantEnter", "NextIteration"});
  *graph.add_node() = CreateNode("Less/y", "Const", {"^Identity"});
  *graph.add_node() = CreateNode(
      "Less", "Less", {"VariantAdd", "less/y", "^InvariantAdd"});
  *graph.add_node() = CreateNode("LoopCond", "LoopCond", {"Less"});
  *graph.add_node() = CreateNode("Switch", "Switch", {"Merge", "LoopCond"});
  *graph.add_node() = CreateNode("Identity", {"Switch:1"});
  *graph.add_node() = CreateNode(
      "NextIteration", "NextIteration", {"VariantAdd"});
  *graph.add_node() = CreateNode("Exit", "Exit", {"Switch"});
  *graph.add_node() = CreateNode("1", {"Exit"});

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::unique_ptr<NodeMap> node_map;
  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;

  node_map.reset(new NodeMap(&graph));
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).back(), 0);

  node_map.reset(new NodeMap(&output));
  EXPECT_TRUE(IdentifyFrames(output, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).back(), 0);
}

TEST_F(LoopOptimizerTest, NestedLoop1) {
  GraphDef graph;
  *graph.add_node() = CreateNode("0", {});
  *graph.add_node() = CreateNode(
      "InvariantEnter", "Enter", "while/while_context", true, 1, {"0"});
  *graph.add_node() = CreateNode(
      "InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"});
  *graph.add_node() = CreateNode(
      "VariantAdd", "Add", {"InvariantAdd", "Identity"});
  *graph.add_node() = CreateNode(
      "VariantEnter", "Enter", "while/while_context", false, 1, {"0"});
  *graph.add_node() = CreateNode(
      "Merge", "Merge", {"VariantEnter", "NextIteration"});
  *graph.add_node() = CreateNode("Less/y", "Const", {"^Identity"});
  *graph.add_node() = CreateNode("Less", "Less", {"Exit2", "less/y"});
  *graph.add_node() = CreateNode("LoopCond", "LoopCond", {"Less"});
  *graph.add_node() = CreateNode("Switch", "Switch", {"Merge", "LoopCond"});
  *graph.add_node() = CreateNode("Identity", {"Switch:1"});
  *graph.add_node() = CreateNode(
      "NextIteration", "NextIteration", {"Exit2"});
  *graph.add_node() = CreateNode("Exit", "Exit", {"Switch"});
  *graph.add_node() = CreateNode("1", {"Exit"});

  *graph.add_node() = CreateNode(
      "InvariantEnter2", "Enter", "while/while/while_context", true, 1,
      {"VariantAdd"});
  *graph.add_node() = CreateNode(
      "InvariantAdd2", "Add", {"InvariantEnter2", "InvariantEnter2"});
  *graph.add_node() = CreateNode(
      "VariantAdd2", "Add", {"InvariantAdd2", "Identity2"});
  *graph.add_node() = CreateNode(
      "VariantEnter2", "Enter", "while/while/while_context", false, 1,
      {"VariantEnter"});
  *graph.add_node() = CreateNode(
      "Merge2", "Merge", {"VariantEnter2", "NextIteration2"});
  *graph.add_node() = CreateNode("Less2/y", "Const", {"^Identity2"});
  *graph.add_node() = CreateNode("Less2", "Less", {"VariantAdd2", "less2/y"});
  *graph.add_node() = CreateNode("LoopCond2", "LoopCond", {"Less2"});
  *graph.add_node() = CreateNode("Switch2", "Switch", {"Merge2", "LoopCond2"});
  *graph.add_node() = CreateNode("Identity2", {"Switch2:1"});
  *graph.add_node() = CreateNode(
      "NextIteration2", "NextIteration", {"VariantAdd2"});
  *graph.add_node() = CreateNode("Exit2", "Exit", {"Switch2"});

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::unique_ptr<NodeMap> node_map;
  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;

  node_map.reset(new NodeMap(&graph));
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).size(), 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).back(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd2")).size(), 2);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd2")).back(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).back(), 0);

  node_map.reset(new NodeMap(&output));
  EXPECT_TRUE(IdentifyFrames(output, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).back(), 0);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd2")).size(), 2);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd2")).back(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd")).size(), 0);
}

TEST_F(LoopOptimizerTest, NestedLoop2) {
  GraphDef graph;
  *graph.add_node() = CreateNode("0", {});
  *graph.add_node() = CreateNode(
      "InvariantEnter", "Enter", "while/while_context", true, 1, {"0"});
  *graph.add_node() = CreateNode(
      "InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"});
  *graph.add_node() = CreateNode(
      "VariantAdd", "Add", {"InvariantAdd", "Identity"});
  *graph.add_node() = CreateNode(
      "VariantEnter", "Enter", "while/while_context", false, 1, {"0"});
  *graph.add_node() = CreateNode(
      "Merge", "Merge", {"VariantEnter", "NextIteration"});
  *graph.add_node() = CreateNode("Less/y", "Const", {"^Identity"});
  *graph.add_node() = CreateNode("Less", "Less", {"Exit2", "less/y"});
  *graph.add_node() = CreateNode("LoopCond", "LoopCond", {"Less"});
  *graph.add_node() = CreateNode("Switch", "Switch", {"Merge", "LoopCond"});
  *graph.add_node() = CreateNode("Identity", {"Switch:1"});
  *graph.add_node() = CreateNode(
      "NextIteration", "NextIteration", {"Exit2"});
  *graph.add_node() = CreateNode("Exit", "Exit", {"Switch"});
  *graph.add_node() = CreateNode("1", {"Exit"});

  *graph.add_node() = CreateNode(
      "InvariantEnter2", "Enter", "while/while/while_context", true, 1,
      {"InvariantAdd"});
  *graph.add_node() = CreateNode(
      "InvariantAdd2", "Add", {"InvariantEnter2", "InvariantEnter2"});
  *graph.add_node() = CreateNode(
      "VariantAdd2", "Add", {"InvariantAdd2", "Identity2"});
  *graph.add_node() = CreateNode(
      "VariantEnter2", "Enter", "while/while/while_context", false, 1,
      {"VariantEnter"});
  *graph.add_node() = CreateNode(
      "Merge2", "Merge", {"VariantEnter2", "NextIteration2"});
  *graph.add_node() = CreateNode("Less2/y", "Const", {"^Identity2"});
  *graph.add_node() = CreateNode("Less2", "Less", {"VariantAdd2", "less2/y"});
  *graph.add_node() = CreateNode("LoopCond2", "LoopCond", {"Less2"});
  *graph.add_node() = CreateNode("Switch2", "Switch", {"Merge2", "LoopCond2"});
  *graph.add_node() = CreateNode("Identity2", {"Switch2:1"});
  *graph.add_node() = CreateNode(
      "NextIteration2", "NextIteration", {"VariantAdd2"});
  *graph.add_node() = CreateNode("Exit2", "Exit", {"Switch2"});

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::unique_ptr<NodeMap> node_map;
  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;

  node_map.reset(new NodeMap(&graph));
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).size(), 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).back(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd2")).size(), 2);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd2")).back(), 1);

  node_map.reset(new NodeMap(&output));
  EXPECT_TRUE(IdentifyFrames(output, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).size(), 0);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd2")).size(), 2);
  EXPECT_EQ(frames.at(node_map->GetNode("VariantAdd2")).back(), 1);
}

TEST_F(LoopOptimizerTest, NestedLoopConst1) {
  GraphDef graph;
  *graph.add_node() = CreateNode("0", {});
  *graph.add_node() = CreateNode(
      "InvariantEnter", "Enter", "while/while_context", true, 1, {"0"});
  *graph.add_node() = CreateNode(
      "InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"});
  *graph.add_node() = CreateNode(
      "VariantAdd", "Add", {"InvariantAdd", "Identity"});
  *graph.add_node() = CreateNode(
      "VariantEnter", "Enter", "while/while_context", false, 1, {"0"});
  *graph.add_node() = CreateNode(
      "Merge", "Merge", {"VariantEnter", "NextIteration"});
  *graph.add_node() = CreateNode("Less/y", "Const", {"^Identity"});
  *graph.add_node() = CreateNode("Less", "Less", {"Exit2", "less/y"});
  *graph.add_node() = CreateNode("LoopCond", "LoopCond", {"Less"});
  *graph.add_node() = CreateNode("Switch", "Switch", {"Merge", "LoopCond"});
  *graph.add_node() = CreateNode("Identity", {"Switch:1"});
  *graph.add_node() = CreateNode(
      "NextIteration", "NextIteration", {"Exit2"});
  *graph.add_node() = CreateNode("Exit", "Exit", {"Switch"});
  *graph.add_node() = CreateNode("1", {"Exit"});

  *graph.add_node() = CreateNode(
      "InvariantEnter2", "Enter", "while/while/while_context", true, 1,
      {"VariantAdd"});
  *graph.add_node() = CreateNode("Const2", "Const", {"^Identity2"});
  *graph.add_node() = CreateNode(
      "InvariantAdd2", "Add", {"InvariantEnter2", "Const2"});
  *graph.add_node() = CreateNode(
      "VariantAdd2", "Add", {"InvariantAdd2", "Identity2"});
  *graph.add_node() = CreateNode(
      "VariantEnter2", "Enter", "while/while/while_context", false, 1,
      {"VariantEnter"});
  *graph.add_node() = CreateNode(
      "Merge2", "Merge", {"VariantEnter2", "NextIteration2"});
  *graph.add_node() = CreateNode("Less2/y", "Const", {"^Identity2"});
  *graph.add_node() = CreateNode("Less2", "Less", {"VariantAdd2", "less2/y"});
  *graph.add_node() = CreateNode("LoopCond2", "LoopCond", {"Less2"});
  *graph.add_node() = CreateNode("Switch2", "Switch", {"Merge2", "LoopCond2"});
  *graph.add_node() = CreateNode("Identity2", {"Switch2:1"});
  *graph.add_node() = CreateNode(
      "NextIteration2", "NextIteration", {"VariantAdd2"});
  *graph.add_node() = CreateNode("Exit2", "Exit", {"Switch2"});

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::unique_ptr<NodeMap> node_map;
  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;

  node_map.reset(new NodeMap(&graph));
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).size(), 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).back(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("Const2")).size(), 2);
  EXPECT_EQ(frames.at(node_map->GetNode("Const2")).back(), 1);

  node_map.reset(new NodeMap(&output));
  EXPECT_TRUE(IdentifyFrames(output, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).back(), 0);
  EXPECT_EQ(frames.at(node_map->GetNode("Const2")).size(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("Const2")).back(), 0);
}

TEST_F(LoopOptimizerTest, NestedLoopConst2) {
  GraphDef graph;
  *graph.add_node() = CreateNode("0", {});
  *graph.add_node() = CreateNode(
      "InvariantEnter", "Enter", "while/while_context", true, 1, {"0"});
  *graph.add_node() = CreateNode(
      "InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"});
  *graph.add_node() = CreateNode(
      "VariantAdd", "Add", {"InvariantAdd", "Identity"});
  *graph.add_node() = CreateNode(
      "VariantEnter", "Enter", "while/while_context", false, 1, {"0"});
  *graph.add_node() = CreateNode(
      "Merge", "Merge", {"VariantEnter", "NextIteration"});
  *graph.add_node() = CreateNode("Less/y", "Const", {"^Identity"});
  *graph.add_node() = CreateNode("Less", "Less", {"Exit2", "less/y"});
  *graph.add_node() = CreateNode("LoopCond", "LoopCond", {"Less"});
  *graph.add_node() = CreateNode("Switch", "Switch", {"Merge", "LoopCond"});
  *graph.add_node() = CreateNode("Identity", {"Switch:1"});
  *graph.add_node() = CreateNode(
      "NextIteration", "NextIteration", {"Exit2"});
  *graph.add_node() = CreateNode("Exit", "Exit", {"Switch"});
  *graph.add_node() = CreateNode("1", {"Exit"});

  *graph.add_node() = CreateNode(
      "InvariantEnter2", "Enter", "while/while/while_context", true, 1,
      {"InvariantAdd"});
  *graph.add_node() = CreateNode("Const2", "Const", {"^Identity2"});
  *graph.add_node() = CreateNode(
      "InvariantAdd2", "Add", {"InvariantEnter2", "Const2"});
  *graph.add_node() = CreateNode(
      "VariantAdd2", "Add", {"InvariantAdd2", "Identity2"});
  *graph.add_node() = CreateNode(
      "VariantEnter2", "Enter", "while/while/while_context", false, 1,
      {"VariantEnter"});
  *graph.add_node() = CreateNode(
      "Merge2", "Merge", {"VariantEnter2", "NextIteration2"});
  *graph.add_node() = CreateNode("Less2/y", "Const", {"^Identity2"});
  *graph.add_node() = CreateNode("Less2", "Less", {"VariantAdd2", "less2/y"});
  *graph.add_node() = CreateNode("LoopCond2", "LoopCond", {"Less2"});
  *graph.add_node() = CreateNode("Switch2", "Switch", {"Merge2", "LoopCond2"});
  *graph.add_node() = CreateNode("Identity2", {"Switch2:1"});
  *graph.add_node() = CreateNode(
      "NextIteration2", "NextIteration", {"VariantAdd2"});
  *graph.add_node() = CreateNode("Exit2", "Exit", {"Switch2"});

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::unique_ptr<NodeMap> node_map;
  std::unordered_map<const NodeDef*, std::vector<int>> frames;
  int num_frames;

  node_map.reset(new NodeMap(&graph));
  EXPECT_TRUE(IdentifyFrames(graph, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).size(), 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).back(), 1);
  EXPECT_EQ(frames.at(node_map->GetNode("Const2")).size(), 2);
  EXPECT_EQ(frames.at(node_map->GetNode("Const2")).back(), 1);

  node_map.reset(new NodeMap(&output));
  EXPECT_TRUE(IdentifyFrames(output, &frames, &num_frames).ok());
  EXPECT_EQ(num_frames, 2);
  EXPECT_EQ(frames.at(node_map->GetNode("InvariantAdd2")).size(), 0);
  EXPECT_EQ(frames.at(node_map->GetNode("Const2")).size(), 0);
}

void VerifyGraphsEqual(const GraphDef& original_graph,
                       const GraphDef& optimized_graph, const string& func) {
  EXPECT_EQ(original_graph.node_size(), optimized_graph.node_size()) << func;
  for (int i = 0; i < original_graph.node_size(); ++i) {
    const NodeDef& original = original_graph.node(i);
    const NodeDef& optimized = optimized_graph.node(i);
    EXPECT_EQ(original.name(), optimized.name()) << func;
    EXPECT_EQ(original.op(), optimized.op()) << func;
    EXPECT_EQ(original.input_size(), optimized.input_size()) << func;
    for (int j = 0; j < original.input_size(); ++j) {
      EXPECT_EQ(original.input(j), optimized.input(j)) << func;
    }
  }
}

TEST_F(LoopOptimizerTest, NoOp) {
  // This trivial graph is so basic there's nothing to optimize.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  VerifyGraphsEqual(item.graph, output, __FUNCTION__);
}

namespace {
NodeDef* AddNode(const string& name, const string& op,
                 const std::vector<string>& inputs,
                 const std::vector<std::pair<string, AttrValue>>& attributes,
                 GraphDef* graph) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op(op);
  for (const string& input : inputs) {
    node->add_input(input);
  }
  for (auto attr : attributes) {
    (*node->mutable_attr())[attr.first] = attr.second;
  }
  return node;
}
}  // namespace

TEST_F(LoopOptimizerTest, RemovePush_NoOp) {
  GrapplerItem item;
  AttrValue frame_name;
  frame_name.set_s("foo");
  AttrValue type;
  type.set_type(DT_RESOURCE);
  GraphDef& graph = item.graph;
  AddNode("c", "Const", {}, {}, &graph);
  // Stack with corresponding push/pop.
  AddNode("stack1", "StackV2", {}, {}, &graph);
  AddNode("push1", "StackPushV2", {"stack1", "c"}, {}, &graph);
  AddNode("pop1", "StackPopV2", {"stack1"}, {}, &graph);
  // Stack with corresponding push/pop behind Enter.
  AddNode("stack2", "StackV2", {}, {}, &graph);
  AddNode("push_enter", "Enter", {"stack2"},
          {{"T", type}, {"frame_name", frame_name}}, &graph);
  AddNode("push2", "StackPushV2", {"push_enter", "c"}, {}, &graph);
  AddNode("pop_enter", "Enter", {"stack2"},
          {{"T", type}, {"frame_name", frame_name}}, &graph);
  AddNode("pop2", "StackPopV2", {"pop_enter"}, {}, &graph);
  // Stack with unexpected op type in fanout of Stack.
  AddNode("stack3", "StackV2", {}, {}, &graph);
  AddNode("push3", "StackPushV2", {"stack3", "c"}, {}, &graph);
  AddNode("stop", "StopGradient", {"stack3"}, {}, &graph);
  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  VerifyGraphsEqual(item.graph, output, __FUNCTION__);
}

TEST_F(LoopOptimizerTest, RemovePushWithoutMatchingPop) {
  GrapplerItem item;
  GraphDef& graph = item.graph;
  AttrValue frame_name;
  frame_name.set_s("foo");
  AttrValue type;
  type.set_type(DT_RESOURCE);
  AddNode("c", "Const", {}, {}, &graph);
  AddNode("stack1", "StackV2", {}, {}, &graph);
  AddNode("push1", "StackPushV2", {"stack1", "c"}, {}, &graph);
  AddNode("stack2", "StackV2", {}, {}, &graph);
  AddNode("push_enter", "Enter", {"stack2"},
          {{"T", type}, {"frame_name", frame_name}}, &graph);
  AddNode("push2", "StackPushV2", {"push_enter", "c"}, {}, &graph);
  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_EQ(6, output.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "push1") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("c", node.input(0));
      EXPECT_EQ("^stack1", node.input(1));
    } else if (node.name() == "push2") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("c", node.input(0));
      EXPECT_EQ("^push_enter", node.input(1));
    } else {
      const NodeDef& orig_node = item.graph.node(i);
      EXPECT_EQ(orig_node.ShortDebugString(), node.ShortDebugString());
    }
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
