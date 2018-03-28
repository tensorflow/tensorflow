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
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class LoopOptimizerTest : public GrapplerTest {
 protected:
  // These helpers always sets T=DT_FLOAT.
  void AddEnterNode(const string& name, const string& frame,
                    const bool is_constant, const int piterations,
                    const std::vector<string>& inputs, GraphDef* graph) const {
    std::vector<std::pair<string, AttrValue>> attributes;
    AttrValue type;
    type.set_type(DT_FLOAT);
    attributes.emplace_back("T", type);
    AttrValue frame_name;
    frame_name.set_s(frame);
    attributes.emplace_back("frame_name", frame_name);
    AttrValue is_const;
    is_const.set_b(is_constant);
    attributes.emplace_back("is_constant", is_const);
    AttrValue parallel_iterations;
    parallel_iterations.set_i(piterations);
    attributes.emplace_back("parallel_iterations", parallel_iterations);
    AddNode(name, "Enter", inputs, attributes, graph);
  }

  void AddSimpleNode(const string& name, const string& op,
                     const std::vector<string>& inputs, GraphDef* graph) const {
    std::vector<std::pair<string, AttrValue>> attributes;
    AttrValue type;
    type.set_type(DT_FLOAT);
    attributes.emplace_back("T", type);
    AddNode(name, op, inputs, attributes, graph);
  }
};

TEST_F(LoopOptimizerTest, Basic) {
  GraphDef graph;
  AddSimpleNode("In", "Identity", {}, &graph);
  AddEnterNode("InvariantEnter", "while/while_context", true, 1, {"In"},
               &graph);
  AddSimpleNode("InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"},
                &graph);
  AddSimpleNode("VariantAdd", "Add", {"InvariantAdd", "Identity"}, &graph);
  AddEnterNode("VariantEnter", "while/while_context", false, 1, {"In"}, &graph);
  AddSimpleNode("Merge", "Merge", {"VariantEnter", "NextIteration"}, &graph);
  AddSimpleNode("Less/y", "Const", {"^Identity"}, &graph);
  AddSimpleNode("Less", "Less", {"VariantAdd", "Less/y"}, &graph);
  AddSimpleNode("LoopCond", "LoopCond", {"Less"}, &graph);
  AddSimpleNode("Switch", "Switch", {"Merge", "LoopCond"}, &graph);
  AddSimpleNode("Identity", "Identity", {"Switch:1"}, &graph);
  AddSimpleNode("NextIteration", "NextIteration", {"VariantAdd"}, &graph);
  AddSimpleNode("Exit", "Exit", {"Switch"}, &graph);
  AddSimpleNode("Out", "Identity", {"Exit"}, &graph);

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE);
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
  AddSimpleNode("In", "Identity", {}, &graph);
  AddEnterNode("InvariantEnter", "while/while_context", true, 1, {"In"},
               &graph);
  AddSimpleNode("Const", "Const", {"^Identity"}, &graph);
  AddSimpleNode("InvariantAdd", "Add", {"InvariantEnter", "Const"}, &graph);
  AddSimpleNode("VariantAdd", "Add", {"InvariantAdd", "Identity"}, &graph);
  AddEnterNode("VariantEnter", "while/while_context", false, 1, {"In"}, &graph);
  AddSimpleNode("Merge", "Merge", {"VariantEnter", "NextIteration"}, &graph);
  AddSimpleNode("Less/y", "Const", {"^Identity"}, &graph);
  AddSimpleNode("Less", "Less", {"VariantAdd", "Less/y"}, &graph);
  AddSimpleNode("LoopCond", "LoopCond", {"Less"}, &graph);
  AddSimpleNode("Switch", "Switch", {"Merge", "LoopCond"}, &graph);
  AddSimpleNode("Identity", "Identity", {"Switch:1"}, &graph);
  AddSimpleNode("NextIteration", "NextIteration", {"VariantAdd"}, &graph);
  AddSimpleNode("Exit", "Exit", {"Switch"}, &graph);
  AddSimpleNode("Out", "Identity", {"Exit"}, &graph);

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE);
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
  AddSimpleNode("In", "Identity", {}, &graph);
  AddEnterNode("InvariantEnter", "while/while_context", true, 1, {"In"},
               &graph);
  AddSimpleNode("InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"},
                &graph);
  AddSimpleNode("VariantAdd", "Add", {"InvariantAdd", "Identity"}, &graph);
  AddEnterNode("VariantEnter", "while/while_context", false, 1, {"In"}, &graph);
  AddSimpleNode("Merge", "Merge", {"VariantEnter", "NextIteration"}, &graph);
  AddSimpleNode("Less/y", "Const", {"^Identity"}, &graph);
  AddSimpleNode("Less", "Less", {"VariantAdd", "Less/y", "^InvariantAdd"},
                &graph);
  AddSimpleNode("LoopCond", "LoopCond", {"Less"}, &graph);
  AddSimpleNode("Switch", "Switch", {"Merge", "LoopCond"}, &graph);
  AddSimpleNode("Identity", "Identity", {"Switch:1"}, &graph);
  AddSimpleNode("NextIteration", "NextIteration", {"VariantAdd"}, &graph);
  AddSimpleNode("Exit", "Exit", {"Switch"}, &graph);
  AddSimpleNode("Out", "Identity", {"Exit"}, &graph);

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE);
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
  AddSimpleNode("In", "Identity", {}, &graph);
  AddEnterNode("InvariantEnter", "while/while_context", true, 1, {"In"},
               &graph);
  AddSimpleNode("InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"},
                &graph);
  AddSimpleNode("VariantAdd", "Add", {"InvariantAdd", "Identity"}, &graph);
  AddEnterNode("VariantEnter", "while/while_context", false, 1, {"In"}, &graph);
  AddSimpleNode("Merge", "Merge", {"VariantEnter", "NextIteration"}, &graph);
  AddSimpleNode("Less/y", "Const", {"^Identity"}, &graph);
  AddSimpleNode("Less", "Less", {"Exit2", "Less/y"}, &graph);
  AddSimpleNode("LoopCond", "LoopCond", {"Less"}, &graph);
  AddSimpleNode("Switch", "Switch", {"Merge", "LoopCond"}, &graph);
  AddSimpleNode("Identity", "Identity", {"Switch:1"}, &graph);
  AddSimpleNode("NextIteration", "NextIteration", {"Exit2"}, &graph);
  AddSimpleNode("Exit", "Exit", {"Switch"}, &graph);
  AddSimpleNode("Out", "Identity", {"Exit"}, &graph);

  AddEnterNode("InvariantEnter2", "while/while/while_context", true, 1,
               {"VariantAdd"}, &graph);
  AddSimpleNode("InvariantAdd2", "Add", {"InvariantEnter2", "InvariantEnter2"},
                &graph);
  AddSimpleNode("VariantAdd2", "Add", {"InvariantAdd2", "Identity2"}, &graph);
  AddEnterNode("VariantEnter2", "while/while/while_context", false, 1,
               {"VariantEnter"}, &graph);
  AddSimpleNode("Merge2", "Merge", {"VariantEnter2", "NextIteration2"}, &graph);
  AddSimpleNode("Less2/y", "Const", {"^Identity2"}, &graph);
  AddSimpleNode("Less2", "Less", {"VariantAdd2", "Less2/y"}, &graph);
  AddSimpleNode("LoopCond2", "LoopCond", {"Less2"}, &graph);
  AddSimpleNode("Switch2", "Switch", {"Merge2", "LoopCond2"}, &graph);
  AddSimpleNode("Identity2", "Identity", {"Switch2:1"}, &graph);
  AddSimpleNode("NextIteration2", "NextIteration", {"VariantAdd2"}, &graph);
  AddSimpleNode("Exit2", "Exit", {"Switch2"}, &graph);

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE);
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
  AddSimpleNode("In", "Identity", {}, &graph);
  AddEnterNode("InvariantEnter", "while/while_context", true, 1, {"In"},
               &graph);
  AddSimpleNode("InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"},
                &graph);
  AddSimpleNode("VariantAdd", "Add", {"InvariantAdd", "Identity"}, &graph);
  AddEnterNode("VariantEnter", "while/while_context", false, 1, {"In"}, &graph);
  AddSimpleNode("Merge", "Merge", {"VariantEnter", "NextIteration"}, &graph);
  AddSimpleNode("Less/y", "Const", {"^Identity"}, &graph);
  AddSimpleNode("Less", "Less", {"Exit2", "Less/y"}, &graph);
  AddSimpleNode("LoopCond", "LoopCond", {"Less"}, &graph);
  AddSimpleNode("Switch", "Switch", {"Merge", "LoopCond"}, &graph);
  AddSimpleNode("Identity", "Identity", {"Switch:1"}, &graph);
  AddSimpleNode("NextIteration", "NextIteration", {"Exit2"}, &graph);
  AddSimpleNode("Exit", "Exit", {"Switch"}, &graph);
  AddSimpleNode("Out", "Identity", {"Exit"}, &graph);

  AddEnterNode("InvariantEnter2", "while/while/while_context", true, 1,
               {"InvariantAdd"}, &graph);
  AddSimpleNode("InvariantAdd2", "Add", {"InvariantEnter2", "InvariantEnter2"},
                &graph);
  AddSimpleNode("VariantAdd2", "Add", {"InvariantAdd2", "Identity2"}, &graph);
  AddEnterNode("VariantEnter2", "while/while/while_context", false, 1,
               {"VariantEnter"}, &graph);
  AddSimpleNode("Merge2", "Merge", {"VariantEnter2", "NextIteration2"}, &graph);
  AddSimpleNode("Less2/y", "Const", {"^Identity2"}, &graph);
  AddSimpleNode("Less2", "Less", {"VariantAdd2", "Less2/y"}, &graph);
  AddSimpleNode("LoopCond2", "LoopCond", {"Less2"}, &graph);
  AddSimpleNode("Switch2", "Switch", {"Merge2", "LoopCond2"}, &graph);
  AddSimpleNode("Identity2", "Identity", {"Switch2:1"}, &graph);
  AddSimpleNode("NextIteration2", "NextIteration", {"VariantAdd2"}, &graph);
  AddSimpleNode("Exit2", "Exit", {"Switch2"}, &graph);

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE);
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
  AddSimpleNode("In", "Identity", {}, &graph);
  AddEnterNode("InvariantEnter", "while/while_context", true, 1, {"In"},
               &graph);
  AddSimpleNode("InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"},
                &graph);
  AddSimpleNode("VariantAdd", "Add", {"InvariantAdd", "Identity"}, &graph);
  AddEnterNode("VariantEnter", "while/while_context", false, 1, {"In"}, &graph);
  AddSimpleNode("Merge", "Merge", {"VariantEnter", "NextIteration"}, &graph);
  AddSimpleNode("Less/y", "Const", {"^Identity"}, &graph);
  AddSimpleNode("Less", "Less", {"Exit2", "Less/y"}, &graph);
  AddSimpleNode("LoopCond", "LoopCond", {"Less"}, &graph);
  AddSimpleNode("Switch", "Switch", {"Merge", "LoopCond"}, &graph);
  AddSimpleNode("Identity", "Identity", {"Switch:1"}, &graph);
  AddSimpleNode("NextIteration", "NextIteration", {"Exit2"}, &graph);
  AddSimpleNode("Exit", "Exit", {"Switch"}, &graph);
  AddSimpleNode("Out", "Identity", {"Exit"}, &graph);

  AddEnterNode("InvariantEnter2", "while/while/while_context", true, 1,
               {"VariantAdd"}, &graph);
  AddSimpleNode("Const2", "Const", {"^Identity2"}, &graph);
  AddSimpleNode("InvariantAdd2", "Add", {"InvariantEnter2", "Const2"}, &graph);
  AddSimpleNode("VariantAdd2", "Add", {"InvariantAdd2", "Identity2"}, &graph);
  AddEnterNode("VariantEnter2", "while/while/while_context", false, 1,
               {"VariantEnter"}, &graph);
  AddSimpleNode("Merge2", "Merge", {"VariantEnter2", "NextIteration2"}, &graph);
  AddSimpleNode("Less2/y", "Const", {"^Identity2"}, &graph);
  AddSimpleNode("Less2", "Less", {"VariantAdd2", "Less2/y"}, &graph);
  AddSimpleNode("LoopCond2", "LoopCond", {"Less2"}, &graph);
  AddSimpleNode("Switch2", "Switch", {"Merge2", "LoopCond2"}, &graph);
  AddSimpleNode("Identity2", "Identity", {"Switch2:1"}, &graph);
  AddSimpleNode("NextIteration2", "NextIteration", {"VariantAdd2"}, &graph);
  AddSimpleNode("Exit2", "Exit", {"Switch2"}, &graph);

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE);
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
  AddSimpleNode("In", "Identity", {}, &graph);
  AddEnterNode("InvariantEnter", "while/while_context", true, 1, {"In"},
               &graph);
  AddSimpleNode("InvariantAdd", "Add", {"InvariantEnter", "InvariantEnter"},
                &graph);
  AddSimpleNode("VariantAdd", "Add", {"InvariantAdd", "Identity"}, &graph);
  AddEnterNode("VariantEnter", "while/while_context", false, 1, {"In"}, &graph);
  AddSimpleNode("Merge", "Merge", {"VariantEnter", "NextIteration"}, &graph);
  AddSimpleNode("Less/y", "Const", {"^Identity"}, &graph);
  AddSimpleNode("Less", "Less", {"Exit2", "Less/y"}, &graph);
  AddSimpleNode("LoopCond", "LoopCond", {"Less"}, &graph);
  AddSimpleNode("Switch", "Switch", {"Merge", "LoopCond"}, &graph);
  AddSimpleNode("Identity", "Identity", {"Switch:1"}, &graph);
  AddSimpleNode("NextIteration", "NextIteration", {"Exit2"}, &graph);
  AddSimpleNode("Exit", "Exit", {"Switch"}, &graph);
  AddSimpleNode("Out", "Identity", {"Exit"}, &graph);

  AddEnterNode("InvariantEnter2", "while/while/while_context", true, 1,
               {"InvariantAdd"}, &graph);
  AddSimpleNode("Const2", "Const", {"^Identity2"}, &graph);
  AddSimpleNode("InvariantAdd2", "Add", {"InvariantEnter2", "Const2"}, &graph);
  AddSimpleNode("VariantAdd2", "Add", {"InvariantAdd2", "Identity2"}, &graph);
  AddEnterNode("VariantEnter2", "while/while/while_context", false, 1,
               {"VariantEnter"}, &graph);
  AddSimpleNode("Merge2", "Merge", {"VariantEnter2", "NextIteration2"}, &graph);
  AddSimpleNode("Less2/y", "Const", {"^Identity2"}, &graph);
  AddSimpleNode("Less2", "Less", {"VariantAdd2", "Less2/y"}, &graph);
  AddSimpleNode("LoopCond2", "LoopCond", {"Less2"}, &graph);
  AddSimpleNode("Switch2", "Switch", {"Merge2", "LoopCond2"}, &graph);
  AddSimpleNode("Identity2", "Identity", {"Switch2:1"}, &graph);
  AddSimpleNode("NextIteration2", "NextIteration", {"VariantAdd2"}, &graph);
  AddSimpleNode("Exit2", "Exit", {"Switch2"}, &graph);

  GrapplerItem item;
  item.graph = graph;

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE);
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

TEST_F(LoopOptimizerTest, RemovePush_NoOp) {
  GrapplerItem item;
  GraphDef& graph = item.graph;
  AddSimpleNode("c", "Const", {}, &graph);
  // Stack with corresponding push/pop.
  AddSimpleNode("stack1", "StackV2", {}, &graph);
  AddSimpleNode("push1", "StackPushV2", {"stack1", "c"}, &graph);
  AddSimpleNode("pop1", "StackPopV2", {"stack1"}, &graph);
  AddSimpleNode("id1", "Identity", {"pop1"}, &graph);
  // Stack with corresponding push/pop behind Enter.
  AddSimpleNode("stack2", "StackV2", {}, &graph);
  AddEnterNode("enter2_c", "frame_name", false, 1, {"c"}, &graph);
  AddEnterNode("enter2_stack2", "frame_name", false, 1, {"stack2"}, &graph);
  AddSimpleNode("push2", "StackPushV2", {"enter2_stack2", "enter2_c"}, &graph);
  AddSimpleNode("pop2", "StackPopV2", {"enter2_stack2"}, &graph);
  AddSimpleNode("id2", "Identity", {"pop2"}, &graph);
  // Stack with unexpected op type in fanout of Stack.
  AddSimpleNode("stack3", "StackV2", {}, &graph);
  AddSimpleNode("push3", "StackPushV2", {"stack3", "c"}, &graph);
  AddSimpleNode("stop", "StopGradient", {"stack3"}, &graph);

  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  VerifyGraphsEqual(item.graph, output, __FUNCTION__);
}

TEST_F(LoopOptimizerTest, RemovePushWithoutMatchingPop) {
  GrapplerItem item;
  GraphDef& graph = item.graph;
  AddSimpleNode("c", "Const", {}, &graph);
  // Push without Pop.
  AddSimpleNode("stack1", "StackV2", {}, &graph);
  AddSimpleNode("push1", "StackPushV2", {"stack1", "c"}, &graph);
  // Push without Pop behind Enter.
  AddSimpleNode("stack2", "StackV2", {}, &graph);
  AddEnterNode("enter_c", "frame_name", false, 1, {"c"}, &graph);
  AddEnterNode("enter_stack2", "frame_name", false, 1, {"stack2"}, &graph);
  AddSimpleNode("push2", "StackPushV2", {"enter_stack2", "enter_c"}, &graph);
  // Pop without consumer.
  AddSimpleNode("stack3", "StackV2", {}, &graph);
  AddSimpleNode("push3", "StackPushV2", {"stack3", "c"}, &graph);
  AddSimpleNode("pop3", "StackPopV2", {"stack3"}, &graph);
  // Push for a Pop without consumer that is fetched should not be removed.
  AddSimpleNode("stack4", "StackV2", {}, &graph);
  AddSimpleNode("push4", "StackPushV2", {"stack4", "c"}, &graph);
  AddSimpleNode("pop4", "StackPopV2", {"stack4"}, &graph);

  item.fetch.push_back("pop4");

  LoopOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(13, output.node_size());
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
      EXPECT_EQ("enter_c", node.input(0));
      EXPECT_EQ("^enter_stack2", node.input(1));
    } else if (node.name() == "push3") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("c", node.input(0));
      EXPECT_EQ("^stack3", node.input(1));
    } else {
      const NodeDef& orig_node = item.graph.node(i);
      EXPECT_EQ(orig_node.ShortDebugString(), node.ShortDebugString());
    }
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
