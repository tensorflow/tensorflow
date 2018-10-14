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
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

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

  void DisableAllStages(LoopOptimizer* optimizer) {
    LoopOptimizer::LoopOptimizerOptions options;
    options.enable_loop_invariant_node_motion = false;
    options.enable_stack_push_removal = false;
    optimizer->options_ = options;
  }

  void EnableOnlyLoopInvariantNodeMotion(LoopOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.enable_loop_invariant_node_motion = true;
  }

  void EnableOnlyStackPushRemoval(LoopOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.enable_stack_push_removal = true;
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

  LoopOptimizer optimizer;
  EnableOnlyLoopInvariantNodeMotion(&optimizer);
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

  LoopOptimizer optimizer;
  EnableOnlyLoopInvariantNodeMotion(&optimizer);
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

  LoopOptimizer optimizer;
  EnableOnlyLoopInvariantNodeMotion(&optimizer);
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

  LoopOptimizer optimizer;
  EnableOnlyLoopInvariantNodeMotion(&optimizer);
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

  LoopOptimizer optimizer;
  EnableOnlyLoopInvariantNodeMotion(&optimizer);
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

  LoopOptimizer optimizer;
  EnableOnlyLoopInvariantNodeMotion(&optimizer);
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

  LoopOptimizer optimizer;
  EnableOnlyLoopInvariantNodeMotion(&optimizer);
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
  EnableOnlyStackPushRemoval(&optimizer);
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
  EnableOnlyStackPushRemoval(&optimizer);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  VerifyGraphsEqual(item.graph, output, __FUNCTION__);
}

TEST_F(LoopOptimizerTest, RemovePush_NoPopButStackLives) {
  GrapplerItem item;
  GraphDef& graph = item.graph;
  AddSimpleNode("c", "Const", {}, &graph);
  // Stack with corresponding push
  AddSimpleNode("stack1", "StackV2", {}, &graph);
  AddSimpleNode("push1", "StackPushV2", {"stack1", "c"}, &graph);
  // Stack with corresponding push behind Enter.
  AddSimpleNode("stack2", "StackV2", {}, &graph);
  AddEnterNode("enter2_c", "frame_name", false, 1, {"c"}, &graph);
  AddEnterNode("enter2_stack2", "frame_name", false, 1, {"stack2"}, &graph);
  AddSimpleNode("push2", "StackPushV2", {"enter2_stack2", "enter2_c"}, &graph);
  item.keep_ops.push_back("stack1");
  item.keep_ops.push_back("stack2");

  LoopOptimizer optimizer;
  EnableOnlyStackPushRemoval(&optimizer);
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
  EnableOnlyStackPushRemoval(&optimizer);
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

TEST_F(LoopOptimizerTest, RemoveDeadBranches_ConstantCondition) {
  Scope scope = Scope::NewRootScope();
  Output v_in = ops::Variable(scope.WithOpName("v_in"), {3}, DT_FLOAT);

  Output ctrl1 = ops::Const(scope.WithOpName("ctrl1"), false, TensorShape({}));
  ops::Switch s1(scope.WithOpName("switch1"), v_in, ctrl1);
  Output square1 = ops::Square(scope.WithOpName("square1"), s1.output_false);
  Output sqrt1 = ops::Sqrt(scope.WithOpName("sqrt1"), s1.output_true);

  Output ctrl2 = ops::Const(scope.WithOpName("ctrl2"), true, TensorShape({}));
  ops::Switch s2(scope.WithOpName("switch2"), v_in, ctrl2);
  Output square2 = ops::Square(scope.WithOpName("square2"), s2.output_false);
  Output sqrt2 = ops::Sqrt(scope.WithOpName("sqrt2"), s2.output_true);

  Output ctrl3 = ops::Const(scope.WithOpName("ctrl3"), false, TensorShape({}));
  ops::Switch s3(scope.WithOpName("switch3"), v_in, ctrl3);
  Output square3 = ops::Square(scope.WithOpName("square3"), s3.output_false);
  Output sqrt3 = ops::Sqrt(scope.WithOpName("sqrt3"), s3.output_true);

  Output ctrl4 = ops::Const(scope.WithOpName("ctrl4"), false, TensorShape({}));
  ops::Switch s4(scope.WithOpName("switch4"), v_in, ctrl4);
  Output square4 = ops::Square(scope.WithOpName("square4"), s4.output_false);
  Output sqrt4 = ops::Sqrt(scope.WithOpName("sqrt4"), s4.output_true);

  ops::Merge m1(scope.WithOpName("m1"), {square1, sqrt1});
  ops::Merge m2(scope.WithOpName("m2"), {v_in, square1});
  ops::Merge m3(scope.WithOpName("m3"), {v_in, sqrt1});
  ops::Merge m4(scope.WithOpName("m4"), {square1, sqrt2});
  ops::Merge m5(scope.WithOpName("m5"), {square2, sqrt1});
  ops::Merge m6(scope.WithOpName("m6").WithControlDependencies(sqrt2),
                {v_in, square1});
  ops::Merge m7(scope.WithOpName("m7").WithControlDependencies(sqrt1),
                {v_in, square1});

  ops::Switch s5(scope.WithOpName("switch5"), v_in, ctrl1);
  Output id1 = ops::Identity(scope.WithOpName("id1"), s5.output_false);
  Output id2 = ops::Identity(scope.WithOpName("id2"), s5.output_true);
  ops::Merge m8(scope.WithOpName("m8"), {id1, id2});

  ops::Switch s6(scope.WithOpName("switch6"), v_in, ctrl1);
  Output id3 = ops::Identity(scope.WithOpName("id3"), s6.output_false);
  Output id4 = ops::Identity(scope.WithOpName("id4"), s6.output_true);
  ops::Merge m9(scope.WithOpName("m9"), {id3, id4});

  GrapplerItem item;
  item.fetch.push_back("m8");
  item.fetch.push_back("id4");

  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE, nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_CHECK_OK(status);

  for (const NodeDef& node : output.node()) {
    // These nodes should have been pruned
    EXPECT_NE("Square1", node.name());
    EXPECT_NE("Sqrt2", node.name());
    EXPECT_NE("m5", node.name());
    EXPECT_NE("m7", node.name());

    if (node.name() == "m1") {
      // sqrt1 is dead
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square1", node.input(0));
    } else if (node.name() == "m2") {
      // both inputs are alive
      EXPECT_EQ("Merge", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("v_in", node.input(0));
      EXPECT_EQ("square1", node.input(1));
    } else if (node.name() == "m3") {
      // sqrt1 is dead
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("v_in", node.input(0));
    } else if (node.name() == "m4") {
      // both inputs are alive
      EXPECT_EQ("Merge", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("square1", node.input(0));
      EXPECT_EQ("sqrt2", node.input(1));
    } else if (node.name() == "m6") {
      // both inputs are alive and the control dependency can get triggered
      EXPECT_EQ("Merge", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("v_in", node.input(0));
      EXPECT_EQ("square1", node.input(1));
      EXPECT_EQ("^sqrt2", node.input(2));
    } else if (node.name() == "m8") {
      // The node is to be preserved because of a fetch
      EXPECT_EQ("Merge", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("id1", node.input(0));
      EXPECT_EQ("id2", node.input(1));
    } else if (node.name() == "m9") {
      // The node is to be preserved because of a fetch
      EXPECT_EQ("Merge", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("id3", node.input(0));
      EXPECT_EQ("id4", node.input(1));
    }
  }
}

TEST_F(LoopOptimizerTest, RemoveDeadBranches_ZeroIterWhile) {
  const string gdef_ascii = R"EOF(
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 20
      }
    }
  }
}
node {
  name: "while/Enter"
  op: "Enter"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "frame_name"
    value {
      s: "while/while/"
    }
  }
  attr {
    key: "is_constant"
    value {
      b: false
    }
  }
  attr {
    key: "parallel_iterations"
    value {
      i: 1
    }
  }
}
node {
  name: "while/Merge"
  op: "Merge"
  input: "while/Enter"
  input: "while/NextIteration"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Less/y"
  op: "Const"
  input: "^while/Merge"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "while/Less"
  op: "Less"
  input: "while/Merge"
  input: "while/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/LoopCond"
  op: "LoopCond"
  input: "while/Less"
}
node {
  name: "while/Switch"
  op: "Switch"
  input: "while/Merge"
  input: "while/LoopCond"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@while/Merge"
      }
    }
  }
}
node {
  name: "while/Identity"
  op: "Identity"
  input: "while/Switch:1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/add/y"
  op: "Const"
  input: "^while/Identity"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "while/add"
  op: "Add"
  input: "while/Identity"
  input: "while/add/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/NextIteration"
  op: "NextIteration"
  input: "while/add"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Exit"
  op: "Exit"
  input: "while/Switch"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
versions {
  producer: 21
}
  )EOF";

  GrapplerItem item;
  CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &item.graph));
  item.fetch = {"while/Exit"};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(1, tensors_expected.size());

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE, nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_CHECK_OK(status);
  auto tensors_got = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors_got.size());
  test::ExpectTensorEqual<int32>(tensors_expected[0], tensors_got[0]);

  int nodes_present = 0;
  for (const NodeDef& node : output.node()) {
    // All nodes connected to Switch's positive check should be pruned.
    if (node.name() == "while/add") {
      LOG(ERROR) << "while/add is present after optimization";
    } else if (node.name() == "while/add/y") {
      LOG(ERROR) << "while/add/y is present after optimization";
    } else if (node.name() == "while/NextIteration") {
      LOG(ERROR) << "while/NextIteration is present after optimization";
    } else if (node.name() == "while/Identity") {
      LOG(ERROR) << "while/Identity is present after optimization";
    }
    ++nodes_present;
  }
  EXPECT_EQ(8, nodes_present);
}

}  // namespace grappler
}  // namespace tensorflow
