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
#include "tensorflow/core/grappler/utils/graph_view.h"
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

  void EnableOnlyLoopInvariantNodeMotion(LoopOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.enable_loop_invariant_node_motion = true;
  }

  void EnableOnlyStackPushRemoval(LoopOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.enable_stack_push_removal = true;
  }

 private:
  void DisableAllStages(LoopOptimizer* optimizer) {
    LoopOptimizer::LoopOptimizerOptions options;
    options.enable_loop_invariant_node_motion = false;
    options.enable_stack_push_removal = false;
    optimizer->options_ = options;
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
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  {  // Original graph.
    Status status;
    utils::GraphView view(&graph, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 1);
    const auto* invariant_add_node = view.GetNode("InvariantAdd");
    ASSERT_NE(invariant_add_node, nullptr);
    const auto* invariant_add_node_def = invariant_add_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*invariant_add_node_def).back(), 0);
    const auto* variant_add_node = view.GetNode("VariantAdd");
    ASSERT_NE(variant_add_node, nullptr);
    const auto* variant_add_node_def = variant_add_node->node();
    ASSERT_EQ(frames.Frames(*variant_add_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*variant_add_node_def).back(), 0);
  }

  {  // Optimized graph.
    Status status;
    utils::GraphView view(&output, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 1);
    const auto* invariant_add_node = view.GetNode("InvariantAdd");
    ASSERT_NE(invariant_add_node, nullptr);
    const auto* invariant_add_node_def = invariant_add_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_node_def).size(), 0);
    const auto* variant_add_node = view.GetNode("VariantAdd");
    ASSERT_NE(variant_add_node, nullptr);
    const auto* variant_add_node_def = variant_add_node->node();
    ASSERT_EQ(frames.Frames(*variant_add_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*variant_add_node_def).back(), 0);
  }
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
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  {  // Original graph.
    Status status;
    utils::GraphView view(&graph, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 1);
    const auto* invariant_add_node = view.GetNode("InvariantAdd");
    ASSERT_NE(invariant_add_node, nullptr);
    const auto* invariant_add_node_def = invariant_add_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*invariant_add_node_def).back(), 0);
    const auto* const_node = view.GetNode("Const");
    ASSERT_NE(const_node, nullptr);
    const auto* const_node_node_def = const_node->node();
    ASSERT_EQ(frames.Frames(*const_node_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*const_node_node_def).back(), 0);
  }

  {  // Optimized graph.
    Status status;
    utils::GraphView view(&output, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 1);
    const auto* invariant_add_node = view.GetNode("InvariantAdd");
    ASSERT_NE(invariant_add_node, nullptr);
    const auto* invariant_add_node_def = invariant_add_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_node_def).size(), 0);
    const auto* const_node = view.GetNode("Const");
    ASSERT_NE(const_node, nullptr);
    const auto* const_node_node_def = const_node->node();
    ASSERT_EQ(frames.Frames(*const_node_node_def).size(), 0);
  }
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
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  {  // Original graph.
    Status status;
    utils::GraphView view(&graph, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 1);
    const auto* invariant_add_node = view.GetNode("InvariantAdd");
    ASSERT_NE(invariant_add_node, nullptr);
    const auto* invariant_add_node_def = invariant_add_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*invariant_add_node_def).back(), 0);
  }

  {  // Optimized graph.
    Status status;
    utils::GraphView view(&output, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 1);
    const auto* invariant_add_node = view.GetNode("InvariantAdd");
    ASSERT_NE(invariant_add_node, nullptr);
    const auto* invariant_add_node_def = invariant_add_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*invariant_add_node_def).back(), 0);
  }
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
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  {  // Original graph.
    Status status;
    utils::GraphView view(&graph, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 2);
    const auto* invariant_add_2_node = view.GetNode("InvariantAdd2");
    ASSERT_NE(invariant_add_2_node, nullptr);
    const auto* invariant_add_2_node_def = invariant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_2_node_def).size(), 2);
    EXPECT_EQ(frames.Frames(*invariant_add_2_node_def).back(), 1);
    const auto* variant_add_2_node = view.GetNode("VariantAdd2");
    ASSERT_NE(variant_add_2_node, nullptr);
    const auto* variant_add_2_node_def = variant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*variant_add_2_node_def).size(), 2);
    EXPECT_EQ(frames.Frames(*variant_add_2_node_def).back(), 1);
    const auto* invariant_add_node = view.GetNode("InvariantAdd");
    ASSERT_NE(invariant_add_node, nullptr);
    const auto* invariant_add_node_def = invariant_add_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*invariant_add_node_def).back(), 0);
  }

  {  // Optimized graph.
    Status status;
    utils::GraphView view(&output, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 2);
    const auto* invariant_add_2_node = view.GetNode("InvariantAdd2");
    ASSERT_NE(invariant_add_2_node, nullptr);
    const auto* invariant_add_2_node_def = invariant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_2_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*invariant_add_2_node_def).back(), 0);
    const auto* variant_add_2_node = view.GetNode("VariantAdd2");
    ASSERT_NE(variant_add_2_node, nullptr);
    const auto* variant_add_2_node_def = variant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*variant_add_2_node_def).size(), 2);
    EXPECT_EQ(frames.Frames(*variant_add_2_node_def).back(), 1);
    const auto* invariant_add_node = view.GetNode("InvariantAdd");
    ASSERT_NE(invariant_add_node, nullptr);
    const auto* invariant_add_node_def = invariant_add_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_node_def).size(), 0);
  }
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
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  {  // Original graph.
    Status status;
    utils::GraphView view(&graph, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 2);
    const auto* invariant_add_2_node = view.GetNode("InvariantAdd2");
    ASSERT_NE(invariant_add_2_node, nullptr);
    const auto* invariant_add_2_node_def = invariant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_2_node_def).size(), 2);
    EXPECT_EQ(frames.Frames(*invariant_add_2_node_def).back(), 1);
    const auto* variant_add_2_node = view.GetNode("VariantAdd2");
    ASSERT_NE(variant_add_2_node, nullptr);
    const auto* variant_add_2_node_def = variant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*variant_add_2_node_def).size(), 2);
    EXPECT_EQ(frames.Frames(*variant_add_2_node_def).back(), 1);
  }

  {  // Optimized graph.
    Status status;
    utils::GraphView view(&output, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 2);
    const auto* invariant_add_2_node = view.GetNode("InvariantAdd2");
    ASSERT_NE(invariant_add_2_node, nullptr);
    const auto* invariant_add_2_node_def = invariant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_2_node_def).size(), 0);
    const auto* variant_add_2_node = view.GetNode("VariantAdd2");
    ASSERT_NE(variant_add_2_node, nullptr);
    const auto* variant_add_2_node_def = variant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*variant_add_2_node_def).size(), 2);
    EXPECT_EQ(frames.Frames(*variant_add_2_node_def).back(), 1);
  }
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
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  {  // Original graph.
    Status status;
    utils::GraphView view(&graph, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 2);
    const auto* invariant_add_2_node = view.GetNode("InvariantAdd2");
    ASSERT_NE(invariant_add_2_node, nullptr);
    const auto* invariant_add_2_node_def = invariant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_2_node_def).size(), 2);
    EXPECT_EQ(frames.Frames(*invariant_add_2_node_def).back(), 1);
    const auto* const_2_node = view.GetNode("Const2");
    ASSERT_NE(const_2_node, nullptr);
    const auto* const_2_node_def = const_2_node->node();
    ASSERT_EQ(frames.Frames(*const_2_node_def).size(), 2);
    EXPECT_EQ(frames.Frames(*const_2_node_def).back(), 1);
  }

  {  // Optimized graph.
    Status status;
    utils::GraphView view(&output, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 2);
    const auto* invariant_add_2_node = view.GetNode("InvariantAdd2");
    ASSERT_NE(invariant_add_2_node, nullptr);
    const auto* invariant_add_2_node_def = invariant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_2_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*invariant_add_2_node_def).back(), 0);
    const auto* const_2_node = view.GetNode("Const2");
    ASSERT_NE(const_2_node, nullptr);
    const auto* const_2_node_def = const_2_node->node();
    ASSERT_EQ(frames.Frames(*const_2_node_def).size(), 1);
    EXPECT_EQ(frames.Frames(*const_2_node_def).back(), 0);
  }
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
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  {  // Original graph.
    Status status;
    utils::GraphView view(&graph, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 2);
    const auto* invariant_add_2_node = view.GetNode("InvariantAdd2");
    ASSERT_NE(invariant_add_2_node, nullptr);
    const auto* invariant_add_2_node_def = invariant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_2_node_def).size(), 2);
    EXPECT_EQ(frames.Frames(*invariant_add_2_node_def).back(), 1);
    const auto* const_2_node = view.GetNode("Const2");
    ASSERT_NE(const_2_node, nullptr);
    const auto* const_2_node_def = const_2_node->node();
    ASSERT_EQ(frames.Frames(*const_2_node_def).size(), 2);
    EXPECT_EQ(frames.Frames(*const_2_node_def).back(), 1);
  }

  {  // Optimized graph.
    Status status;
    utils::GraphView view(&output, &status);
    TF_ASSERT_OK(status);
    FrameView frames;
    TF_EXPECT_OK(frames.InferFromGraphView(view));

    EXPECT_EQ(frames.num_frames(), 2);
    const auto* invariant_add_2_node = view.GetNode("InvariantAdd2");
    ASSERT_NE(invariant_add_2_node, nullptr);
    const auto* invariant_add_2_node_def = invariant_add_2_node->node();
    ASSERT_EQ(frames.Frames(*invariant_add_2_node_def).size(), 0);
    const auto* const_2_node = view.GetNode("Const2");
    ASSERT_NE(const_2_node, nullptr);
    const auto* const_2_node_def = const_2_node->node();
    ASSERT_EQ(frames.Frames(*const_2_node_def).size(), 0);
  }
}

void VerifyGraphsEqual(const GraphDef& original_graph,
                       const GraphDef& optimized_graph, const string& func) {
  EXPECT_EQ(original_graph.node_size(), optimized_graph.node_size()) << func;
  for (int i = 0; i < original_graph.node_size(); ++i) {
    const NodeDef& original = original_graph.node(i);
    const NodeDef& optimized = optimized_graph.node(i);
    EXPECT_EQ(optimized.name(), original.name()) << func;
    EXPECT_EQ(optimized.op(), original.op()) << func;
    ASSERT_EQ(optimized.input_size(), original.input_size()) << func;
    for (int j = 0; j < original.input_size(); ++j) {
      EXPECT_EQ(optimized.input(j), original.input(j)) << func;
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

TEST_F(LoopOptimizerTest, RemovePushNoOp) {
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

TEST_F(LoopOptimizerTest, RemovePushNoPopButStackLives) {
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

  EXPECT_EQ(output.node_size(), 13);
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "push1") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "c");
      EXPECT_EQ(node.input(1), "^stack1");
    } else if (node.name() == "push2") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "enter_c");
      EXPECT_EQ(node.input(1), "^enter_stack2");
    } else if (node.name() == "push3") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "c");
      EXPECT_EQ(node.input(1), "^stack3");
    } else {
      const NodeDef& orig_node = item.graph.node(i);
      EXPECT_EQ(node.ShortDebugString(), orig_node.ShortDebugString());
    }
  }
}

TEST_F(LoopOptimizerTest, RemoveDeadBranchesConstantCondition) {
  Scope scope = Scope::NewRootScope();
  Output v_in = ops::Const<float>(scope.WithOpName("v_in"), {123.0}, {});

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
    EXPECT_NE(node.name(), "Square1");
    EXPECT_NE(node.name(), "Sqrt2");
    EXPECT_NE(node.name(), "m5");

    if (node.name() == "m1") {
      // sqrt1 is dead
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "square1");
    } else if (node.name() == "m2") {
      // both inputs are alive
      EXPECT_EQ(node.op(), "Merge");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "v_in");
      EXPECT_EQ(node.input(1), "square1");
    } else if (node.name() == "m3") {
      // sqrt1 is dead
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "v_in");
    } else if (node.name() == "m4") {
      // both inputs are alive
      EXPECT_EQ(node.op(), "Merge");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "square1");
      EXPECT_EQ(node.input(1), "sqrt2");
    } else if (node.name() == "m8") {
      // The node is to be preserved because of a fetch
      EXPECT_EQ(node.op(), "Merge");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "id1");
      EXPECT_EQ(node.input(1), "id2");
    } else if (node.name() == "m9") {
      // The node is to be preserved because of a fetch
      EXPECT_EQ(node.op(), "Merge");
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ(node.input(0), "id3");
      EXPECT_EQ(node.input(1), "id4");
    }
  }

  auto tensors_expected = EvaluateNodes(item.graph, {"m8", "m9"});
  ASSERT_EQ(tensors_expected.size(), 2);

  auto tensors = EvaluateNodes(output, {"m8", "m9"});
  ASSERT_EQ(tensors.size(), 2);

  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
  test::ExpectTensorNear<float>(tensors_expected[1], tensors[1], 1e-6);
}

TEST_F(LoopOptimizerTest, RemoveDeadBranchesFullyRemoveDeadBranches) {
  const string gdef_ascii = R"EOF(
node {
  name: "episodicreplaybuffer_add_readvariableop_resource"
  op: "_Arg"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "index"
    value {
      i: 0
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/and_1/x"
  op: "Const"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_BOOL
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_BOOL
        tensor_shape {
        }
        bool_val: true
      }
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/begin_episode"
  op: "Const"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_BOOL
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_BOOL
        tensor_shape {
        }
        bool_val: false
      }
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Switch"
  op: "Switch"
  input: "EpisodicReplayBuffer/add/and_1/x"
  input: "EpisodicReplayBuffer/add/and_1/x"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/NoOp"
  op: "NoOp"
  input: "^EpisodicReplayBuffer/add/and_1/x"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
}
node {
  name: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Assert/Switch"
  op: "Switch"
  input: "EpisodicReplayBuffer/add/and_1/x"
  input: "EpisodicReplayBuffer/add/and_1/x"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@EpisodicReplayBuffer/add/assert_equal/All"
      }
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Assert/Switch_1"
  op: "Switch"
  input: "EpisodicReplayBuffer/add/begin_episode"
  input: "EpisodicReplayBuffer/add/and_1/x"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@EpisodicReplayBuffer/add/begin_episode"
      }
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Assert/Switch_2"
  op: "Switch"
  input: "EpisodicReplayBuffer/add/begin_episode"
  input: "EpisodicReplayBuffer/add/and_1/x"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@EpisodicReplayBuffer/add/end_episode"
      }
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/switch_f"
  op: "Identity"
  input: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Switch"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/control_dependency"
  op: "Const"
  input: "^EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/NoOp"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_BOOL
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_BOOL
        tensor_shape {
        }
        tensor_content: "\001"
      }
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Assert"
  op: "Assert"
  input: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Assert/Switch"
  input: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Assert/Switch_1"
  input: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Assert/Switch_2"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      list {
        type: DT_BOOL
        type: DT_BOOL
      }
    }
  }
  attr {
    key: "summarize"
    value {
      i: 3
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/control_dependency_1"
  op: "Identity"
  input: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/switch_f"
  input: "^EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Assert"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/switch_f"
      }
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Merge"
  op: "Merge"
  input: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/control_dependency_1"
  input: "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/control_dependency"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/FloorMod/y"
  op: "Const"
  input: "^EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Merge"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT64
        tensor_shape {
        }
        int64_val: 5000
      }
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/ReadVariableOp"
  op: "ReadVariableOp"
  input: "episodicreplaybuffer_add_readvariableop_resource"
  input: "^EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Merge"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/Less/y"
  op: "Const"
  input: "^EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Merge"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT64
        tensor_shape {
        }
        int64_val: 0
      }
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/Less"
  op: "Less"
  input: "EpisodicReplayBuffer/add/ReadVariableOp"
  input: "EpisodicReplayBuffer/add/Less/y"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/or"
  op: "LogicalOr"
  input: "EpisodicReplayBuffer/add/begin_episode"
  input: "EpisodicReplayBuffer/add/Less"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
}
node {
  name: "EpisodicReplayBuffer/add/get_episode_id/pred_id"
  op: "Identity"
  input: "EpisodicReplayBuffer/add/or"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/get_episode_id/Switch"
  op: "Switch"
  input: "EpisodicReplayBuffer/add/or"
  input: "EpisodicReplayBuffer/add/or"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/get_episode_id/critical_section_execute/AssignVariableOp/Switch"
  op: "Switch"
  input: "episodicreplaybuffer_add_readvariableop_resource"
  input: "EpisodicReplayBuffer/add/get_episode_id/pred_id"
  input: "^EpisodicReplayBuffer/add/ReadVariableOp"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@EpisodicReplayBuffer/add/ReadVariableOp/resource"
      }
    }
  }
}
node {
  name: "EpisodicReplayBuffer/add/get_episode_id/critical_section_execute/ReadVariableOp_3"
  op: "ReadVariableOp"
  input: "^EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Merge"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
library {
}
versions {
  producer: 27
}
  )EOF";

  GrapplerItem item;
  CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &item.graph));
  item.fetch = {
      "EpisodicReplayBuffer/add/get_episode_id/critical_section_execute/"
      "ReadVariableOp_3"};

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE, nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_CHECK_OK(status);

  bool found_merge = false;
  for (const auto& node : output.node()) {
    if (node.name() ==
        "EpisodicReplayBuffer/add/assert_equal/Assert/AssertGuard/Merge") {
      found_merge = true;
    }
  }

  EXPECT_TRUE(found_merge)
      << "Merge node was deleted, but it shouldn't have been.";
}

TEST_F(LoopOptimizerTest, RemoveDeadBranchesZeroIterWhile) {
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
  ASSERT_EQ(tensors_expected.size(), 1);

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE, nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_CHECK_OK(status);
  auto tensors_got = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors_got.size(), 1);
  test::ExpectTensorEqual<int32>(tensors_got[0], tensors_expected[0]);

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
  EXPECT_EQ(nodes_present, 8);
}

TEST_F(LoopOptimizerTest, RemoveDeadBranchesConstantFeed) {
  const string gdef_ascii = R"EOF(
node {
  name: "Const"
  op: "Const"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "I\'m a value!"
      }
    }
  }
}
node {
  name: "cond/Switch_1"
  op: "Switch"
  input: "Const"
  input: "Const_1"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Const"
      }
    }
  }
}
node {
  name: "Const_1"
  op: "Const"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_BOOL
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_BOOL
        tensor_shape {
        }
        bool_val: true
      }
    }
  }
}
node {
  name: "cond/Switch"
  op: "Switch"
  input: "Const_1"
  input: "Const_1"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "cond/switch_t"
  op: "Identity"
  input: "cond/Switch:1"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "cond/Const"
  op: "Const"
  input: "^cond/switch_t"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "cond/Merge"
  op: "Merge"
  input: "cond/Switch_1"
  input: "cond/Const"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
}
node {
  name: "Identity"
  op: "Identity"
  input: "cond/Merge"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
}
library {
}
versions {
  producer: 27
}
  )EOF";

  GrapplerItem item;
  CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &item.graph));
  item.fetch = {"Identity"};
  Tensor feed_tensor(DT_BOOL, {});
  feed_tensor.flat<bool>()(0) = false;
  item.feed.push_back({"Const_1", feed_tensor});
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  LoopOptimizer optimizer(RewriterConfig::AGGRESSIVE, nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_CHECK_OK(status);
  auto tensors_got = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors_got.size(), 1);
  test::ExpectTensorEqual<tstring>(tensors_got[0], tensors_expected[0]);

  EXPECT_EQ(output.node_size(), 8);

  // No rewrite because branch has a constant feed node.
  bool found = false;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "cond/Merge") {
      EXPECT_EQ(node.op(), "Merge");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "cond/Switch_1");
      EXPECT_EQ(node.input(1), "cond/Const");
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

}  // namespace grappler
}  // namespace tensorflow
