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

#include "tensorflow/core/grappler/optimizers/graph_optimizer_stage.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

class GraphOptimizerStageTest : public ::testing::Test {};

struct FakeResult {};

// NoOp optimizer stage that supports all the node types and does nothing
class FakeOptimizerStage : public GraphOptimizerStage<FakeResult> {
 public:
  explicit FakeOptimizerStage(const string& optimizer_name,
                              const string& stage_name,
                              const GraphOptimizerContext& ctx)
      : GraphOptimizerStage(optimizer_name, stage_name, ctx) {}
  ~FakeOptimizerStage() override = default;

  bool IsSupported(const NodeDef* node) const override { return true; }
  Status TrySimplify(NodeDef* node, FakeResult* result) override {
    return Status::OK();
  }
};

TEST_F(GraphOptimizerStageTest, ParseNodeNameAndScope_InRoot) {
  const auto scope_and_name = ParseNodeScopeAndName("Add");
  EXPECT_EQ("", scope_and_name.scope);
  EXPECT_EQ("Add", scope_and_name.name);
}

TEST_F(GraphOptimizerStageTest, ParseNodeNameAndScope_InScope) {
  const auto scope_and_name = ParseNodeScopeAndName("a/b/c/Add");
  EXPECT_EQ("a/b/c", scope_and_name.scope);
  EXPECT_EQ("Add", scope_and_name.name);
}

TEST_F(GraphOptimizerStageTest, OptimizedNodeName) {
  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ nullptr,
                            /*graph_properties*/ nullptr,
                            /*node_name*/ nullptr,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  const auto node = ParseNodeScopeAndName("a/b/c/Add");

  // Without rewrite rule
  EXPECT_EQ("a/b/c/my_opt/my_stg_Add", stage.OptimizedNodeName(node));
  EXPECT_EQ(
      "a/b/c/my_opt/my_stg_Add_Mul_Sqrt",
      stage.OptimizedNodeName(node, std::vector<string>({"Mul", "Sqrt"})));

  // With rewrite rule
  const string rewrite = "my_rewrite";
  EXPECT_EQ("a/b/c/my_opt/my_stg_my_rewrite_Add",
            stage.OptimizedNodeName(node, rewrite));
}

TEST_F(GraphOptimizerStageTest, GetInputNodeAndProperties) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto add = ops::Add(s.WithOpName("Add"), a, b);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(/*assume_valid_feeds*/ false));

  NodeMap node_map(&item.graph);

  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ &item.graph,
                            /*graph_properties*/ &properties,
                            /*node_name*/ &node_map,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  NodeDef* add_node;
  TF_CHECK_OK(stage.GetInputNode("Add", &add_node));
  EXPECT_EQ("a", add_node->input(0));
  EXPECT_EQ("b", add_node->input(1));

  OpInfo::TensorProperties add_properties;
  TF_CHECK_OK(stage.GetTensorProperties("Add", &add_properties));
  EXPECT_EQ(DT_FLOAT, add_properties.dtype());

  OpInfo::TensorProperties a_properties;
  TF_CHECK_OK(stage.GetTensorProperties("a:0", &a_properties));
  EXPECT_EQ(DT_FLOAT_REF, a_properties.dtype());

  OpInfo::TensorProperties b_properties;
  TF_CHECK_OK(stage.GetTensorProperties("b:0", &b_properties));
  EXPECT_EQ(DT_FLOAT_REF, b_properties.dtype());
}

TEST_F(GraphOptimizerStageTest, AddNodes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto add = ops::Add(s.WithOpName("Add"), a, b);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(/*assume_valid_feeds*/ false));

  NodeMap node_map(&item.graph);

  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ &item.graph,
                            /*graph_properties*/ &properties,
                            /*node_name*/ &node_map,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  NodeDef* add_node;
  TF_CHECK_OK(stage.GetInputNode("Add", &add_node));

  // Add a new copy node
  NodeDef* add_node_copy = stage.AddCopyNode("Add_1", add_node);
  EXPECT_EQ("Add_1", add_node_copy->name());
  EXPECT_EQ("Add", add_node_copy->op());
  EXPECT_EQ("a", add_node_copy->input(0));
  EXPECT_EQ("b", add_node_copy->input(1));

  // It must be available for by-name lookup
  NodeDef* add_node_copy_by_name;
  TF_CHECK_OK(stage.GetInputNode("Add_1", &add_node_copy_by_name));
  EXPECT_EQ(add_node_copy, add_node_copy_by_name);

  // Add new empty node
  NodeDef* empty_node = stage.AddEmptyNode("Add_2");
  EXPECT_EQ("Add_2", empty_node->name());

  // It must be available for by-name lookup
  NodeDef* empty_node_by_name;
  TF_CHECK_OK(stage.GetInputNode("Add_2", &empty_node_by_name));
  EXPECT_EQ(empty_node, empty_node_by_name);
}

}  // namespace
}  // end namespace grappler
}  // end namespace tensorflow
