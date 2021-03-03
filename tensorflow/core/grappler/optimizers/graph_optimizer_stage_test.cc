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
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;

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

TEST_F(GraphOptimizerStageTest, ParseNodeNameAndScopeInRoot) {
  const auto scope_and_name = ParseNodeScopeAndName("Add");
  EXPECT_EQ(scope_and_name.scope, "");
  EXPECT_EQ(scope_and_name.name, "Add");
}

TEST_F(GraphOptimizerStageTest, ParseNodeNameAndScopeInScope) {
  const auto scope_and_name = ParseNodeScopeAndName("a/b/c/Add");
  EXPECT_EQ(scope_and_name.scope, "a/b/c");
  EXPECT_EQ(scope_and_name.name, "Add");
}

TEST_F(GraphOptimizerStageTest, OptimizedNodeName) {
  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ nullptr,
                            /*graph_properties*/ nullptr,
                            /*node_map*/ nullptr,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  const auto node = ParseNodeScopeAndName("a/b/c/Add");

  // Without rewrite rule
  EXPECT_EQ(stage.OptimizedNodeName(node), "a/b/c/my_opt/my_stg_Add");
  EXPECT_EQ(stage.OptimizedNodeName(node, std::vector<string>({"Mul", "Sqrt"})),
            "a/b/c/my_opt/my_stg_Add_Mul_Sqrt");

  // With rewrite rule
  const string rewrite = "my_rewrite";
  EXPECT_EQ(stage.OptimizedNodeName(node, rewrite),
            "a/b/c/my_opt/my_stg_my_rewrite_Add");
}

TEST_F(GraphOptimizerStageTest, UniqueOptimizedNodeName) {
  GraphDef graph =
      GDef({NDef("a/b/c/A", "NotImportant", {}),
            NDef("a/b/c/my_opt/my_stg_A", "NotImportant", {}),
            NDef("a/b/c/my_opt/my_stg_my_rewrite_A", "NotImportant", {})},
           /*funcs=*/{});

  NodeMap node_map(&graph);
  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ nullptr,
                            /*graph_properties*/ nullptr,
                            /*node_map*/ &node_map,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  const auto node = ParseNodeScopeAndName("a/b/c/A");

  EXPECT_EQ(stage.UniqueOptimizedNodeName(node),
            "a/b/c/my_opt/my_stg_A_unique0");

  // With rewrite rule
  const string rewrite = "my_rewrite";
  EXPECT_EQ(stage.UniqueOptimizedNodeName(node, rewrite),
            "a/b/c/my_opt/my_stg_my_rewrite_A_unique1");
}

TEST_F(GraphOptimizerStageTest, UniqueOptimizedNodeNameWithUsedNodeNames) {
  GraphDef graph = GDef(
      {NDef("a/b/c/A", "NotImportant", {}),
       NDef("a/b/c/my_opt/my_stg_A", "NotImportant", {}),
       NDef("a/b/c/my_opt/my_stg_A_unique0", "NotImportant", {}),
       NDef("a/b/c/my_opt/my_stg_my_rewrite_A", "NotImportant", {}),
       NDef("a/b/c/my_opt/my_stg_my_rewrite_A_unique1", "NotImportant", {})},
      /*funcs=*/{});

  NodeMap node_map(&graph);
  GraphOptimizerContext ctx(/*nodes_to_preserve*/ nullptr,
                            /*optimized_graph*/ nullptr,
                            /*graph_properties*/ nullptr,
                            /*node_map*/ &node_map,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  const auto node = ParseNodeScopeAndName("a/b/c/A");

  EXPECT_EQ(stage.UniqueOptimizedNodeName(node),
            "a/b/c/my_opt/my_stg_A_unique1");

  // With rewrite rule
  const string rewrite = "my_rewrite";
  EXPECT_EQ(stage.UniqueOptimizedNodeName(node, rewrite),
            "a/b/c/my_opt/my_stg_my_rewrite_A_unique2");
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
                            /*node_map*/ &node_map,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  NodeDef* add_node;
  TF_CHECK_OK(stage.GetInputNode("Add", &add_node));
  ASSERT_EQ(add_node->input_size(), 2);
  EXPECT_EQ(add_node->input(0), "a");
  EXPECT_EQ(add_node->input(1), "b");

  const OpInfo::TensorProperties* add_properties;
  TF_CHECK_OK(stage.GetTensorProperties("Add", &add_properties));
  EXPECT_EQ(add_properties->dtype(), DT_FLOAT);

  const OpInfo::TensorProperties* a_properties;
  TF_CHECK_OK(stage.GetTensorProperties("a:0", &a_properties));
  EXPECT_EQ(a_properties->dtype(), DT_FLOAT_REF);

  const OpInfo::TensorProperties* b_properties;
  TF_CHECK_OK(stage.GetTensorProperties("b:0", &b_properties));
  EXPECT_EQ(b_properties->dtype(), DT_FLOAT_REF);
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
                            /*node_map*/ &node_map,
                            /*feed_nodes*/ nullptr,
                            /*opt_level*/ RewriterConfig::ON);
  FakeOptimizerStage stage("my_opt", "my_stg", ctx);

  NodeDef* add_node;
  TF_CHECK_OK(stage.GetInputNode("Add", &add_node));

  // Add a new copy node
  NodeDef* add_node_copy = stage.AddCopyNode("Add_1", add_node);
  EXPECT_EQ(add_node_copy->name(), "Add_1");
  EXPECT_EQ(add_node_copy->op(), "Add");
  ASSERT_EQ(add_node->input_size(), 2);
  EXPECT_EQ(add_node_copy->input(0), "a");
  EXPECT_EQ(add_node_copy->input(1), "b");

  // It must be available for by-name lookup
  NodeDef* add_node_copy_by_name;
  TF_CHECK_OK(stage.GetInputNode("Add_1", &add_node_copy_by_name));
  EXPECT_EQ(add_node_copy, add_node_copy_by_name);

  // Add new empty node
  NodeDef* empty_node = stage.AddEmptyNode("Add_2");
  EXPECT_EQ(empty_node->name(), "Add_2");
  EXPECT_EQ(empty_node->input_size(), 0);

  // It must be available for by-name lookup
  NodeDef* empty_node_by_name;
  TF_CHECK_OK(stage.GetInputNode("Add_2", &empty_node_by_name));
  EXPECT_EQ(empty_node, empty_node_by_name);

  // Check that AddEmptyNode adds a unique suffix if the node already exists.
  NodeDef* unique_empty_node = stage.AddEmptyNode("Add_2");
  EXPECT_EQ(unique_empty_node->name(), "Add_2_0");
}

}  // namespace
}  // end namespace grappler
}  // end namespace tensorflow
