/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/disable_functional_ops_lowering_for_xla_pass.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

absl::Status AddWhileNode(Graph* graph, absl::string_view name, Node** node) {
  return NodeBuilder(name, "While")
      .Input(std::vector<NodeBuilder::NodeOut>{})
      .Attr("_lower_using_switch_merge", true)
      .Attr("cond", NameAttrList())
      .Attr("body", NameAttrList())
      .Attr("T", DataTypeVector{})
      .Finalize(graph, node);
}

absl::Status AddPlaceholder(Graph* graph, absl::string_view name,
                            DataType dtype, Node** node) {
  return NodeBuilder(name, "Placeholder")
      .Attr("dtype", dtype)
      .Finalize(graph, node);
}

absl::Status AddIfNode(Graph* graph, absl::string_view name, Node* cond,
                       Node** node) {
  return NodeBuilder(name, "If")
      .Input(cond, 0)
      .Input(std::vector<NodeBuilder::NodeOut>{})
      .Attr("_lower_using_switch_merge", true)
      .Attr("Tcond", DT_BOOL)
      .Attr("Tin", DataTypeVector{})
      .Attr("Tout", DataTypeVector{})
      .Attr("then_branch", NameAttrList())
      .Attr("else_branch", NameAttrList())
      .Finalize(graph, node);
}

absl::Status AddCaseNode(Graph* graph, absl::string_view name,
                         Node* branch_index, Node** node) {
  return NodeBuilder(name, "Case")
      .Input(branch_index, 0)
      .Input(std::vector<NodeBuilder::NodeOut>{})
      .Attr("_lower_using_switch_merge", true)
      .Attr("Tin", DataTypeVector{})
      .Attr("Tout", DataTypeVector{})
      .Attr("branches", std::vector<NameAttrList>{NameAttrList()})
      .Finalize(graph, node);
}

bool HasLowerUsingSwitchMergeAttr(const Node* n) {
  bool lower = false;
  return TryGetNodeAttr(n->attrs(), "_lower_using_switch_merge", &lower) &&
        lower;
}

absl::Status RunPass(std::unique_ptr<Graph>* graph,
                     SessionOptions* session_options) {
  GraphOptimizationPassOptions options;
  options.graph = graph;
  options.session_options = session_options;
  DisableFunctionalOpsLoweringForXlaPass pass;
  return pass.Run(options);
}

TEST(DisableFunctionalOpsLoweringForXlaPassTest,
    ClearsAttrOnFunctionalNodesWhenGlobalJitEnabled) {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Node* while_node;
  TF_ASSERT_OK(AddWhileNode(graph.get(), "my_while", &while_node));
  Node* cond;
  TF_ASSERT_OK(AddPlaceholder(graph.get(), "cond", DT_BOOL, &cond));
  Node* if_node;
  TF_ASSERT_OK(AddIfNode(graph.get(), "my_if", cond, &if_node));
  Node* branch_index;
  TF_ASSERT_OK(
      AddPlaceholder(graph.get(), "branch_index", DT_INT32, &branch_index));
  Node* case_node;
  TF_ASSERT_OK(AddCaseNode(graph.get(), "my_case", branch_index, &case_node));

  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_1);

  TF_ASSERT_OK(RunPass(&graph, &session_options));

  EXPECT_FALSE(HasLowerUsingSwitchMergeAttr(while_node));
  EXPECT_FALSE(HasLowerUsingSwitchMergeAttr(if_node));
  EXPECT_FALSE(HasLowerUsingSwitchMergeAttr(case_node));
}

TEST(DisableFunctionalOpsLoweringForXlaPassTest,
    PreservesAttrWhenGlobalJitDisabled) {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Node* while_node;
  TF_ASSERT_OK(AddWhileNode(graph.get(), "my_while", &while_node));

  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::OFF);

  TF_ASSERT_OK(RunPass(&graph, &session_options));

  EXPECT_TRUE(HasLowerUsingSwitchMergeAttr(while_node));
}

TEST(DisableFunctionalOpsLoweringForXlaPassTest, NoOpWhenGraphIsMissing) {
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);

  GraphOptimizationPassOptions options;
  options.session_options = &session_options;
  DisableFunctionalOpsLoweringForXlaPass pass;
  TF_ASSERT_OK(pass.Run(options));
}

class DisableFunctionalOpsLoweringForXlaPassFlagsTest
    : public ::testing::Test {
 protected:
  void SetUp() override {
    flags_ = GetMarkForCompilationPassFlags();
    original_ = flags_->xla_auto_jit_flag;
  }
  void TearDown() override { flags_->xla_auto_jit_flag = original_; }

  MarkForCompilationPassFlags* flags_;
  XlaAutoJitFlag original_;
};

TEST_F(DisableFunctionalOpsLoweringForXlaPassFlagsTest,
      FallsBackToFlagsWhenSessionOptionsIsNull) {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Node* while_node;
  TF_ASSERT_OK(AddWhileNode(graph.get(), "my_while", &while_node));

  flags_->xla_auto_jit_flag.optimization_level_single_gpu =
      OptimizerOptions::ON_2;
  flags_->xla_auto_jit_flag.optimization_level_general =
      OptimizerOptions::ON_2;

  TF_ASSERT_OK(RunPass(&graph, /*session_options=*/nullptr));

  EXPECT_FALSE(HasLowerUsingSwitchMergeAttr(while_node));
}

TEST_F(DisableFunctionalOpsLoweringForXlaPassFlagsTest,
      PreservesAttrWhenSessionOptionsIsNullAndFlagsAreOff) {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Node* while_node;
  TF_ASSERT_OK(AddWhileNode(graph.get(), "my_while", &while_node));

  flags_->xla_auto_jit_flag.optimization_level_single_gpu =
      OptimizerOptions::OFF;
  flags_->xla_auto_jit_flag.optimization_level_general = OptimizerOptions::OFF;

  TF_ASSERT_OK(RunPass(&graph, /*session_options=*/nullptr));

  EXPECT_TRUE(HasLowerUsingSwitchMergeAttr(while_node));
}

}  // namespace
}  // namespace tensorflow
