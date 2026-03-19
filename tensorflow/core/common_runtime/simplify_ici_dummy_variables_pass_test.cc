/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/simplify_ici_dummy_variables_pass.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "tensorflow/cc/framework/scope.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/config/flags.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Return the node with the specified name
Node* GetNode(const Graph& graph, const std::string& name) {
  for (Node* node : graph.nodes()) {
    if (node->name() == name) return node;
  }
  return nullptr;
}

std::string TestDataPath() {
  return tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/common_runtime/testdata/");
}

using SimplifyIciDummyVariablesPassTest = ::testing::TestWithParam<bool>;

// Test the case enable_tf2min_ici_weight is false.
TEST(SimplifyIciDummyVariablesPassTest, flag_is_false) {
  flags::Global().enable_tf2min_ici_weight.reset(false);
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  std::string graph_path =
      TestDataPath() + "simplify_ici_dummy_variables_pass_before.pbtxt";
  tensorflow::GraphDef graph_def;
  absl::Status load_graph_status =
      ReadTextProto(tensorflow::Env::Default(), graph_path, &graph_def);
  EXPECT_EQ(load_graph_status.ok(), true);
  TF_EXPECT_OK(ConvertGraphDefToGraph(GraphConstructorOptions(), graph_def,
                                      graph.get()));

  GraphOptimizationPassOptions options;
  options.graph = &graph;
  SimplifyIciDummyVariablesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  Node* fill_1 =
      GetNode(*graph, "tpu_dummy_input_ici_specific_index_0_task_id_2");
  EXPECT_EQ(fill_1, nullptr);

  Node* fill_2 =
      GetNode(*graph, "tpu_dummy_input_ici_specific_index_1_task_id_2");
  EXPECT_EQ(fill_2, nullptr);
}

// Test the case enable_tf2min_ici_weight is true, graph after pass will have
// dummy variables on task 2.
// The bool test parameter decides whether to load a graph with TPUExecute or
// TPUExecuteAndUpdateVariables ops.
TEST_P(SimplifyIciDummyVariablesPassTest, replace_dummy_variable) {
  flags::Global().enable_tf2min_ici_weight.reset(true);
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  const std::string graph_file_name =
      GetParam() ? "simplify_ici_dummy_variables_pass_updatevars_before.pbtxt"
                 : "simplify_ici_dummy_variables_pass_before.pbtxt";
  std::string graph_path = TestDataPath() + graph_file_name;
  tensorflow::GraphDef graph_def;
  absl::Status load_graph_status =
      ReadTextProto(tensorflow::Env::Default(), graph_path, &graph_def);
  EXPECT_EQ(load_graph_status.ok(), true);
  TF_EXPECT_OK(ConvertGraphDefToGraph(GraphConstructorOptions(), graph_def,
                                      graph.get()));

  GraphOptimizationPassOptions options;
  options.graph = &graph;
  SimplifyIciDummyVariablesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  Node* tpu_dummy_input_1 =
      GetNode(*graph, "tpu_dummy_input_ici_specific_index_0_task_id_2");
  EXPECT_NE(tpu_dummy_input_1, nullptr);
  EXPECT_EQ(tpu_dummy_input_1->requested_device(),
            "/job:tpu_host_worker/replica:0/task:2/device:CPU:0");

  Node* tpu_dummy_input_2 =
      GetNode(*graph, "tpu_dummy_input_ici_specific_index_1_task_id_2");
  EXPECT_NE(tpu_dummy_input_2, nullptr);
  EXPECT_EQ(tpu_dummy_input_2->requested_device(),
            "/job:tpu_host_worker/replica:0/task:2/device:CPU:0");

  // Check that all remaining TPUDummyInput nodes have at least one out edge.
  for (auto n : graph->nodes()) {
    if (n->type_string() == "TPUDummyInput") {
      EXPECT_FALSE(n->out_edges().empty());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(All, SimplifyIciDummyVariablesPassTest,
                         ::testing::Values(false, true),
                         ::testing::PrintToStringParamName());

}  // namespace tensorflow
