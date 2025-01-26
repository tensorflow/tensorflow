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
#include "tsl/platform/test.h"

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

  Node* fill_1_dim = GetNode(*graph, "const_1_ici_specific_index_0_task_id_2");
  Node* fill_1_value =
      GetNode(*graph, "const_2_ici_specific_index_0_task_id_2");
  Node* fill_1 = GetNode(*graph, "fill_ici_specific_index_0_task_id_2");
  EXPECT_EQ(fill_1_dim, nullptr);
  EXPECT_EQ(fill_1_value, nullptr);
  EXPECT_EQ(fill_1, nullptr);

  Node* fill_2_dim = GetNode(*graph, "const_1_ici_specific_index_1_task_id_2");
  Node* fill_2_value =
      GetNode(*graph, "const_2_ici_specific_index_1_task_id_2");
  Node* fill_2 = GetNode(*graph, "fill_ici_specific_index_1_task_id_2");
  EXPECT_EQ(fill_2_dim, nullptr);
  EXPECT_EQ(fill_2_value, nullptr);
  EXPECT_EQ(fill_2, nullptr);
}

// Test the case enable_tf2min_ici_weight is true, graph after pass will have
// dummy variables on task 2.
TEST(SimplifyIciDummyVariablesPassTest, replace_dummy_variable) {
  flags::Global().enable_tf2min_ici_weight.reset(true);
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

  Node* fill_1_dim = GetNode(*graph, "const_1_ici_specific_index_0_task_id_2");
  Node* fill_1_value =
      GetNode(*graph, "const_2_ici_specific_index_0_task_id_2");
  Node* fill_1 = GetNode(*graph, "fill_ici_specific_index_0_task_id_2");
  EXPECT_NE(fill_1_dim, nullptr);
  EXPECT_NE(fill_1_value, nullptr);
  EXPECT_NE(fill_1, nullptr);
  EXPECT_EQ(fill_1_dim->requested_device(),
            "/job:tpu_host_worker/replica:0/task:2/device:CPU:0");
  EXPECT_EQ(fill_1_value->requested_device(),
            "/job:tpu_host_worker/replica:0/task:2/device:CPU:0");
  EXPECT_EQ(fill_1->requested_device(),
            "/job:tpu_host_worker/replica:0/task:2/device:CPU:0");

  Node* fill_2_dim = GetNode(*graph, "const_1_ici_specific_index_1_task_id_2");
  Node* fill_2_value =
      GetNode(*graph, "const_2_ici_specific_index_1_task_id_2");
  Node* fill_2 = GetNode(*graph, "fill_ici_specific_index_1_task_id_2");
  EXPECT_NE(fill_2_dim, nullptr);
  EXPECT_NE(fill_2_value, nullptr);
  EXPECT_NE(fill_2, nullptr);
  EXPECT_EQ(fill_2_dim->requested_device(),
            "/job:tpu_host_worker/replica:0/task:2/device:CPU:0");
  EXPECT_EQ(fill_2_value->requested_device(),
            "/job:tpu_host_worker/replica:0/task:2/device:CPU:0");
  EXPECT_EQ(fill_2->requested_device(),
            "/job:tpu_host_worker/replica:0/task:2/device:CPU:0");
}

}  // namespace tensorflow
