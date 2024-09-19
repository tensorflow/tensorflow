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

#include "tensorflow/cc/framework/scope.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/test.h"

namespace tensorflow {

TEST(SimplifyIciDummyVariablesPassTest, SimplifyIciDummyVariables) {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  SimplifyIciDummyVariablesPass pass;
  TF_ASSERT_OK(pass.Run(options));
}

}  // namespace tensorflow
