/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/graph_def_builder.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

TEST(GraphDefBuilderTest, Version) {

  // Verify that our assertions will be nontrivial
  ASSERT_LT(0, TF_GRAPH_DEF_VERSION);

  // Newly built graphs should use the current version
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);

  // Check version when we convert to a Graph
  Graph graph(OpRegistry::Global());
  TF_EXPECT_OK(builder.ToGraph(&graph));
  ASSERT_EQ(graph.versions().producer(), TF_GRAPH_DEF_VERSION);
  ASSERT_EQ(graph.versions().min_consumer(), TF_GRAPH_DEF_VERSION_MIN_CONSUMER);

  // Check version when we convert to a GraphDef
  GraphDef graph_def;
  TF_EXPECT_OK(builder.ToGraphDef(&graph_def));
  ASSERT_EQ(graph_def.versions().producer(), TF_GRAPH_DEF_VERSION);
  ASSERT_EQ(graph_def.versions().min_consumer(),
            TF_GRAPH_DEF_VERSION_MIN_CONSUMER);
}

}  // namespace
}  // namespace tensorflow
