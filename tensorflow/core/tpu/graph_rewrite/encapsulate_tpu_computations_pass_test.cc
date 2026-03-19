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

#include "tensorflow/core/tpu/graph_rewrite/encapsulate_tpu_computations_pass.h"

#include <memory>

#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

std::unique_ptr<Graph> CreateGraph() {
  // c = a + b
  auto g = std::make_unique<Graph>(OpRegistry::Global());
  auto in0 = test::graph::Arg(g.get(), 0, DT_FLOAT);
  auto in1 = test::graph::Arg(g.get(), 1, DT_FLOAT);
  auto tmp = test::graph::Add(g.get(), in0, in1);
  auto ret = test::graph::Retval(g.get(), 0, tmp);
  g->AddControlEdge(in1, ret);
  FixupSourceAndSinkEdges(g.get());
  return g;
}

TEST(EncapsulateTPUComputationsPassTest, NonTPUGraph) {
  auto g = CreateGraph();
  GraphOptimizationPassOptions options;
  options.graph = &g;
  options.flib_def = g->mutable_flib_def();

  EncapsulateTPUComputationsPass pass;
  TF_ASSERT_OK(pass.Run(options));

  int nodes_meeting_expectations = 0;

  for (const auto* node : g->nodes()) {
    if (!IsSource(node) && !IsSink(node)) {
      ASSERT_TRUE(node->attrs().Find("_xla_inferred_shapes"));
      ++nodes_meeting_expectations;
    }
  }
  EXPECT_EQ(nodes_meeting_expectations, 4);
}

TEST(EncapsulateTPUComputationsPassTest, SkipEncapsulationForNonTPUGraph) {
  flags::Global().enable_skip_encapsulation_for_non_tpu_graphs.reset(true);
  auto g = CreateGraph();
  GraphOptimizationPassOptions options;
  options.graph = &g;
  options.flib_def = g->mutable_flib_def();

  EncapsulateTPUComputationsPass pass;
  TF_ASSERT_OK(pass.Run(options));

  int nodes_meeting_expectations = 0;

  for (const auto* node : g->nodes()) {
    if (!IsSource(node) && !IsSink(node)) {
      ASSERT_FALSE(node->attrs().Find("_xla_inferred_shapes"));
      ++nodes_meeting_expectations;
    }
  }
  EXPECT_EQ(nodes_meeting_expectations, 4);

  flags::Global().enable_skip_encapsulation_for_non_tpu_graphs.reset(false);
}

}  // namespace

}  // namespace tensorflow
