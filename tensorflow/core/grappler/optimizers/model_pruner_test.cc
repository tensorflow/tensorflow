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

#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class ModelPrunerTest : public ::testing::Test {};

TEST_F(ModelPrunerTest, NoPruning) {
  // This trivial graph is so basic there's nothing to prune.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  ModelPruner pruner;
  GraphDef output;
  Status s = pruner.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(s);

  EXPECT_EQ(item.graph.node_size(), output.node_size());
  for (int i = 0; i < item.graph.node_size(); ++i) {
    const NodeDef& original = item.graph.node(i);
    const NodeDef& optimized = output.node(i);
    EXPECT_EQ(original.name(), optimized.name());
    EXPECT_EQ(original.op(), optimized.op());
    EXPECT_EQ(original.input_size(), optimized.input_size());
    for (int j = 0; j < original.input_size(); ++j) {
      EXPECT_EQ(original.input(j), optimized.input(j));
    }
  }
}

TEST_F(ModelPrunerTest, SimplePruning) {
  // Build a simple graph with a few trivially prunable ops.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
  Output b = ops::AddN(s.WithOpName("b"), {a});
  Output c = ops::StopGradient(s.WithOpName("c"), b);
  Output d = ops::StopGradient(s.WithOpName("d"), c);
  Output e = ops::AddN(s.WithOpName("e"), {d});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ModelPruner pruner;
  GraphDef output;
  Status status = pruner.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(3, output.node_size());
  const NodeDef& new_a = output.node(0);
  EXPECT_EQ(NodeName(a.name()), new_a.name());
  const NodeDef& new_b = output.node(1);
  EXPECT_EQ(NodeName(b.name()), new_b.name());
  const NodeDef& new_e = output.node(2);
  EXPECT_EQ(NodeName(e.name()), new_e.name());

  EXPECT_EQ(1, new_e.input_size());
  EXPECT_EQ(NodeName(b.name()), new_e.input(0));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
