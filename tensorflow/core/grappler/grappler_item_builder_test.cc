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

#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

class GrapplerItemBuilderTest : public ::testing::Test {};

// Create a sample graph with a symbolic gradient for sum.
void SampleSumSymbolicGradientGraphdef(
    GraphDef *def, CollectionDef *fetches,
    std::vector<string> *names_of_ops_of_inline) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  auto dummy_variable = Variable(scope, {2, 2}, DT_FLOAT);
  auto x = Const(scope, 1.0f);
  auto y = Const(scope, 2);
  auto z = Const(scope, 3.0f);
  TF_ASSERT_OK(scope.status());

  NameAttrList fn;
  fn.set_name("Sum");
  (*fn.mutable_attr())["T"].set_type(DT_FLOAT);
  auto g0 = SymbolicGradient(scope, std::initializer_list<Input>{x, y, z},
                             {DT_FLOAT, DT_INT32}, fn);

  // TODO(bsteiner): we should rewrite the feed/fetch nodes to reflect the
  // inlining that's done in the item builder
  // fetches->mutable_node_list()->add_value(g0[0].name());
  fetches->mutable_node_list()->add_value("SymbolicGradient/dx");
  fetches->mutable_node_list()->add_value("SymbolicGradient/dy_reshaped");

  TF_CHECK_OK(scope.ToGraphDef(def));

  // Add names of the ops that replace the Mul symbolic gradient during
  // inlining. This is for validation.
  *names_of_ops_of_inline = {
      "SymbolicGradient/dx",          "SymbolicGradient/tile_scaling",
      "SymbolicGradient/dy_reshaped", "SymbolicGradient/y_shape",
      "SymbolicGradient/x_shape",     "SymbolicGradient/stitch_idx0",
      "SymbolicGradient/x_rank",      "SymbolicGradient/stitch_val1",
      "SymbolicGradient/i_shape",     "SymbolicGradient/di",
      "SymbolicGradient/zero",        "SymbolicGradient/one"};
}

std::unique_ptr<GrapplerItem> CreateGrapplerItem(const GraphDef &def,
                                                 const CollectionDef &fetches) {
  MetaGraphDef meta_def;
  ItemConfig cfg;
  cfg.inline_functions = true;
  *meta_def.mutable_graph_def() = def;
  (*meta_def.mutable_collection_def())["train_op"] = fetches;
  return GrapplerItemFromMetaGraphDef("0", meta_def, cfg);
}

int CountSymbolicGradientOps(const std::unique_ptr<GrapplerItem> &item) {
  int n_symb_grads = 0;
  for (const auto &node : item->graph.node()) {
    if (node.op() == FunctionLibraryDefinition::kGradientOp) {
      n_symb_grads++;
    }
  }
  return n_symb_grads;
}

int CountOpsWithNames(const std::unique_ptr<GrapplerItem> &item,
                      const std::vector<string> &names) {
  std::set<string> names_set(names.begin(), names.end());
  int n_with_names = 0;
  for (const auto &node : item->graph.node()) {
    if (names_set.find(node.name()) != names_set.end()) {
      n_with_names++;
    }
  }
  return n_with_names;
}

TEST_F(GrapplerItemBuilderTest, SymbolicGradientInlining) {
  // Create sample sum symbolic gradient graph.
  GraphDef def;
  CollectionDef fetches;
  std::vector<string> ops_of_inline;
  SampleSumSymbolicGradientGraphdef(&def, &fetches, &ops_of_inline);

  // Create the inlined graph.
  std::unique_ptr<GrapplerItem> with_inline = CreateGrapplerItem(def, fetches);

  // For the inlined graph, there should be 0 symbolic gradient ops.
  EXPECT_EQ(0, CountSymbolicGradientOps(with_inline));

  // For the inlined graph, make sure all the required expanded op’s are in the
  // graph.
  EXPECT_EQ(ops_of_inline.size(),
            CountOpsWithNames(with_inline, ops_of_inline));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
