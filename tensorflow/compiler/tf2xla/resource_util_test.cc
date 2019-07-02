/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/resource_util.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

void AnalyzeAndVerify(
    const GraphDef& graphdef,
    absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>*
        expected) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  TF_EXPECT_OK(
      ConvertGraphDefToGraph(GraphConstructorOptions(), graphdef, graph.get()));

  absl::flat_hash_map<const Node*, absl::flat_hash_set<const Node*>>
      sources_paths;
  TF_EXPECT_OK(AnalyzeResourceOpSourcePath(graph.get(), &sources_paths));

  EXPECT_EQ(sources_paths.size(), expected->size());

  for (const auto it : sources_paths) {
    const std::string& src_name = it.first->name();
    const auto& expected_path = expected->at(src_name);
    EXPECT_EQ(it.second.size(), expected_path.size());
    for (const Node* n : it.second) {
      EXPECT_TRUE(expected_path.find(n->name()) != expected_path.end());
    }
  }
}

}  // anonymous namespace

TEST(ResourceOpAnalyzerTest, SingleResourceSingleUserNoPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     * stack_size -> stack_op -> stack_close
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder stack_close_builder("stack_close", "StackCloseV2", op_reg);
    stack_close_builder.Input(stack_op);
    opts.FinalizeBuilder(&stack_close_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected["stack_op"] = absl::flat_hash_set<std::string>({"stack_close"});
  AnalyzeAndVerify(graphdef, &expected);
}

TEST(ResourceOpAnalyzerTest, SingleResourceSingleUserWithPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     * stack_size -> stack_op -> resource_identity -> stack_close
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder resource_identity_builder("resource_identity", "Identity",
                                          op_reg);
    resource_identity_builder.Input(stack_op);
    Node* resource_identity = opts.FinalizeBuilder(&resource_identity_builder);

    NodeBuilder stack_close_builder("stack_close", "StackCloseV2", op_reg);
    stack_close_builder.Input(resource_identity);
    opts.FinalizeBuilder(&stack_close_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected["stack_op"] =
      absl::flat_hash_set<std::string>({"resource_identity", "stack_close"});
  AnalyzeAndVerify(graphdef, &expected);
}

TEST(ResourceOpAnalyzerTest, SingleResourceMultipleUserNoPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     *                        stack_close0
     *                       /
     * stack_size -> stack_op
     *                       \
     *                        stack_close1
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder stack_close0_builder("stack_close0", "StackCloseV2", op_reg);
    stack_close0_builder.Input(stack_op);
    opts.FinalizeBuilder(&stack_close0_builder);

    NodeBuilder stack_close1_builder("stack_close1", "StackCloseV2", op_reg);
    stack_close1_builder.Input(stack_op);
    opts.FinalizeBuilder(&stack_close1_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected["stack_op"] =
      absl::flat_hash_set<std::string>({"stack_close0", "stack_close1"});
  AnalyzeAndVerify(graphdef, &expected);
}

TEST(ResourceOpAnalyzerTest, SingleResourceMultipleUserWithPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     *                                              stack_close0
     *                                             /
     * stack_size -> stack_op -> resource_identity
     *                                             \
     *                                              stack_close1
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder resource_identity_builder("resource_identity", "Identity",
                                          op_reg);
    resource_identity_builder.Input(stack_op);
    Node* resource_identity = opts.FinalizeBuilder(&resource_identity_builder);

    NodeBuilder stack_close0_builder("stack_close0", "StackCloseV2", op_reg);
    stack_close0_builder.Input(resource_identity);
    opts.FinalizeBuilder(&stack_close0_builder);

    NodeBuilder stack_close1_builder("stack_close1", "StackCloseV2", op_reg);
    stack_close1_builder.Input(resource_identity);
    opts.FinalizeBuilder(&stack_close1_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected["stack_op"] = absl::flat_hash_set<std::string>(
      {"resource_identity", "stack_close0", "stack_close1"});
  AnalyzeAndVerify(graphdef, &expected);
}

TEST(ResourceOpAnalyzerTest, MultipleResourceMultipleUserNoPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     *                        stack_close0
     *                       /
     *               stack_op0
     *             /         \
     *            /           stack_close1
     * stack_size
     *            \           stack_close2
     *             \         /
     *               stack_op1
     *                       \
     *                         stack_close3
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op0_builder("stack_op0", "StackV2", op_reg);
    stack_op0_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op0 = opts.FinalizeBuilder(&stack_op0_builder);

    NodeBuilder stack_close0_builder("stack_close0", "StackCloseV2", op_reg);
    stack_close0_builder.Input(stack_op0);
    opts.FinalizeBuilder(&stack_close0_builder);

    NodeBuilder stack_close1_builder("stack_close1", "StackCloseV2", op_reg);
    stack_close1_builder.Input(stack_op0);
    opts.FinalizeBuilder(&stack_close1_builder);

    NodeBuilder stack_op1_builder("stack_op1", "StackV2", op_reg);
    stack_op1_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op1 = opts.FinalizeBuilder(&stack_op1_builder);

    NodeBuilder stack_close2_builder("stack_close2", "StackCloseV2", op_reg);
    stack_close2_builder.Input(stack_op1);
    opts.FinalizeBuilder(&stack_close2_builder);

    NodeBuilder stack_close3_builder("stack_close3", "StackCloseV2", op_reg);
    stack_close3_builder.Input(stack_op1);
    opts.FinalizeBuilder(&stack_close3_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected["stack_op0"] =
      absl::flat_hash_set<std::string>({"stack_close0", "stack_close1"});
  expected["stack_op1"] =
      absl::flat_hash_set<std::string>({"stack_close2", "stack_close3"});
  AnalyzeAndVerify(graphdef, &expected);
}

TEST(ResourceOpAnalyzerTest, MultipleResourceMultipleUserWithPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*                                I
     *               stack_op0  ----> d  --->  stack_close0
     *             /                  e
     *            /                   n
     * stack_size ------------------> t
     *            \                   i
     *             \                  t
     *               stack_op1  ----> y  --->  stack_close0
     *                                N
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op0_builder("stack_op0", "StackV2", op_reg);
    stack_op0_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op0 = opts.FinalizeBuilder(&stack_op0_builder);

    NodeBuilder stack_op1_builder("stack_op1", "StackV2", op_reg);
    stack_op1_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op1 = opts.FinalizeBuilder(&stack_op1_builder);

    NodeBuilder identity_n_builder("identity_n", "IdentityN", op_reg);
    identity_n_builder.Input({stack_op0, stack_size_placeholder, stack_op1});

    NodeBuilder stack_close0_builder("stack_close0", "StackCloseV2", op_reg);
    stack_close0_builder.Input(stack_op0);
    opts.FinalizeBuilder(&stack_close0_builder);

    NodeBuilder stack_close1_builder("stack_close1", "StackCloseV2", op_reg);
    stack_close1_builder.Input(stack_op0);
    opts.FinalizeBuilder(&stack_close1_builder);

    NodeBuilder stack_close2_builder("stack_close2", "StackCloseV2", op_reg);
    stack_close2_builder.Input(stack_op1);
    opts.FinalizeBuilder(&stack_close2_builder);

    NodeBuilder stack_close3_builder("stack_close3", "StackCloseV2", op_reg);
    stack_close3_builder.Input(stack_op1);
    opts.FinalizeBuilder(&stack_close3_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected["stack_op0"] =
      absl::flat_hash_set<std::string>({"stack_close0", "stack_close1"});
  expected["stack_op1"] =
      absl::flat_hash_set<std::string>({"stack_close2", "stack_close3"});
  AnalyzeAndVerify(graphdef, &expected);
}

}  // namespace tensorflow
