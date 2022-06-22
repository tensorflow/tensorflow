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
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {
ResourceUsageAnalysis::NodeInfo node_info_from_string(absl::string_view s) {
  std::vector<std::string> tokens = absl::StrSplit(s, ':');
  EXPECT_EQ(tokens.size(), 3);

  ResourceUsageAnalysis::NodeInfo node_info;
  if (tokens[0].empty()) {
    node_info.function_name_ = std::nullopt;
  } else {
    node_info.function_name_ = std::move(tokens[0]);
  }
  node_info.node_name_ = std::move(tokens[1]);
  node_info.op_ = std::move(tokens[2]);
  return node_info;
}

void AnalyzeAndVerify(
    const GraphDef& graphdef, FunctionLibraryDefinition* flib_def,
    const absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>&
        expected) {
  auto graph = absl::make_unique<Graph>(flib_def);
  TF_EXPECT_OK(
      ConvertGraphDefToGraph(GraphConstructorOptions(), graphdef, graph.get()));

  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, Env::Default(), /*config=*/nullptr, TF_GRAPH_DEF_VERSION,
      flib_def, OptimizerOptions());
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  absl::flat_hash_map<ResourceUsageAnalysis::NodeInfo,
                      absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>
      source_to_path;
  TF_EXPECT_OK(ResourceUsageAnalysis::Analyze(graph.get(), lib_runtime,
                                              &source_to_path));

  absl::flat_hash_map<ResourceUsageAnalysis::NodeInfo,
                      absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>
      expected_source_to_path;
  for (auto it : expected) {
    auto src_node_info = node_info_from_string(it.first);
    for (const std::string& user : it.second) {
      expected_source_to_path[src_node_info].emplace(
          node_info_from_string(user));
    }
  }

  EXPECT_EQ(source_to_path, expected_source_to_path);
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
  expected[":stack_op:StackV2"] =
      absl::flat_hash_set<std::string>({":stack_close:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
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
  expected[":stack_op:StackV2"] = absl::flat_hash_set<std::string>(
      {":resource_identity:Identity", ":stack_close:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
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
  expected[":stack_op:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close0:StackCloseV2", ":stack_close1:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
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
  expected[":stack_op:StackV2"] = absl::flat_hash_set<std::string>(
      {":resource_identity:Identity", ":stack_close0:StackCloseV2",
       ":stack_close1:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
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
  expected[":stack_op0:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close0:StackCloseV2", ":stack_close1:StackCloseV2"});
  expected[":stack_op1:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close2:StackCloseV2", ":stack_close3:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
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
  expected[":stack_op0:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close0:StackCloseV2", ":stack_close1:StackCloseV2"});
  expected[":stack_op1:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close2:StackCloseV2", ":stack_close3:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, ResourcePassThroughFunction) {
  auto library = absl::make_unique<FunctionDefLibrary>();
  /*
   *  pass_through_function:
   *
   *  _Arg -> Identity -> _Retval
   */
  *library->add_function() = FunctionDefHelper::Define(
      /*function_name=*/"pass_through_function",
      /*arg_def=*/{"in: resource"},
      /*ret_def=*/{"out: resource"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"out"}, "Identity", {"in"}, {{"T", DataType::DT_RESOURCE}}}});

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), *library);
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     * stack_size -> stack_op -> pass_through_function -> stack_close
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder pass_through_fn_builder("pass_through_fn",
                                        "pass_through_function", op_reg);
    pass_through_fn_builder.Input(stack_op);
    Node* pass_through_fn = opts.FinalizeBuilder(&pass_through_fn_builder);

    NodeBuilder stack_close_builder("stack_close", "StackCloseV2", op_reg);
    stack_close_builder.Input(pass_through_fn);
    opts.FinalizeBuilder(&stack_close_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected[":stack_op:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close:StackCloseV2", ":pass_through_fn:pass_through_function",
       "pass_through_function:out:Identity"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, ResourceUserInFunction) {
  auto library = absl::make_unique<FunctionDefLibrary>();
  /*
   *  resource_user_function:
   *
   *  _Arg -> Identity -> StackCloseV2
   */
  *library->add_function() = FunctionDefHelper::Define(
      /*function_name=*/"resource_user_function",
      /*arg_def=*/{"in: resource"},
      /*ret_def=*/{},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"stack_close"},
        "StackCloseV2",
        {"in"},
        {{"T", DataType::DT_RESOURCE}}}});

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), *library);
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     * stack_size -> stack_op -> resource_user_function
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder resource_user_fn_builder("resource_user_function",
                                         "resource_user_function", op_reg);
    resource_user_fn_builder.Input(stack_op);
    opts.FinalizeBuilder(&resource_user_fn_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected[":stack_op:StackV2"] = absl::flat_hash_set<std::string>(
      {":resource_user_function:resource_user_function",
       "resource_user_function:stack_close:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, ResourceSourceInFunction) {
  auto library = absl::make_unique<FunctionDefLibrary>();
  /*
   *  resource_source_function:
   *
   *  _Arg -> StackV2 -> _Retval
   */
  *library->add_function() = FunctionDefHelper::Define(
      /*function_name=*/"resource_source_function",
      /*arg_def=*/{"in: int32"},
      /*ret_def=*/{"out: resource"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"out"}, "StackV2", {"in"}, {{"elem_type", DataType::DT_FLOAT}}}});

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), *library);
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     * stack_size -> resource_source_function -> stack_close
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder resource_source_fn_builder("resource_source_function",
                                           "resource_source_function", op_reg);
    resource_source_fn_builder.Input(stack_size_placeholder);
    Node* resource_source_function =
        opts.FinalizeBuilder(&resource_source_fn_builder);

    NodeBuilder stack_close_builder("stack_close", "StackCloseV2", op_reg);
    stack_close_builder.Input(resource_source_function);
    opts.FinalizeBuilder(&stack_close_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected["resource_source_function:out:StackV2"] =
      absl::flat_hash_set<std::string>({":stack_close:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

}  // namespace tensorflow
