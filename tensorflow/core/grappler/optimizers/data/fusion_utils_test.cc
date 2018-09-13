/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/fusion_utils.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace fusion_utils {
namespace {

string ParseNodeConnection(const string &name) {
  return name.substr(0, name.find(':'));
}

void CheckUniqueNames(const FunctionDef &function) {
  std::unordered_set<string> inputs;
  for (const auto &input_arg : function.signature().input_arg())
    inputs.insert(input_arg.name());
  EXPECT_EQ(inputs.size(), function.signature().input_arg_size());

  std::unordered_set<string> outputs;
  for (const auto &output_arg : function.signature().output_arg())
    outputs.insert(output_arg.name());
  EXPECT_EQ(outputs.size(), function.signature().output_arg_size());

  std::unordered_set<string> nodes;
  for (const auto &node : function.node_def()) nodes.insert(node.name());

  EXPECT_EQ(nodes.size(), function.node_def_size());
}

TEST(FusionUtilsTest, FuseFunctionsByComposition) {
  GraphDef graph;
  auto *parent_function = graph.mutable_library()->add_function();
  *parent_function = test::function::XTimesTwo();
  auto *function = graph.mutable_library()->add_function();
  *function = test::function::XTimesTwo();

  auto *fused_function = FuseFunctions(
      *parent_function, *function, "fused_maps", fusion_utils::ComposeSignature,
      fusion_utils::ComposeInput, fusion_utils::ComposeOutput,
      fusion_utils::MergeNodes, graph.mutable_library());

  EXPECT_EQ(fused_function->signature().name(), "fused_maps");
  EXPECT_EQ(fused_function->signature().input_arg_size(), 1);
  EXPECT_EQ(fused_function->signature().output_arg_size(), 1);
  EXPECT_EQ(fused_function->ret_size(), 1);
  std::cerr << fused_function->DebugString();
  CheckUniqueNames(*fused_function);

  const NodeDef *parent_mul = nullptr, *output_mul = nullptr;
  for (const auto &fused_node : fused_function->node_def()) {
    if (fused_node.op() == "Mul") {
      if (fused_node.name() == "y")
        parent_mul = &fused_node;
      else
        output_mul = &fused_node;
    }
  }
  ASSERT_NE(parent_mul, nullptr);
  ASSERT_NE(output_mul, nullptr);
  EXPECT_EQ(ParseNodeConnection(output_mul->input(0)), parent_mul->name());

  auto output_value = fused_function->ret().at(
      fused_function->signature().output_arg(0).name());

  EXPECT_EQ(ParseNodeConnection(output_value), output_mul->name());
}

TEST(FusionUtilsTest, FuseFunctionWithPredicate) {
  GraphDef graph;
  auto *xtimes_two = graph.mutable_library()->add_function();
  *xtimes_two = test::function::XTimesTwo();
  auto *is_zero = graph.mutable_library()->add_function();
  *is_zero = test::function::IsZero();

  auto *fused_function =
      FuseFunctions(*xtimes_two, *is_zero, "fused_map_and_filter_function",
                    fusion_utils::CombineSignature, fusion_utils::ComposeInput,
                    fusion_utils::CombineOutput, fusion_utils::MergeNodes,
                    graph.mutable_library());

  EXPECT_EQ(fused_function->signature().name(),
            "fused_map_and_filter_function");

  EXPECT_EQ(fused_function->signature().input_arg_size(), 1);
  EXPECT_EQ(fused_function->signature().output_arg_size(), 2);
  EXPECT_EQ(fused_function->ret_size(), 2);
  CheckUniqueNames(*fused_function);

  ASSERT_TRUE(
      graph_utils::ContainsFunctionNodeWithOp("Equal", *fused_function));
  const auto &equal_node = fused_function->node_def(
      graph_utils::FindFunctionNodeWithOp("Equal", *fused_function));

  EXPECT_EQ(xtimes_two->signature().output_arg(0).name(),
            fused_function->signature().output_arg(0).name());

  EXPECT_EQ(fused_function->signature().output_arg(1).name(),
            equal_node.name());

  EXPECT_EQ(ParseNodeConnection(equal_node.input(0)),
            fused_function->signature().output_arg(0).name());

  auto output_value = fused_function->ret().at(
      fused_function->signature().output_arg(1).name());
  EXPECT_EQ(ParseNodeConnection(output_value), equal_node.name());
}

TEST(FusionUtilsTest, FuseSameFunctionWithExtraOutput) {
  GraphDef graph;
  auto *parent_function = graph.mutable_library()->add_function();
  *parent_function = test::function::XTimesTwo();
  auto *function = graph.mutable_library()->add_function();
  *function = test::function::XTimesTwo();

  auto *fused_function = FuseFunctions(
      *parent_function, *function, "fused_maps", fusion_utils::CombineSignature,
      fusion_utils::ComposeInput, fusion_utils::CombineOutput,
      fusion_utils::MergeNodes, graph.mutable_library());

  EXPECT_EQ(fused_function->signature().input_arg_size(), 1);
  EXPECT_EQ(fused_function->signature().output_arg_size(), 2);
  EXPECT_EQ(fused_function->ret_size(), 2);
  CheckUniqueNames(*fused_function);
}

TEST(FusionUtilsTest, ZipFusion) {
  GraphDef graph;
  auto *function = graph.mutable_library()->add_function();
  *function = test::function::XTimesTwo();

  auto zip_signature = [](const OpDef &parent_function_signature,
                          const OpDef &function_signature,
                          OpDef *fused_function_signature) {
    *fused_function_signature = parent_function_signature;
    fused_function_signature->mutable_input_arg()->MergeFrom(
        function_signature.input_arg());
    fused_function_signature->mutable_output_arg()->MergeFrom(
        function_signature.output_arg());
  };

  auto zip_input = [](const StringCollection &parent_inputs,
                      const StringCollection &function_inputs,
                      const StringCollection &parent_outputs, int arg_num) {
    // Take corresponding parent output.
    return function_inputs.at(arg_num);
  };

  auto *fused_function =
      FuseFunctions(*function, *function, "zip_maps", zip_signature, zip_input,
                    fusion_utils::CombineOutput, fusion_utils::MergeNodes,
                    graph.mutable_library());

  EXPECT_EQ(fused_function->signature().input_arg_size(), 2);
  EXPECT_EQ(fused_function->signature().output_arg_size(), 2);
  EXPECT_EQ(fused_function->ret_size(), 2);
  CheckUniqueNames(*fused_function);
}

}  // namespace
}  // namespace fusion_utils
}  // namespace grappler
}  // namespace tensorflow
