/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/jit/xla_compile_util.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/kernels/ops_testutil.h"

namespace tensorflow {
namespace {

TEST_F(OpsTestBase, CreateSingleOpGraph) {
  TF_EXPECT_OK(NodeDefBuilder("identity_op", "Identity")
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DT_FLOAT)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 2}), {6.9, 4.2});
  TF_EXPECT_OK(RunOpKernel());

  XlaCompiler::SingleOpCompileArgument single_op_arg(*context_);

  std::vector<XlaArgument> args(1);
  args[0].kind = XlaArgument::kConstant;
  args[0].type = DT_FLOAT;
  args[0].shape = TensorShape({1, 2});
  args[0].constant_value = GetInput(0);
  args[0].initialized = true;

  TF_ASSERT_OK_AND_ASSIGN(
      auto graph,
      CreateSingleOpGraph(*node_def(), args, single_op_arg.output_dtypes));

  const auto& node_name_index = graph->BuildNodeNameIndex();

  const Node* identity_node = node_name_index.at("identity_op");
  EXPECT_EQ(identity_node->op_def().name(), "Identity");
  EXPECT_EQ(identity_node->attrs().FindByString("T")->type(), DT_FLOAT);

  EXPECT_EQ(identity_node->num_inputs(), 1);
  const Node* identity_input_node = nullptr;
  TF_EXPECT_OK(identity_node->input_node(0, &identity_input_node));
  EXPECT_EQ(identity_input_node->name(), "_arg0");

  const Node* arg_node = node_name_index.at("_arg0");
  EXPECT_EQ(arg_node->op_def().name(), "_Arg");
  EXPECT_EQ(arg_node->attrs().FindByString("T")->type(), DT_FLOAT);

  const Node* retval_node = node_name_index.at("_retval0");
  EXPECT_EQ(retval_node->op_def().name(), "_Retval");
  EXPECT_EQ(retval_node->attrs().FindByString("T")->type(), DT_FLOAT);

  EXPECT_EQ(identity_node->num_outputs(), 1);
  EXPECT_EQ(retval_node->num_inputs(), 1);
  const Node* retval_input_node = nullptr;
  TF_EXPECT_OK(retval_node->input_node(0, &retval_input_node));
  EXPECT_EQ(retval_input_node->name(), "identity_op");
}

}  // namespace
}  // namespace tensorflow
