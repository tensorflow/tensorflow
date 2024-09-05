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

#include "absl/strings/match.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/tf2xla/rearrange_function_argument.h"
#include "xla/test.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

TEST(RearrangeFunctionArgumentForFunctionTest, Basic) {
  FunctionDefLibrary fdl;
  {
    // Function for StatefulPartitionedCall's "f", If's
    // "then_branch"/"else_branch".
    // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_BOOL)
    // "ret0" = "arg1"
    // "ret1" = "arg0"
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
    Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_BOOL, 1);
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), arg1, 0);
    auto ret1 = ops::_Retval(s.WithOpName("ret1"), arg0, 1);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "f1", xla_fdef));
  }
  {
    // Function for While's "body".
    // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_BOOL)
    // "ret0" = "arg0"
    // "ret1" = "arg1"
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
    Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_BOOL, 1);
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), arg0, 0);
    auto ret1 = ops::_Retval(s.WithOpName("ret1"), arg1, 1);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "f2", xla_fdef));
  }
  {
    // Function for While's "cond".
    // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_BOOL)
    // "ret0" = "arg1"
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
    Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_BOOL, 1);
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), arg1, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "f3", xla_fdef));
  }
  FunctionLibraryDefinition fld(OpRegistry::Global(), fdl);

  // Build the XLA computation graph.
  // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_INT32)
  // "arg0", "arg1" -> "if" (If) -> "ret0", "ret1"
  // "arg0", "arg1" -> "while" (While) -> "ret2", "ret3"
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
  Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_BOOL, 1);
  NameAttrList f;
  f.set_name("f1");
  auto if_op = ops::If(s.WithOpName("if"), arg1,
                       std::initializer_list<Input>{arg0, arg1},
                       {DT_BOOL, DT_RESOURCE}, f, f);
  auto ret0 = ops::_Retval(s.WithOpName("ret0"), if_op.output[0], 0);
  auto ret1 = ops::_Retval(s.WithOpName("ret1"), if_op.output[1], 1);
  NameAttrList cond_fn, body_fn;
  cond_fn.set_name("f3");
  body_fn.set_name("f2");
  auto while_op =
      ops::While(s.WithOpName("while"),
                 std::initializer_list<Input>{arg0, arg1}, cond_fn, body_fn);
  auto ret2 = ops::_Retval(s.WithOpName("ret2"), while_op.output[0], 2);
  auto ret3 = ops::_Retval(s.WithOpName("ret3"), while_op.output[1], 3);
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(s.ToGraph(g.get()));

  std::vector<std::unique_ptr<FunctionBody>> fbodies;
  TF_CHECK_OK(RearrangeFunctionArguments(
      [&](const NameAttrList &function, const FunctionBody **fbody) {
        std::unique_ptr<FunctionBody> new_fbody;
        TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fld.Find(function.name()),
                                                   AttrSlice(&function.attr()),
                                                   &fld, &new_fbody));
        *fbody = new_fbody.get();
        fbodies.push_back(std::move(new_fbody));
        return absl::OkStatus();
      },
      g.get(), &fld));

  // Check function f1_rearrange_0, input types should be {DT_BOOL, DT_RESOURCE}
  // and output types should be {DT_BOOL}.
  const FunctionDef *f1_rewritten = fld.Find("f1_rearrange_0");
  CHECK_NE(f1_rewritten, nullptr);
  ASSERT_EQ(f1_rewritten->signature().input_arg_size(), 2);
  EXPECT_EQ(f1_rewritten->signature().input_arg(0).type(), DT_BOOL);
  EXPECT_EQ(f1_rewritten->signature().input_arg(1).type(), DT_RESOURCE);
  ASSERT_EQ(f1_rewritten->signature().output_arg_size(), 1);
  EXPECT_EQ(f1_rewritten->signature().output_arg(0).type(), DT_BOOL);

  // Check node "if" input and output edges.
  auto node_name_index = g->BuildNodeNameIndex();
  const Node *if_node = node_name_index.at("if");
  ASSERT_NE(if_node, nullptr);
  const Node *input_node;
  TF_CHECK_OK(if_node->input_node(1, &input_node));
  EXPECT_EQ(input_node->name(), "arg1");
  TF_CHECK_OK(if_node->input_node(2, &input_node));
  EXPECT_EQ(input_node->name(), "arg0");
  const Node *ret0_node = node_name_index.at("ret0");
  ASSERT_NE(ret0_node, nullptr);
  TF_CHECK_OK(ret0_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "if");
  const Node *ret1_node = node_name_index.at("ret1");
  ASSERT_NE(ret1_node, nullptr);
  TF_CHECK_OK(ret1_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "arg0");

  // Check node "while" input and output edges.
  const Node *while_node = node_name_index.at("while");
  ASSERT_NE(while_node, nullptr);
  TF_CHECK_OK(while_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "arg1");
  TF_CHECK_OK(while_node->input_node(1, &input_node));
  EXPECT_EQ(input_node->name(), "arg0");
  const Node *ret2_node = node_name_index.at("ret2");
  ASSERT_NE(ret2_node, nullptr);
  TF_CHECK_OK(ret2_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "arg0");
  const Node *ret3_node = node_name_index.at("ret3");
  ASSERT_NE(ret3_node, nullptr);
  TF_CHECK_OK(ret3_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "while");
}

TEST(RearrangeFunctionArgumentForFunctionTest,
     WhileResourceRetvalFromDifferentArgUnimplemented) {
  FunctionDefLibrary fdl;
  {
    // Function for While's "body".
    // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_RESOURCE), "arg2" (T=DT_INT32)
    // "ret0" = "arg1"
    // "ret1" = "arg0"
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
    Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_RESOURCE, 1);
    Output arg2 = ops::_Arg(s.WithOpName("arg2"), DT_INT32, 2);
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), arg1, 0);
    auto ret1 = ops::_Retval(s.WithOpName("ret1"), arg0, 1);
    auto ret2 = ops::_Retval(s.WithOpName("ret2"), arg2, 2);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "f2", xla_fdef));
  }
  {
    // Function for While's "cond".
    // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_RESOURCE), "arg2" (T=DT_INT32)
    // "ret0" = true
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
    Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_RESOURCE, 1);
    Output arg2 = ops::_Arg(s.WithOpName("arg2"), DT_INT32, 2);
    Output cond = ops::Const(s.WithOpName("const"), true, TensorShape({}));
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), cond, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "f1", xla_fdef));
  }
  FunctionLibraryDefinition fld(OpRegistry::Global(), fdl);

  // Build the XLA computation graph.
  // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_RESOURCE), "arg2" (T=DT_INT32)
  // "arg0", "arg1" -> "while" (While)
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
  Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_RESOURCE, 1);
  Output arg2 = ops::_Arg(s.WithOpName("arg2"), DT_INT32, 2);
  NameAttrList cond_fn, body_fn;
  cond_fn.set_name("f1");
  body_fn.set_name("f2");
  auto while_op = ops::While(s.WithOpName("while"),
                             std::initializer_list<Input>{arg0, arg1, arg2},
                             cond_fn, body_fn);
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(s.ToGraph(g.get()));

  std::vector<std::unique_ptr<FunctionBody>> fbodies;
  Status status = RearrangeFunctionArguments(
      [&](const NameAttrList &function, const FunctionBody **fbody) {
        std::unique_ptr<FunctionBody> new_fbody;
        TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fld.Find(function.name()),
                                                   AttrSlice(&function.attr()),
                                                   &fld, &new_fbody));
        *fbody = new_fbody.get();
        fbodies.push_back(std::move(new_fbody));
        return absl::OkStatus();
      },
      g.get(), &fld);
  EXPECT_EQ(status.code(), error::UNIMPLEMENTED);
}

}  // namespace tensorflow
