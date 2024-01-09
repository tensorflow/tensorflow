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

#include "tensorflow/core/framework/graph_to_functiondef.h"

#include <utility>
#include <vector>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/base64.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

FunctionDef RemoveDebugInfo(const FunctionDef& def) {
  FunctionDef copy = def;
  for (auto& node_def : *copy.mutable_node_def()) {
    node_def.clear_experimental_debug_info();
  }
  return copy;
}

bool EqualFunctionDef(const FunctionDef& a, const FunctionDef& b,
                      string* diff) {
  // TODO(phawkins) use a more sophisticated equality test.
  if (a.DebugString() != b.DebugString()) {
    if (diff) {
      *diff = strings::StrCat("Definition mismatch for function ",
                              a.signature().name(), ":\n", a.DebugString(),
                              "\n ---- vs. ----\n", b.DebugString());
    }
    return false;
  }
  return true;
}

TEST(GraphToFunctionDefTest, Basics) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  auto b = ops::_Arg(root.WithOpName("B"), DT_FLOAT, 1);
  auto c = ops::_Arg(root.WithOpName("C"), DT_FLOAT, 2);
  auto d = ops::Add(root.WithOpName("D"), a, b);
  auto e = ops::Add(root.WithOpName("b"), d, c);
  auto f = ops::Neg(root.WithOpName("h"), e);
  auto g = ops::AddN(root.WithOpName("G"), std::initializer_list<Output>{e, f});
  auto h = ops::_Retval(root.WithOpName("H"), g, 0);

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphConstructorOptions options;
  TF_EXPECT_OK(ConvertGraphDefToGraph(options, graph_def, graph.get()));

  FunctionDef fdef;
  TF_EXPECT_OK(GraphToFunctionDef(*graph, "test_fn", &fdef));

  FunctionDef fdef_expected = FunctionDefHelper::Create(
      "test_fn",                             // function name
      {"a: float", "b: float", "c: float"},  // inputs
      {"h: float"},                          // outputs
      {},                                    // attrs
      {
          // nodes in the function body
          {{"D"}, "Add", {"a", "b"}, {{"T", DT_FLOAT}}},
          {{"b_0"}, "Add", {"D:z:0", "c"}, {{"T", DT_FLOAT}}},
          {{"h_0"}, "Neg", {"b_0:z:0"}, {{"T", DT_FLOAT}}},
          {{"G"}, "AddN", {"b_0:z:0", "h_0:y:0"}, {{"N", 2}, {"T", DT_FLOAT}}},
      },
      {{"h", "G:sum:0"}});  // return values

  string diff;
  bool fdefs_equal =
      EqualFunctionDef(fdef_expected, RemoveDebugInfo(fdef), &diff);

  EXPECT_TRUE(fdefs_equal) << diff;
}

TEST(GraphToFunctionDefTest, OverrideOutputNames) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  auto b = ops::_Retval(root.WithOpName("H"), a, 0);

  FunctionDef fdef;
  // Override the output name from h to b.
  TF_EXPECT_OK(GraphToFunctionDef(*root.graph(), "test_fn", {"b"}, &fdef));

  FunctionDef fdef_expected =
      FunctionDefHelper::Create("test_fn",      // function name
                                {"a: float"},   // inputs
                                {"b: float"},   // outputs
                                {},             // attrs
                                {},             // body
                                {{"b", "a"}});  // return values

  string diff;
  bool fdefs_equal =
      EqualFunctionDef(fdef_expected, RemoveDebugInfo(fdef), &diff);

  EXPECT_TRUE(fdefs_equal) << diff;
}

TEST(GraphToFunctionDefTest, DuplicatedOutputNames) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  auto b = ops::_Retval(root.WithOpName("B"), a, 0);
  auto c = ops::_Retval(root.WithOpName("C"), a, 1);

  FunctionDef fdef;
  // Duplicated output names.
  auto status = GraphToFunctionDef(*root.graph(), "test_fn", {"d", "d"}, &fdef);

  EXPECT_THAT(status, tensorflow::testing::StatusIs(
                          error::INVALID_ARGUMENT,
                          "Cannot have duplicate output names. Name 'd' "
                          "appears more than once in 'output_names' array."));
}

TEST(GraphToFunctionDefTest, ArgAttrShape) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  // Attr "shape" is auto renamed to "_output_shapes".
  AttrValue shape_attr;
  *(shape_attr.mutable_shape()) = TensorShape({1, 2}).AsProto();
  a.node()->AddAttr("shape", shape_attr);
  auto b = ops::_Retval(root.WithOpName("B"), a, 0);

  FunctionDef fdef;
  TF_EXPECT_OK(GraphToFunctionDef(*root.graph(), "test_fn", &fdef));

  FunctionDef fdef_expected =
      FunctionDefHelper::Create("test_fn",      // function name
                                {"a: float"},   // inputs
                                {"b: float"},   // outputs
                                {},             // attrs
                                {},             // body
                                {{"b", "a"}});  // return values

  FunctionDef::ArgAttrs attrs;
  AttrValue output_shapes;
  *(output_shapes.mutable_list()->add_shape()) = TensorShape({1, 2}).AsProto();
  attrs.mutable_attr()->insert({"_output_shapes", output_shapes});
  (*fdef_expected.mutable_arg_attr())[0] = std::move(attrs);

  string diff;
  bool fdefs_equal =
      EqualFunctionDef(fdef_expected, RemoveDebugInfo(fdef), &diff);

  EXPECT_TRUE(fdefs_equal) << diff;
}

TEST(GraphToFunctionDefTest, ArgAttrPrivateAttr) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  // Private arg attr starting with "_" are copied to fdef arg_attr.
  AttrValue private_attr;
  *(private_attr.mutable_s()) = "value";
  a.node()->AddAttr("_name", private_attr);
  auto b = ops::_Retval(root.WithOpName("B"), a, 0);

  FunctionDef fdef;
  TF_EXPECT_OK(GraphToFunctionDef(*root.graph(), "test_fn", &fdef));

  FunctionDef fdef_expected =
      FunctionDefHelper::Create("test_fn",      // function name
                                {"a: float"},   // inputs
                                {"b: float"},   // outputs
                                {},             // attrs
                                {},             // body
                                {{"b", "a"}});  // return values

  FunctionDef::ArgAttrs attrs;
  attrs.mutable_attr()->insert({"_name", private_attr});
  (*fdef_expected.mutable_arg_attr())[0] = std::move(attrs);

  string diff;
  bool fdefs_equal =
      EqualFunctionDef(fdef_expected, RemoveDebugInfo(fdef), &diff);

  EXPECT_TRUE(fdefs_equal) << diff;
}

TEST(GraphToFunctionDefTest, ArgAttrConstInput) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::Const(root.WithOpName("A"), 0.0f, {2, 2});
  // Attr "shape" with dtype other than DT_RESOURCE is copied to fdef arg_attr.
  Tensor t(DT_FLOAT, TensorShape({2, 2}));
  TensorProto t_proto;
  t.AsProtoField(&t_proto);
  AttrValue attr;
  *(attr.mutable_tensor()) = std::move(t_proto);
  a.node()->AddAttr("value", attr);
  a.node()->AddAttr("index", 0);
  auto b = ops::_Retval(root.WithOpName("B"), a, 0);

  std::vector<OutputTensor> inputs;
  std::vector<OutputTensor> outputs;
  auto add_arg_or_retval = [](Node* node,
                              std::vector<OutputTensor>* args_or_retvals) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
    if (index >= args_or_retvals->size()) {
      args_or_retvals->resize(index + 1);
    }
    (*args_or_retvals)[index].node = node;
    return OkStatus();
  };
  for (Node* node : root.graph()->op_nodes()) {
    // Set const as the input node.
    if (node->IsConstant()) {
      TF_EXPECT_OK(add_arg_or_retval(node, &inputs));
    } else {
      TF_EXPECT_OK(add_arg_or_retval(node, &outputs));
    }
  }

  FunctionDef fdef;
  // Adds description.
  TF_EXPECT_OK(GraphToFunctionDef(
      *root.graph(), "test_fn", /*append_hash_to_fn_name=*/false,
      /*set_stateful_from_nodes=*/false,
      /*copy_placeholder_attrs_from_nodes=*/false, /*body_nodes*/ {}, inputs,
      outputs,
      /*output_names*/ {}, /*control_outputs=*/{}, /*control_output_names=*/{},
      /*description=*/"ArgAttrConstInput", &fdef));

  FunctionDef fdef_expected =
      FunctionDefHelper::Create("test_fn",      // function name
                                {"a: float"},   // inputs
                                {"b: float"},   // outputs
                                {},             // attrs
                                {},             // body
                                {{"b", "a"}});  // return values

  AttrValue value;
  *(value.mutable_list()->add_shape()) = TensorShape({2, 2}).AsProto();
  FunctionDef::ArgAttrs attrs;
  attrs.mutable_attr()->insert({"_output_shapes", value});
  (*fdef_expected.mutable_arg_attr())[0] = std::move(attrs);
  (*fdef_expected.mutable_signature()->mutable_description()) =
      "ArgAttrConstInput";

  string diff;
  bool fdefs_equal =
      EqualFunctionDef(fdef_expected, RemoveDebugInfo(fdef), &diff);

  EXPECT_TRUE(fdefs_equal) << diff;
}

TEST(GraphToFunctionDefTest, AppendHashToFnName) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::Const(root.WithOpName("A"), 0.0f, {2, 2});
  AttrValue foo;
  *foo.mutable_placeholder() = "foo";
  a.node()->AddAttr("attr_name_not_found", foo);

  std::vector<const Node*> body_nodes;
  for (Node* node : root.graph()->op_nodes()) {
    body_nodes.push_back(node);
  }

  FunctionDef fdef;
  // Set append_hash_to_fn_name to true.
  TF_EXPECT_OK(GraphToFunctionDef(
      *root.graph(), "test_fn", /*append_hash_to_fn_name=*/true,
      /*set_stateful_from_nodes=*/false,
      /*copy_placeholder_attrs_from_nodes=*/false, /*body_nodes*/ body_nodes,
      /*inputs*/ {},
      /*outputs*/ {},
      /*output_names*/ {}, /*control_outputs=*/{}, /*control_output_names=*/{},
      /*description=*/nullptr, &fdef));

  // Hash appended after "test_fn".
  EXPECT_TRUE(absl::StartsWith(fdef.signature().name(), "test_fn_"));
}

TEST(GraphToFunctionDefTest, CopyPlaceholderAttrsFromNodes) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::VarHandleOp(root.WithOpName("var"), DT_FLOAT, {});
  AttrValue foo;
  *foo.mutable_placeholder() = "foo";
  // The op_def of VarHandleOp has a "shared_name" attribute.
  a.node()->AddAttr("shared_name", foo);
  std::vector<const Node*> body_nodes;
  for (Node* node : root.graph()->op_nodes()) {
    body_nodes.push_back(node);
  }

  FunctionDef fdef;
  TF_EXPECT_OK(GraphToFunctionDef(
      *root.graph(), "test_fn", /*append_hash_to_fn_name=*/false,
      /*set_stateful_from_nodes=*/false,
      /*copy_placeholder_attrs_from_nodes=*/true, body_nodes, /*inputs*/ {},
      /*outputs*/ {},
      /*output_names*/ {}, /*control_outputs=*/{}, /*control_output_names=*/{},
      /*description=*/nullptr, &fdef));
}

TEST(GraphToFunctionDefTest, CopyPlaceholderAttrsFromNodesUnImplemented) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::Const(root.WithOpName("A"), 0.0f, {2, 2});
  AttrValue foo;
  *foo.mutable_placeholder() = "foo";
  // Const op_def doesn't have a "attr_name_not_found" attr.
  a.node()->AddAttr("attr_name_not_found", foo);
  std::vector<const Node*> body_nodes;
  for (Node* node : root.graph()->op_nodes()) {
    body_nodes.push_back(node);
  }

  FunctionDef fdef;
  auto status = GraphToFunctionDef(
      *root.graph(), "test_fn", /*append_hash_to_fn_name=*/false,
      /*set_stateful_from_nodes=*/false,
      /*copy_placeholder_attrs_from_nodes=*/true, body_nodes, /*inputs*/ {},
      /*outputs*/ {},
      /*output_names*/ {}, /*control_outputs=*/{}, /*control_output_names=*/{},
      /*description=*/nullptr, &fdef);

  EXPECT_EQ(status.code(), error::UNIMPLEMENTED);
}

// Regression test for a crash if there was a control edge to a _Retval node.
TEST(GraphToFunctionDefTest, ControlDependencies) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("a"), DT_FLOAT, 0);
  auto b = ops::Neg(root.WithOpName("b").WithControlDependencies(a), a);
  auto c = ops::_Retval(root.WithOpName("c").WithControlDependencies(b), b, 0);

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphConstructorOptions options;
  TF_EXPECT_OK(ConvertGraphDefToGraph(options, graph_def, graph.get()));

  FunctionDef fdef;
  TF_EXPECT_OK(GraphToFunctionDef(*graph, "test_fn", &fdef));

  FunctionDef fdef_expected = FunctionDefHelper::Create(
      "test_fn",     // function name
      {"a: float"},  // inputs
      {"c: float"},  // outputs
      {},            // attrs
      {
          // nodes in the function body
          {{"b"}, "Neg", {"a", "^a"}, {{"T", DT_FLOAT}}},
      },
      {{"c", "b:y:0"}});  // return values

  string diff;
  bool fdefs_equal =
      EqualFunctionDef(fdef_expected, RemoveDebugInfo(fdef), &diff);

  EXPECT_TRUE(fdefs_equal) << diff;
}

TEST(GraphToFunctionDefTest, ControlOutputs) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("a"), DT_FLOAT, 0);
  auto b = ops::Neg(root.WithOpName("b"), a);
  auto c = ops::_Retval(root.WithOpName("c"), b, 0);

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphConstructorOptions options;
  TF_EXPECT_OK(ConvertGraphDefToGraph(options, graph_def, graph.get()));

  // Add a 'b' node to the control return set.
  const auto control_ret = [](const Node* n) -> absl::optional<string> {
    if (n->name() == "b") return absl::make_optional<string>("must_execute");
    return absl::nullopt;
  };

  FunctionDef fdef;
  TF_EXPECT_OK(GraphToFunctionDef(*graph, "test_fn", control_ret, &fdef));

  FunctionDef fdef_expected =
      FunctionDefHelper::Create("test_fn",     // function name
                                {"a: float"},  // inputs
                                {"c: float"},  // outputs
                                {},            // attrs
                                {
                                    // nodes in the function body
                                    {{"b"}, "Neg", {"a"}, {{"T", DT_FLOAT}}},
                                },
                                {{"c", "b:y:0"}},          // return values
                                {{"must_execute", "b"}});  // control returns

  string diff;
  bool fdefs_equal =
      EqualFunctionDef(fdef_expected, RemoveDebugInfo(fdef), &diff);

  EXPECT_TRUE(fdefs_equal) << diff;
}

}  // namespace
}  // namespace tensorflow
