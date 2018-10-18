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

#include "tensorflow/compiler/jit/extract_outside_compilation_pass.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(RewriteOutsideCompilationSubgraphFnTest, Basic) {
  // Build the graph:
  // "add" = "arg0" + "arg1"
  // "ret0" = "add"
  // "ret1" = "arg1"
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_INT32, 0);
  Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_FLOAT, 1);
  Output arg2 = ops::_Arg(s.WithOpName("arg2"), DT_INT32, 2);
  Output add = ops::Add(s.WithOpName("add"), arg0, arg0);
  auto ret0 = ops::_Retval(s.WithOpName("ret0"), add, 0);
  auto ret1 = ops::_Retval(s.WithOpName("ret1"), arg1, 1);
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(s.ToGraph(g.get()));
  auto node_name_image = g->BuildNodeNameIndex();
  Node *add_node = node_name_image["add"];
  EXPECT_NE(add_node, nullptr);
  add_node->AddAttr(kXlaConnectedToXlaComputationAttrName, "cluster");
  add_node->AddAttr(kXlaConnectedFromXlaComputationAttrName, "cluster");

  RewriteOutsideCompilationSubgraphFn rewrite_fn("_xla", "_oc", "cluster");
  std::vector<OutputTensor> arg_source_tensors;
  NodeDef call_node_def;
  call_node_def.set_op("0");
  TF_CHECK_OK(
      rewrite_fn(arg_source_tensors, &g, nullptr, nullptr, &call_node_def));
  node_name_image = g->BuildNodeNameIndex();

  // Verify step 1: add key placeholder node.
  Node *key_placeholder = node_name_image["cluster_key_placeholder"];
  EXPECT_NE(key_placeholder, nullptr);
  // Verify step 2: replace _Arg nodes with XlaRecvAtHost.
  for (Node *n : g->nodes()) {
    EXPECT_NE(n->type_string(), "_Arg");
  }
  Node *recv_at_host = node_name_image["outside_compilation_cluster_0_recv"];
  EXPECT_NE(recv_at_host, nullptr);
  std::vector<DataType> recv_at_host_dtypes;
  TF_CHECK_OK(
      GetNodeAttr(recv_at_host->attrs(), "Toutputs", &recv_at_host_dtypes));
  EXPECT_EQ(recv_at_host_dtypes.size(), 3);
  EXPECT_EQ(recv_at_host_dtypes[0], DT_INT32);
  EXPECT_EQ(recv_at_host_dtypes[1], DT_FLOAT);
  EXPECT_EQ(recv_at_host_dtypes[2], DT_INT32);
  // Verify step 3: replace _Retval nodes with XlaSendFromHost.
  for (Node *n : g->nodes()) {
    EXPECT_NE(n->type_string(), "_Retval");
  }
  Node *send_from_host = node_name_image["outside_compilation_cluster_0_send"];
  EXPECT_NE(send_from_host, nullptr);
  std::vector<DataType> send_from_host_dtypes;
  TF_CHECK_OK(
      GetNodeAttr(send_from_host->attrs(), "Tinputs", &send_from_host_dtypes));
  EXPECT_EQ(send_from_host_dtypes.size(), 2);
  EXPECT_EQ(send_from_host_dtypes[0], DT_INT32);
  EXPECT_EQ(send_from_host_dtypes[1], DT_FLOAT);
  // Verify step 4: nodes marked with XLA cluster and outside compilation attr.
  add_node = node_name_image["add"];
  EXPECT_NE(add_node, nullptr);
  EXPECT_TRUE(HasNodeAttr(add_node->def(), "_xla"));
  EXPECT_TRUE(HasNodeAttr(add_node->def(), "_oc"));
  // Verify step 5: control edges added.
  bool has_control_edge_from_recv_at_host = false;
  for (auto e : add_node->in_edges()) {
    if (e->IsControlEdge() && e->src() == recv_at_host) {
      has_control_edge_from_recv_at_host = true;
    }
  }
  EXPECT_TRUE(has_control_edge_from_recv_at_host);
  bool has_control_edge_to_send_from_host = false;
  for (auto e : add_node->out_edges()) {
    if (e->IsControlEdge() && e->dst() == send_from_host) {
      has_control_edge_to_send_from_host = true;
    }
  }
  EXPECT_TRUE(has_control_edge_to_send_from_host);
  // Verify step 7: necessary attrs added to call_node_def.
  string shape_inference_graph;
  TF_CHECK_OK(GetNodeAttr(AttrSlice(&call_node_def.attr()),
                          "shape_inference_graph", &shape_inference_graph));
  EXPECT_EQ(shape_inference_graph,
            "_outside_compilation_shape_inference_cluster_0");
}

TEST(RewriteOutsideCompilationSubgraphFnTest, NoSendFromHost) {
  // Build the graph: only 1 node: "arg0"
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_INT32, 0);
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(s.ToGraph(g.get()));

  RewriteOutsideCompilationSubgraphFn rewrite_fn("_xla", "_oc", "cluster");
  std::vector<OutputTensor> arg_source_tensors;
  NodeDef call_node_def;
  call_node_def.set_op("0");
  TF_CHECK_OK(
      rewrite_fn(arg_source_tensors, &g, nullptr, nullptr, &call_node_def));
  auto node_name_image = g->BuildNodeNameIndex();

  // Check key placeholder and RecvAtHost is present, but SendFromHost is not.
  Node *key_placeholder = node_name_image["cluster_key_placeholder"];
  EXPECT_NE(key_placeholder, nullptr);
  Node *recv_at_host = node_name_image["outside_compilation_cluster_0_recv"];
  EXPECT_NE(recv_at_host, nullptr);
  Node *send_from_host = node_name_image["outside_compilation_cluster_0_send"];
  EXPECT_EQ(send_from_host, nullptr);
}

TEST(RewriteOutsideCompilationSubgraphFnTest, NoRecvAtHost) {
  // Build the graph:
  // "ret" = "const0"
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output const0 = ops::Const(s.WithOpName("const0"), 1, {2});
  auto ret = ops::_Retval(s.WithOpName("ret"), const0, 0);
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(s.ToGraph(g.get()));

  RewriteOutsideCompilationSubgraphFn rewrite_fn("_xla", "_oc", "cluster");
  std::vector<OutputTensor> arg_source_tensors;
  NodeDef call_node_def;
  call_node_def.set_op("0");
  TF_CHECK_OK(
      rewrite_fn(arg_source_tensors, &g, nullptr, nullptr, &call_node_def));
  auto node_name_image = g->BuildNodeNameIndex();

  // Check key placeholder and SendFromHost is present, but RecvAtHost is not.
  Node *key_placeholder = node_name_image["cluster_key_placeholder"];
  EXPECT_NE(key_placeholder, nullptr);
  Node *recv_at_host = node_name_image["outside_compilation_cluster_0_recv"];
  EXPECT_EQ(recv_at_host, nullptr);
  Node *send_from_host = node_name_image["outside_compilation_cluster_0_send"];
  EXPECT_NE(send_from_host, nullptr);
}

TEST(RewriteOutsideCompilationSubgraphFnTest, NoKeyPlaceholder) {
  // Build the graph: only 1 node: "const0"
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output const0 = ops::Const(s.WithOpName("const0"), 1, {2});
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(s.ToGraph(g.get()));

  RewriteOutsideCompilationSubgraphFn rewrite_fn("_xla", "_oc", "cluster");
  std::vector<OutputTensor> arg_source_tensors;
  NodeDef call_node_def;
  call_node_def.set_op("0");
  TF_CHECK_OK(
      rewrite_fn(arg_source_tensors, &g, nullptr, nullptr, &call_node_def));
  auto node_name_image = g->BuildNodeNameIndex();

  // Check key placeholder/RecvAtHost/SendFromHost are not present.
  Node *key_placeholder = node_name_image["cluster_key_placeholder"];
  EXPECT_EQ(key_placeholder, nullptr);
  Node *recv_at_host = node_name_image["outside_compilation_cluster_0_recv"];
  EXPECT_EQ(recv_at_host, nullptr);
  Node *send_from_host = node_name_image["outside_compilation_cluster_0_send"];
  EXPECT_EQ(send_from_host, nullptr);
}

TEST(RewriteOutsideCompilationSubgraphFnTest, ShapesInferred) {
  // Build the graph:
  // "ret" = "const0"
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output const0 = ops::Const(s.WithOpName("const0"), 1, {2});
  auto ret = ops::_Retval(s.WithOpName("ret"), const0, 0);
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(s.ToGraph(g.get()));
  auto node_name_image = g->BuildNodeNameIndex();
  Node *const0_node = node_name_image["const0"];
  EXPECT_NE(const0_node, nullptr);
  PartialTensorShape shape({2});
  const0_node->AddAttr(kXlaInferredShapesAttrName,
                       std::vector<PartialTensorShape>{shape});

  RewriteOutsideCompilationSubgraphFn rewrite_fn("_xla", "_oc", "cluster");
  std::vector<OutputTensor> arg_source_tensors;
  NodeDef call_node_def;
  call_node_def.set_op("0");
  TF_CHECK_OK(
      rewrite_fn(arg_source_tensors, &g, nullptr, nullptr, &call_node_def));
  node_name_image = g->BuildNodeNameIndex();

  // Check "shape" attr is available in call_node_def.
  std::vector<TensorShapeProto> shapes;
  TF_CHECK_OK(GetNodeAttr(AttrSlice(&call_node_def.attr()), "shapes", &shapes));
  EXPECT_EQ(shapes.size(), 1);
  EXPECT_EQ(shapes[0].dim_size(), 1);
}

}  // namespace tensorflow
