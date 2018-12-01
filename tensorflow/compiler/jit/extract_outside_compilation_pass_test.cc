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

#include "absl/strings/match.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
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

TEST(ExtractOutsideCompilationForFunctionTest, Basic) {
  // Build the XLA computation func.
  // "const0"
  // "identity0" = "const0" (outside compilation cluster "0")
  // "identity1" = "identity0" (outside compilation cluster "1")
  // "identity2" = "identity1"
  FunctionDefLibrary fdl;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output const0 = ops::Const(s.WithOpName("const0"), 1, {2});
    Output identity0 = ops::Identity(s.WithOpName("identity0"), const0);
    Output identity1 = ops::Identity(s.WithOpName("identity1"), identity0);
    Output identity2 = ops::Identity(s.WithOpName("identity2"), identity1);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    auto node_name_image = g->BuildNodeNameIndex();
    node_name_image["identity0"]->AddAttr("_oc", "0");
    node_name_image["identity1"]->AddAttr("_oc", "1");
    PartialTensorShape shape({2});
    node_name_image["identity1"]->AddAttr(
        kXlaInferredShapesAttrName, std::vector<PartialTensorShape>{shape});

    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "cluster", xla_fdef));
  }
  FunctionLibraryDefinition fld(OpRegistry::Global(), fdl);

  protobuf::Map<string, tensorflow::AttrValue> attrs;
  std::map<string, int> host_compute_core = {{"0", 1}, {"1", 0}};
  std::unique_ptr<Graph> host_graph;
  std::vector<string> shape_inference_graphs;
  bool has_outside_compilation;
  NameAttrList name_attrs;
  name_attrs.set_name("cluster");
  *name_attrs.mutable_attr() = attrs;
  TF_CHECK_OK(ExtractOutsideCompilationForFunction(
      "_xla", "_oc", "cluster", name_attrs, "cluster_rewritten",
      host_compute_core, &fld, &host_graph, &shape_inference_graphs,
      &has_outside_compilation));

  // Get rewritten XLA computation function.
  FunctionBody *fbody = nullptr;
  TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("cluster_rewritten"),
                                      AttrSlice(), &fld,
                                      [&](const string &op, const OpDef **sig) {
                                        return fld.LookUpOpDef(op, sig);
                                      },
                                      &fbody));
  std::unique_ptr<FunctionBody> fbody_deleter(fbody);
  auto node_name_index = fbody->graph->BuildNodeNameIndex();

  // Check XlaHostCompute nodes.
  Node *host_compute_0 = node_name_index["outside_compilation_0_host_compute"];
  EXPECT_NE(host_compute_0, nullptr);
  Node *host_compute_1 = node_name_index["outside_compilation_1_host_compute"];
  EXPECT_NE(host_compute_1, nullptr);
  // Check XlaHostCompute nodes' "tpu_core" attr.
  int tpu_core;
  TF_CHECK_OK(GetNodeAttr(host_compute_0->attrs(), "tpu_core", &tpu_core));
  EXPECT_EQ(tpu_core, 1);
  TF_CHECK_OK(GetNodeAttr(host_compute_1->attrs(), "tpu_core", &tpu_core));
  EXPECT_EQ(tpu_core, 0);
  // Check XlaHostCompute nodes' "shapes" attr. "0" should not have shapes, and
  // "1" should have shapes.
  std::vector<TensorShapeProto> shapes;
  TF_CHECK_OK(GetNodeAttr(host_compute_0->attrs(), "shapes", &shapes));
  EXPECT_EQ(shapes.size(), 0);
  TF_CHECK_OK(GetNodeAttr(host_compute_1->attrs(), "shapes", &shapes));
  EXPECT_EQ(shapes.size(), 1);
  EXPECT_EQ(shapes[0].dim_size(), 1);
  // Check XlaHostCompute nodes' "shape_inference_graph" attr. Both should have
  // empty values.
  string shape_inference_graph;
  TF_CHECK_OK(GetNodeAttr(host_compute_0->attrs(), "shape_inference_graph",
                          &shape_inference_graph));
  EXPECT_EQ(shape_inference_graph, "");
  TF_CHECK_OK(GetNodeAttr(host_compute_1->attrs(), "shape_inference_graph",
                          &shape_inference_graph));
  EXPECT_EQ(shape_inference_graph, "");

  // Check `shape_inference_graphs`.
  EXPECT_EQ(shape_inference_graphs.size(), 0);

  // Check `host_graph`: verify we have key placeholder and sequencer.
  Node *key_placeholder = nullptr, *sequencer = nullptr;
  for (Node *n : host_graph->nodes()) {
    if (n->type_string() == "Placeholder" &&
        absl::EndsWith(n->name(), "_key_placeholder")) {
      EXPECT_EQ(key_placeholder, nullptr);
      key_placeholder = n;
    } else if (HasNodeAttr(n->def(), "_xla_host_transfer_sequencer")) {
      EXPECT_EQ(sequencer, nullptr);
      sequencer = n;
    }
  }
  EXPECT_NE(key_placeholder, nullptr);
  EXPECT_NE(sequencer, nullptr);
  // Check SendFromHost and RecvAtHost has key placeholder as input, and have
  // control edge to sequencer.
  int num_send_from_host = 0, num_recv_at_host = 0;
  std::vector<Node *> send_recv_nodes;
  for (Node *n : host_graph->nodes()) {
    if (n->type_string() == "_XlaSendFromHost") {
      num_send_from_host++;
      send_recv_nodes.push_back(n);
    } else if (n->type_string() == "_XlaRecvAtHost") {
      num_recv_at_host++;
      send_recv_nodes.push_back(n);
    }
  }
  EXPECT_EQ(num_send_from_host, 1);
  EXPECT_EQ(num_recv_at_host, 1);
  for (Node *n : send_recv_nodes) {
    Node *input_node;
    TF_CHECK_OK(n->input_node(n->num_inputs() - 1, &input_node));
    EXPECT_EQ(input_node, key_placeholder);

    bool has_control_edge_to_sequencer = false;
    for (const Edge *e : n->out_edges()) {
      if (e->IsControlEdge() && e->dst() == sequencer) {
        has_control_edge_to_sequencer = true;
        break;
      }
    }
    EXPECT_TRUE(has_control_edge_to_sequencer);
  }
}

TEST(ExtractOutsideCompilationForFunctionTest, NoHostGraph) {
  // Build the XLA computation func.
  // "const0"
  FunctionDefLibrary fdl;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output const0 = ops::Const(s.WithOpName("const0"), 1, {2});
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));

    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "cluster", xla_fdef));
  }
  FunctionLibraryDefinition fld(OpRegistry::Global(), fdl);

  protobuf::Map<string, tensorflow::AttrValue> attrs;
  std::map<string, int> host_compute_core = {{"0", 1}, {"1", 0}};
  std::unique_ptr<Graph> host_graph;
  std::vector<string> shape_inference_graphs;
  bool has_outside_compilation;
  NameAttrList name_attrs;
  name_attrs.set_name("cluster");
  *name_attrs.mutable_attr() = attrs;
  TF_CHECK_OK(ExtractOutsideCompilationForFunction(
      "_xla", "_oc", "cluster", name_attrs, "cluster_rewritten",
      host_compute_core, &fld, &host_graph, &shape_inference_graphs,
      &has_outside_compilation));

  // Check `host_graph` is empty.
  EXPECT_FALSE(host_graph);
}

TEST(ExtractOutsideCompilationForFunctionTest, XlaHostComputeRemoved) {
  // Build the XLA computation func.
  // "const0"
  // "const1" (outside compilation clsuter "0")
  FunctionDefLibrary fdl;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output const0 = ops::Const(s.WithOpName("const0"), 1, {2});
    Output const1 = ops::Const(s.WithOpName("const1"), 1, {2});
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    auto node_name_image = g->BuildNodeNameIndex();
    node_name_image["const1"]->AddAttr("_oc", "0");

    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "cluster", xla_fdef));
  }
  FunctionLibraryDefinition fld(OpRegistry::Global(), fdl);

  protobuf::Map<string, tensorflow::AttrValue> attrs;
  std::map<string, int> host_compute_core = {{"0", 1}, {"1", 0}};
  std::unique_ptr<Graph> host_graph;
  std::vector<string> shape_inference_graphs;
  bool has_outside_compilation;
  NameAttrList name_attrs;
  name_attrs.set_name("cluster");
  *name_attrs.mutable_attr() = attrs;
  TF_CHECK_OK(ExtractOutsideCompilationForFunction(
      "_xla", "_oc", "cluster", name_attrs, "cluster_rewritten",
      host_compute_core, &fld, &host_graph, &shape_inference_graphs,
      &has_outside_compilation));

  // Check rewritten XLA graph: verify that we have no XlaHostCompute.
  FunctionBody *fbody = nullptr;
  TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("cluster_rewritten"),
                                      AttrSlice(), &fld,
                                      [&](const string &op, const OpDef **sig) {
                                        return fld.LookUpOpDef(op, sig);
                                      },
                                      &fbody));
  std::unique_ptr<FunctionBody> fbody_deleter(fbody);
  for (Node *n : fbody->graph->nodes()) {
    EXPECT_NE(n->type_string(), "XlaHostCompute");
  }

  // Check `host_graph`: verify we have no placeholder, but we have "const1".
  int num_key_placeholders = 0;
  for (Node *n : host_graph->nodes()) {
    if (n->type_string() == "Placeholder" &&
        absl::EndsWith(n->name(), "_key_placeholder")) {
      num_key_placeholders++;
    }
  }
  EXPECT_EQ(num_key_placeholders, 0);
  auto node_name_index = host_graph->BuildNodeNameIndex();
  EXPECT_NE(node_name_index.find("const1"), node_name_index.end());
}

}  // namespace tensorflow
