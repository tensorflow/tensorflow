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
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

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

  RewriteOutsideCompilationSubgraphFn rewrite_fn("_xla", "_oc", "cluster", "");
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
  Node *recv_at_host = node_name_image["outside_compilation_cluster__0_recv"];
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
  Node *send_from_host = node_name_image["outside_compilation_cluster__0_send"];
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
  NameAttrList shape_inference_graph;
  TF_CHECK_OK(GetNodeAttr(AttrSlice(&call_node_def.attr()),
                          "shape_inference_graph", &shape_inference_graph));
  EXPECT_EQ(shape_inference_graph.name(),
            "_outside_compilation_shape_inference_cluster__0");
}

TEST(RewriteOutsideCompilationSubgraphFnTest, NoSendFromHost) {
  // Build the graph: only 1 node: "arg0"
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_INT32, 0);
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(s.ToGraph(g.get()));

  RewriteOutsideCompilationSubgraphFn rewrite_fn("_xla", "_oc", "cluster", "");
  std::vector<OutputTensor> arg_source_tensors;
  NodeDef call_node_def;
  call_node_def.set_op("0");
  TF_CHECK_OK(
      rewrite_fn(arg_source_tensors, &g, nullptr, nullptr, &call_node_def));
  auto node_name_image = g->BuildNodeNameIndex();

  // Check key placeholder and RecvAtHost is present, but SendFromHost is not.
  Node *key_placeholder = node_name_image["cluster_key_placeholder"];
  EXPECT_NE(key_placeholder, nullptr);
  Node *recv_at_host = node_name_image["outside_compilation_cluster__0_recv"];
  EXPECT_NE(recv_at_host, nullptr);
  Node *send_from_host = node_name_image["outside_compilation_cluster__0_send"];
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

  RewriteOutsideCompilationSubgraphFn rewrite_fn("_xla", "_oc", "cluster", "");
  std::vector<OutputTensor> arg_source_tensors;
  NodeDef call_node_def;
  call_node_def.set_op("0");
  TF_CHECK_OK(
      rewrite_fn(arg_source_tensors, &g, nullptr, nullptr, &call_node_def));
  auto node_name_image = g->BuildNodeNameIndex();

  // Check key placeholder and SendFromHost is present, but RecvAtHost is not.
  Node *key_placeholder = node_name_image["cluster_key_placeholder"];
  EXPECT_NE(key_placeholder, nullptr);
  Node *recv_at_host = node_name_image["outside_compilation_cluster__0_recv"];
  EXPECT_EQ(recv_at_host, nullptr);
  Node *send_from_host = node_name_image["outside_compilation_cluster__0_send"];
  EXPECT_NE(send_from_host, nullptr);
}

TEST(RewriteOutsideCompilationSubgraphFnTest, NoKeyPlaceholder) {
  // Build the graph: only 1 node: "const0"
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output const0 = ops::Const(s.WithOpName("const0"), 1, {2});
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(s.ToGraph(g.get()));

  RewriteOutsideCompilationSubgraphFn rewrite_fn("_xla", "_oc", "cluster", "");
  std::vector<OutputTensor> arg_source_tensors;
  NodeDef call_node_def;
  call_node_def.set_op("0");
  TF_CHECK_OK(
      rewrite_fn(arg_source_tensors, &g, nullptr, nullptr, &call_node_def));
  auto node_name_image = g->BuildNodeNameIndex();

  // Check key placeholder/RecvAtHost/SendFromHost are not present.
  Node *key_placeholder = node_name_image["cluster_key_placeholder"];
  EXPECT_EQ(key_placeholder, nullptr);
  Node *recv_at_host = node_name_image["outside_compilation_cluster__0_recv"];
  EXPECT_EQ(recv_at_host, nullptr);
  Node *send_from_host = node_name_image["outside_compilation_cluster__0_send"];
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

  RewriteOutsideCompilationSubgraphFn rewrite_fn("_xla", "_oc", "cluster", "");
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

class ExtractOutsideCompilationForFunctionTest : public ::testing::Test {
 public:
  void SetUp() override {
    SessionOptions session_options;
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        session_options, "/job:localhost/replica:0/task:0", &devices));
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
  }

  Status ExtractOutsideCompilationTest(
      const string &xla_cluster_attr_name,
      const string &outside_compilation_attr_name,
      const string &xla_cluster_name, const NameAttrList &func_name_attrs,
      const string &new_func_name, const string &host_graph_func_name,
      const std::map<string, int> &host_compute_core,
      FunctionLibraryDefinition *fld,
      std::vector<string> *shape_inference_graphs,
      bool *has_outside_compilation) {
    OptimizerOptions opts;
    pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, fld, opts,
        /*default_thread_pool=*/nullptr);
    auto flr = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
    return ExtractOutsideCompilationForFunction(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        func_name_attrs, new_func_name, host_graph_func_name, host_compute_core,
        flr, fld, shape_inference_graphs, has_outside_compilation);
  }

 private:
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
};

TEST_F(ExtractOutsideCompilationForFunctionTest, Basic) {
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
  std::vector<string> shape_inference_graphs;
  bool has_outside_compilation;
  NameAttrList name_attrs;
  name_attrs.set_name("cluster");
  *name_attrs.mutable_attr() = attrs;
  TF_CHECK_OK(ExtractOutsideCompilationTest(
      "_xla", "_oc", "cluster", name_attrs, "cluster_rewritten", "host_graph",
      host_compute_core, &fld, &shape_inference_graphs,
      &has_outside_compilation));

  // Get rewritten XLA computation function.
  std::unique_ptr<FunctionBody> xla_fbody;
  TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("cluster_rewritten"),
                                      AttrSlice(), &fld, &xla_fbody));
  auto node_name_index = xla_fbody->graph->BuildNodeNameIndex();

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
  NameAttrList shape_inference_graph;
  TF_CHECK_OK(GetNodeAttr(host_compute_0->attrs(), "shape_inference_graph",
                          &shape_inference_graph));
  EXPECT_EQ(shape_inference_graph.name(), "");
  TF_CHECK_OK(GetNodeAttr(host_compute_1->attrs(), "shape_inference_graph",
                          &shape_inference_graph));
  EXPECT_EQ(shape_inference_graph.name(), "");

  // Check `shape_inference_graphs`.
  EXPECT_EQ(shape_inference_graphs.size(), 0);

  // Check host graph: verify we have key placeholder and sequencer.
  std::unique_ptr<FunctionBody> host_fbody;
  AttrValue device_ordinal_temp_value;
  device_ordinal_temp_value.set_i(0);
  protobuf::Map<string, AttrValue> host_func_attrs;
  host_func_attrs["_device_ordinal"] = device_ordinal_temp_value;
  TF_CHECK_OK(FunctionDefToBodyHelper(
      *fld.Find("host_graph"), AttrSlice(&host_func_attrs), &fld, &host_fbody));
  Graph *host_graph = host_fbody->graph;
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

TEST_F(ExtractOutsideCompilationForFunctionTest, NoHostGraph) {
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
  std::vector<string> shape_inference_graphs;
  bool has_outside_compilation;
  NameAttrList name_attrs;
  name_attrs.set_name("cluster");
  *name_attrs.mutable_attr() = attrs;
  TF_CHECK_OK(ExtractOutsideCompilationTest(
      "_xla", "_oc", "cluster", name_attrs, "cluster_rewritten", "host_graph",
      host_compute_core, &fld, &shape_inference_graphs,
      &has_outside_compilation));

  // Check host graph is not created.
  EXPECT_EQ(fld.Find("host_graph"), nullptr);
}

TEST_F(ExtractOutsideCompilationForFunctionTest, OutsideCompilationInIf) {
  // Build the XLA computation func.
  // "const0" (bool)
  // "const1" (int32)
  // "if0" (pred = "const0", input = "const1", then_branch = "true_fn",
  //        else_branch = "false_fn")
  FunctionDefLibrary fdl;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg = ops::_Arg(s.WithOpName("arg"), DT_INT32, 0);
    Output identity = ops::Identity(s.WithOpName("identity_true_fn"), arg);
    ops::_Retval retval(s.WithOpName("retval"), identity, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    auto node_name_image = g->BuildNodeNameIndex();
    node_name_image["identity_true_fn"]->AddAttr("_oc", "0");
    PartialTensorShape shape({2});
    node_name_image["identity_true_fn"]->AddAttr(
        kXlaInferredShapesAttrName, std::vector<PartialTensorShape>{shape});

    FunctionDef *true_fn_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "true_fn", true_fn_fdef));
  }
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg = ops::_Arg(s.WithOpName("arg"), DT_INT32, 0);
    Output identity = ops::Identity(s.WithOpName("identity_false_fn"), arg);
    ops::_Retval retval(s.WithOpName("retval"), identity, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    auto node_name_image = g->BuildNodeNameIndex();
    node_name_image["identity_false_fn"]->AddAttr("_oc", "0");
    PartialTensorShape shape({2});
    node_name_image["identity_false_fn"]->AddAttr(
        kXlaInferredShapesAttrName, std::vector<PartialTensorShape>{shape});

    FunctionDef *false_fn_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "false_fn", false_fn_fdef));
  }
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output cond = ops::Const(s.WithOpName("const0"), true, {2});
    Output input = ops::Const(s.WithOpName("const1"), 1, {2});
    NameAttrList true_fn;
    true_fn.set_name("true_fn");
    NameAttrList false_fn;
    false_fn.set_name("false_fn");
    auto if_op = ops::If(s.WithOpName("if"), cond,
                         std::initializer_list<Input>{cond, input}, {DT_INT32},
                         true_fn, false_fn);
    ops::_Retval retval(s.WithOpName("retval"), if_op.output[0], 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));

    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "cluster", xla_fdef));
  }
  FunctionLibraryDefinition fld(OpRegistry::Global(), fdl);

  protobuf::Map<string, tensorflow::AttrValue> attrs;
  std::map<string, int> host_compute_core;
  std::vector<string> shape_inference_graphs;
  bool has_outside_compilation;
  NameAttrList name_attrs;
  name_attrs.set_name("cluster");
  *name_attrs.mutable_attr() = attrs;
  TF_CHECK_OK(ExtractOutsideCompilationTest(
      "_xla", "_oc", "cluster", name_attrs, "cluster_rewritten", "host_graph",
      host_compute_core, &fld, &shape_inference_graphs,
      &has_outside_compilation));

  // Check host graph.
  {
    std::unique_ptr<FunctionBody> host_fbody;
    AttrValue device_ordinal_temp_value;
    device_ordinal_temp_value.set_i(0);
    protobuf::Map<string, AttrValue> host_func_attrs;
    host_func_attrs["_device_ordinal"] = device_ordinal_temp_value;
    TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("host_graph"),
                                        AttrSlice(&host_func_attrs), &fld,
                                        &host_fbody));
    Graph *host_graph = host_fbody->graph;
    auto node_name_index = host_graph->BuildNodeNameIndex();

    // Verify we have XlaRecvAtHost to receive "If" predicate.
    Node *recv_if_pred_node = node_name_index["recv_oc_if_pred_if"];
    EXPECT_NE(recv_if_pred_node, nullptr);

    // Verify we have an "If" to choose outside compilation between then_branch
    // and else_branch, and it has `recv_if_pred_node` as cond input.
    Node *if_oc_node = node_name_index["oc_if_if"];
    EXPECT_NE(if_oc_node, nullptr);
    Node *if_oc_node_cond_input;
    TF_CHECK_OK(if_oc_node->input_node(0, &if_oc_node_cond_input));
    EXPECT_EQ(if_oc_node_cond_input, recv_if_pred_node);

    // Check that then_branch outside compilation has node "identity_true_fn".
    const FunctionDef *true_def = fld.Find("oc_then_branch_host_if_true_fn");
    EXPECT_NE(true_def, nullptr);
    bool has_identity_true_fn_node = false;
    for (const auto &node_def : true_def->node_def()) {
      if (node_def.name() == "identity_true_fn") {
        has_identity_true_fn_node = true;
        break;
      }
    }
    EXPECT_TRUE(has_identity_true_fn_node);

    // Check that else_branch outside compilation has node "identity_false_fn".
    const FunctionDef *false_def = fld.Find("oc_else_branch_host_if_false_fn");
    EXPECT_NE(false_def, nullptr);
    bool has_identity_false_fn_node = false;
    for (const auto &node_def : false_def->node_def()) {
      if (node_def.name() == "identity_false_fn") {
        has_identity_false_fn_node = true;
        break;
      }
    }
    EXPECT_TRUE(has_identity_false_fn_node);
  }

  // Check XLA graph.
  {
    std::unique_ptr<FunctionBody> xla_fbody;
    TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("cluster_rewritten"),
                                        AttrSlice(), &fld, &xla_fbody));
    Graph *xla_graph = xla_fbody->graph;
    auto node_name_index = xla_graph->BuildNodeNameIndex();

    // Check that we have XlaSendToHost to send cond predicate to host, and
    // there is a control edge to If node.
    Node *send_if_pred_node = node_name_index["send_oc_if_pred_if"];
    EXPECT_NE(send_if_pred_node, nullptr);
    bool has_control_edge_to_if = false;
    for (const Edge *e : send_if_pred_node->out_edges()) {
      if (e->IsControlEdge() && e->dst()->name() == "if") {
        has_control_edge_to_if = true;
        break;
      }
    }
    EXPECT_TRUE(has_control_edge_to_if);

    // Check that the "If" node now has `send_if_pred_node` as attribute
    // _xla_token_input_nodes.
    Node *if_node = node_name_index["if"];
    EXPECT_NE(if_node, nullptr);
    std::vector<string> token_inputs;
    TF_CHECK_OK(
        GetNodeAttr(if_node->def(), "_xla_token_input_nodes", &token_inputs));
    EXPECT_THAT(token_inputs, ::testing::ElementsAre("send_oc_if_pred_if"));
  }
}

TEST_F(ExtractOutsideCompilationForFunctionTest, OutsideCompilationInWhile) {
  // Build the XLA computation func.
  // "const0" (bool)
  // "while0" (input = "const0", cond = "cond_fn", body = "body_fn")
  FunctionDefLibrary fdl;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg = ops::_Arg(s.WithOpName("arg"), DT_BOOL, 0);
    Output identity = ops::Identity(s.WithOpName("identity_cond_fn"), arg);
    ops::_Retval retval(s.WithOpName("retval"), identity, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    auto node_name_image = g->BuildNodeNameIndex();
    node_name_image["identity_cond_fn"]->AddAttr("_oc", "0");
    PartialTensorShape shape({2});
    node_name_image["identity_cond_fn"]->AddAttr(
        kXlaInferredShapesAttrName, std::vector<PartialTensorShape>{shape});

    FunctionDef *cond_fn_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "cond_fn", cond_fn_fdef));
  }
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg = ops::_Arg(s.WithOpName("arg"), DT_BOOL, 0);
    Output identity = ops::Identity(s.WithOpName("identity_body_fn"), arg);
    ops::_Retval retval(s.WithOpName("retval"), identity, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    auto node_name_image = g->BuildNodeNameIndex();
    node_name_image["identity_body_fn"]->AddAttr("_oc", "0");
    PartialTensorShape shape({2});
    node_name_image["identity_body_fn"]->AddAttr(
        kXlaInferredShapesAttrName, std::vector<PartialTensorShape>{shape});

    FunctionDef *body_fn_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "body_fn", body_fn_fdef));
  }
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output input = ops::Const(s.WithOpName("const0"), true, {2});
    NameAttrList cond_fn;
    cond_fn.set_name("cond_fn");
    NameAttrList body_fn;
    body_fn.set_name("body_fn");
    auto while_op =
        ops::While(s.WithOpName("while"), std::initializer_list<Input>{input},
                   cond_fn, body_fn);
    ops::_Retval retval(s.WithOpName("retval"), while_op.output[0], 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));

    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "cluster", xla_fdef));
  }
  FunctionLibraryDefinition fld(OpRegistry::Global(), fdl);

  protobuf::Map<string, tensorflow::AttrValue> attrs;
  std::map<string, int> host_compute_core;
  std::vector<string> shape_inference_graphs;
  bool has_outside_compilation;
  NameAttrList name_attrs;
  name_attrs.set_name("cluster");
  *name_attrs.mutable_attr() = attrs;
  TF_CHECK_OK(ExtractOutsideCompilationTest(
      "_xla", "_oc", "cluster", name_attrs, "cluster_rewritten", "host_graph",
      host_compute_core, &fld, &shape_inference_graphs,
      &has_outside_compilation));

  // Check host graph.
  {
    std::unique_ptr<FunctionBody> host_fbody;
    AttrValue device_ordinal_temp_value;
    device_ordinal_temp_value.set_i(0);
    protobuf::Map<string, AttrValue> host_func_attrs;
    host_func_attrs["_device_ordinal"] = device_ordinal_temp_value;
    TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("host_graph"),
                                        AttrSlice(&host_func_attrs), &fld,
                                        &host_fbody));
    Graph *host_graph = host_fbody->graph;
    auto node_name_index = host_graph->BuildNodeNameIndex();

    // Verify we have an "While" to execute outside compilation.
    Node *while_oc_node = node_name_index["oc_while_while"];
    EXPECT_NE(while_oc_node, nullptr);

    // Check that cond outside compilation has node "identity_cond_fn".
    const FunctionDef *cond_def = fld.Find("oc_cond_host_while_cond_fn");
    EXPECT_NE(cond_def, nullptr);
    bool has_identity_cond_fn_node = false;
    for (const auto &node_def : cond_def->node_def()) {
      if (node_def.name() == "identity_cond_fn") {
        has_identity_cond_fn_node = true;
        break;
      }
    }
    EXPECT_TRUE(has_identity_cond_fn_node);

    // Check that body outside compilation has node "identity_body_fn".
    const FunctionDef *body_def = fld.Find("oc_body_host_while_body_fn");
    EXPECT_NE(body_def, nullptr);
    bool has_identity_body_fn_node = false;
    for (const auto &node_def : body_def->node_def()) {
      if (node_def.name() == "identity_body_fn") {
        has_identity_body_fn_node = true;
        break;
      }
    }
    EXPECT_TRUE(has_identity_body_fn_node);
  }

  // Check XLA graph.
  {
    // Verify that rewritten cond fn has XlaSendToHost to send loop predicate to
    // host.
    const FunctionDef *cond_def = fld.Find("cond_fn_oc");
    EXPECT_NE(cond_def, nullptr);
    bool has_send_oc_while_cond_node = false;
    for (const auto &node_def : cond_def->node_def()) {
      if (node_def.name() == "send_oc_while_cond_while") {
        has_send_oc_while_cond_node = true;
        break;
      }
    }
    EXPECT_TRUE(has_send_oc_while_cond_node);
  }
}

TEST_F(ExtractOutsideCompilationForFunctionTest, OutsideCompilationInFunction) {
  // Build the XLA computation func.
  // "const0" (int32)
  // "fn" (input = "const0")
  FunctionDefLibrary fdl;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg = ops::_Arg(s.WithOpName("arg"), DT_INT32, 0);
    Output identity = ops::Identity(s.WithOpName("identity"), arg);
    ops::_Retval retval(s.WithOpName("retval"), identity, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    auto node_name_image = g->BuildNodeNameIndex();
    node_name_image["identity"]->AddAttr("_oc", "0");
    PartialTensorShape shape({2});
    node_name_image["identity"]->AddAttr(
        kXlaInferredShapesAttrName, std::vector<PartialTensorShape>{shape});

    FunctionDef *true_fn_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "fn", true_fn_fdef));
  }
  FunctionLibraryDefinition fld(OpRegistry::Global(), fdl);
  {
    std::unique_ptr<Graph> g(new Graph(&fld));

    tensorflow::TensorProto tensor_proto;
    tensor_proto.set_dtype(tensorflow::DT_INT32);
    tensorflow::TensorShapeProto shape;
    shape.add_dim()->set_size(2);
    *tensor_proto.mutable_tensor_shape() = shape;
    for (int i = 0; i < 2; ++i) {
      tensor_proto.add_int_val(1);
    }
    NodeDef const_def;
    TF_CHECK_OK(NodeDefBuilder("const", "Const")
                    .Attr("dtype", DT_INT32)
                    .Attr("value", tensor_proto)
                    .Finalize(&const_def));
    Status s;
    Node *const_node = g->AddNode(const_def, &s);
    TF_CHECK_OK(s);

    NodeDef fn_def;
    TF_CHECK_OK(NodeDefBuilder("fn", "fn", &fld)
                    .Input("const", 0, DT_INT32)
                    .Finalize(&fn_def));
    Node *fn_node = g->AddNode(fn_def, &s);
    TF_CHECK_OK(s);
    g->AddEdge(const_node, 0, fn_node, 0);

    NodeDef ret_def;
    TF_CHECK_OK(NodeDefBuilder("ret", "_Retval")
                    .Attr("index", 0)
                    .Attr("T", DT_INT32)
                    .Input("fn", 0, DT_INT32)
                    .Finalize(&ret_def));
    Node *ret_node = g->AddNode(ret_def, &s);
    TF_CHECK_OK(s);
    g->AddEdge(fn_node, 0, ret_node, 0);

    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "cluster", xla_fdef));
    TF_CHECK_OK(fld.AddFunctionDef(*xla_fdef));
  }

  protobuf::Map<string, tensorflow::AttrValue> attrs;
  std::map<string, int> host_compute_core;
  std::vector<string> shape_inference_graphs;
  bool has_outside_compilation;
  NameAttrList name_attrs;
  name_attrs.set_name("cluster");
  *name_attrs.mutable_attr() = attrs;
  TF_CHECK_OK(ExtractOutsideCompilationTest(
      "_xla", "_oc", "cluster", name_attrs, "cluster_rewritten", "host_graph",
      host_compute_core, &fld, &shape_inference_graphs,
      &has_outside_compilation));

  // Check host graph.
  {
    std::unique_ptr<FunctionBody> host_fbody;
    AttrValue device_ordinal_temp_value;
    device_ordinal_temp_value.set_i(0);
    protobuf::Map<string, AttrValue> host_func_attrs;
    host_func_attrs["_device_ordinal"] = device_ordinal_temp_value;
    TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("host_graph"),
                                        AttrSlice(&host_func_attrs), &fld,
                                        &host_fbody));
    Graph *host_graph = host_fbody->graph;
    auto node_name_index = host_graph->BuildNodeNameIndex();

    // Verify we have call node for outside compilation in `fn`.
    Node *call_node = node_name_index["oc_call_fn"];
    EXPECT_NE(call_node, nullptr);

    std::unique_ptr<FunctionBody> call_fbody;
    TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("oc_func_call_host_fn"),
                                        AttrSlice(&host_func_attrs), &fld,
                                        &call_fbody));

    // Verify we have _XlaRecvAtHost and _XlaSendFromHost nodes.
    bool has_recv = false, has_send = false;
    for (Node *n : call_fbody->graph->nodes()) {
      if (n->type_string() == "_XlaRecvAtHost") {
        has_recv = true;
      } else if (n->type_string() == "_XlaSendFromHost") {
        has_send = true;
      }
    }
    EXPECT_TRUE(has_recv);
    EXPECT_TRUE(has_send);
  }

  // Check XLA graph.
  {
    std::unique_ptr<FunctionBody> xla_fbody;
    TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("cluster_rewritten"),
                                        AttrSlice(), &fld, &xla_fbody));
    Graph *xla_graph = xla_fbody->graph;
    auto node_name_index = xla_graph->BuildNodeNameIndex();

    // Check that we have call node.
    Node *fn_node = node_name_index["fn"];
    EXPECT_NE(fn_node, nullptr);
    EXPECT_EQ(fn_node->type_string(), "fn_oc");

    std::unique_ptr<FunctionBody> call_fbody;
    TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("fn_oc"), AttrSlice(), &fld,
                                        &call_fbody));

    // Verify we have XlaHostCompute nodes.
    bool has_hc = false;
    for (Node *n : call_fbody->graph->nodes()) {
      if (n->type_string() == "XlaHostCompute") {
        has_hc = true;
      }
    }
    EXPECT_TRUE(has_hc);
  }
}

TEST_F(ExtractOutsideCompilationForFunctionTest,
       OutsideCompilationClusterDataDependency) {
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
    std::cout << "Graph is " << (*g).ToGraphDefDebug().DebugString()
              << std::endl;
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
  std::vector<string> shape_inference_graphs;
  bool has_outside_compilation;
  NameAttrList name_attrs;
  name_attrs.set_name("cluster");
  *name_attrs.mutable_attr() = attrs;
  TF_CHECK_OK(ExtractOutsideCompilationTest(
      "_xla", "_oc", "cluster", name_attrs, "cluster_rewritten", "host_graph",
      host_compute_core, &fld, &shape_inference_graphs,
      &has_outside_compilation));

  // Get rewritten XLA computation function.
  std::unique_ptr<FunctionBody> xla_fbody;
  TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("cluster_rewritten"),
                                      AttrSlice(), &fld, &xla_fbody));
  auto node_name_index = xla_fbody->graph->BuildNodeNameIndex();

  // Check XlaHostCompute nodes.
  Node *host_compute_0 = node_name_index["outside_compilation_0_host_compute"];
  EXPECT_NE(host_compute_0, nullptr);
  Node *host_compute_1 = node_name_index["outside_compilation_1_host_compute"];
  EXPECT_NE(host_compute_1, nullptr);

  // Check XlaHostCompute nodes' "_xla_token_input_nodes" attr.
  std::vector<string> token_input_nodes;
  TF_CHECK_OK(GetNodeAttr(AttrSlice(host_compute_0->attrs()),
                          "_xla_token_input_nodes", &token_input_nodes));

  std::vector<string> expected_token_input_nodes_0({"_xla_token_arg_node"});
  EXPECT_EQ(token_input_nodes, expected_token_input_nodes_0);
  token_input_nodes.clear();
  std::vector<string> expected_token_input_nodes_1(
      {"_xla_token_arg_node", "outside_compilation_0_host_compute"});
  TF_CHECK_OK(GetNodeAttr(AttrSlice(host_compute_1->attrs()),
                          "_xla_token_input_nodes", &token_input_nodes));
  EXPECT_EQ(token_input_nodes, expected_token_input_nodes_1);

  // Check there is a control edge from host_compute_0 to host_compute_1.
  bool has_control_edge = false;
  for (const Edge *e : host_compute_1->in_edges()) {
    if (e->IsControlEdge() && e->src() == host_compute_0) {
      has_control_edge = true;
      break;
    }
  }
  EXPECT_TRUE(has_control_edge);
}

TEST_F(ExtractOutsideCompilationForFunctionTest,
       OutsideCompilationClusterControlDependency) {
  // Build the XLA computation func.
  // "const0"
  // "identity0" = "const0" (outside compilation cluster "0")
  // "identity1" = "const0" "^identity0" (outside compilation cluster "1",
  //                                      control dependent on cluster "0")
  // "identity2" = "identity1"
  FunctionDefLibrary fdl;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output const0 = ops::Const(s.WithOpName("const0"), 1, {2});
    Output identity0 = ops::Identity(s.WithOpName("identity0"), const0);
    Output identity1 = ops::Identity(
        s.WithOpName("identity1").WithControlDependencies(identity0), const0);
    Output identity2 = ops::Identity(s.WithOpName("identity2"), identity1);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    std::cout << "Graph is " << (*g).ToGraphDefDebug().DebugString()
              << std::endl;
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
  std::vector<string> shape_inference_graphs;
  bool has_outside_compilation;
  NameAttrList name_attrs;
  name_attrs.set_name("cluster");
  *name_attrs.mutable_attr() = attrs;
  TF_CHECK_OK(ExtractOutsideCompilationTest(
      "_xla", "_oc", "cluster", name_attrs, "cluster_rewritten", "host_graph",
      host_compute_core, &fld, &shape_inference_graphs,
      &has_outside_compilation));

  // Get rewritten XLA computation function.
  std::unique_ptr<FunctionBody> xla_fbody;
  TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("cluster_rewritten"),
                                      AttrSlice(), &fld, &xla_fbody));
  auto node_name_index = xla_fbody->graph->BuildNodeNameIndex();

  // Check XlaHostCompute nodes.
  Node *host_compute_0 = node_name_index["outside_compilation_0_host_compute"];
  EXPECT_NE(host_compute_0, nullptr);
  Node *host_compute_1 = node_name_index["outside_compilation_1_host_compute"];
  EXPECT_NE(host_compute_1, nullptr);

  // Check XlaHostCompute nodes' "_xla_token_input_nodes" attr.
  std::vector<string> token_input_nodes;
  TF_CHECK_OK(GetNodeAttr(AttrSlice(host_compute_0->attrs()),
                          "_xla_token_input_nodes", &token_input_nodes));

  std::vector<string> expected_token_input_nodes_0({"_xla_token_arg_node"});
  EXPECT_EQ(token_input_nodes, expected_token_input_nodes_0);
  token_input_nodes.clear();
  std::vector<string> expected_token_input_nodes_1(
      {"_xla_token_arg_node", "outside_compilation_0_host_compute"});
  TF_CHECK_OK(GetNodeAttr(AttrSlice(host_compute_1->attrs()),
                          "_xla_token_input_nodes", &token_input_nodes));
  EXPECT_EQ(token_input_nodes, expected_token_input_nodes_1);

  // Check there is a control edge from host_compute_0 to host_compute_1.
  bool has_control_edge = false;
  for (const Edge *e : host_compute_1->in_edges()) {
    if (e->IsControlEdge() && e->src() == host_compute_0) {
      has_control_edge = true;
      break;
    }
  }
  EXPECT_TRUE(has_control_edge);
}
}  // namespace tensorflow
