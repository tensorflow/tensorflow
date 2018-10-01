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

#include "tensorflow/compiler/jit/build_xla_ops_pass.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::tensorflow::testing::FindNodeByName;
using ::tensorflow::testing::matchers::CtrlDeps;
using ::tensorflow::testing::matchers::NodeWith;
using ::tensorflow::testing::matchers::Op;

Status BuildXlaOps(const Scope& s, std::unique_ptr<Graph>* result) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  TF_RETURN_IF_ERROR(s.ToGraph(graph.get()));

  // Assign all nodes to the CPU device.
  static const char* kCpuDevice = "/job:localhost/replica:0/task:0/cpu:0";
  for (Node* n : graph->nodes()) {
    if (n->assigned_device_name().empty()) {
      n->set_assigned_device_name(kCpuDevice);
    }
  }

  GraphOptimizationPassOptions opt_options;
  opt_options.graph = &graph;
  BuildXlaOpsPass pass;
  TF_RETURN_IF_ERROR(pass.Run(opt_options));
  *result = std::move(graph);
  return Status::OK();
}

Status MakeXlaCompiledKernel(Graph* graph, const string& callee_name,
                             const string& node_name, Node** result) {
  NodeDef call_node;
  call_node.set_name(node_name);
  call_node.set_op(callee_name);
  AddNodeAttr(kXlaCompiledKernelAttr, true, &call_node);
  AddNodeAttr(kXlaNumConstantArgsAttr, 0, &call_node);
  AddNodeAttr(kXlaNumResourceArgsAttr, 0, &call_node);
  Status s;
  *result = graph->AddNode(call_node, &s);
  return s;
}

Node* MakeWrite(const Scope& scope, const string& id) {
  Output var_handle =
      ops::VarHandleOp(scope.WithOpName("Var" + id), DT_FLOAT, TensorShape({}));
  Output value_to_write =
      ops::Const(scope.WithOpName("ValueToAssign" + id), 1.0f);
  ops::AssignVariableOp assign_op(scope.WithOpName("Assignee" + id), var_handle,
                                  value_to_write);
  return assign_op.operation.node();
}

FunctionDefLibrary CreateFunctionDefLibWithConstFunction(const string& name) {
  FunctionDefLibrary flib_def;
  FunctionDef func = FunctionDefHelper::Create(
      /*function_name=*/name, /*in_def=*/{}, /*out_def=*/{"out: float"},
      /*attr_def*/
      {}, /*node_def=*/{FunctionDefHelper::Const("one", 1.0f)},
      /*ret_def=*/{{"out", "out:output:0"}});
  *flib_def.add_function() = std::move(func);
  return flib_def;
}

TEST(BuildXlaOps, ControlDepsPreserved) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("cluster_0");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));
  Node* call;
  TF_ASSERT_OK(MakeXlaCompiledKernel(root.graph(), "cluster_0", "C", &call));
  Node* write_op = MakeWrite(root, "write");
  root.graph()->AddControlEdge(call, write_op);

  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(BuildXlaOps(root, &graph));

  Node* write_op_new = FindNodeByName(graph.get(), write_op->name());
  ASSERT_NE(write_op_new, nullptr);
  EXPECT_THAT(write_op_new, NodeWith(CtrlDeps(NodeWith(Op("_XlaRun")))));
}

}  // namespace
}  // namespace tensorflow
