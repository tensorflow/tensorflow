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

#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

Node* MakeRead(const Scope& scope, const string& id) {
  Output var_handle =
      ops::VarHandleOp(scope.WithOpName("Var" + id), DT_FLOAT, TensorShape({}));
  Output read =
      ops::ReadVariableOp(scope.WithOpName("Read" + id), var_handle, DT_FLOAT);
  return read.node();
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

Node* MakeModify(const Scope& scope, const string& id) {
  Output var_handle =
      ops::VarHandleOp(scope.WithOpName("Var" + id), DT_FLOAT, TensorShape({}));
  Output value_to_write = ops::Const(scope.WithOpName("Increment" + id), 1.0f);
  ops::AssignAddVariableOp assign_add_op(scope.WithOpName("Increment" + id),
                                         var_handle, value_to_write);
  return assign_add_op.operation.node();
}

Node* MakeNeutral(const Scope& scope, const string& id) {
  return ops::Const(scope.WithOpName("Const" + id), 42.0f).node();
}

Status ComputeIncompatiblePairs(Graph* g,
                                std::vector<std::pair<int, int>>* result) {
  FixupSourceAndSinkEdges(g);
  return ComputeIncompatibleResourceOperationPairs(*g, &g->flib_def(), {},
                                                   result);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteRead) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(write, read);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> write_read_pair = {write->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], write_read_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, ReadWrite) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(read, write);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, ReadWriteNoEdges) {
  Scope root = Scope::NewRootScope().ExitOnError();

  MakeRead(root, "R");
  MakeWrite(root, "W");

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, ReadModify) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* modify = MakeModify(root, "M");

  root.graph()->AddControlEdge(read, modify);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, ModifyRead) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* modify = MakeModify(root, "M");

  root.graph()->AddControlEdge(modify, read);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> modify_read_pair = {modify->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], modify_read_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, ModifyWrite) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* modify = MakeModify(root, "M");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(modify, write);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteModify) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* modify = MakeModify(root, "M");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(write, modify);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> write_modify_pair = {write->id(), modify->id()};
  EXPECT_EQ(incompatible_pairs[0], write_modify_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, ReadModifyWrite) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* modify = MakeModify(root, "M");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(read, modify);
  root.graph()->AddControlEdge(modify, write);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteModifyRead) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* modify = MakeModify(root, "M");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(write, modify);
  root.graph()->AddControlEdge(modify, read);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 3);

  std::pair<int, int> write_modify_pair = {write->id(), modify->id()};
  std::pair<int, int> modify_read_pair = {modify->id(), read->id()};
  std::pair<int, int> write_read_pair = {write->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], modify_read_pair);
  EXPECT_EQ(incompatible_pairs[1], write_read_pair);
  EXPECT_EQ(incompatible_pairs[2], write_modify_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteReadModify) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* modify = MakeModify(root, "M");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(write, read);
  root.graph()->AddControlEdge(read, modify);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 2);

  std::pair<int, int> write_modify_pair = {write->id(), modify->id()};
  std::pair<int, int> write_read_pair = {write->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], write_read_pair);
  EXPECT_EQ(incompatible_pairs[1], write_modify_pair);
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

Node* MakeCall(Graph* graph, const string& callee_name, const string& node_name,
               Status* status) {
  NodeDef call_node;
  call_node.set_name(node_name);
  call_node.set_op(callee_name);
  return graph->AddNode(call_node, status);
}

TEST(ResourceOperationSafetyAnalysisTest, CallRead) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* read = MakeRead(root, "R");
  Status status;
  Node* call = MakeCall(root.graph(), "Const_func", "C", &status);
  TF_ASSERT_OK(status);

  root.graph()->AddControlEdge(call, read);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> call_read_edge = {call->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], call_read_edge);
}

TEST(ResourceOperationSafetyAnalysisTest, ReadCall) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* read = MakeRead(root, "R");
  Status status;
  Node* call = MakeCall(root.graph(), "Const_func", "C", &status);
  TF_ASSERT_OK(status);

  root.graph()->AddControlEdge(read, call);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, CallWrite) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* write = MakeWrite(root, "W");
  Status status;
  Node* call = MakeCall(root.graph(), "Const_func", "C", &status);
  TF_ASSERT_OK(status);

  root.graph()->AddControlEdge(call, write);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteCall) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* write = MakeWrite(root, "W");
  Status status;
  Node* call = MakeCall(root.graph(), "Const_func", "C", &status);
  TF_ASSERT_OK(status);

  root.graph()->AddControlEdge(write, call);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> write_call_edge = {write->id(), call->id()};
  EXPECT_EQ(incompatible_pairs[0], write_call_edge);
}

TEST(ResourceOperationSafetyAnalysisTest, SymbolicGradientRead) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* read = MakeRead(root, "R");
  NameAttrList fn;
  fn.set_name("Const_func");
  Node* symbolic_gradient =
      ops::SymbolicGradient(root, /*input=*/{ops::Const(root, 1.0f)},
                            /*Tout=*/{DT_FLOAT}, fn)
          .output[0]
          .node();

  root.graph()->AddControlEdge(symbolic_gradient, read);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> symbolic_gradient_read_edge = {symbolic_gradient->id(),
                                                     read->id()};
  EXPECT_EQ(incompatible_pairs[0], symbolic_gradient_read_edge);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteSymbolicGradient) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* write = MakeWrite(root, "W");
  NameAttrList fn;
  fn.set_name("Const_func");
  Node* symbolic_gradient =
      ops::SymbolicGradient(root, /*input=*/{ops::Const(root, 1.0f)},
                            /*Tout=*/{DT_FLOAT}, fn)
          .output[0]
          .node();

  root.graph()->AddControlEdge(write, symbolic_gradient);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> write_symbolic_gradient_edge = {write->id(),
                                                      symbolic_gradient->id()};
  EXPECT_EQ(incompatible_pairs[0], write_symbolic_gradient_edge);
}

TEST(ResourceOperationSafetyAnalysisTest, ChainOfOps) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* write_0 = MakeWrite(root, "W0");
  Node* neutral_0 = MakeNeutral(root, "N0");
  Node* read_0 = MakeRead(root, "R0");
  Node* write_1 = MakeWrite(root, "W1");
  Node* neutral_1 = MakeNeutral(root, "N1");
  Node* read_1 = MakeRead(root, "R1");

  root.graph()->AddControlEdge(write_0, neutral_0);
  root.graph()->AddControlEdge(neutral_0, read_0);
  root.graph()->AddControlEdge(read_0, write_1);
  root.graph()->AddControlEdge(write_1, neutral_1);
  root.graph()->AddControlEdge(neutral_1, read_1);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 3);
  std::pair<int, int> write_0_read_0_pair = {write_0->id(), read_0->id()};
  std::pair<int, int> write_0_read_1_pair = {write_0->id(), read_1->id()};
  std::pair<int, int> write_1_read_1_pair = {write_1->id(), read_1->id()};

  EXPECT_EQ(incompatible_pairs[0], write_0_read_0_pair);
  EXPECT_EQ(incompatible_pairs[1], write_0_read_1_pair);
  EXPECT_EQ(incompatible_pairs[2], write_1_read_1_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, DagOfOps) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* write_0 = MakeWrite(root, "W0");
  Node* write_1 = MakeWrite(root, "W1");
  Node* neutral = MakeNeutral(root, "N");
  Node* read_0 = MakeRead(root, "R0");
  Node* read_1 = MakeRead(root, "R1");

  root.graph()->AddControlEdge(write_0, neutral);
  root.graph()->AddControlEdge(write_1, neutral);
  root.graph()->AddControlEdge(neutral, read_0);
  root.graph()->AddControlEdge(neutral, read_1);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 4);
  std::pair<int, int> write_0_read_0_pair = {write_0->id(), read_0->id()};
  std::pair<int, int> write_0_read_1_pair = {write_0->id(), read_1->id()};
  std::pair<int, int> write_1_read_0_pair = {write_1->id(), read_0->id()};
  std::pair<int, int> write_1_read_1_pair = {write_1->id(), read_1->id()};

  EXPECT_EQ(incompatible_pairs[0], write_0_read_0_pair);
  EXPECT_EQ(incompatible_pairs[1], write_0_read_1_pair);
  EXPECT_EQ(incompatible_pairs[2], write_1_read_0_pair);
  EXPECT_EQ(incompatible_pairs[3], write_1_read_1_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, DagOfOpsWithRepeatedPaths) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* write_0 = MakeWrite(root, "W0");
  Node* write_1 = MakeWrite(root, "W1");
  Node* neutral = MakeNeutral(root, "N");
  Node* read_0 = MakeRead(root, "R0");
  Node* read_1 = MakeRead(root, "R1");

  root.graph()->AddControlEdge(write_0, neutral);
  root.graph()->AddControlEdge(write_1, neutral);
  root.graph()->AddControlEdge(neutral, read_0);
  root.graph()->AddControlEdge(neutral, read_1);
  root.graph()->AddControlEdge(write_1, read_1);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 4);
  std::pair<int, int> write_0_read_0_pair = {write_0->id(), read_0->id()};
  std::pair<int, int> write_0_read_1_pair = {write_0->id(), read_1->id()};
  std::pair<int, int> write_1_read_0_pair = {write_1->id(), read_0->id()};
  std::pair<int, int> write_1_read_1_pair = {write_1->id(), read_1->id()};

  EXPECT_EQ(incompatible_pairs[0], write_0_read_0_pair);
  EXPECT_EQ(incompatible_pairs[1], write_0_read_1_pair);
  EXPECT_EQ(incompatible_pairs[2], write_1_read_0_pair);
  EXPECT_EQ(incompatible_pairs[3], write_1_read_1_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, Loop) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output init_value = ops::Placeholder(root.WithOpName("init"), DT_FLOAT);
  Output loop_cond = ops::Placeholder(root.WithOpName("init"), DT_BOOL);
  Output enter_value =
      ops::internal::Enter(root.WithOpName("enter"), init_value, "fr");
  ops::Merge iv(root.WithOpName("iv"), {enter_value, enter_value});
  ops::Switch latch(root.WithOpName("latch"), iv.output, loop_cond);
  ops::internal::Exit exit(root.WithOpName("exit"), iv.output);
  Output next_iteration =
      ops::NextIteration(root.WithOpName("next_iteration"), latch.output_true);
  TF_ASSERT_OK(
      root.graph()->UpdateEdge(next_iteration.node(), 0, iv.output.node(), 1));

  Node* write = MakeWrite(root, "W");
  Node* read = MakeRead(root, "R");

  root.graph()->AddControlEdge(iv.output.node(), write);
  root.graph()->AddControlEdge(write, read);
  root.graph()->AddControlEdge(read, next_iteration.node());

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);

  std::pair<int, int> write_read_pair = {write->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], write_read_pair);
}

bool IsResourceArgDef(const OpDef::ArgDef& arg_def) {
  return arg_def.type() == DT_RESOURCE;
}
}  // namespace
}  // namespace tensorflow
