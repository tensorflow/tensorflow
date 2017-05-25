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

#include "tensorflow/core/kernels/remote_fused_graph_execute_op_test_utils.h"

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
/* static */ Output RemoteFusedGraphExecuteOpTestUtils::BuildAddOp(
    const Scope& scope, const Input& x, const Input& y) {
  CHECK(scope.ok());
  auto _x = ops::AsNodeOut(scope, x);
  CHECK(scope.ok());
  auto _y = ops::AsNodeOut(scope, y);
  CHECK(scope.ok());
  Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("Add");
  auto builder = NodeBuilder(unique_name, "Add").Input(_x).Input(_y);
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  CHECK(scope.ok()) << scope.status();
  return Output(ret, 0);
}

/* static */ Status RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
    const string& name0, const float val0, const string& name1,
    const float val1, const string& name_out, GraphDef* graph_def) {
  Scope root = Scope::NewRootScope();
  Output node0 = ops::Const(root.WithOpName(name0), val0);
  Output node1 = ops::Const(root.WithOpName(name1), val1);
  RemoteFusedGraphExecuteOpTestUtils::BuildAddOp(root.WithOpName(name_out),
                                                 node0, node1);
  TF_RETURN_IF_ERROR(root.ToGraphDef(graph_def));
  return Status::OK();
}

/* static */ Status RemoteFusedGraphExecuteOpTestUtils::BuildMultipleAddGraph(
    GraphDef* graph_def) {
  Scope root = tensorflow::Scope::NewRootScope();

  Tensor a_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&a_data, 1.0f);
  Output a_const = ops::Const(root.WithOpName("A"), Input::Initializer(a_data));

  Tensor b_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&b_data, 1.0f);
  Output b_const = ops::Const(root.WithOpName("B"), Input::Initializer(b_data));

  Tensor c_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&c_data, 1.0f);
  Output c_const = ops::Const(root.WithOpName("C"), Input::Initializer(c_data));

  Tensor d_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&d_data, 1.0f);
  Output d_const = ops::Const(root.WithOpName("D"), Input::Initializer(d_data));

  Tensor e_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&e_data, 1.0f);
  Output e_const = ops::Const(root.WithOpName("E"), Input::Initializer(e_data));

  Output f_add = ops::Add(root.WithOpName("F"), a_const, b_const);

  Output g_add = ops::Add(root.WithOpName("G"), d_const, e_const);

  Output h_add = ops::Add(root.WithOpName("H"), f_add, c_const);

  Output i_add = ops::Add(root.WithOpName("I"), c_const, g_add);

  Output j_add = ops::Add(root.WithOpName("J"), h_add, i_add);

  Output k_add = ops::Add(root.WithOpName("K"), j_add, g_add);

  TF_RETURN_IF_ERROR(root.ToGraphDef(graph_def));

  return Status::OK();
}

}  // namespace tensorflow
