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

#include "tensorflow/cc/ops/const_op.h"
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

/* static */ GraphDef RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
    const string& name0, const float val0, const string& name1,
    const float val1, const string& name_out) {
  Scope root = Scope::NewRootScope();
  Output node0 = ops::Const(root.WithOpName(name0), val0);
  Output node1 = ops::Const(root.WithOpName(name1), val1);
  RemoteFusedGraphExecuteOpTestUtils::BuildAddOp(root.WithOpName(name_out),
                                                 node0, node1);
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

}  // namespace tensorflow
