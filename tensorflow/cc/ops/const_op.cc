/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace ops {

namespace {
template <typename T>
Output ConstHelper(const Scope& scope, const T& value, DataType dtype) {
  if (!scope.ok()) return Output();

  Node* ret;
  Graph* graph = scope.graph();
  const string unique_name = scope.GetUniqueNameForOp("Const");
  auto builder = NodeBuilder(unique_name, "Const")
                     .Attr("value", value)
                     .Attr("dtype", dtype);
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(graph, &ret));
  if (!scope.ok()) return Output();

  scope.UpdateStatus(scope.DoShapeInference(ret));
  if (!scope.ok()) return Output();

  return Output(ret);
}
}  // namespace

Output Const(const Scope& scope, const Input::Initializer& val) {
  if (!val.status.ok()) {
    scope.UpdateStatus(val.status);
    return Output();
  }
  return ConstHelper(scope, val.tensor, val.tensor.dtype());
}

Output ConstFromProto(const Scope& scope, const TensorProto& proto) {
  return ConstHelper(scope, proto, proto.dtype());
}

NodeBuilder::NodeOut AsNodeOut(const Scope& scope, const Input& inp) {
  if (!inp.status().ok()) {
    scope.UpdateStatus(inp.status());
    return NodeBuilder::NodeOut(inp.node(), inp.index());
  }
  if (inp.node()) {
    return NodeBuilder::NodeOut(inp.node(), inp.index());
  }
  if (!inp.node_name().empty()) {
    return NodeBuilder::NodeOut(inp.node_name(), inp.index(), inp.data_type());
  }
  auto transformed = Input{
      Const(scope.NewSubScope("Const"), Input::Initializer(inp.tensor()))};
  return NodeBuilder::NodeOut{transformed.node(), transformed.index()};
}

std::vector<NodeBuilder::NodeOut> AsNodeOutList(const Scope& scope,
                                                const InputList& inp) {
  std::vector<NodeBuilder::NodeOut> out;
  for (const auto& i : inp) {
    const auto node_out = AsNodeOut(scope, i);
    if (!scope.ok()) {
      return {};
    }
    out.push_back(node_out);
  }
  return out;
}

}  // namespace ops
}  // namespace tensorflow
