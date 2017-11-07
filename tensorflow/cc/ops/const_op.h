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

#ifndef THIRD_PARTY_TENSORFLOW_CC_OPS_CONST_OP_H_
#define THIRD_PARTY_TENSORFLOW_CC_OPS_CONST_OP_H_

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

/// @defgroup const_op Const Op
/// @{

Output Const(const Scope& scope, const Input::Initializer& val);

Output ConstFromProto(const Scope& scope, const TensorProto& proto);

NodeBuilder::NodeOut AsNodeOut(const Scope& scope, const Input& inp);

template <typename T>
Output Const(const Scope& scope, const Input::Initializer& val) {
  auto orig_const_output = Const(scope, val);
  if (!scope.ok()) return Output();

  typedef typename Input::Initializer::RealType<T>::type DstT;

  if (val.tensor.dtype() == DataTypeToEnum<DstT>::v()) {
    return orig_const_output;
  }
  if (val.tensor.NumElements() == 0) {
    Tensor t(DataTypeToEnum<DstT>::v(), val.tensor.shape());
    return Const(scope, Input::Initializer(t));
  }

  // TODO(keveman): Refactor Cast op's kernel implementation such that the code
  // can be directly called here instead of adding the Cast op to the graph.
  auto orig_const = AsNodeOut(scope, orig_const_output);
  const auto cast_op_name = scope.GetUniqueNameForOp("Cast");

  auto cast_builder = NodeBuilder(cast_op_name, "Cast")
                          .Input(orig_const)
                          .Attr("DstT", DataTypeToEnum<DstT>::v());
  scope.UpdateBuilder(&cast_builder);
  Node* ret;
  scope.UpdateStatus(cast_builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return Output();
  scope.UpdateStatus(scope.DoShapeInference(ret));
  return Output(ret, 0);
}

template <typename T>
Output Const(const Scope& scope, const T& v, const TensorShape shape) {
  return Const(scope, Input::Initializer(v, shape));
}

template <typename T>
Output Const(const Scope& scope, const std::initializer_list<T>& v,
             const TensorShape shape) {
  return Const(scope, Input::Initializer(v, shape));
}

std::vector<NodeBuilder::NodeOut> AsNodeOutList(const Scope& scope,
                                                const InputList& inp);

/// }@

}  // namespace ops
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_OPS_CONST_OP_H_
