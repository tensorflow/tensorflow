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

Output Const(const Scope& scope, const Input::Initializer& val);

template <typename T>
Output Const(const Scope& scope, const Input::Initializer& val) {
  if (!scope.ok()) return Output();
  if (!val.status.ok()) {
    scope.UpdateStatus(val.status);
    return Output();
  }
  typedef typename Input::Initializer::RealType<T>::type DstT;
  if (val.tensor.NumElements() > 0) {
    // TODO(keveman): Implement the in-situ cast.
    scope.UpdateStatus(errors::Unimplemented(
        "Explict cast of a non-empty tensor not implemented yet"));
    return Output();
  }
  Tensor t(DataTypeToEnum<DstT>::v(), val.tensor.shape());
  return Const(scope, Input::Initializer(t));
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

NodeBuilder::NodeOut AsNodeOut(const Scope& scope, const Input& inp);

std::vector<NodeBuilder::NodeOut> AsNodeOutList(const Scope& scope,
                                                const InputList& inp);

}  // namespace ops
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_OPS_CONST_OP_H_
