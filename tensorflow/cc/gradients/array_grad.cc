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

#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"

namespace tensorflow {
namespace ops {
namespace {

REGISTER_NO_GRADIENT_OP("Const");
REGISTER_NO_GRADIENT_OP("StopGradient");
REGISTER_NO_GRADIENT_OP("ConcatOffset");
REGISTER_NO_GRADIENT_OP("EditDistance");
REGISTER_NO_GRADIENT_OP("ZerosLike");
REGISTER_NO_GRADIENT_OP("InvertPermutation");
REGISTER_NO_GRADIENT_OP("Shape");
REGISTER_NO_GRADIENT_OP("ShapeN");
REGISTER_NO_GRADIENT_OP("Rank");
REGISTER_NO_GRADIENT_OP("Size");
REGISTER_NO_GRADIENT_OP("BroadcastGradientArgs");
REGISTER_NO_GRADIENT_OP("OneHot");

Status PackGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  int N;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->def(), "N", &N));
  int axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->def(), "axis", &axis));

  grad_outputs->reserve(N);
  auto grad_op = Unpack(scope, grad_inputs[0], N, Unpack::Axis(axis));
  for (const Output& o : grad_op.output) {
    grad_outputs->emplace_back(o);
  }
  return Status::OK();
}
REGISTER_GRADIENT_OP("Pack", PackGrad);

Status UnpackGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
  int axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->def(), "axis", &axis));
  grad_outputs->push_back(Pack(scope, grad_inputs, Pack::Axis(axis)));
  return Status::OK();
}
REGISTER_GRADIENT_OP("Unpack", UnpackGrad);

Status IdentityGrad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return Status::OK();
}
REGISTER_GRADIENT_OP("Identity", IdentityGrad);

Status RefIdentityGrad(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return Status::OK();
}
REGISTER_GRADIENT_OP("RefIdentity", RefIdentityGrad);

Status SplitGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(Concat(scope, op.input(0), grad_inputs));
  return Status::OK();
}
REGISTER_GRADIENT_OP("Split", SplitGrad);

Status DiagGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(DiagPart(scope, grad_inputs[0]));
  return Status::OK();
}
REGISTER_GRADIENT_OP("Diag", DiagGrad);

Status DiagPartGrad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Diag(scope, grad_inputs[0]));
  return Status::OK();
}
REGISTER_GRADIENT_OP("DiagPart", DiagPartGrad);

Status MatrixDiagGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(MatrixDiagPart(scope, grad_inputs[0]));
  return Status::OK();
}
REGISTER_GRADIENT_OP("MatrixDiag", MatrixDiagGrad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
