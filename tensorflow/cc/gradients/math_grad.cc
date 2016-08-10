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

#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/cc/framework/grad_op_registry.h"

namespace tensorflow {
namespace ops {
namespace {

// TODO(andydavis) Move this to a more appropriate file.
REGISTER_NO_GRADIENT_OP("Const");

// MatMulGrad helper function used to compute two MatMul operations
// based on input matrix transposition combinations.
Status MatMulGradHelper(const Scope& scope, const Output& x0, const bool adj_x0,
                        const Output& x1, const bool adj_x1, const Output& y0,
                        const bool adj_y0, const Output& y1, const bool adj_y1,
                        std::vector<Output>* grad_outputs) {
  auto dx =
      MatMul(scope, x0, x1, MatMul::TransposeA(adj_x0).TransposeB(adj_x1));
  grad_outputs->push_back(dx);
  auto dy =
      MatMul(scope, y0, y1, MatMul::TransposeA(adj_y0).TransposeB(adj_y1));
  grad_outputs->push_back(dy);
  return Status::OK();
}

// MatMulGrad common used to read and check node attr state, and determine
// proper MatMul products for gradients based on input matrix transposition
// combinations.
// TODO(andydavis) Re-use this function for BatchMatMulGrad.
Status MatMulGradCommon(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        const string& attr_adj_x, const string& attr_adj_y,
                        std::vector<Output>* grad_outputs) {
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.output(0).node()->def(), "T", &dtype));
  if (dtype == DT_COMPLEX64 || dtype == DT_COMPLEX128) {
    return errors::Unimplemented(
        "MatMul gradient for complex data type is not supported yet.");
  }

  bool ta;
  bool tb;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.output(0).node()->def(), attr_adj_x, &ta));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.output(0).node()->def(), attr_adj_y, &tb));

  if (!ta && !tb) {
    return MatMulGradHelper(scope, grad_inputs[0], false, op.input(1), true,
                            op.input(0), true, grad_inputs[0], false,
                            grad_outputs);
  } else if (!ta && tb) {
    return MatMulGradHelper(scope, grad_inputs[0], false, op.input(1), false,
                            grad_inputs[0], true, op.input(0), false,
                            grad_outputs);
  } else if (ta && !tb) {
    return MatMulGradHelper(scope, op.input(1), false, grad_inputs[0], true,
                            op.input(0), false, grad_inputs[0], false,
                            grad_outputs);
  }
  return MatMulGradHelper(scope, op.input(1), true, grad_inputs[0], true,
                          grad_inputs[0], true, op.input(0), true,
                          grad_outputs);
}

Status MatMulGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
  return MatMulGradCommon(scope, op, grad_inputs, "transpose_a", "transpose_b",
                          grad_outputs);
}

REGISTER_GRADIENT_OP("MatMul", MatMulGrad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
