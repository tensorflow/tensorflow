/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/uniform_op_quant_spec.h"

#include <memory>

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::quant {

std::unique_ptr<OpQuantSpec> GetUniformOpQuantSpec(Operation* op) {
  auto spec = std::make_unique<OpQuantSpec>();
  if (auto call_op = dyn_cast<TF::UniformQuantizedConvolutionHybridOp>(op)) {
    spec->coeff_op_quant_dim[1] = 3;
  } else if (auto call_op = dyn_cast<TF::UniformQuantizedDotHybridOp>(op)) {
    spec->coeff_op_quant_dim[1] = -1;
  }

  for (auto quantizable_operand : spec->coeff_op_quant_dim) {
    spec->quantizable_operands.insert(quantizable_operand.first);
  }
  return spec;
}

}  // namespace mlir::quant
