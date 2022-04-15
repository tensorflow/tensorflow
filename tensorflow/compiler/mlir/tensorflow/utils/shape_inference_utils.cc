/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/shape_inference_utils.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"

#define DEBUG_TYPE "tf-shape-inference-utils"

namespace mlir {
namespace TF {

LogicalResult InferReturnTypeComponentsForTFOp(
    Optional<Location> location, Operation* op, int64_t graph_version,
    tfg::OperandAsConstantFn operand_as_constant_fn,
    tfg::OpResultAsShapeFn op_result_as_shape_fn,
    tfg::ResultElementTypeFn result_element_type_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes) {
  assert(op->getName().getDialectNamespace() ==
         TensorFlowDialect::getDialectNamespace());
  return tfg::InferReturnTypeComponentsForTFOp(
      location, op, op->getOperands(), graph_version, operand_as_constant_fn,
      op_result_as_shape_fn, result_element_type_fn,
      tensorflow::GetAttrValuesFromOperation, inferred_return_shapes);
}

}  // namespace TF
}  // namespace mlir
