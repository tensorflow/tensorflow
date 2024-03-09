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

#include "tensorflow/dtensor/mlir/sparse_expander_common.h"

#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::TF::SparseToDenseOp> GetSparseToDenseOp(mlir::Value value) {
  // Travel back until we see a TF op. We generally expect this value
  // to be connected by a series of DTensor ops like DTensorLayout or
  // various DTensorRelayout ops, so skip past the tf.DTensor ops.
  auto op = value.getDefiningOp();
  while (op && op->getName().getStringRef().starts_with("tf.DTensor")) {
    op = op->getOperand(0).getDefiningOp();
  }

  if (op && llvm::isa<mlir::TF::SparseToDenseOp>(op))
    return llvm::dyn_cast_or_null<mlir::TF::SparseToDenseOp>(op);
  return errors::NotFound("SparseToDenseOp not found from value.");
}

bool IsSparseValue(mlir::Value value) { return GetSparseToDenseOp(value).ok(); }

bool HasAnySparseInput(mlir::Operation* op) {
  for (auto operand : op->getOperands())
    if (IsSparseValue(operand)) return true;
  return false;
}

bool AllSparseInput(mlir::Operation* op) {
  for (auto operand : op->getOperands())
    if (!IsSparseValue(operand)) return false;
  return true;
}

StatusOr<mlir::Value> GetIndicesFromSparseTensor(mlir::Value value) {
  auto sparse_op = GetSparseToDenseOp(value);
  if (!sparse_op.ok())
    return errors::NotFound(
        "Indices tensor not found from value because it was not from a "
        "SparseTensor.");
  return sparse_op->getOperand(0);
}

StatusOr<mlir::Value> GetValuesFromSparseTensor(mlir::Value value) {
  auto sparse_op = GetSparseToDenseOp(value);
  if (!sparse_op.ok())
    return errors::NotFound(
        "Values tensor not found from value because it was not from a "
        "SparseTensor.");
  return sparse_op->getOperand(2);
}

StatusOr<mlir::Value> GetDenseShapesFromSparseTensor(mlir::Value value) {
  auto sparse_op = GetSparseToDenseOp(value);
  if (!sparse_op.ok())
    return errors::NotFound(
        "Dense shape tensor not found from value because it was not from a "
        "SparseTensor.");
  return sparse_op->getOperand(1);
}

}  // namespace dtensor
}  // namespace tensorflow
