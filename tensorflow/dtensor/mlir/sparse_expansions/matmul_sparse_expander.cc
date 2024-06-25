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

#include "tensorflow/dtensor/mlir/sparse_expansions/matmul_sparse_expander.h"

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/sparse_expander_common.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> MatMulSparseExpander::ExpandOp(mlir::Operation* op) {
  mlir::TF::MatMulOp mm = mlir::cast<mlir::TF::MatMulOp>(op);
  // If any of the transpose attributes are true, then return original op.
  if (mm.getTransposeA() || mm.getTransposeB()) return op;

  // Expand to SparseTensorDenseMatMul Op only if the left operand
  // is a SparseTensor.
  if (IsSparseValue(op->getOperand(0)) && !IsSparseValue(op->getOperand(1))) {
    mlir::OpBuilder builder(op);
    // Since operand 0 is a SparseValue, we don't need to check that
    // the indices, values, and dense_shapes exist.
    mlir::TF::SparseTensorDenseMatMulOp new_op =
        builder.create<mlir::TF::SparseTensorDenseMatMulOp>(
            op->getLoc(), op->getResultTypes(),
            mlir::ValueRange{
                GetIndicesFromSparseTensor(op->getOperand(0)).value(),
                GetValuesFromSparseTensor(op->getOperand(0)).value(),
                GetDenseShapesFromSparseTensor(op->getOperand(0)).value(),
                op->getOperand(1)});

    op->getResult(0).replaceAllUsesWith(new_op.getResult());
    op->erase();
    return new_op.getOperation();
  }

  // Any other case, return the original op.
  return op;
}

}  // namespace dtensor
}  // namespace tensorflow
