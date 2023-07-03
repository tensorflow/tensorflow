/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/constant_fold.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold_utils.h"

namespace mlir {
namespace quant {
namespace {

// Folds the operation recursively and return the results.
LogicalResult FoldOperation(OpBuilder& builder, Operation* op,
                            SmallVector<Value>& results) {
  SmallVector<ElementsAttr> inputs;
  for (auto operand : op->getOperands()) {
    auto preceding_const_op = operand.getDefiningOp<TF::ConstOp>();
    if (preceding_const_op) {
      inputs.push_back(preceding_const_op.getValue());
      continue;
    }

    Operation* preceding_op = operand.getDefiningOp();
    int preceding_result_id = -1;
    for (auto preceding_result : preceding_op->getResults()) {
      if (operand == preceding_result) {
        preceding_result_id = preceding_result.getResultNumber();
        break;
      }
    }
    SmallVector<Value> preceding_results;
    if (failed(FoldOperation(builder, preceding_op, preceding_results))) {
      return failure();
    }
    auto preceding_result = preceding_results[preceding_result_id];
    preceding_const_op = preceding_result.getDefiningOp<TF::ConstOp>();
    inputs.push_back(preceding_const_op.getValue());
  }

  SmallVector<Attribute> result_values;
  if (failed(TF::EvaluateOperation(op, inputs, result_values))) {
    return failure();
  }

  results.clear();
  builder.setInsertionPointAfter(op);
  for (const auto& result_value : result_values) {
    results.push_back(builder.create<TF::ConstOp>(op->getLoc(), result_value));
  }
  return success();
}

bool IsOperationFoldable(Operation* op) {
  if (isa<TF::ConstOp>(op)) return true;

  if (!op->getDialect()->getNamespace().equals("tf") || !TF::CanBeFolded(op)) {
    return false;
  }

  // Check if the operands are foldable as well.
  for (auto operand : op->getOperands()) {
    auto preceding_op = operand.getDefiningOp();
    if (!preceding_op || !IsOperationFoldable(preceding_op)) {
      return false;
    }
  }

  return true;
}
}  // namespace

SmallVector<Value> ConstantFoldOpIfPossible(Operation* op) {
  if (!IsOperationFoldable(op)) return op->getResults();

  OpBuilder builder(op);
  SmallVector<Value> results;
  if (failed(FoldOperation(builder, op, results))) {
    return op->getResults();
  }
  return results;
}

}  // namespace quant
}  // namespace mlir
