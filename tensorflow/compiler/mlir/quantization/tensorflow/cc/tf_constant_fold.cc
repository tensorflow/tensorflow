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
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/tf_constant_fold.h"

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_lift_as_function_call.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold_utils.h"

namespace mlir {
namespace tf_quant {
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

  if (op->getDialect()->getNamespace() != "tf" || !TF::CanBeFolded(op)) {
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

// TODO: b/289744814 - Refactor to have a single source of truth of TF Quant
// specs.
absl::flat_hash_set<int> GetQuantizableOperands(Operation* op) {
  absl::flat_hash_set<int> quantizable_operands;
  if (isa<TF::DepthwiseConv2dNativeOp, TF::Conv2DOp, TF::Conv3DOp, TF::MatMulOp,
          TF::BatchMatMulOp>(op)) {
    quantizable_operands.insert(1);
  } else if (isa<TF::GatherOp>(op)) {
    quantizable_operands.insert(0);
  } else if (auto einsum_op = dyn_cast<TF::EinsumOp>(op)) {
    if (IsEinsumSupportedByXlaDotV2(einsum_op.getEquationAttr())) {
      quantizable_operands.insert(1);
    }
  }
  return quantizable_operands;
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

LogicalResult ConstantFoldQuantizableOperands::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  absl::flat_hash_set<int> quantizable_operands = GetQuantizableOperands(op);
  if (quantizable_operands.empty()) return failure();

  bool has_change = false;
  for (auto operand_idx : quantizable_operands) {
    Value operand = op->getOperand(operand_idx);
    Operation* preceding_op = operand.getDefiningOp();
    if (!preceding_op || isa<TF::ConstOp>(preceding_op)) continue;

    int preceding_result_idx = -1;
    for (auto preceding_result : preceding_op->getResults()) {
      if (operand == preceding_result) {
        preceding_result_idx = preceding_result.getResultNumber();
        break;
      }
    }

    has_change = has_change || IsOperationFoldable(preceding_op);
    SmallVector<Value> folded_results = ConstantFoldOpIfPossible(preceding_op);
    op->setOperand(operand_idx, folded_results[preceding_result_idx]);
  }

  return success(/*isSuccess=*/has_change);
}

}  // namespace tf_quant
}  // namespace mlir
