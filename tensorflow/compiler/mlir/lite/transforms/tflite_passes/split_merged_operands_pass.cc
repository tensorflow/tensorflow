/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/tflite_passes/split_merged_operands_pass.h"

#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/stateful_ops_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {
namespace {
LogicalResult DuplicateValueIfNeeded(Operation* op,
                                     llvm::DenseSet<Value>* values,
                                     OpBuilder* builder) {
  std::vector<int> stateful_operands_index;
  if (!IsStatefulOp(op, &stateful_operands_index)) return success();

  for (int index : stateful_operands_index) {
    Value operand = op->getOperand(index);
    auto inserted_value = values->insert(operand).second;
    if (inserted_value) continue;
    // We can only clone the constant op or const->dequantize combo. The latter
    // case is useful for float16 quantization. Since all ops have been
    // legalized to tflite ops, so we only care about ConstOp or QConstOp or
    // mlir constant op.
    Operation* input_op = operand.getDefiningOp();
    if (input_op == nullptr) return failure();

    Attribute attr;
    if (matchPattern(input_op, m_Constant(&attr))) {
      // Constant case.
      builder->setInsertionPoint(op);
      Operation* duplicated_input_op = builder->clone(*input_op);

      // Rewire the inputs.
      op->setOperand(index, duplicated_input_op->getResult(0));
    } else if (auto dq = dyn_cast<DequantizeOp>(input_op);
               dq && matchPattern(dq.getInput(), m_Constant(&attr))) {
      // Constant -> Dequantize case.
      builder->setInsertionPoint(op);
      Operation* duplicated_input_op =
          builder->clone(*dq.getInput().getDefiningOp());
      Operation* duplicated_dq_op = builder->clone(*dq);
      // Rewire the inputs.
      duplicated_dq_op->setOperand(0, duplicated_input_op->getResult(0));
      op->setOperand(index, duplicated_dq_op->getResult(0));
    } else {
      op->emitError()
          << "We cannot duplicate the value since it's not constant.\n";
      return failure();
    }
  }
  return success();
}
}  // namespace

void SplitMergedOperandsPass::runOnOperation() {
  llvm::DenseSet<Value> stateful_values;
  auto func = getOperation();
  OpBuilder builder(func);
  for (auto& bb : func.getBody()) {
    for (auto& op : bb) {
      if (failed(DuplicateValueIfNeeded(&op, &stateful_values, &builder))) {
        func.emitError() << "Failed to duplicate values for the stateful op\n";
        return signalPassFailure();
      }
    }
  }
}


}  // namespace TFL
}  // namespace mlir
