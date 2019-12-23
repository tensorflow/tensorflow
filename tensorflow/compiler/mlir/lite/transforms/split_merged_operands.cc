/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/SymbolTable.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/stateful_ops_utils.h"

// Background info:
// Currently the model taken to MLIRConverter is frozen (all the variables have
// been converted to constants, all the assign ops are gone, etc.). However,
// TFLite has these variable tensors semantics. So the variable mapping from TF
// to TFLite is actually broken here, we sort of hard-code the variable tensors
// based on the actual ops using them, such as unidirectional_sequence_lstm.
//
// MLIRConverter also benefits from lots of typical compiler optimization like
// merging same input values if they're identical. These optimizations are
// desirable but not for those TFLite ops which have variable tensors as inputs.
// Yes, they have identical input values, but those identical values are
// "stateful", their values can change during invocations.
//
// A typical example is unidirectional_sequence_lstm have two variable tensor
// inputs: activation state & cell state. They may have same initial values
// (typical zero-initialized), but their values will be changed. So we cannot
// just merge those values.
//
// This pass is more like short-term workaround since we don't have a good
// variable representation right now.
//
// This pass will duplicate input values for those variable tensor inputs.

namespace mlir {
namespace TFL {
namespace {

struct SplitMergedOperandsPass : public FunctionPass<SplitMergedOperandsPass> {
  void runOnFunction() override;
};

LogicalResult DuplicateValueIfNeeded(Operation* op,
                                     llvm::DenseSet<ValuePtr>* values,
                                     OpBuilder* builder) {
  std::vector<int> stateful_operands_index;
  if (!IsStatefulOp(op, &stateful_operands_index)) return success();

  for (int index : stateful_operands_index) {
    ValuePtr operand = op->getOperand(index);
    auto inserted_value = values->insert(operand).second;
    if (inserted_value) continue;
    // We can only clone the constant op at this point.
    // Since all ops have been legalized to tflite ops, so we only care about
    // ConstOp or QConstOp or mlir constant op/
    Operation* input_op = operand->getDefiningOp();
    if (input_op == nullptr) return failure();

    Attribute attr;
    if (!matchPattern(input_op, m_Constant(&attr))) {
      op->emitError()
          << "We cannot duplicate the value since it's not constant.\n";
      return failure();
    }
    builder->setInsertionPoint(op);
    Operation* duplicated_input_op = builder->clone(*input_op);

    // Rewire the inputs.
    op->setOperand(index, duplicated_input_op->getResult(0));
  }
  return success();
}

void SplitMergedOperandsPass::runOnFunction() {
  llvm::DenseSet<ValuePtr> stateful_values;
  auto func = getFunction();
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

}  // namespace

/// Creates an instance of the TensorFlow Lite dialect SplitMergedOperands
/// pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateSplitMergedOperandsPass() {
  return std::make_unique<SplitMergedOperandsPass>();
}

static PassRegistration<SplitMergedOperandsPass> pass(
    "tfl-split-merged-operands",
    "Split merged stateful operands for tfl operations.");

}  // namespace TFL
}  // namespace mlir
