/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/subgraph.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

// This pass is used to fold tfl.const ops to each subgraph (func::FuncOp):
// See the example below:
//
// In main:
// %0 = tfl.const...
// %1 = tfl.const...
// %2 = call func_1(..., %0,...)
// %3 = call func_2(..., %0, ..., %1...)
// ...
//
// Then those consts will be copied into each function and replace their usage.
// func_1:
//   %0 = tfl.const...
// func_2:
//   %0 = tfl.const...
//   %1 = tfl.const...
class FoldConstantsToSubgraphPass
    : public mlir::PassWrapper<FoldConstantsToSubgraphPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldConstantsToSubgraphPass)

  llvm::StringRef getArgument() const final {
    return "tfl-fold-constants-to-subgraph";
  }
  llvm::StringRef getDescription() const final {
    return "Fold constants into each subgraph.";
  }
  FoldConstantsToSubgraphPass() = default;
  FoldConstantsToSubgraphPass(const FoldConstantsToSubgraphPass& other) {
    this->fold_all_constants_flag_ = other.fold_all_constants_flag_;
  }
  explicit FoldConstantsToSubgraphPass(bool fold_all_constants) {
    fold_all_constants_flag_ = fold_all_constants;
  }

 private:
  void runOnOperation() override;

  Option<bool> fold_all_constants_flag_{
      *this, "fold-all-constants",
      llvm::cl::desc("Whether to fold all constants or just i32."),
      llvm::cl::init(false)};
};

void CopyConstantIntoFunc(int argument_index, Operation* const_op,
                          func::FuncOp func) {
  assert((llvm::isa<TFL::ConstOp, TFL::QConstOp>(const_op)) &&
         "Expect QConst or Const op.");
  OpBuilder builder(func.getBody());
  auto cloned_const_op = const_op->clone();
  cloned_const_op->setLoc(func.getBody().getLoc());
  builder.insert(cloned_const_op);
  // Rewire the usage.
  func.getArgument(argument_index)
      .replaceAllUsesWith(cloned_const_op->getResult(0));
}

bool IsConstOrQConstInt(Operation* op) {
  if (!llvm::isa<TFL::ConstOp, TFL::QConstOp>(op)) return false;

  if (auto const_op = dyn_cast_or_null<TFL::ConstOp>(op)) {
    // ConstOp path.
    auto type = const_op.getType()
                    .dyn_cast_or_null<RankedTensorType>()
                    .getElementType();
    if (!type.isInteger(32) && !type.isInteger(64)) return false;
  } else {
    // QConstOp path.
    auto qconst_op = dyn_cast<TFL::QConstOp>(op);
    auto type =
        quant::QuantizedType::getQuantizedElementType(qconst_op.getType());
    if (type.getStorageTypeIntegralWidth() != 32) {
      return false;
    }
  }
  return true;
}

void FoldConstantsToSubgraphPass::runOnOperation() {
  auto module = getOperation();

  for (auto fn : module.getOps<func::FuncOp>()) {
    fn.walk([&](Operation* op) {
      if (!llvm::isa<TFL::ConstOp, TFL::QConstOp>(op)) return;

      // We only fold int32/int64 for Const and i32 for QConst if not specify
      // all constants flag. (Since they're more like "configs" or i32 biases.)
      // We will fold every const ops (and q_const ops) if we speicfy the
      // fold_all_constants_flag.
      if (!fold_all_constants_flag_) {
        if (!IsConstOrQConstInt(op)) return;
      }

      for (auto consumer : op->getResult(0).getUsers()) {
        auto consumer_call = llvm::dyn_cast_or_null<func::CallOp>(consumer);

        if (!consumer_call) continue;

        auto function_name = consumer_call.getCallee();

        // Locate the argument position of the use.
        int argument_index = -1;
        for (int i = 0; i < consumer_call.getNumOperands(); ++i) {
          if (consumer_call.getOperand(i) == op->getResult(0)) {
            argument_index = i;
            break;
          }
        }

        // Copy the const into the consumer func and replace their usages.
        func::FuncOp func = module.lookupSymbol<func::FuncOp>(function_name);

        CopyConstantIntoFunc(argument_index, op, func);
      }
    });
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateFoldConstantsToSubgraphPass(
    bool fold_all_constants) {
  return std::make_unique<FoldConstantsToSubgraphPass>(fold_all_constants);
}

static PassRegistration<FoldConstantsToSubgraphPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
