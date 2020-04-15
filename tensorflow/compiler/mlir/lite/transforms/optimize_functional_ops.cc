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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

using FuncSet = llvm::SmallSet<FuncOp, 4>;

// Module pass to optimize TensorFlow functional ops.
struct OptimizeFunctionalOpsPass
    : public PassWrapper<OptimizeFunctionalOpsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// Updates function return type of the given functions to match the terminator
// op operands' types.
//
// Requires the function has exactly one block.
static void UpdateFuncType(FuncOp func) {
  Operation* terminator = &func.getBlocks().front().back();
  auto return_types = llvm::to_vector<4>(terminator->getOperandTypes());

  FunctionType func_type = func.getType();
  if (llvm::makeArrayRef(return_types) == func_type.getResults()) return;

  auto updated_type =
      FunctionType::get(func_type.getInputs(), return_types, func.getContext());
  func.setType(updated_type);
}

// TODO(jpienaar): Remove when recursive side-effect modeling is added.
static bool IsSideEffectFree(FuncOp func) {
  return !func.getBody()
              .walk([&](Operation* op) {
                if (!MemoryEffectOpInterface::hasNoEffect(op) &&
                    !op->isKnownTerminator())
                  return WalkResult::interrupt();
                return WalkResult::advance();
              })
              .wasInterrupted();
}

// Folds TensorFlow If op with constant conditional operand by inlining the
// function body based on the conditional value.
class FoldIfOp : public OpRewritePattern<TF::IfOp> {
 public:
  explicit FoldIfOp(MLIRContext* context, FuncSet* inlined_funcs)
      : OpRewritePattern<TF::IfOp>(context), inlined_funcs_(inlined_funcs) {}

  LogicalResult matchAndRewrite(TF::IfOp op,
                                PatternRewriter& rewriter) const override {
    // This pattern is restricted to if ops in functions with exactly one block
    // and therefore one terminator op. So, that function return type can be
    // updated if operands' shapes change after inlining. Without this
    // restriction, it would require tensor cast ops.
    FuncOp parent_op = op.getParentOfType<FuncOp>();
    if (parent_op.getBlocks().size() != 1) return failure();

    // Find the then and else branch functions.
    SymbolTable table(op.getParentOfType<ModuleOp>());
    FuncOp then_branch = table.lookup<FuncOp>(op.then_branch());
    FuncOp else_branch = table.lookup<FuncOp>(op.else_branch());

    // If the If has no uses and its functions are side-effect free, then
    // remove.
    // TODO(jpienaar): Remove once recusive side-effects are supported.
    if (op.use_empty() &&
        (op.is_stateless() ||
         (IsSideEffectFree(then_branch) && IsSideEffectFree(else_branch)))) {
      inlined_funcs_->insert(then_branch);
      inlined_funcs_->insert(else_branch);
      rewriter.eraseOp(op.getOperation());
      return success();
    }

    // Extract the constant cond value.
    DenseElementsAttr cond;
    if (!matchPattern(op.cond(), m_Constant(&cond))) return failure();

    // TODO(hinsu): Handle constants that are not scalar booleans.
    auto cond_type = cond.getType().dyn_cast<RankedTensorType>();
    if (!cond_type || !cond_type.getShape().equals({}) ||
        !cond_type.getElementType().isInteger(/*width=*/1))
      return failure();

    // Identify the branch to inline.
    bool cond_value = (*cond.int_value_begin()).getSExtValue();
    FuncOp func = cond_value ? then_branch : else_branch;

    // Make sure that the function has exactly one block to simplify inlining.
    // TFLite doesn't use control flow with blocks so functions with more than
    // one blocks are not encountered in practice.
    if (func.getBody().getBlocks().size() != 1) return failure();

    BlockAndValueMapping mapper;
    for (int i = 0, e = func.getNumArguments(); i != e; ++i)
      mapper.map(func.getArgument(i), op.getOperand(i + 1));

    llvm::SmallVector<Value, 4> updated_results;
    for (auto& op_to_inline : func.getBody().front()) {
      // If this is a terminator, identify the values to use to replace the
      // original If op.
      if (op_to_inline.isKnownTerminator()) {
        updated_results.reserve(op_to_inline.getNumOperands());
        for (Value operand : op_to_inline.getOperands())
          updated_results.push_back(mapper.lookup(operand));
        break;
      }

      // Otherwise, clone the op here.
      rewriter.clone(op_to_inline, mapper);
    }
    rewriter.replaceOp(op, updated_results);

    // Here, shapes of the updated_results may not match the original values. If
    // any of the values are operands of the terminator op, then the function
    // return type should be updated.
    UpdateFuncType(parent_op);

    // Track functions that could be erased if this op was the last reference
    // of the function.
    inlined_funcs_->insert(then_branch);
    inlined_funcs_->insert(else_branch);
    return success();
  }

 private:
  FuncSet* inlined_funcs_;
};

// Erases functions from the given candidates that are not referenced by any of
// the ops in the module.
static void EraseDeadFuncs(const FuncSet& candidate_funcs, ModuleOp module) {
  if (candidate_funcs.empty()) return;

  SymbolTable manager(module);

  // Identify the functions that are used as symbols in the module and shouldn't
  // be erased.
  FuncSet in_use_funcs;
  manager.getOp()->walk([&](Operation* op) {
    for (auto attr : op->getAttrs()) {
      if (auto symbol = attr.second.dyn_cast<FlatSymbolRefAttr>()) {
        auto func = manager.lookup<FuncOp>(symbol.getValue());
        in_use_funcs.insert(func);
      }
    }
  });

  for (FuncOp func : candidate_funcs) {
    if (!in_use_funcs.count(func)) manager.erase(func);
  }
}

void OptimizeFunctionalOpsPass::runOnOperation() {
  OwningRewritePatternList patterns;

  FuncSet inlined_funcs;
  patterns.insert<FoldIfOp>(&getContext(), &inlined_funcs);

  ModuleOp module = getOperation();
  applyPatternsAndFoldGreedily(module, patterns);

  // Erase inlined functions that don't have any references.
  //
  // TODO(hinsu): Update this to not erase entry points once TFLite support to
  // have multiple entry points is implemented. Until then, it is safe to
  // erase these functions.
  EraseDeadFuncs(inlined_funcs, module);
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateOptimizeFunctionalOpsPass() {
  return std::make_unique<OptimizeFunctionalOpsPass>();
}

static PassRegistration<OptimizeFunctionalOpsPass> pass(
    "tfl-optimize-functional-ops", "Optimize TensorFlow functional ops");
}  // namespace TFL
}  // namespace mlir
