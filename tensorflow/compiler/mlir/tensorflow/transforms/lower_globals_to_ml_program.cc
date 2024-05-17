/* Copyright 2022 Google Inc. All Rights Reserved.

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

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Dialect/Affine/Utils.h"  // from @llvm-project
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"  // from @llvm-project
#include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/RegionGraphTraits.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_saved_model {

// Trace upwards until we hit a (block) argument a value originates from.
static LogicalResult traceUpwardsToArgument(Value v, llvm::DenseSet<Value> seen,
                                            BlockArgument *out) {
  if (seen.contains(v)) {
    return failure();  // infinite loop
  }
  seen.insert(v);

  if (auto blockArg = mlir::dyn_cast<BlockArgument>(v)) {
    Operation *op = blockArg.getOwner()->getParentOp();

    // If we're in the first block, then the argument to that block is the
    // one we're looking for.
    if (auto func = cast<func::FuncOp>(op)) {
      if (blockArg.getOwner()->isEntryBlock()) {
        *out = blockArg;
        return success();
      }
    }

    // If we're in an inner block, the we have to find all ops that branch
    // to that block, and trace through them.
    llvm::DenseSet<BlockArgument> options;
    for (Block *block : blockArg.getOwner()->getPredecessors()) {
      if (!block->empty()) {
        Operation *back = &block->back();
        if (BranchOpInterface branchOp = dyn_cast<BranchOpInterface>(back)) {
          for (int i = 0; i < branchOp->getNumSuccessors(); i++) {
            if (branchOp->getSuccessor(i) != blockArg.getOwner()) continue;
            SuccessorOperands operands = branchOp.getSuccessorOperands(i);
            BlockArgument arg;
            if (traceUpwardsToArgument(operands[blockArg.getArgNumber()], seen,
                                       &arg)
                    .succeeded()) {
              options.insert(arg);
            }
          }
        } else {
          op->emitOpError("Predecessor op doesn't implement BranchOpInterface");
          return failure();
        }
      }
    }
    if (!options.empty()) {
      if (options.size() != 1) {
        op->emitOpError("Incompatible code paths.");
        return failure();
      } else {
        *out = *options.begin();
        return success();
      }
    }
    return op->emitOpError("Block has no predecessor");
  }

  if (v.getDefiningOp()->getNumOperands() == 1) {
    // If the value is originating from an unary op, assume it's something
    // simple like "cast" and keep tracing.
    return traceUpwardsToArgument(v.getDefiningOp()->getOperand(0), seen, out);
  } else {
    // Typically a tf.VarHandle op.
    return v.getDefiningOp()->emitOpError("Non constant predecessor");
  }
}

// For the "resource" attribute in a ReadVariable or AssignVariable op,
// determine the symbol reference to the (new) ml_program::GlobalOp.
SymbolRefAttr lookupGlobalTensor(func::FuncOp func, Value resource,
                                 SymbolTable &syms,
                                 DenseMap<Operation *, std::string> opToName) {
  llvm::DenseSet<Value> seen;
  BlockArgument arg;
  if (traceUpwardsToArgument(resource, seen, &arg).failed()) return nullptr;
  Operation *global =
      tf_saved_model::LookupBoundInputOfType<tf_saved_model::GlobalTensorOp>(
          func, arg.getArgNumber(), syms);
  auto name = opToName[global];
  if (name.empty()) return nullptr;
  return SymbolRefAttr::get(func->getContext(), opToName[global]);
}

static LogicalResult convertTFGlobals(ModuleOp module) {
  OpBuilder globalBuilder(module.getBodyRegion());
  DenseMap<Operation *, std::string> opToName;
  for (auto globalTensor : module.getOps<tf_saved_model::GlobalTensorOp>()) {
    auto exportedNames = tf_saved_model::GetExportedNames(globalTensor);
    std::string name;
    if (exportedNames.empty()) {
      name = "global_ml_" + globalTensor.getSymName().str();
    } else if (exportedNames.size() == 1) {
      name = exportedNames[0].str();
    } else {
      return globalTensor.emitError()
             << "Multiple exported names for global tensor not supported yet";
    }
    Attribute initial_value;
    if (globalTensor.getValue()) {
      initial_value = *globalTensor.getValue();
    } else {
      initial_value = mlir::Attribute();
    }
    opToName[globalTensor] = name;
    auto variableOp = globalBuilder.create<ml_program::GlobalOp>(
        globalTensor.getLoc(), name, globalTensor.getType(),
        globalTensor.getIsMutable(), initial_value,
        /*visibility=*/globalBuilder.getStringAttr("private"));
    variableOp.setPrivate();
  }

  SymbolTable syms(module);
  for (auto func : module.getOps<func::FuncOp>()) {
    if (!tf_saved_model::IsExported(func)) {
      continue;
    }
    bool success = true;
    func.walk([&](mlir::TF::ReadVariableOp op) {
      auto sym = lookupGlobalTensor(func, op.getResource(), syms, opToName);
      success &= !!sym;
      if (!success) return;
      OpBuilder builder(op);
      auto load = builder.create<mlir::ml_program::GlobalLoadOp>(
          op.getLoc(), op.getValue().getType(), sym);
      op.getValue().replaceAllUsesWith(load.getResult());
      op.erase();
    });
    func.walk([&](mlir::TF::AssignVariableOp op) {
      auto sym = lookupGlobalTensor(func, op.getResource(), syms, opToName);
      success &= !!sym;
      if (!success) return;
      OpBuilder builder(op);
      builder.create<mlir::ml_program::GlobalStoreOp>(op.getLoc(), sym,
                                                      op.getValue());
      op.erase();
    });
    if (!success) return failure();

    // Erase tf_saved_model attributes we consumed. We can't delete them
    // right away, since they might weave through blocks, but we can replace
    // them with a dummy which DCE can pick up later.
    llvm::BitVector argsToErase(func.getNumArguments());
    for (int i = 0; i < func.getNumArguments(); i++) {
      if (auto global = tf_saved_model::LookupBoundInputOfType<
              tf_saved_model::GlobalTensorOp>(func, i, syms)) {
        OpBuilder builder(func.getBody());
        auto dummy = builder.create<TF::VarHandleOp>(
            global.getLoc(), func.getArgument(i).getType(), "dummy", "dummy");
        func.getArgument(i).replaceAllUsesWith(dummy.getResult());
        argsToErase.set(i);
      }
    }
    func.eraseArguments(argsToErase);
  }

  // Erase all the global tensors.
  for (auto globalTensor : llvm::make_early_inc_range(
           module.getOps<tf_saved_model::GlobalTensorOp>())) {
    globalTensor.erase();
  }
  return success();
}

#define GEN_PASS_DEF_LOWERGLOBALSTOMLPROGRAMPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"
class LowerGlobalsToMlProgram
    : public impl::LowerGlobalsToMlProgramPassBase<LowerGlobalsToMlProgram> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::tf_saved_model::TensorFlowSavedModelDialect,
                    ml_program::MLProgramDialect>();
  }

  void runOnOperation() override {
    if (failed(convertTFGlobals(getOperation()))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> CreateLowerGlobalsToMlProgramPass() {
  return std::make_unique<LowerGlobalsToMlProgram>();
}

void RegisterLowerGlobalsToMlProgramPass() {
  registerPass(CreateLowerGlobalsToMlProgramPass);
}

}  // namespace tf_saved_model
}  // namespace mlir
