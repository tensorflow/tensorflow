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

#include <memory>

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_dataflow.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"

#define DEBUG_TYPE "freeze-global-tensor"

namespace mlir {
namespace tf_saved_model {

namespace {

#define GEN_PASS_DEF_FREEZEGLOBALTENSORSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"
struct FreezeGlobalTensorsPass
    : public impl::FreezeGlobalTensorsPassBase<FreezeGlobalTensorsPass> {
  explicit FreezeGlobalTensorsPass(bool allow_mutable_tensors) {
    this->allow_mutable_tensors = allow_mutable_tensors;
  }
  void runOnOperation() override;
};

void FreezeGlobalTensorsPass::runOnOperation() {
  auto module = getOperation();
  if (!tf_saved_model::HasTfSavedModelSemantics(module)) return;

  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  TF::LoadResourceDataflowAnalysis(solver);
  if (failed(solver.initializeAndRun(module))) return signalPassFailure();

  DenseSet<GlobalTensorOp> remaining_global_tensor_ops;
  {
    auto ops = module.getOps<GlobalTensorOp>();
    remaining_global_tensor_ops.insert(ops.begin(), ops.end());
  }

  for (auto global_tensor : remaining_global_tensor_ops) {
    // This pass assumes that all global tensors as immutable (e.g. by a
    // previous optimize global tensors pass). If not, this pass has to fail
    // since it cannot perform one of its goals.
    if (global_tensor.getIsMutable()) {
      if (allow_mutable_tensors) continue;
      global_tensor.emitError()
          << "is not immutable, try removing mutable variables in your model "
             "since mutable variables are currently not supported through "
             "this converter";
      return signalPassFailure();
    }
  }

  // Collect all those freezable. This is an extra scan but allows for the
  // partial behavior from `allow_mutable_tensor`.
  DenseMap<BlockArgument, bool> freezeable;
  for (auto func : module.getOps<func::FuncOp>()) {
    for (BlockArgument val : func.getArguments()) {
      if (!mlir::isa<TF::ResourceType>(getElementTypeOrSelf(val.getType())))
        continue;

      // Check that there is only a single global tensor associated with arg.
      const TF::ResourceDataflowState *latticeElement =
          solver.lookupState<TF::ResourceDataflowState>(val);
      if (!latticeElement || latticeElement->getValue().ops.size() != 1)
        continue;

      // Don't freeze mutable tensors.
      Operation *op = *latticeElement->getValue().ops.begin();
      GlobalTensorOp globalTensor = llvm::dyn_cast<GlobalTensorOp>(op);

      if (!globalTensor)
        continue;  // happens if the name is e.g. in a VarHandleOp.

      if (globalTensor.getIsMutable()) {
        freezeable[val] = false;
        continue;
      }

      freezeable[val] = true;

      // Verify users are supported kind.
      for (Operation *user : val.getUsers()) {
        if (!(isa<TF::ReadVariableOp>(user) || isa<CallOpInterface>(user))) {
          freezeable[val] = false;
          // Error out early if possible.
          if (!allow_mutable_tensors) {
            user->emitError()
                << "could not rewrite use of immutable bound input";
            return signalPassFailure();
          }
        }
      }
    }
  }

  DenseSet<GlobalTensorOp> frozen_global_tensors;
  for (auto func : module.getOps<func::FuncOp>()) {
    llvm::BitVector args_to_erase(func.getNumArguments());
    DenseMap<Operation *, llvm::BitVector> remove_operands;
    OpBuilder builder(func.getBody());

    for (BlockArgument val : func.getArguments()) {
      if (!freezeable[val]) continue;

      const TF::ResourceDataflowState *latticeElement =
          solver.lookupState<TF::ResourceDataflowState>(val);
      Operation *op = *latticeElement->getValue().ops.begin();
      GlobalTensorOp global_tensor = llvm::dyn_cast<GlobalTensorOp>(op);
      if (!global_tensor)
        continue;  // happens if the name is e.g. in a VarHandleOp.

      if (!global_tensor.getValue())
        continue;  // a value wasn't loaded for this tensor.

      SmallVector<TF::ReadVariableOp, 4> read_variable_ops_to_erase;
      frozen_global_tensors.insert(global_tensor);

      for (Operation *user : val.getUsers()) {
        if (auto read_op = llvm::dyn_cast<TF::ReadVariableOp>(user)) {
          // Collect all read variable ops so that all its uses can be replaced
          // with the tf.constant corresponding to the global tensor op.
          read_variable_ops_to_erase.push_back(read_op);
        } else {
          llvm::BitVector &bvector = remove_operands[user];
          bvector.resize(user->getNumOperands());
          for (OpOperand &use : user->getOpOperands())
            bvector.set(use.getOperandNumber());
        }
      }

      // Replace the arg with a tf.Const op in the function body.
      builder.setInsertionPointToStart(&func.getBody().front());
      auto const_op = builder.create<TF::ConstOp>(global_tensor.getLoc(),
                                                  *global_tensor.getValue());
      args_to_erase.set(val.getArgNumber());
      for (auto read_op : read_variable_ops_to_erase) {
        read_op.getResult().replaceAllUsesWith(const_op.getResult());
        read_op.erase();
      }
    }
    // As the other uses are call operations, we simply remove the arguments
    // as the function arguments will be removed below once that function is
    // processed.
    for (auto it : remove_operands) {
      it.first->eraseOperands(it.second);
    }

    func.eraseArguments(args_to_erase);
  }

  // Erase all global tensors that were frozen.
  for (auto global_tensor : frozen_global_tensors) {
    remaining_global_tensor_ops.erase(global_tensor);
    global_tensor->erase();
  }

  // Verify that there are no remaining global tensors.
  if (!allow_mutable_tensors && !remaining_global_tensor_ops.empty()) {
    module.emitError() << "could not freeze all global tensors in the module";
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateFreezeGlobalTensorsPass(
    bool allow_mutable_tensors) {
  return std::make_unique<FreezeGlobalTensorsPass>(allow_mutable_tensors);
}

}  // namespace tf_saved_model
}  // namespace mlir
