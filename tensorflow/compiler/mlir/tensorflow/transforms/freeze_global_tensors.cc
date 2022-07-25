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

#include <algorithm>
#include <vector>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/savedmodel_passes_detail.h"

#define DEBUG_TYPE "freeze-global-tensor"

namespace mlir {
namespace tf_saved_model {

// The value of our lattice represents the GlobalTensorOp matching the value.
struct ResourceLatticeValue {
  explicit ResourceLatticeValue(GlobalTensorOp op = nullptr) {
    if (op) ops.insert(op);
  }

  static ResourceLatticeValue getPessimisticValueState(MLIRContext *context) {
    return ResourceLatticeValue();
  }
  static ResourceLatticeValue getPessimisticValueState(Value value) {
    if (auto barg = value.dyn_cast<BlockArgument>()) {
      if (func::FuncOp func =
              dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp())) {
        SymbolTable symbol_table(func->getParentOfType<ModuleOp>());
        auto global_tensor = LookupBoundInputOfType<GlobalTensorOp>(
            func, barg.getArgNumber(), symbol_table);
        return ResourceLatticeValue(global_tensor);
      }
    }
    return ResourceLatticeValue();
  }

  bool operator==(const ResourceLatticeValue &rhs) const {
    return ops == rhs.ops;
  }

  static ResourceLatticeValue join(const ResourceLatticeValue &lhs,
                                   const ResourceLatticeValue &rhs) {
    // Take union of both sets of possible GlobalTensorOp values that can be
    // referenced here.
    ResourceLatticeValue ret;
    ret.ops.insert(lhs.ops.begin(), lhs.ops.end());
    ret.ops.insert(rhs.ops.begin(), rhs.ops.end());
    return ret;
  }

  void print(raw_ostream &os) const {
    llvm::interleaveComma(ops, os << "["), os << "]";
  }

  // The location which originated the int value.
  // IR constructs (i.e., GlobalTensorOp) are not const-correct.
  mutable DenseSet<GlobalTensorOp> ops;
};

namespace {
class ResourceAnalysis : public dataflow::SparseDataFlowAnalysis<
                             dataflow::Lattice<ResourceLatticeValue>> {
 public:
  using StateT = dataflow::Lattice<ResourceLatticeValue>;
  using dataflow::SparseDataFlowAnalysis<StateT>::SparseDataFlowAnalysis;
  ~ResourceAnalysis() override = default;

  void visitOperation(Operation *op, ArrayRef<const StateT *> operands,
                      ArrayRef<StateT *> results) override {
    LLVM_DEBUG(llvm::dbgs() << "ResAn: Visiting operation: " << *op << "\n");
    markAllPessimisticFixpoint(results);
  }
};

struct FreezeGlobalTensorsPass
    : public FreezeGlobalTensorsPassBase<FreezeGlobalTensorsPass> {
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
  solver.load<ResourceAnalysis>();
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
    if (global_tensor.is_mutable()) {
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
      if (!getElementTypeOrSelf(val.getType()).isa<TF::ResourceType>())
        continue;

      // Check that there is only a single global tensor associated with arg.
      const ResourceAnalysis::StateT *latticeElement =
          solver.lookupState<ResourceAnalysis::StateT>(val);
      if (!latticeElement || latticeElement->getValue().ops.size() != 1)
        continue;

      // Don't freeze mutable tensors.
      if (latticeElement->getValue().ops.begin()->is_mutable()) {
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

      const ResourceAnalysis::StateT *latticeElement =
          solver.lookupState<ResourceAnalysis::StateT>(val);
      GlobalTensorOp global_tensor = *latticeElement->getValue().ops.begin();

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
                                                  global_tensor.value());
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
