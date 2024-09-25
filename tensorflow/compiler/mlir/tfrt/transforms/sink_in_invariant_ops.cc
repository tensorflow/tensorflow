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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace tensorflow {
namespace {

// TODO(b/262610234): Generalize the sinking conditions.
// Check if the op qualifies to sink to the callee.
bool IsSinkCandidate(mlir::Operation *op) {
  return op && llvm::isa<mlir::TF::VarHandleOp, mlir::TF::ConstOp,
                         mlir::TF::HashTableV2Op>(op);
}

// Check if the op is allowed to be sinked. We are being conservative here to
// whilelist very limited set of ops here.
struct AllowSinkHelper {
  explicit AllowSinkHelper(mlir::Operation *op, int arg_index) {
    if (llvm::isa<mlir::TF::BatchFunctionOp,
                  mlir::TF::StatefulPartitionedCallOp>(op)) {
      allow_sink_to = true;
      callee_arg_index = arg_index;
      return;
    }

    if (llvm::isa<mlir::TF::IfOp>(op) && arg_index > 0) {
      allow_sink_to = true;
      callee_arg_index = arg_index - 1;
      return;
    }
  }

  bool allow_sink_to = false;
  int callee_arg_index = 0;
};

llvm::SmallVector<mlir::Value> FindValueInCallees(
    const mlir::SymbolTable &symbol_table,
    const mlir::SymbolUserMap &symbol_users, mlir::Operation *caller,
    int arg_index) {
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallDenseSet<llvm::StringRef> callees;
  for (const auto &named_attr : caller->getAttrs()) {
    if (auto symbol_attr =
            mlir::dyn_cast<mlir::FlatSymbolRefAttr>(named_attr.getValue())) {
      auto symbol = symbol_attr.getValue();

      auto callee = symbol_table.lookup<mlir::func::FuncOp>(symbol);
      if (!callee) continue;

      // One callee invoked by multiple caller is skipped for simplicity.
      // Consider adding support if more usage are observed from production.
      if (llvm::ArrayRef<mlir::Operation *> users =
              symbol_users.getUsers(callee);
          users.size() > 1)
        continue;

      // Invoked by same caller multiple times, only process the first one.
      if (!callees.insert(symbol).second) continue;

      values.push_back(callee.getArgument(arg_index));
    }
  }
  return values;
}

void FindSinkTarget(
    const mlir::SymbolTable &symbol_table,
    const mlir::SymbolUserMap &symbol_users, mlir::OpResult original,
    mlir::Value value,
    llvm::DenseMap<mlir::OpOperand *, llvm::SmallDenseSet<mlir::OpResult>>
        &targets) {
  for (mlir::OpOperand &use : value.getUses()) {
    auto *user = use.getOwner();

    AllowSinkHelper helper(user, use.getOperandNumber());

    if (helper.allow_sink_to) {
      auto values = FindValueInCallees(symbol_table, symbol_users, user,
                                       helper.callee_arg_index);
      for (auto value : values) {
        FindSinkTarget(symbol_table, symbol_users, original, value, targets);
      }
    } else if (value != original) {
      targets[&use].insert(original);
    }
  }
}

// Sink in invariant ops like tf.Const, tf.VarHandleOp and tf.HashTableV2 ops
// into sinkable calls like tf.BatchFunction and tf.If. If there are nested
// calls, the invariant ops will only be copied at the target.
void SinkInInvariantOps(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  mlir::SymbolTableCollection symbol_table_collection;
  mlir::SymbolUserMap symbol_users(symbol_table_collection, module);

  // TODO(b/263191534): Replace with CallOpInterface to handle callees.
  // Identify the invariant Op, Caller, Callee FuncOp to update.

  llvm::DenseMap<mlir::OpOperand *, llvm::SmallDenseSet<mlir::OpResult>>
      targets;
  module.walk([&](mlir::Operation *op) {
    if (IsSinkCandidate(op)) {
      for (auto value : op->getOpResults()) {
        FindSinkTarget(symbol_table, symbol_users, value, value, targets);
      }
    }
  });

  // Clone the sinkable op associated with the func op to the func op
  mlir::OpBuilder builder(module);
  for (const auto &p : targets) {
    if (p.second.size() != 1) continue;

    auto *use = p.first;

    builder.setInsertionPointToStart(use->getOwner()->getBlock());

    mlir::OpResult original = *p.second.begin();
    auto *new_op = builder.clone(*original.getDefiningOp());

    use->get().replaceAllUsesWith(
        new_op->getResult(original.getResultNumber()));
  }
}

class SinkInInvariantOpsPass
    : public mlir::PassWrapper<SinkInInvariantOpsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkInInvariantOpsPass)

  llvm::StringRef getArgument() const final {
    return "tfrt-sink-in-invariant-ops";
  }
  llvm::StringRef getDescription() const final {
    return "Sink in the invariant ops to facilitate invariant ops hoisting.";
  }

  void runOnOperation() override {
    auto module = getOperation();
    SinkInInvariantOps(module);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateSinkInInvariantOpsPass() {
  return std::make_unique<SinkInInvariantOpsPass>();
}

static mlir::PassRegistration<SinkInInvariantOpsPass>
    sink_in_invariant_ops_pass;

}  // namespace tensorflow
