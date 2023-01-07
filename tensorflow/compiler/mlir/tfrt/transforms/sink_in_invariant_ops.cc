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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace tensorflow {
namespace {

// Clone the sinkable op associated with the func op to the func op
void CloneOpIntoFuncOp(
    const llvm::DenseMap<mlir::func::FuncOp,
                         llvm::DenseMap<int32_t, mlir::Operation *>>
        &func_op_operands_to_sink) {
  for (auto &iter : func_op_operands_to_sink) {
    auto func = iter.first;
    mlir::OpBuilder builder(func);
    builder.setInsertionPointToStart(&func.getBody().front());

    for (auto &operand_iter : iter.second) {
      auto *cloned_op = operand_iter.second->clone();
      func.getArgument(operand_iter.first)
          .replaceAllUsesWith(*cloned_op->getResults().begin());
      builder.insert(cloned_op);
      builder.setInsertionPointAfter(cloned_op);
    }
  }
}

// TODO(b/262610234): Generalize the sinking conditions.
// Check if the op qualifies to sink to the callee.
bool IsSinkCandidate(mlir::Operation *op) {
  return op && llvm::isa<mlir::TF::VarHandleOp, mlir::TF::ConstOp,
                         mlir::TF::HashTableOp>(op);
}

// Check if the op is allowed to be sinked. We are being conservative here to
// whilelist very limited set of ops here.
bool AllowSinkTo(mlir::Operation *op) {
  return llvm::isa<mlir::TF::BatchFunctionOp, mlir::TF::IfOp,
                   mlir::TF::StatefulPartitionedCallOp>(op);
}

// There are following cases:
// #1, sink v1
// @func1 { v1 = VarHandleOp, v2 = CallerOp{ f=@func2 }(v1) }
// @func2(arg0) { v2 = ReadVariableOp }
//
// #2, copy v1 to callee, still keep in func1
// @func1 { v1 = VarHandleOp, v2 = ReadVariableOp, v3 = CallerOp{ f=@func2 }(v1)
// }
// @func2(arg0) { v2 = ReadVariableOp(arg0) }
//
// #3, sink v1 and v2
// @func1 { v1 = VarHandleOp, v2 = ReadVariableOp, v3 = CallerOp{ f=@func2 }(v2)
// }
// @func2(arg0) { v2 = OtherOp(arg0) }
//
// #4, copy v1 and v2 to func2, keep in func1
// @func1 { v1 = VarHandleOp, v2 = ReadVariableOp, v3 = OtherOp(v2), v4 =
// CallerOp{ f=@func2 }(v2) }
// @func2(arg0) { v2 = OtherOp(arg0) }
//
// We only support #1 for now as that's the most common pattern from production.
// If we implement #2 and #4 in the future, should consider dedupe in the
// tfrt_resource_init because multiple resource handle will be created on the
// same resource.

void SinkInInvariantOps(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  mlir::SymbolTableCollection symbol_table_collection;
  mlir::SymbolUserMap symbol_users(symbol_table_collection, module);

  // Maps from function op, to the operand index to erase, to the caller op.
  llvm::DenseMap<mlir::func::FuncOp, llvm::DenseMap<int32_t, mlir::Operation *>>
      func_op_operands_to_sink;

  // TODO(b/263191534): Replace with CallOpInterface to handle callees.
  // Identify the invariant Op, Caller, Callee FuncOp to update.
  module.walk([&](mlir::Operation *op) {
    if (!AllowSinkTo(op)) return;

    auto track_callee = [&](mlir::func::FuncOp &func_op) {
      auto diff = op->getNumOperands() - func_op.getNumArguments();
      for (int i = 0; i < func_op.getNumArguments(); ++i) {
        auto arg_op = op->getOperand(diff + i).getDefiningOp();
        if (!IsSinkCandidate(arg_op)) continue;
        func_op_operands_to_sink[func_op][i] = arg_op;
      }
    };

    llvm::DenseSet<llvm::StringRef> callees;
    for (const auto &named_attr : op->getAttrs()) {
      if (auto symbol_attr =
              named_attr.getValue().dyn_cast<mlir::FlatSymbolRefAttr>()) {
        auto symbol = symbol_attr.getValue();

        auto callee = symbol_table.lookup<mlir::func::FuncOp>(symbol);
        if (!callee) continue;

        // One callee invoked by multiple caller is skipped for simplicity.
        // Consider adding support if more usage are observed from production.
        if (const llvm::ArrayRef<mlir::Operation *> users =
                symbol_users.getUsers(callee);
            users.size() > 1)
          continue;

        // Invoked by same caller multiple times, only process the first one.
        if (callees.count(symbol)) continue;
        track_callee(callee);
        callees.insert(symbol);
      }
    }
  });

  CloneOpIntoFuncOp(func_op_operands_to_sink);
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
