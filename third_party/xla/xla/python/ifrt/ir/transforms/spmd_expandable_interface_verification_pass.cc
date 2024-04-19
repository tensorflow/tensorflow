/* Copyright 2023 The OpenXLA Authors.

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

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_SPMDEXPANDABLEINTERFACEVERIFICATIONPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

class SpmdExpandableInterfaceVerificationPass
    : public impl::SpmdExpandableInterfaceVerificationPassBase<
          SpmdExpandableInterfaceVerificationPass> {
 public:
  using impl::SpmdExpandableInterfaceVerificationPassBase<
      SpmdExpandableInterfaceVerificationPass>::
      SpmdExpandableInterfaceVerificationPassBase;

  mlir::LogicalResult initialize(mlir::MLIRContext* context) override {
    dialects_require_no_spmd_interface_.insert(excluded_dialects_.begin(),
                                               excluded_dialects_.end());
    return mlir::success();
  }

  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    llvm::SmallSet<mlir::func::FuncOp, 4> visited_callees;
    mlir::SymbolTableCollection symbol_table;

    auto result = module_op.walk([&](CallOp call_op) -> mlir::WalkResult {
      llvm::ArrayRef<int> devices = call_op.getDevices();
      DCHECK_GT(devices.size(), 0);
      // CallOp with only 1 device need no SPMD expansion, so skip checking.
      if (devices.size() == 1) {
        return mlir::WalkResult::advance();
      }

      mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);

      if (auto [unused, inserted] = visited_callees.insert(callee); !inserted) {
        return mlir::WalkResult::advance();
      }

      // Check each op in the callee function.
      if (HasOpWithUnimplementedInterface(callee)) {
        return mlir::WalkResult::interrupt();
      }

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

 private:
  bool HasOpWithUnimplementedInterface(mlir::func::FuncOp func_op) {
    auto result = func_op->walk([&](mlir::Operation* op) -> mlir::WalkResult {
      if (llvm::isa<mlir::func::FuncOp>(op) ||
          dialects_require_no_spmd_interface_.contains(
              op->getName().getDialectNamespace())) {
        return mlir::WalkResult::advance();
      }

      // Other ops should implement the interface.
      if (!llvm::isa<IfrtSpmdExpandable>(op)) {
        return op->emitOpError()
               << "requires op to have `IfrtSpmdExpandable` "
                  "OpInterface implemented or the dialect `"
               << op->getName().getDialectNamespace().str()
               << "` to be added to the excluded-dialects list.";
      }
      return mlir::WalkResult::advance();
    });
    return result.wasInterrupted();
  }

  absl::flat_hash_set<std::string> dialects_require_no_spmd_interface_;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateSpmdExpandableInterfaceVerificationPass(
    SpmdExpandableInterfaceVerificationPassOptions options) {
  return std::make_unique<SpmdExpandableInterfaceVerificationPass>(options);
}
}  // namespace ifrt
}  // namespace xla
