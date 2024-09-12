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

#include "llvm/ADT/Hashing.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTDUPLICATEDCALLEEELIMINATIONPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

// Compares FuncOps except symbol name.
struct FuncInfo : llvm::DenseMapInfo<mlir::func::FuncOp> {
  using OperationEquivalence = ::mlir::OperationEquivalence;

  static unsigned getHashValue(mlir::func::FuncOp func_op) {
    llvm::hash_code hash = {};
    for (mlir::NamedAttribute attr : func_op->getAttrs()) {
      if (attr.getName() == func_op.getSymNameAttrName()) {
        continue;
      }
      hash = llvm::hash_combine(hash, attr);
    }
    func_op.getBody().walk([&](mlir::Operation* op) {
      hash = llvm::hash_combine(
          hash, OperationEquivalence::computeHash(
                    op, /*hashOperands=*/OperationEquivalence::ignoreHashValue,
                    /*hashResults=*/OperationEquivalence::ignoreHashValue,
                    OperationEquivalence::IgnoreLocations));
    });
    return hash;
  }

  static bool isEqual(mlir::func::FuncOp lhs, mlir::func::FuncOp rhs) {
    if (lhs == rhs) {
      return true;
    }
    if (lhs == getEmptyKey() || lhs == getTombstoneKey() ||
        rhs == getEmptyKey() || rhs == getTombstoneKey()) {
      return false;
    }
    if (lhs.getFunctionType() != rhs.getFunctionType()) {
      return false;
    }
    mlir::NamedAttrList lattrs = lhs->getAttrDictionary();
    mlir::NamedAttrList rattrs = rhs->getAttrDictionary();
    lattrs.erase(lhs.getSymNameAttrName());
    rattrs.erase(rhs.getSymNameAttrName());
    if (lattrs != rattrs) {
      return false;
    }
    return OperationEquivalence::isRegionEquivalentTo(
        &lhs.getBody(), &rhs.getBody(), OperationEquivalence::IgnoreLocations);
  }
};

class IfrtDuplicatedCalleeEliminationPass
    : public impl::IfrtDuplicatedCalleeEliminationPassBase<
          IfrtDuplicatedCalleeEliminationPass> {
 public:
  void runOnOperation() override;
};

void IfrtDuplicatedCalleeEliminationPass::runOnOperation() {
  mlir::SymbolTableCollection symbol_table;
  mlir::DenseMap<mlir::func::FuncOp, mlir::SymbolRefAttr, FuncInfo>
      unique_funcs;
  getOperation().walk([&](xla::ifrt::CallOp call_op) {
    mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
    auto [it, inserted] =
        unique_funcs.insert({callee, call_op.getCalleeAttr()});
    if (!inserted) {
      call_op.setCalleeAttr(it->second);
    }
  });
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtDuplicatedCalleeEliminationPass() {
  return std::make_unique<IfrtDuplicatedCalleeEliminationPass>();
}

}  // namespace ifrt
}  // namespace xla
