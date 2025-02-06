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
#include <iterator>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

using ::mlir::ArrayRef;
using ::mlir::ModuleOp;
using ::mlir::Operation;
using ::mlir::SymbolTable;
using ::mlir::SymbolTableCollection;
using ::mlir::SymbolUserMap;

// This only includes some preliminary checks as this is a short term solution.
bool AreEquivalent(mlir::func::FuncOp& lhs, mlir::func::FuncOp& rhs) {
  if (lhs.getFunctionType() != rhs.getFunctionType()) return false;

  for (auto arg_pair : llvm::zip(lhs.getArguments(), rhs.getArguments())) {
    auto& lhs_arg = std::get<0>(arg_pair);
    auto& rhs_arg = std::get<1>(arg_pair);
    if (lhs_arg.getType() != rhs_arg.getType()) return false;
  }

  auto lhs_ops = lhs.getBody().getOps();
  auto rhs_ops = rhs.getBody().getOps();
  if (std::distance(lhs_ops.begin(), lhs_ops.end()) !=
      std::distance(rhs_ops.begin(), rhs_ops.end()))
    return false;

  for (auto op_pair : llvm::zip(lhs_ops, rhs_ops)) {
    auto& lhs_op = std::get<0>(op_pair);
    auto& rhs_op = std::get<1>(op_pair);
    if (lhs_op.getName() != rhs_op.getName()) return false;
    if (lhs_op.getNumRegions() != rhs_op.getNumRegions()) return false;
    if (lhs_op.getNumSuccessors() != rhs_op.getNumSuccessors()) return false;
    if (!std::equal(lhs_op.getOperandTypes().begin(),
                    lhs_op.getOperandTypes().end(),
                    rhs_op.getOperandTypes().begin()))
      return false;
    if (!std::equal(lhs_op.getResultTypes().begin(),
                    lhs_op.getResultTypes().end(),
                    rhs_op.getResultTypes().begin()))
      return false;
  }

  return true;
}

// Deduplicate the functions if all users are BatchFunctionOp and have the same
// shared_name.
//
// TODO(b/192463730): this is the short term solution and not needed anymore
// after the shape inference pass is revamped with ideal solution
// (b/192463730#comment11).
class DeduplicateFunctionsInovkedByBatchFunction
    : public mlir::PassWrapper<DeduplicateFunctionsInovkedByBatchFunction,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      DeduplicateFunctionsInovkedByBatchFunction)

 private:
  llvm::StringRef getArgument() const final {
    return "tfrt-deduplicate-functions-invoked-by-batch-function";
  }
  llvm::StringRef getDescription() const final {
    return "Deduplicate the functions invoked by tf.BatchFunction with the "
           "same shared_name";
  }
  void runOnOperation() override {
    if (failed(Run())) {
      signalPassFailure();
    }
  }

  mlir::LogicalResult Run();
};

mlir::LogicalResult DeduplicateFunctionsInovkedByBatchFunction::Run() {
  ModuleOp module = getOperation();
  SymbolTableCollection symbol_table_collection;
  SymbolTable& symbol_table = symbol_table_collection.getSymbolTable(module);
  SymbolUserMap symbol_users(symbol_table_collection, module);

  // Categorize the functions invoked by BatchFunctionOp by its shared_name.
  llvm::StringMap<llvm::SmallVector<mlir::func::FuncOp, 2>>
      shared_name_to_func_ops;

  for (auto func :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
    ArrayRef<Operation*> users = symbol_users.getUsers(func);
    llvm::StringRef shared_name;
    // Deduplicate the function only if all users are BatchFunctionOp and have
    // the same shared_name
    if (!users.empty() && llvm::all_of(users, [&shared_name](Operation* user) {
          auto op = llvm::dyn_cast_or_null<mlir::TF::BatchFunctionOp>(user);
          // User is not a BatchFunctionOp
          if (!op) return false;
          if (shared_name.empty()) {
            shared_name = op.getSharedName();
            return true;
          }
          return shared_name == op.getSharedName();
        })) {
      shared_name_to_func_ops[shared_name].push_back(func);
    }
  }

  for (auto& it : shared_name_to_func_ops) {
    auto& func_ops = it.second;
    mlir::func::FuncOp& func_op_to_keep = func_ops.front();
    for (mlir::func::FuncOp& func_op_to_remove : llvm::drop_begin(func_ops)) {
      if (!AreEquivalent(func_op_to_keep, func_op_to_remove)) {
        return func_op_to_remove.emitError(
            "func_ops for BatchFunctionOp with the same shared name are "
            "different");
      }
      if (failed(SymbolTable::replaceAllSymbolUses(
              func_op_to_remove, func_op_to_keep.getSymNameAttr(), module))) {
        return func_op_to_remove.emitError("unable to replace the symbol use");
      }
      symbol_table.erase(func_op_to_remove);
    }
  }

  return mlir::success();
}
}  // namespace

std::unique_ptr<mlir::OperationPass<ModuleOp>>
CreateDeduplicateFunctionsInovkedByBatchFunctionPass() {
  return std::make_unique<DeduplicateFunctionsInovkedByBatchFunction>();
}

static mlir::PassRegistration<DeduplicateFunctionsInovkedByBatchFunction>
    register_pass;

}  // namespace tfrt_compiler
}  // namespace tensorflow
