/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/constants.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORFUNCTIONRENAMING
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

struct DTensorFunctionRenaming
    : public impl::DTensorFunctionRenamingBase<DTensorFunctionRenaming> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    const std::string append =
        module->getAttrOfType<mlir::StringAttr>(dtensor::kCacheKey)
            .getValue()
            .str();

    // If the cache key isn't set, simply return without renameing functions.
    if (append.empty()) return;

    mlir::SymbolTableCollection symbol_table;
    mlir::SymbolUserMap symbolUsers(symbol_table, module);

    for (mlir::func::FuncOp func_op :
         llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
      // Only rename private functions, functions which are public (i.e. the
      // main function of the module), must have stable names since they are
      // public and may be used by other modules/pieces of code.
      if (func_op.getVisibility() != mlir::SymbolTable::Visibility::Private)
        continue;
      std::string new_name = absl::StrCat(
          mlir::SymbolTable::getSymbolName(func_op).getValue().str(), append);
      symbolUsers.replaceAllUsesWith(
          func_op, mlir::StringAttr::get(&getContext(), new_name));
      mlir::SymbolTable::setSymbolName(func_op, new_name);
    }
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateFunctionRenamingPass() {
  return std::make_unique<DTensorFunctionRenaming>();
}

}  // namespace dtensor
}  // namespace tensorflow
