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
#include <string_view>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_ops.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/passes.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

#define GEN_PASS_DEF_EXPORTFUNCTIONS
#include "tensorflow/compiler/xla/mlir/runtime/transforms/passes.h.inc"

class ExportFunctionsPass
    : public impl::ExportFunctionsBase<ExportFunctionsPass> {
  void runOnOperation() override;
};

static void ConvertReturnOperations(func::FuncOp func, Value exec_ctx) {
  // Convert all returns to the Runtime API calls.
  func.walk([&](func::ReturnOp ret) {
    ImplicitLocOpBuilder b(ret.getLoc(), ret);

    // Return all outputs via the `rt.set_output` operation.
    for (const auto& pair : llvm::enumerate(ret.getOperands())) {
      b.create<SetOutputOp>(exec_ctx, pair.index(), pair.value());
    }

    // Replace original return with an empty one.
    b.create<func::ReturnOp>();
    ret.erase();
  });

  // Update function type to the function with empty results.
  auto type = FunctionType::get(func.getContext(), func.getArgumentTypes(), {});
  func.setType(type);
}

static Value PrependExecutionContextArgument(func::FuncOp func) {
  Type new_type = ExecutionContextType::get(func.getContext());
  DictionaryAttr attr = DictionaryAttr::get(func.getContext());
  func.insertArguments({0}, {new_type}, {attr}, {func.getLoc()});
  return func.getArgument(0);
}

static void ConvertExportedFunction(ExportOp exported, func::FuncOp func) {
  Value exec_ctx = PrependExecutionContextArgument(func);
  ConvertReturnOperations(func, exec_ctx);

  // After conversion mark exported function with an attribute.
  func->setAttr(kExportedAttrName, exported.getOrdinalAttr());
}

void ExportFunctionsPass::runOnOperation() {
  llvm::SmallVector<std::pair<ExportOp, func::FuncOp>> exported;

  // Collect exported functions.
  SymbolTable sym_table(getOperation());
  getOperation().walk([&](ExportOp op) {
    if (op.getOrdinal().has_value()) {
      func::FuncOp func = sym_table.lookup<func::FuncOp>(op.getFunctionRef());
      exported.emplace_back(op, func);
    }
  });

  // Convert all exported functions.
  llvm::for_each(exported, [](std::pair<ExportOp, func::FuncOp>& pair) {
    ConvertExportedFunction(pair.first, pair.second);
    pair.first.erase();
  });
}

std::unique_ptr<OperationPass<ModuleOp>> CreateExportRuntimeFunctionsPass() {
  return std::make_unique<ExportFunctionsPass>();
}

}  // namespace runtime
}  // namespace xla
