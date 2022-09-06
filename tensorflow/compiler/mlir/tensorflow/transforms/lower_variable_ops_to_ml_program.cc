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
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"  // from @llvm-project
#include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_dataflow.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/savedmodel_passes_detail.h"

namespace mlir {
namespace tf_saved_model {

namespace {

std::string GetVariableName(Operation* op) {
  if (auto handle = dyn_cast<TF::VarHandleOp>(op)) {
    std::string container = handle.container().str();
    std::string shared_name = handle.shared_name().str();
    if (container.empty()) {
      return absl::StrCat("vars.", shared_name);
    } else {
      return absl::StrCat("vars.", container, ".", shared_name);
    }
  } else if (auto global = dyn_cast<tf_saved_model::GlobalTensorOp>(op)) {
    return absl::StrCat("vars.", global.sym_name().str());
  }
  return "<no name>";
}

Operation* GetHandleSource(Operation* op, DataFlowSolver& solver) {
  Value resource;
  if (auto read = llvm::dyn_cast<TF::ReadVariableOp>(op)) {
    resource = read.resource();
  } else if (auto write = llvm::dyn_cast<TF::AssignVariableOp>(op)) {
    resource = write.resource();
  }
  const TF::ResourceDataflowAnalysis::StateT* state =
      solver.lookupState<TF::ResourceDataflowAnalysis::StateT>(resource);
  if (!state) {
    return nullptr;
  }
  auto ops = state->getValue().ops;
  if (ops.size() != 1) {
    return nullptr;
  }
  Operation* source = *ops.begin();
  return source;
}

ml_program::GlobalOp CreateGlobalOpFromOp(Type type, Operation* source,
                                          OpBuilder& builder,
                                          SymbolTable& symbol_table) {
  std::string name = GetVariableName(source);
  if (auto existing = symbol_table.lookup<ml_program::GlobalOp>(name)) {
    // TODO(kramm): Verify that this is the same type.
    return existing;
  }
  auto globalOp = builder.create<ml_program::GlobalOp>(
      builder.getBlock()->getParentOp()->getLoc(), name, type, true,
      builder.getZeroAttr(type), nullptr);
  symbol_table.insert(globalOp);
  // TODO(kramm): If we're converting from a GlobalTensorOp, also
  // convert the initial value.
  return globalOp;
}

}  // namespace

struct LowerVariableOpsToMlProgramPass
    : public LowerVariableOpsToMlProgramPassBase<
          LowerVariableOpsToMlProgramPass> {
  explicit LowerVariableOpsToMlProgramPass() {}
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::tf_saved_model::TensorFlowSavedModelDialect,
                    ml_program::MLProgramDialect>();
  }
  void runOnOperation() override {
    auto module = getOperation();
    if (!tf_saved_model::HasTfSavedModelSemantics(module)) return;

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<TF::ResourceDataflowAnalysis>();
    if (failed(solver.initializeAndRun(module))) return signalPassFailure();

    SymbolTable symbol_table(module);

    OpBuilder globalBuilder(module.getBodyRegion());

    module.walk([&](TF::ReadVariableOp op) {
      Operation* source = GetHandleSource(op, solver);
      if (!source) return;
      ml_program::GlobalOp globalOp = CreateGlobalOpFromOp(
          op->getResult(0).getType(), source, globalBuilder, symbol_table);
      if (!globalOp) return;
      OpBuilder builder(op);
      auto loadOp = builder.create<mlir::ml_program::GlobalLoadOp>(
          op.getLoc(), op.value().getType(),
          SymbolRefAttr::get(op->getContext(), globalOp.getSymName()));
      op.getResult().replaceAllUsesWith(loadOp.getResult());
      op.erase();
    });

    module.walk([&](TF::AssignVariableOp op) {
      Operation* source = GetHandleSource(op, solver);
      if (!source) return;
      ml_program::GlobalOp globalOp = CreateGlobalOpFromOp(
          op.value().getType(), source, globalBuilder, symbol_table);
      if (!globalOp) return;
      symbol_table.insert(globalOp);
      OpBuilder builder(op);
      builder.create<mlir::ml_program::GlobalStoreOp>(
          op.getLoc(),
          SymbolRefAttr::get(op->getContext(), globalOp.getSymName()),
          op.value());
      op.erase();
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
CreateLowerVariableOpsToMlProgramPass() {
  return std::make_unique<LowerVariableOpsToMlProgramPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
