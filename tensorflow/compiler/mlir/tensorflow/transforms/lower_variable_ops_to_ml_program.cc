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
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
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

namespace mlir {
namespace tf_saved_model {

namespace {

std::string GetVariableName(Operation* op) {
  if (auto handle = dyn_cast<TF::VarHandleOp>(op)) {
    std::string container = handle.getContainer().str();
    std::string shared_name = handle.getSharedName().str();
    if (container.empty()) {
      return absl::StrCat("vars.", shared_name);
    } else {
      return absl::StrCat("vars.", container, ".", shared_name);
    }
  } else if (auto global = dyn_cast<tf_saved_model::GlobalTensorOp>(op)) {
    return absl::StrCat("vars.", global.getSymName().str());
  }
  return "<no name>";
}

Operation* GetHandleSource(Operation* op, DataFlowSolver& solver) {
  Value resource;
  if (auto read = llvm::dyn_cast<TF::ReadVariableOp>(op)) {
    resource = read.getResource();
  } else if (auto write = llvm::dyn_cast<TF::AssignVariableOp>(op)) {
    resource = write.getResource();
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

Attribute GetInitialValue(Operation* source) {
  if (auto global = dyn_cast<tf_saved_model::GlobalTensorOp>(source)) {
    if (global.getValue()) {
      return *global.getValue();
    }
  }
  return nullptr;
}

Type GetGlobalType(Operation* source) {
  if (auto var_handle_op = dyn_cast<TF::VarHandleOp>(source)) {
    // Resources are represented as tensor<resource<tensor<...>>>, so
    // unwrap until we get to the inner tensor<...>.
    auto tensor =
        llvm::dyn_cast<TensorType>(var_handle_op.getResource().getType());
    if (!tensor) return nullptr;
    TF::ResourceType resource =
        llvm::dyn_cast<TF::ResourceType>(tensor.getElementType());
    if (!resource || resource.getSubtypes().size() != 1) return nullptr;
    return resource.getSubtypes().front();
  } else if (auto global_tensor_op =
                 dyn_cast<tf_saved_model::GlobalTensorOp>(source)) {
    return global_tensor_op.getType();
  }
  // Likely can't actually happen, assuming tf_saved_model.semantics checks
  // already ran.
  return nullptr;
}

ml_program::GlobalOp CreateGlobalOpFromOp(Operation* source, OpBuilder& builder,
                                          SymbolTable& symbol_table) {
  Type type = GetGlobalType(source);
  std::string name = GetVariableName(source);
  if (auto existing = symbol_table.lookup<ml_program::GlobalOp>(name)) {
    // This might be of a different type, but we'll do a Cast later.
    return existing;
  }

  Attribute initial_value = GetInitialValue(source);
  if (!initial_value) {
    initial_value = builder.getZeroAttr(type);
  }

  if (!type) return nullptr;

  auto globalOp = builder.create<ml_program::GlobalOp>(
      builder.getBlock()->getParentOp()->getLoc(), name, type, false,
      initial_value, nullptr);
  symbol_table.insert(globalOp);

  return globalOp;
}

}  // namespace

#define GEN_PASS_DEF_LOWERVARIABLEOPSTOMLPROGRAMPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"

struct LowerVariableOpsToMlProgramPass
    : public impl::LowerVariableOpsToMlProgramPassBase<
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
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TF::ResourceDataflowAnalysis>();
    if (failed(solver.initializeAndRun(module))) return signalPassFailure();

    SymbolTable symbol_table(module);

    OpBuilder globalBuilder(module.getBodyRegion());

    module.walk([&](TF::ReadVariableOp op) {
      Operation* source = GetHandleSource(op, solver);
      if (!source) return;
      ml_program::GlobalOp globalOp =
          CreateGlobalOpFromOp(source, globalBuilder, symbol_table);
      if (!globalOp) return;
      OpBuilder builder(op);
      Operation* load = builder.create<mlir::ml_program::GlobalLoadOp>(
          op.getLoc(), globalOp.getType(),
          SymbolRefAttr::get(op->getContext(), globalOp.getSymName()));
      if (globalOp.getType() != op.getValue().getType()) {
        load = builder.create<TF::CastOp>(op.getLoc(), op.getValue().getType(),
                                          load->getResult(0));
      }
      op.getResult().replaceAllUsesWith(load->getResult(0));
      op.erase();
    });

    module.walk([&](TF::AssignVariableOp op) {
      Operation* source = GetHandleSource(op, solver);
      if (!source) return;
      ml_program::GlobalOp globalOp =
          CreateGlobalOpFromOp(source, globalBuilder, symbol_table);
      if (!globalOp) return;
      symbol_table.insert(globalOp);
      OpBuilder builder(op);
      globalOp.setIsMutableAttr(builder.getUnitAttr());
      Value value_to_store = op.getValue();
      if (globalOp.getType() != op.getValue().getType()) {
        value_to_store = builder.create<TF::CastOp>(
            op.getLoc(), globalOp.getType(), value_to_store);
      }
      builder.create<mlir::ml_program::GlobalStoreOp>(
          op.getLoc(),
          SymbolRefAttr::get(op->getContext(), globalOp.getSymName()),
          value_to_store);
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
