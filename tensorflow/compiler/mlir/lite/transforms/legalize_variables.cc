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

#include "llvm/ADT/None.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {
namespace {
// This file has Legalize variable pass which is responsible for:
// - Converting all TF::ReadVariableOp and TF::AssignVariableOp to the
//   TFLite equivalent ops.
// Note that, this pass assumes all variables are already available as
// GlobalTensorOp and all varHandle are converted already to a function
// arguments with bounded_input attribute.
// Also all other ops are already legalized to TFLite.
// TODO(b/149099381): Handle flex support use cases.

// Returns Value representing the resource_id.
Value GetResourceIDAsI32(int resource_id, Location loc,
                         mlir::OpBuilder& rewriter) {
  return rewriter.create<ConstOp>(
      loc, DenseElementsAttr::get(
               RankedTensorType::get({1}, rewriter.getIntegerType(32)),
               resource_id));
}

// Helper method that fetches the global tensor that 'op' points to it.
template <typename T>
tf_saved_model::GlobalTensorOp GetGlobalTensor(const SymbolTable& symbol_table,
                                               T op, FuncOp func) {
  auto block_arg = op.resource().template dyn_cast<BlockArgument>();
  if (!block_arg) return nullptr;
  int index = block_arg.getArgNumber();
  auto sym = func.template getArgAttrOfType<FlatSymbolRefAttr>(
      index, "tf_saved_model.bound_input");
  if (!sym) {
    return nullptr;
  }
  return symbol_table.lookup<tf_saved_model::GlobalTensorOp>(sym.getValue());
}

mlir::Operation* GetAssignVariableOp(int variable_id,
                                     TF::AssignVariableOp assign_op,
                                     mlir::OpBuilder builder) {
  return builder.create<TFL::AssignVariableOp>(
      assign_op.getLoc(),
      GetResourceIDAsI32(variable_id, assign_op.getLoc(), builder),
      assign_op.value());
}

mlir::Operation* GetReadVariableOp(int variable_id, TF::ReadVariableOp read_op,
                                   mlir::OpBuilder builder) {
  return builder.create<TFL::ReadVariableOp>(
      read_op.getLoc(), read_op.getResult().getType(),
      GetResourceIDAsI32(variable_id, read_op.getLoc(), builder));
}

template <typename T>
class LegalizeVariablesPattern : public mlir::OpConversionPattern<T> {
 public:
  LegalizeVariablesPattern(mlir::MLIRContext* context,
                           const std::map<std::string, int>* global_tensor_id,
                           SymbolTable symbol_table)
      : mlir::OpConversionPattern<T>(context),
        global_tensor_id_(global_tensor_id),
        symbol_table_(symbol_table) {}

  LogicalResult matchAndRewrite(
      T var_op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    auto* op = var_op.getOperation();
    auto func = var_op->template getParentOfType<FuncOp>();
    if (!func) return failure();
    auto global_tensor = GetGlobalTensor<T>(symbol_table_, var_op, func);
    if (!global_tensor) return failure();
    auto variable_id = global_tensor_id_->at(global_tensor.sym_name().str());
    mlir::OpBuilder builder(var_op);
    mlir::Operation* tfl_var_op = nullptr;
    if (llvm::isa<TF::AssignVariableOp>(op)) {
      auto assign_op = llvm::cast<TF::AssignVariableOp>(op);
      tfl_var_op = GetAssignVariableOp(variable_id, assign_op, builder);
    } else {
      auto read_op = llvm::cast<TF::ReadVariableOp>(op);
      tfl_var_op = GetReadVariableOp(variable_id, read_op, builder);
    }
    var_op->replaceAllUsesWith(tfl_var_op);
    rewriter.eraseOp(var_op);
    return success();
  }

 private:
  const std::map<std::string, int>* global_tensor_id_;
  SymbolTable symbol_table_;
};

// Pass which legalizes TF variables which are already passed as bounded
// arguments to functions, to TFLite variables.
class LegalizeVariables
    : public PassWrapper<LegalizeVariables, OperationPass<ModuleOp>> {
 public:
  LegalizeVariables() = default;
  LegalizeVariables(const LegalizeVariables&) {}

  void runOnOperation() override {
    auto module = getOperation();
    // Use ordered container to make sure ids are deterministic if we got tensor
    // ids from different part, also easier to debug.
    // TODO(b/149099381): Remove integer IDs after adding the new variable
    // handle type.
    std::map<std::string, int> global_tensor_id;
    for (auto global_tensor : module.getOps<tf_saved_model::GlobalTensorOp>()) {
      global_tensor_id[global_tensor.sym_name().str()];
    }
    int id = 0;
    for (auto& tensor : global_tensor_id) tensor.second = id++;

    SymbolTable symbol_table(module);
    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<LegalizeVariablesPattern<TF::ReadVariableOp>,
                    LegalizeVariablesPattern<TF::AssignVariableOp>>(
        &getContext(), &global_tensor_id, symbol_table);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeVariablesPass() {
  return std::make_unique<LegalizeVariables>();
}

static PassRegistration<LegalizeVariables> pass(
    "tfl-legalize-variables-tf",
    "Legalize TensorFlow variables to TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
