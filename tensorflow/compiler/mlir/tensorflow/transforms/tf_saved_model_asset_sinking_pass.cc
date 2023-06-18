/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_asset_sinking_pass.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/tsl/platform/path.h"

namespace mlir {
namespace tf_saved_model {
namespace {

#define GEN_PASS_DEF_ASSETSINKINGPASS
#define GEN_PASS_DECL_ASSETSINKINGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"

class AssetSinkingPass : public impl::AssetSinkingPassBase<AssetSinkingPass> {
 public:
  AssetSinkingPass() = default;

  explicit AssetSinkingPass(llvm::StringRef saved_model_dir) {
    saved_model_dir_ = saved_model_dir.str();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    if (!mlir::tf_saved_model::HasTfSavedModelSemantics(module)) {
      return;
    }

    auto init_op = mlir::tf_saved_model::GetSessionInitializerOp(module);
    if (init_op == nullptr || init_op.getInitializers().empty()) {
      return;
    }

    mlir::SymbolTable symbol_table(module);
    for (auto initializer : init_op.getInitializers()) {
      auto func = symbol_table.lookup<mlir::func::FuncOp>(
          initializer.cast<mlir::FlatSymbolRefAttr>().getValue());
      RewriteFunction(symbol_table, func);
    }

    // Clean up unused asset ops.
    for (auto asset : llvm::make_early_inc_range(
             module.getOps<mlir::tf_saved_model::AssetOp>())) {
      if (symbol_table.symbolKnownUseEmpty(asset, module)) {
        asset.erase();
      }
    }
  }

 private:
  // Replaces bounded-input arguments of the function with constant ops in the
  // body and removes the arguments.
  void RewriteFunction(const mlir::SymbolTable& symbol_table,
                       mlir::func::FuncOp func) {
    if (func.getNumArguments() == 0) {
      return;
    }

    auto builder = mlir::OpBuilder::atBlockBegin(&func.front());

    llvm::SmallDenseMap<llvm::StringRef, mlir::TF::ConstOp> const_ops;
    llvm::BitVector arg_indexes_to_remove(func.getNumArguments());

    // Replace arguments with const ops.
    for (mlir::BlockArgument argument : func.getArguments()) {
      auto asset = mlir::tf_saved_model::LookupBoundInputOfType<
          mlir::tf_saved_model::AssetOp>(func, argument.getArgNumber(),
                                         symbol_table);
      if (asset == nullptr) {
        continue;
      }

      // Create a const op for the asset if it doesn't already exist.
      auto it = const_ops.find(asset.getSymName());
      if (it == const_ops.end()) {
        // Asset filenames are relative to the SavedModel directory.
        const std::string filename = tsl::io::JoinPath(
            saved_model_dir_, absl::string_view(asset.getFilename()));

        mlir::RankedTensorType type = mlir::RankedTensorType::get(
            {}, mlir::TF::StringType::get(builder.getContext()));
        auto const_op = builder.create<mlir::TF::ConstOp>(
            builder.getUnknownLoc(),
            mlir::DenseStringElementsAttr::get(type, {filename}));

        it = const_ops.insert({asset.getSymName(), const_op}).first;
      }

      argument.replaceAllUsesWith(it->second.getOutput());
      arg_indexes_to_remove.set(argument.getArgNumber());
    }

    // Erase function arguments with bounded input.
    func.eraseArguments(arg_indexes_to_remove);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateAssetSinkingPass(
    llvm::StringRef saved_model_dir) {
  return std::make_unique<AssetSinkingPass>(saved_model_dir);
}

}  // namespace tf_saved_model
}  // namespace mlir
