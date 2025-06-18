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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tsl/platform/path.h"

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
    ModuleOp module = getOperation();
    if (!HasTfSavedModelSemantics(module)) {
      return;
    }

    auto init_op = GetSessionInitializerOp(module);
    if (init_op == nullptr || init_op.getInitializers().empty()) {
      return;
    }

    SymbolTable symbol_table(module);
    for (auto initializer : init_op.getInitializers()) {
      auto func = symbol_table.lookup<func::FuncOp>(
          mlir::cast<FlatSymbolRefAttr>(initializer).getValue());
      RewriteFunction(symbol_table, func);
    }

    // Clean up unused asset ops.
    for (auto asset : llvm::make_early_inc_range(module.getOps<AssetOp>())) {
      if (symbol_table.symbolKnownUseEmpty(asset, module)) {
        asset.erase();
      }
    }
  }

 private:
  // Replaces bounded-input arguments of the function with constant ops in the
  // body and removes the arguments.
  void RewriteFunction(const SymbolTable& symbol_table, func::FuncOp func) {
    if (func.getNumArguments() == 0) {
      return;
    }

    auto builder = OpBuilder::atBlockBegin(&func.front());

    llvm::SmallDenseMap<llvm::StringRef, TF::ConstOp> const_ops;
    llvm::BitVector arg_indexes_to_remove(func.getNumArguments());

    // Replace arguments with const ops.
    for (BlockArgument argument : func.getArguments()) {
      auto asset = LookupBoundInputOfType<AssetOp>(
          func, argument.getArgNumber(), symbol_table);
      if (asset == nullptr) {
        continue;
      }

      // Create a const op for the asset if it doesn't already exist.
      auto it = const_ops.find(asset.getSymName());
      if (it == const_ops.end()) {
        // Asset filenames are relative to the SavedModel directory.
        const std::string filename = tsl::io::JoinPath(
            saved_model_dir_, absl::string_view(asset.getFilename()));

        RankedTensorType type = RankedTensorType::get(
            {}, TF::StringType::get(builder.getContext()));
        auto const_op = builder.create<TF::ConstOp>(
            builder.getUnknownLoc(),
            DenseStringElementsAttr::get(type, {filename}));

        it = const_ops.insert({asset.getSymName(), const_op}).first;
      }

      argument.replaceAllUsesWith(it->second.getOutput());
      arg_indexes_to_remove.set(argument.getArgNumber());
    }

    // Erase function arguments with bounded input.
    CHECK(llvm::succeeded(func.eraseArguments(arg_indexes_to_remove)));
  }
};

}  // namespace

absl::Status AddSessionInitializerAndInlineCheckpoint(
    ModuleOp module, absl::string_view checkpoint_path) {
  // The main function should be the only public function.
  func::FuncOp main_func = nullptr;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isPublic()) {
      if (main_func != nullptr) {
        return absl::InternalError("Only one public function is allowed.");
      }
      main_func = func;
    }
  }
  if (main_func.getNumArguments() != 1) {
    return absl::InternalError("Expected 1 argument for the main function.");
  }
  OpBuilder builder(module.getBodyRegion());
  // Create SessionInitializerOp; should reference main function.
  StringAttr func_name = main_func.getSymNameAttr();
  llvm::SmallVector<mlir::Attribute, 2> func_names = {
      mlir::SymbolRefAttr::get(builder.getContext(), func_name)};
  builder.create<tf_saved_model::SessionInitializerOp>(
      module->getLoc(), builder.getArrayAttr(func_names));
  // Create AssetOp; this holds the checkpoint_path.
  // TODO(b/318761632): Cleanup usage of string literals, instead use constants.
  auto asset_op = builder.create<tf_saved_model::AssetOp>(
      module->getLoc(),
      /*sym_name=*/
      builder.getStringAttr("__tf_saved_model_variables"),  // Val unimportant.
      /*filename=*/
      builder.getStringAttr(checkpoint_path));
  // Marks the input to be inlined.
  main_func.setArgAttr(
      0, "tf_saved_model.bound_input",
      SymbolRefAttr::get(builder.getContext(), asset_op.getName()));
  // Bound arguments are expected to be of type tensor<!tf_type.string>, not
  // tensor<x!tf_type.string>.
  auto tensor_string_type =
      RankedTensorType::get({}, TF::StringType::get(builder.getContext()));
  main_func.getArguments().front().setType(tensor_string_type);
  main_func.setType(
      FunctionType::get(builder.getContext(), {tensor_string_type},
                        main_func.getFunctionType().getResults()));
  // Name of the main function for the eventual executable needs to be set.
  main_func->setAttr(kTfSavedModelExportedNamesAttr,
                     builder.getStrArrayAttr({func_name}));
  main_func->setAttr(
      kTfSavedModelInitializerTypeAttr,
      builder.getStringAttr(kTfSavedModelInitializerRestoreType));
  module->setAttr("tf_saved_model.semantics",
                  UnitAttr::get(builder.getContext()));
  return absl::OkStatus();
}

std::unique_ptr<OperationPass<ModuleOp>> CreateAssetSinkingPass(
    llvm::StringRef saved_model_dir) {
  return std::make_unique<AssetSinkingPass>(saved_model_dir);
}

}  // namespace tf_saved_model
}  // namespace mlir
