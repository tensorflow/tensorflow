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

#include <memory>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/core/platform/path.h"

namespace mlir {
namespace tf_saved_model {
namespace {

#define GEN_PASS_DEF_FREEZEASSETSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"

// This pass will replace a func's saved model asset bound inputs which are
// bound to tf.InitializeTableFromTextFileV2Op ops with tf.Const ops inside the
// func's body.
struct FreezeAssetsPass : public impl::FreezeAssetsPassBase<FreezeAssetsPass> {
  FreezeAssetsPass() = default;

  FreezeAssetsPass(const FreezeAssetsPass& pass) {}
  explicit FreezeAssetsPass(std::string saved_model_dir) {
    this->saved_model_dir = saved_model_dir;
  }
  void runOnOperation() override;

 private:
  std::string saved_model_dir;
};

void FreezeAssetsPass::runOnOperation() {
  auto module = getOperation();
  if (!tf_saved_model::HasTfSavedModelSemantics(module)) {
    return;
  }
  SymbolTable symbol_table(module);

  for (auto func : module.getOps<func::FuncOp>()) {
    llvm::BitVector args_to_erase(func.getNumArguments());
    OpBuilder builder(func.getBody());

    for (int i = 0, e = func.getNumArguments(); i < e; ++i) {
      SmallVector<TF::InitializeTableFromTextFileV2Op, 4>
          init_table_from_text_file_ops_to_erase;
      auto asset = LookupBoundInputOfType<AssetOp>(func, i, symbol_table);

      if (!asset) continue;

      auto arg = func.getArgument(i);
      bool arg_is_deletable = true;
      for (auto user : arg.getUsers()) {
        if (auto read_op =
                llvm::dyn_cast<TF::InitializeTableFromTextFileV2Op>(user)) {
          init_table_from_text_file_ops_to_erase.push_back(read_op);
        } else {
          arg_is_deletable = false;
          continue;
        }
      }
      if (arg_is_deletable) {
        args_to_erase.set(i);
      }

      // Replace the arg with a tf.Const op in the function body.
      builder.setInsertionPointToStart(&func.getBody().front());

      std::string asset_filename = asset.getFilename().str();
      std::string filename =
          tensorflow::io::JoinPath(saved_model_dir, asset_filename);
      ShapedType shaped_type =
          RankedTensorType::get({1}, TF::StringType::get(builder.getContext()));
      auto const_op = builder.create<TF::ConstOp>(
          asset.getLoc(),
          DenseStringElementsAttr::get(shaped_type, {filename}));
      for (auto init_op : init_table_from_text_file_ops_to_erase) {
        // Replace the InitializeTableFromTextFileV2Op to use the saved model's
        // asset filepath.
        builder.setInsertionPoint(init_op);
        builder.create<TF::InitializeTableFromTextFileV2Op>(
            init_op.getLoc(), init_op.getTableHandle(), const_op.getResult(),
            init_op.getKeyIndex(), init_op.getValueIndex(),
            init_op.getVocabSize(), init_op.getDelimiter());
        init_op.erase();
      }
    }
    func.eraseArguments(args_to_erase);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateFreezeAssetsPass(
    std::string saved_model_dir) {
  return std::make_unique<FreezeAssetsPass>(saved_model_dir);
}

}  // namespace tf_saved_model
}  // namespace mlir
