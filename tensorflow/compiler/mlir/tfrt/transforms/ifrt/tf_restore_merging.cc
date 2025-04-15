/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>

#include <memory>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_TFRESTOREMERGINGPASS
#define GEN_PASS_DECL_TFRESTOREMERGINGPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

class TfRestoreMergingPass
    : public impl::TfRestoreMergingPassBase<TfRestoreMergingPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    for (mlir::Block& block : func) {
      // Group `tf.RestoreV2` ops by prefixes and merge each group.
      llvm::SmallDenseMap<mlir::Value, std::vector<mlir::TF::RestoreV2Op>>
          restore_groups;
      for (auto restore : block.getOps<mlir::TF::RestoreV2Op>()) {
        restore_groups[restore.getPrefix()].push_back(restore);
      }
      for (const auto& restores : llvm::make_second_range(restore_groups)) {
        if (mlir::failed(MergeRestores(restores))) {
          return signalPassFailure();
        }
      }
    }
  }

 private:
  mlir::DenseStringElementsAttr GetStringTensorAttr(
      llvm::ArrayRef<llvm::StringRef> values) {
    const int size = values.size();
    const auto type = mlir::RankedTensorType::get(
        {size}, mlir::TF::StringType::get(&getContext()));
    return mlir::DenseStringElementsAttr::get(type, values);
  }

  // Merges `tf.RestoreV2` ops with the same prefix. Ignores restore ops with
  // non-constant `tensor_names` and/or `shape_and_slices`.
  mlir::LogicalResult MergeRestores(
      llvm::ArrayRef<mlir::TF::RestoreV2Op> restores) {
    if (restores.size() <= 1) {
      return mlir::success();
    }

    // All restore ops must have the same prefix.
    const mlir::Value prefix =
        mlir::TF::RestoreV2Op(restores.front()).getPrefix();

    std::vector<mlir::TF::RestoreV2Op> restores_to_merge;
    std::vector<mlir::Value> values_to_replace;
    std::vector<llvm::StringRef> merged_tensor_names;
    std::vector<llvm::StringRef> merged_shape_and_slices;

    std::vector<mlir::Location> restore_locs;
    std::vector<mlir::Location> tensor_names_locs;
    std::vector<mlir::Location> shape_and_slices_locs;

    for (mlir::TF::RestoreV2Op restore : restores) {
      mlir::DenseStringElementsAttr tensor_names;
      mlir::DenseStringElementsAttr shape_and_slices;
      if (!mlir::matchPattern(restore,
                              mlir::m_Op<mlir::TF::RestoreV2Op>(
                                  mlir::matchers::m_Val(prefix),
                                  mlir::m_Constant(&tensor_names),
                                  mlir::m_Constant(&shape_and_slices)))) {
        continue;
      }
      if (tensor_names.size() != restore.getNumResults() ||
          shape_and_slices.size() != restore.getNumResults()) {
        return restore.emitOpError()
               << "returns an inconsistent number of results";
      }

      restores_to_merge.push_back(restore);
      llvm::append_range(values_to_replace, restore.getTensors());
      llvm::append_range(merged_tensor_names,
                         tensor_names.getValues<llvm::StringRef>());
      llvm::append_range(merged_shape_and_slices,
                         shape_and_slices.getValues<llvm::StringRef>());

      restore_locs.push_back(restore.getLoc());
      tensor_names_locs.push_back(restore.getTensorNames().getLoc());
      shape_and_slices_locs.push_back(restore.getShapeAndSlices().getLoc());
    }
    if (restores_to_merge.size() <= 1) {
      return mlir::success();
    }

    // Insert the merged restore op right before the first restore op to be
    // merged in order to keep the dominance property.
    mlir::OpBuilder builder(restores_to_merge.front());

    auto new_tensor_names = builder.create<mlir::TF::ConstOp>(
        builder.getFusedLoc(tensor_names_locs),
        GetStringTensorAttr(merged_tensor_names));
    auto new_shape_and_slices = builder.create<mlir::TF::ConstOp>(
        builder.getFusedLoc(shape_and_slices_locs),
        GetStringTensorAttr(merged_shape_and_slices));

    auto new_restore = builder.create<mlir::TF::RestoreV2Op>(
        builder.getFusedLoc(restore_locs),
        mlir::TypeRange(mlir::ValueRange(values_to_replace)), prefix,
        new_tensor_names, new_shape_and_slices);
    for (auto [old_value, new_value] :
         llvm::zip(values_to_replace, new_restore.getTensors())) {
      old_value.replaceAllUsesWith(new_value);
    }

    for (mlir::TF::RestoreV2Op restore : restores_to_merge) {
      restore.erase();
    }
    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfRestoreMergingPass() {
  return std::make_unique<TfRestoreMergingPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
