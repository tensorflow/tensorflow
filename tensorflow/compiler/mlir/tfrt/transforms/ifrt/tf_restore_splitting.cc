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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_TFRESTORESPLITTINGPASS
#define GEN_PASS_DECL_TFRESTORESPLITTINGPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

class TfRestoreSplittingPass
    : public impl::TfRestoreSplittingPassBase<TfRestoreSplittingPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    const mlir::WalkResult result =
        func.walk([&](mlir::TF::RestoreV2Op restore) {
          if (mlir::failed(SplitRestore(restore))) {
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      return signalPassFailure();
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

  // Splits the `tf.RestoreV2` op into per-variable restore ops if its
  // `tensor_name` and `shape_and_slices` are constant.
  mlir::LogicalResult SplitRestore(mlir::TF::RestoreV2Op restore) {
    mlir::DenseStringElementsAttr tensor_names;
    mlir::DenseStringElementsAttr shape_and_slices;
    if (!mlir::matchPattern(restore,
                            mlir::m_Op<mlir::TF::RestoreV2Op>(
                                /*prefix=*/mlir::matchers::m_Any(),
                                mlir::m_Constant(&tensor_names),
                                mlir::m_Constant(&shape_and_slices)))) {
      return mlir::success();
    }
    if (tensor_names.size() != restore.getNumResults() ||
        shape_and_slices.size() != restore.getNumResults()) {
      return restore.emitOpError()
             << "returns an inconsistent number of results";
    }

    mlir::OpBuilder builder(restore);
    for (auto [tensor_name, shape_and_slice, result] :
         llvm::zip(tensor_names.getValues<llvm::StringRef>(),
                   shape_and_slices.getValues<llvm::StringRef>(),
                   restore.getTensors())) {
      auto new_tensor_names =
          builder.create<mlir::TF::ConstOp>(restore.getTensorNames().getLoc(),
                                            GetStringTensorAttr({tensor_name}));

      auto new_shape_and_slices = builder.create<mlir::TF::ConstOp>(
          restore.getShapeAndSlices().getLoc(),
          GetStringTensorAttr({shape_and_slice}));

      auto new_restore = builder.create<mlir::TF::RestoreV2Op>(
          restore.getLoc(), mlir::TypeRange({result.getType()}),
          restore.getPrefix(), new_tensor_names, new_shape_and_slices);
      result.replaceAllUsesWith(new_restore.getTensors()[0]);
    }

    restore.erase();
    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfRestoreSplittingPass() {
  return std::make_unique<TfRestoreSplittingPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
