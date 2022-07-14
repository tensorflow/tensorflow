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

#include <algorithm>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {

namespace {

// MLIR pass that undoes unintended const merging across different meshes within
// the same Block by canonicalization passes.
struct DTensorUndoMergeConstAcrossMesh
    : public DTensorUndoMergeConstAcrossMeshBase<
          DTensorUndoMergeConstAcrossMesh> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);
    getOperation().walk([&builder](mlir::TF::ConstOp const_op) {
      llvm::SmallVector<Mesh> known_meshes;
      llvm::SmallVector<mlir::TF::DTensorLayout> unique_layout_ops;
      for (mlir::Operation* consumer : const_op->getUsers()) {
        mlir::TF::DTensorLayout layout_op =
            mlir::dyn_cast<mlir::TF::DTensorLayout>(consumer);
        if (!layout_op) continue;

        const Layout layout = layout_op.layout();  // keep-alive for mesh.
        const Mesh& mesh = layout.mesh();
        if (std::find(known_meshes.begin(), known_meshes.end(), mesh) ==
            known_meshes.end()) {
          if (!known_meshes.empty()) {
            // We skip the first layout_op to preserve its original ConstOp.
            unique_layout_ops.push_back(layout_op);
          }
          known_meshes.emplace_back(mesh);
        }
      }
      for (auto& layout_op : unique_layout_ops) {
        builder.setInsertionPoint(layout_op);
        layout_op->replaceUsesOfWith(const_op,
                                     builder.cloneWithoutRegions(const_op));
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorUndoMergeConstAcrossMesh() {
  return std::make_unique<DTensorUndoMergeConstAcrossMesh>();
}
}  // namespace dtensor
}  // namespace tensorflow
