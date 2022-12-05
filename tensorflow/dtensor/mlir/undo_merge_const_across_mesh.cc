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
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORUNDOMERGECONSTACROSSMESH
#define GEN_PASS_DEF_DTENSORELIDEIDENTITYBEFORECOPYTOMESH
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// MLIR pass that undoes unintended const merging across different meshes within
// the same Block by canonicalization passes.
struct DTensorUndoMergeConstAcrossMesh
    : public impl::DTensorUndoMergeConstAcrossMeshBase<
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

        const Layout layout = layout_op.getLayout();  // keep-alive for mesh.
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

struct DTensorElideIdentityBeforeCopyToMesh
    : public impl::DTensorElideIdentityBeforeCopyToMeshBase<
          DTensorElideIdentityBeforeCopyToMesh> {
  void runOnOperation() override {
    getOperation().walk([](mlir::TF::CopyToMeshGradOp op) {
      mlir::Value input_value = op->getOperand(0);
      mlir::Operation* defining_op = input_value.getDefiningOp();
      if (!mlir::isa<mlir::TF::IdentityOp>(defining_op)) {
        return;
      }
      op->replaceUsesOfWith(input_value, defining_op->getOperand(0));
      if (!defining_op->use_empty()) {
        return;
      }
      defining_op->erase();
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorUndoMergeConstAcrossMesh() {
  return std::make_unique<DTensorUndoMergeConstAcrossMesh>();
}
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorElideIdentityBeforeCopyToMesh() {
  return std::make_unique<DTensorElideIdentityBeforeCopyToMesh>();
}
}  // namespace dtensor
}  // namespace tensorflow
