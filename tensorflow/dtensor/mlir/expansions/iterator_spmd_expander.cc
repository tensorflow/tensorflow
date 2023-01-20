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

#include "tensorflow/dtensor/mlir/expansions/iterator_spmd_expander.h"

#include <algorithm>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> IteratorGetNextSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  mlir::TF::IteratorGetNextOp original_op =
      mlir::cast<mlir::TF::IteratorGetNextOp>(op);
  mlir::OpBuilder builder(op);

  TF_ASSIGN_OR_RETURN(std::vector<Layout> output_layouts,
                      ExtractRequiredLayoutFromOp(op));

  llvm::SmallVector<mlir::Type, 4> local_types(original_op->getNumResults());

  for (int i = 0; i < original_op->getNumResults(); ++i) {
    mlir::TensorType global_output_type =
        original_op.getResult(i).getType().cast<mlir::TensorType>();
    std::vector<int64_t> local_shape =
        output_layouts[i].LocalShapeFromGlobalShape(
            global_output_type.getShape());
    local_types[i] = mlir::RankedTensorType::get(
        local_shape, global_output_type.getElementType());
  }

  auto new_op = builder.create<mlir::TF::IteratorGetNextOp>(
      DT_LOC(op->getLoc()), local_types, original_op->getOperand(0));

  for (int i = 0; i < original_op->getNumResults(); ++i) {
    original_op.getResult(i).replaceAllUsesWith(new_op.getResult(i));
  }
  original_op.erase();
  return InferSPMDExpandedLocalShape(new_op);
}

StatusOr<llvm::DenseMap<int, Layout>>
IteratorGetNextSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // Extract the output element layouts from the `tf._element_layouts` attribute
  // of the iterator resource tensor.
  TF_ASSIGN_OR_RETURN(const auto layouts,
                      ExtractElementLayoutsFromOperand(op->getOpOperand(0)));

  llvm::DenseMap<int, Layout> output_layouts(op->getNumResults());
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_layouts[i] = layouts[i];
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>>
IteratorGetNextSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Iterator resource tensors are always 0-dimensional.
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)}});
}

}  // namespace dtensor
}  // namespace tensorflow
