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

#include "tensorflow/dtensor/mlir/expansions/sparse_to_dense_spmd_expander.h"

#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> SparseToDenseSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  // Set the op's shape as the local shape of the input tensors from the
  // layouts.
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> computed_layout,
                      ExtractSingleLayoutFromOp(op));
  auto local_shape = computed_layout->LocalShapeFromGlobalShape(
      ExtractGlobalOutputShape(op->getResult(0)).ValueOrDie());
  auto op_result = op->getResult(0);

  const auto element_type =
      op_result.getType().cast<mlir::TensorType>().getElementType();
  op_result.setType(mlir::RankedTensorType::get(local_shape, element_type));
  // No-op
  return op;
}

StatusOr<llvm::DenseMap<int, Layout>>
SparseToDenseSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  return llvm::DenseMap<int, Layout>();
}

StatusOr<llvm::DenseMap<int, Layout>>
SparseToDenseSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If there is no output layout present then do not infer any operand layouts.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  Layout output_layout = output_layouts.lookup(0);
  if (output_layout.mesh().is_tpu_mesh()) {
    return errors::InvalidArgument(
        "Layout for SparseToDenseOp must not be on TPU Mesh.");
  }
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

}  // namespace dtensor
}  // namespace tensorflow
