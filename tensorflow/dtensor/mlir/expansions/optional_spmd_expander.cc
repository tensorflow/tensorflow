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

#include "tensorflow/dtensor/mlir/expansions/optional_spmd_expander.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> OptionalGetValueSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  auto original_op = mlir::cast<mlir::TF::OptionalGetValueOp>(op);
  mlir::OpBuilder builder(op);

  TF_ASSIGN_OR_RETURN(std::vector<Layout> output_layouts,
                      ExtractRequiredLayoutFromOp(op));

  llvm::SmallVector<mlir::Type, 4> local_types(original_op->getNumResults());

  for (int i = 0; i < original_op->getNumResults(); ++i) {
    mlir::TensorType global_output_type =
        mlir::cast<mlir::TensorType>(original_op.getResult(i).getType());
    TF_ASSIGN_OR_RETURN(
        mlir::TensorType local_type,
        LocalTypeFromGlobalType(output_layouts[i], global_output_type));
    local_types[i] = local_type;
  }

  auto new_op = builder.create<mlir::TF::OptionalGetValueOp>(
      DT_LOC(op->getLoc()), local_types, original_op->getOperand(0));

  for (int i = 0; i < original_op->getNumResults(); ++i) {
    original_op.getResult(i).replaceAllUsesWith(new_op.getResult(i));
  }
  original_op.erase();
  return InferSPMDExpandedLocalShape(new_op);
}

StatusOr<llvm::DenseMap<int, Layout>>
OptionalGetValueSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // Extract the output element layouts from some op in the input chain that has
  // the `tf._element_layouts` attribute set.
  TF_ASSIGN_OR_RETURN(const auto layouts,
                      ExtractElementLayoutsFromOperand(op->getOpOperand(0)));

  llvm::DenseMap<int, Layout> output_layouts(op->getNumResults());
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_layouts[i] = layouts[i];
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>>
OptionalGetValueSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(0)))}});
}

StatusOr<mlir::Operation*> OptionalHasValueSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>>
OptionalHasValueSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)}});
}

StatusOr<llvm::DenseMap<int, Layout>>
OptionalHasValueSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(0)))}});
}

}  // namespace dtensor
}  // namespace tensorflow
