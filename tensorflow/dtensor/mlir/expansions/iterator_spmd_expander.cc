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

#include <cstdint>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
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
        mlir::cast<mlir::TensorType>(original_op.getResult(i).getType());
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

StatusOr<mlir::Operation*> IteratorGetNextAsOptionalSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  // Extract the output element layouts from the `tf._element_layouts` attribute
  // of the iterator resource tensor.
  TF_ASSIGN_OR_RETURN(const auto output_layouts,
                      ExtractElementLayoutsFromOperand(op->getOpOperand(0)));

  auto array_attr = op->getAttrOfType<mlir::ArrayAttr>(kIteratorOutputShapes);
  if (!array_attr)
    return errors::InvalidArgument(
        llvm::formatv("Could not find `{0}` attribute of op: {1}",
                      kIteratorOutputShapes, op->getName())
            .str());

  llvm::SmallVector<mlir::Attribute, 4> output_shape_attrs(array_attr.size());
  for (int i = 0; i < array_attr.size(); ++i) {
    std::vector<int64_t> local_shape =
        output_layouts[i].LocalShapeFromGlobalShape(
            mlir::cast<mlir::TF::ShapeAttr>(array_attr[i]).getShape());
    output_shape_attrs[i] = mlir::cast<mlir::Attribute>(
        mlir::TF::ShapeAttr::get(op->getContext(), {local_shape}));
  }

  // Update the `output_shapes` attribute on the op to match the local shape
  // based on the iterator element layouts.
  op->setAttr(kIteratorOutputShapes,
              mlir::ArrayAttr::get(op->getContext(), output_shape_attrs));
  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>>
IteratorGetNextAsOptionalSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Variant tensors are always 0-dimensional.
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)}});
}

StatusOr<llvm::DenseMap<int, Layout>>
IteratorGetNextAsOptionalSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Iterator resource tensors are always 0-dimensional.
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)}});
}

}  // namespace dtensor
}  // namespace tensorflow
