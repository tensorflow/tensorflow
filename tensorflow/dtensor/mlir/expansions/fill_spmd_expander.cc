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

#include "tensorflow/dtensor/mlir/expansions/fill_spmd_expander.h"

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/convert_op_folder.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> FillSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto original_fill = mlir::cast<mlir::TF::FillOp>(op);
  TF_ASSIGN_OR_RETURN(auto dims_layout,
                      ExtractLayoutFromOperand(original_fill.dims()));
  if (!dims_layout.has_value()) {
    return errors::InvalidArgument(
        "Failed during SPMD expansion of tf.FillOp. Layout of dimension "
        "input must be known.");
  }

  if (!dims_layout->IsFullyReplicated()) {
    return errors::InvalidArgument(
        "Expected the layout for fill's `dims` argument to be fully "
        "replicated.");
  }
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> output_layout,
                      ExtractSingleLayoutFromOp(op));
  if (!output_layout.has_value())
    return errors::Internal(
        "FillOp doesn't have a layout after layout propagation");
  if (output_layout->IsFullyReplicated()) {
    // For fully replicated layouts the local shape on each device is the same
    // as the global shape.
    return InferSPMDExpandedLocalShape(op);
  }

  // For sharded outputs, the `dims` just needs to be translated from the
  // global to the local shape.
  mlir::OpBuilder builder(op->getBlock(), ++mlir::Block::iterator(op));

  // Create a tensor from the sharding spec, with the dtype of the original
  // attribute.
  auto shard_values = output_layout->num_shards();
  auto int_type = mlir::RankedTensorType::get(
      static_cast<int64>(shard_values.size()), builder.getIntegerType(32));
  auto int_attr = mlir::DenseIntElementsAttr::get(int_type, shard_values);
  auto target_type_attr = mlir::hlo::convertElementsAttr(
      int_attr,
      original_fill.dims().getType().cast<mlir::TensorType>().getElementType());

  auto location = DT_LOC(op);
  auto num_shards =
      builder.create<mlir::TF::ConstOp>(location, target_type_attr);
  // Divide the global shape by the sharding spec.
  auto div = builder.create<mlir::TF::DivOp>(location, original_fill.dims(),
                                             num_shards.getResult());
  // Copy over static shape information if available
  auto global_output_type =
      original_fill.getResult().getType().cast<mlir::TensorType>();
  TF_ASSIGN_OR_RETURN(
      auto local_type,
      LocalTypeFromGlobalType(output_layout.value(), global_output_type));

  auto new_fill = builder.create<mlir::TF::FillOp>(
      location, local_type, div.getResult(), original_fill.value());
  original_fill.getResult().replaceAllUsesWith(new_fill.output());
  original_fill.erase();
  return InferSPMDExpandedLocalShape(new_fill);
}

StatusOr<llvm::DenseMap<int, Layout>> FillSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always set replicated layout for output.
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(
               mesh, ValueRank(llvm::cast<mlir::TF::FillOp>(op).output()))}});
}

StatusOr<llvm::DenseMap<int, Layout>> FillSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always set replicated layout for dims / value operand of Fill op.
  return llvm::DenseMap<int, Layout>({{0, Layout::ReplicatedOnMesh(mesh, 1)},
                                      {1, Layout::ReplicatedOnMesh(mesh, 0)}});
}

}  // namespace dtensor
}  // namespace tensorflow
