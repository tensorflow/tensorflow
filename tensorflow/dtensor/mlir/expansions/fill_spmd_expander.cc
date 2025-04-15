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

#include <optional>

#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/mlir_hlo/utils/convert_op_folder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> FillSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto original_fill = mlir::cast<mlir::TF::FillOp>(op);
  TF_ASSIGN_OR_RETURN(auto dims_layout,
                      ExtractLayoutFromOperand(original_fill.getDims()));
  if (!dims_layout.has_value()) {
    return errors::InvalidArgument(
        "Failed during SPMD expansion of tf.FillOp. Layout of dimension "
        "input must be known.");
  }

  if (!dims_layout->IsFullyReplicated()) {
    return errors::InvalidArgument(
        "Expected the layout for fill's `dims` argument to be fully "
        "replicated. Got ",
        dims_layout->ToString());
  }
  TF_ASSIGN_OR_RETURN(std::optional<Layout> output_layout,
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
      int_attr, mlir::cast<mlir::TensorType>(original_fill.getDims().getType())
                    .getElementType());

  auto location = DT_LOC(op);
  auto num_shards =
      builder.create<mlir::TF::ConstOp>(location, target_type_attr);
  // Divide the global shape by the sharding spec.
  auto div = builder.create<mlir::TF::DivOp>(location, original_fill.getDims(),
                                             num_shards.getResult());
  // Copy over static shape information if available
  auto global_output_type =
      mlir::cast<mlir::TensorType>(original_fill.getResult().getType());
  TF_ASSIGN_OR_RETURN(
      auto local_type,
      LocalTypeFromGlobalType(output_layout.value(), global_output_type));

  auto new_fill = builder.create<mlir::TF::FillOp>(
      location, local_type, div.getResult(), original_fill.getValue());
  original_fill.getResult().replaceAllUsesWith(new_fill.getOutput());
  original_fill.erase();
  return InferSPMDExpandedLocalShape(new_fill);
}

StatusOr<llvm::DenseMap<int, Layout>> FillSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always set replicated layout for output.
  return llvm::DenseMap<int, Layout>(
      {{0,
        Layout::ReplicatedOnMesh(
            mesh, ValueRank(llvm::cast<mlir::TF::FillOp>(op).getOutput()))}});
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
