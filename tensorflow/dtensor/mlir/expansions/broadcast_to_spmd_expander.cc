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

#include "tensorflow/dtensor/mlir/expansions/broadcast_to_spmd_expander.h"

#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> BroadcastToSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  auto broadcast_op = llvm::cast<mlir::TF::BroadcastToOp>(op);
  TF_ASSIGN_OR_RETURN(
      const Layout shape_layout,
      ExtractRequiredLayoutFromOperand(broadcast_op.getShape()));
  if (!shape_layout.IsFullyReplicated()) {
    return errors::InvalidArgument(
        "Error during BroadcastOp SPMD Expansion. Shape input of broadcast op "
        "must be fully replicated.");
  }

  TF_ASSIGN_OR_RETURN(
      const Layout input_layout,
      ExtractRequiredLayoutFromOperand(broadcast_op.getInput()));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(broadcast_op));

  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> input_global_size,
      GetGlobalShapeOfValueFromDTensorLayout(broadcast_op.getInput()));

  llvm::SmallVector<int64_t, 4> broadcast_to_shape;
  TF_RETURN_IF_ERROR(ExtractConstVectorFromValue(
      GetForwardedDTensorLayoutInput(broadcast_op.getShape()),
      &broadcast_to_shape));

  // Input to BroadcastTo op requires all to all if non-broadcasted-dimensions
  // are not same.
  const int broadcasted_dimensions = output_layout.rank() - input_layout.rank();
  bool requires_all_to_all = false;
  const auto output_num_shards = output_layout.num_shards();
  for (int i = 0; i < input_layout.rank(); ++i) {
    const int output_dim_index = i + broadcasted_dimensions;
    const std::string& output_layout_dim =
        output_layout.sharding_spec(output_dim_index);
    if (input_global_size[i] > 1 &&
        input_layout.sharding_spec(i) != output_layout_dim) {
      requires_all_to_all = true;
    }
    if (output_layout_dim != Layout::kUnshardedDim) {
      broadcast_to_shape[output_dim_index] /=
          output_num_shards[output_dim_index];
    }
  }

  // Insert all-to-all operations just before Broadcast op to ensure all inputs
  // in correct local values.
  mlir::OpBuilder builder(op);
  mlir::Value input_data = broadcast_op.getInput();
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  const Layout all_to_all_input_layout =
      Layout::ReplicatedOnMesh(mesh, input_layout.rank());

  if (requires_all_to_all) {
    TF_ASSIGN_OR_RETURN(auto input_data,
                        EmitAllGather(builder, input_data, input_layout,
                                      all_to_all_input_layout));
    op->setOperand(0, input_data);
  } else {
    // When all-to-all is not needed, output of BroadcastTo operation may be
    // sharded. In that case, we must ensure that `shape` input of BroadcastTo
    // op has correct local sharded shape.
    // Note that we include the sharding on the first
    for (int i = 0; i < broadcasted_dimensions; ++i)
      if (output_layout.sharding_spec(i) != Layout::kUnshardedDim)
        broadcast_to_shape[i] /= output_num_shards[i];
    mlir::Value new_broadcast_to_shape =
        Int64Const(builder, op->getLoc(), broadcast_to_shape);
    op->setOperand(1, new_broadcast_to_shape);
  }

  op = InferSPMDExpandedLocalShape(op);
  if (!requires_all_to_all) return op;

  // If we all-to-all'ed, we may need to split after the local BroadcastTo op
  // has been created in graph.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  builder.setInsertionPointAfter(op);
  TF_ASSIGN_OR_RETURN(
      auto final_output,
      EmitAllScatter(builder, op->getOpResult(0),
                     all_to_all_input_layout.LeftPad(output_layout.rank()),
                     output_layout, &newly_created_ops));
  op->getOpResult(0).replaceAllUsesExcept(final_output, newly_created_ops);
  return final_output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
BroadcastToSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If we do not have an input layout then do not infer an output layout.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  auto broadcast_op = llvm::cast<mlir::TF::BroadcastToOp>(op);
  TF_ASSIGN_OR_RETURN(
      const auto broadcasted_output_shape,
      GetShapeOfValue(broadcast_op.getOutput(), /*fail_on_dynamic=*/true));
  TF_ASSIGN_OR_RETURN(
      const auto input_shape,
      GetShapeOfValue(broadcast_op.getInput(), /*fail_on_dynamic=*/true));

  // Broadcasting works from trailing dimensions and dimensions are broadcasted
  // in forward direction.
  const int output_shape_rank = broadcasted_output_shape.size();
  const int input_shape_rank = input_shape.size();
  const int broadcasted_dimensions = output_shape_rank - input_shape_rank;

  if (broadcasted_dimensions < 0)
    return errors::FailedPrecondition("Broadcasted dimension was less than 0.");

  Layout input_layout = input_layouts.lookup(0);

  std::vector<std::string> layout_sharding;
  for (int i = 0; i < output_shape_rank; ++i) {
    if (i < broadcasted_dimensions) {
      layout_sharding.push_back(Layout::kUnshardedDim);
    } else {
      layout_sharding.push_back(
          input_layout.sharding_spec(i - broadcasted_dimensions));
    }
  }
  TF_ASSIGN_OR_RETURN(Layout inferred_output_layout,
                      Layout::GetLayout(layout_sharding, mesh));
  return llvm::DenseMap<int, Layout>({{0, inferred_output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
BroadcastToSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  // If output layout is not set, then we can only infer the `shape` input
  // which should always be replicated.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>(
        {{1, Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(1)))}});

  auto output_layout = output_layouts.lookup(0);

  auto broadcast_op = llvm::cast<mlir::TF::BroadcastToOp>(op);
  TF_ASSIGN_OR_RETURN(
      const auto broadcasted_output_shape,
      GetShapeOfValue(broadcast_op.getOutput(), /*fail_on_dynamic=*/true));
  TF_ASSIGN_OR_RETURN(
      const auto input_shape,
      GetShapeOfValue(broadcast_op.getInput(), /*fail_on_dynamic=*/true));

  // Broadcasting works from trailing dimensions and dimensions are broadcasted
  // in forward direction.
  const int output_shape_rank = broadcasted_output_shape.size();
  const int input_shape_rank = input_shape.size();
  const int broadcasted_dimensions = output_shape_rank - input_shape_rank;

  std::vector<std::string> sharding_specs;
  for (int i = 0; i < input_shape_rank; ++i) {
    if (input_shape[i] == 1) {
      sharding_specs.push_back(Layout::kUnshardedDim);
    } else {
      sharding_specs.push_back(
          output_layout.sharding_spec(i + broadcasted_dimensions));
    }
  }
  TF_ASSIGN_OR_RETURN(Layout inferred_operand_layout,
                      Layout::GetLayout(sharding_specs, mesh));
  // `shape` input of BroadcastTo is always set as replicated.
  return llvm::DenseMap<int, Layout>(
      {{0, inferred_operand_layout},
       {1, Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(1)))}});
}

}  // namespace dtensor
}  // namespace tensorflow
