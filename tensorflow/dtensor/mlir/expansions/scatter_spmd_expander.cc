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

#include "tensorflow/dtensor/mlir/expansions/scatter_spmd_expander.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

StatusOr<Layout> GetOutputLayout(const absl::optional<Layout>& tensor_layout,
                                 int tensor_rank,
                                 const absl::optional<Layout>& updates_layout,
                                 int updates_rank, const Mesh& mesh) {
  // The first tensor_rank - update_rank dimensions of the output should be set
  // to replicated. The remainder are set from tensor_layout and updates_layout
  // with tensor_layout taking priority, as it is generally larger than updates
  // (as unsharding updates is faster).
  std::vector<std::string> output_specs(tensor_rank);

  // The number of dimensions at the start of the tensor input that are used
  // for the index, also the size of the second dimension of the indices tensor.
  const int index_dimensions = tensor_rank - (updates_rank - 1);

  for (int i = 0; i < tensor_rank; ++i) output_specs[i] = Layout::kUnshardedDim;

  absl::flat_hash_set<std::string> used_mesh_dims;

  if (tensor_layout) {
    for (int i = index_dimensions; i < tensor_rank; ++i) {
      output_specs[i] = tensor_layout->sharding_spec(i);
      if (Layout::IsShardedDimension(output_specs[i]))
        used_mesh_dims.emplace(output_specs[i]);
    }
  }

  if (updates_layout) {
    for (int i = index_dimensions; i < tensor_rank; ++i) {
      const auto& update_spec =
          updates_layout->sharding_spec(i - index_dimensions + 1);

      if (Layout::IsUnshardedDimension(output_specs[i]) &&
          Layout::IsShardedDimension(update_spec) &&
          !used_mesh_dims.contains(update_spec))
        output_specs[i] = update_spec;
    }
  }

  return Layout::GetLayout(output_specs, mesh);
}

template <typename OpType>
StatusOr<mlir::Operation*> TensorScatterOpExpand(mlir::Operation* op) {
  auto scatter_op = llvm::cast<OpType>(op);
  TF_ASSIGN_OR_RETURN(auto tensor_layout,
                      ExtractLayoutFromOperand(scatter_op.getTensor()));
  TF_ASSIGN_OR_RETURN(auto indices_layout,
                      ExtractLayoutFromOperand(scatter_op.getIndices()));
  TF_ASSIGN_OR_RETURN(auto updates_layout,
                      ExtractLayoutFromOperand(scatter_op.getUpdates()));
  TF_ASSIGN_OR_RETURN(auto output_layout,
                      ExtractSingleLayoutFromOp(scatter_op));

  const int tensor_rank = ValueRank(scatter_op.getTensor());
  const int updates_rank = ValueRank(scatter_op.getUpdates());

  if (tensor_rank == -1 || updates_rank == -1)
    return errors::InvalidArgument("all inputs must have valid rank.");

  // Get the global shape of all inputs as we need them for the Relayout
  // operations.
  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> tensor_shape,
      GetGlobalShapeOfValueFromDTensorLayout(scatter_op.getTensor()));
  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> indices_shape,
      GetGlobalShapeOfValueFromDTensorLayout(scatter_op.getIndices()));
  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> updates_shape,
      GetGlobalShapeOfValueFromDTensorLayout(scatter_op.getUpdates()));

  // Start by relaying out the inputs. Indices should replicated.
  TF_ASSIGN_OR_RETURN(
      mlir::Value new_indices,
      EmitRelayout(scatter_op.getIndices(), *indices_layout,
                   Layout::ReplicatedOnMesh(indices_layout->mesh(),
                                            indices_shape.size())));

  // Create intermediate layouts for tensors and updates. Since the layout of
  // tensor and the output of the local tensor-scatter are the same we can reuse
  // GetOutputLayout.
  // If the true output layout is even more sharded, we could forward those
  // shardings here for even better performance.
  TF_ASSIGN_OR_RETURN(
      Layout pre_output_layout,
      GetOutputLayout(tensor_layout, tensor_rank, updates_layout, updates_rank,
                      tensor_layout->mesh()));

  std::vector<std::string> updates_specs(updates_rank);
  updates_specs[0] = Layout::kUnshardedDim;

  const int index_dimensions = tensor_rank - (updates_rank - 1);

  for (int i = 0; i < updates_rank - 1; ++i)
    updates_specs[i + 1] =
        pre_output_layout.sharding_spec(index_dimensions + i);

  TF_ASSIGN_OR_RETURN(Layout new_updates_layout,
                      Layout::GetLayout(updates_specs, updates_layout->mesh()));
  TF_ASSIGN_OR_RETURN(
      mlir::Value new_tensor,
      EmitRelayout(scatter_op.getTensor(), *tensor_layout, pre_output_layout));
  TF_ASSIGN_OR_RETURN(mlir::Value new_updates,
                      EmitRelayout(scatter_op.getUpdates(), *updates_layout,
                                   new_updates_layout));

  mlir::OpBuilder builder(op);
  OpType new_scatter = builder.create<OpType>(
      op->getLoc(), new_tensor.getType(), new_tensor, new_indices, new_updates);

  TF_ASSIGN_OR_RETURN(
      mlir::Value new_output,
      EmitRelayout(new_scatter.getOutput(), pre_output_layout, *output_layout));

  op->getResult(0).replaceAllUsesWith(new_output);
  op->erase();

  return new_output.getDefiningOp();
}

template <typename OpType>
StatusOr<llvm::DenseMap<int, Layout>> TensorScatterOpComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto scatter_op = llvm::cast<OpType>(op);

  const int tensor_rank = ValueRank(scatter_op.getTensor());
  const int updates_rank = ValueRank(scatter_op.getUpdates());
  if (tensor_rank == -1 || updates_rank == -1)
    return errors::InvalidArgument("all inputs must have valid rank.");

  std::optional<Layout> tensor_layout;
  if (input_layouts.find(0) != input_layouts.end())
    tensor_layout.emplace(input_layouts.lookup(0));
  std::optional<Layout> updates_layout;
  if (input_layouts.find(2) != input_layouts.end())
    updates_layout.emplace(input_layouts.lookup(2));

  if (tensor_layout || updates_layout) {
    TF_ASSIGN_OR_RETURN(const Layout output_layout,
                        GetOutputLayout(tensor_layout, tensor_rank,
                                        updates_layout, updates_rank, mesh));
    return llvm::DenseMap<int, Layout>({{0, output_layout}});
  }

  return llvm::DenseMap<int, Layout>();
}

template <typename OpType>
StatusOr<llvm::DenseMap<int, Layout>> TensorScatterOpComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto scatter_op = llvm::cast<OpType>(op);

  const int tensor_rank = ValueRank(scatter_op.getTensor());
  const int indices_rank = ValueRank(scatter_op.getIndices());
  const int updates_rank = ValueRank(scatter_op.getUpdates());
  if (tensor_rank == -1 || indices_rank == -1 || updates_rank == -1)
    return errors::InvalidArgument("all inputs must have valid rank.");

  // The number of dimensions at the start of the tensor input that are used
  // for the index, also the size of the second dimension of the indices tensor.
  const int index_dimensions = tensor_rank - (updates_rank - 1);

  llvm::DenseMap<int, Layout> input_layouts(scatter_op.getNumOperands());

  // Always set indices layout to replicated.
  const Layout indices_layout = Layout::ReplicatedOnMesh(mesh, indices_rank);
  input_layouts[1] = indices_layout;

  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout output_layout = output_layouts.lookup(0);

    std::vector<std::string> tensor_sharding_specs(tensor_rank);
    std::vector<std::string> updates_sharding_specs(updates_rank);

    for (int i = 0; i < index_dimensions; ++i)
      tensor_sharding_specs[i] = Layout::kUnshardedDim;
    updates_sharding_specs[0] = Layout::kUnshardedDim;

    for (int i = index_dimensions; i < tensor_rank; ++i) {
      tensor_sharding_specs[i] = output_layout.sharding_spec(i);
      updates_sharding_specs[i - index_dimensions + 1] =
          output_layout.sharding_spec(i);
    }

    TF_ASSIGN_OR_RETURN(const Layout tensor_layout,
                        Layout::GetLayout(tensor_sharding_specs, mesh));
    TF_ASSIGN_OR_RETURN(const Layout updates_layout,
                        Layout::GetLayout(updates_sharding_specs, mesh));
    input_layouts[0] = tensor_layout;
    input_layouts[2] = updates_layout;
  }

  return input_layouts;
}

}  // namespace

StatusOr<mlir::Operation*> TensorScatterOpSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  if (llvm::isa<mlir::TF::TensorScatterUpdateOp>(op)) {
    return TensorScatterOpExpand<mlir::TF::TensorScatterUpdateOp>(op);
  }
  if (llvm::isa<mlir::TF::TensorScatterAddOp>(op)) {
    return TensorScatterOpExpand<mlir::TF::TensorScatterAddOp>(op);
  }
  return errors::Unimplemented(absl::StrCat(
      "SPMD expansion for op : ", OpName(op), " is not implemented"));
}

StatusOr<llvm::DenseMap<int, Layout>>
TensorScatterOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (llvm::isa<mlir::TF::TensorScatterUpdateOp>(op)) {
    return TensorScatterOpComputeLayoutForward<mlir::TF::TensorScatterUpdateOp>(
        op, input_layouts);
  }
  if (llvm::isa<mlir::TF::TensorScatterAddOp>(op)) {
    return TensorScatterOpComputeLayoutForward<mlir::TF::TensorScatterAddOp>(
        op, input_layouts);
  }
  return errors::Unimplemented(absl::StrCat(
      "Layout propagation for op : ", OpName(op), " is not implemented"));
}

StatusOr<llvm::DenseMap<int, Layout>>
TensorScatterOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  if (llvm::isa<mlir::TF::TensorScatterUpdateOp>(op)) {
    return TensorScatterOpComputeLayoutBackward<
        mlir::TF::TensorScatterUpdateOp>(op, output_layouts);
  }
  if (llvm::isa<mlir::TF::TensorScatterAddOp>(op)) {
    return TensorScatterOpComputeLayoutBackward<mlir::TF::TensorScatterAddOp>(
        op, output_layouts);
  }
  return errors::Unimplemented(absl::StrCat(
      "Layout propagation for op : ", OpName(op), " is not implemented"));
}

StatusOr<mlir::Operation*> ScatterNdOpSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(const std::vector<Layout>& operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));
  TF_ASSIGN_OR_RETURN(const Layout& output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));

  const Layout& indices_layout = operand_layouts[0];
  const Layout& updates_layout = operand_layouts[1];

  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto scatter_op = llvm::cast<mlir::TF::ScatterNdOp>(op);

  const int output_rank = ValueRank(scatter_op.getResult());
  const int updates_rank = ValueRank(scatter_op.getUpdates());
  const int indices_rank = ValueRank(scatter_op.getIndices());

  if (output_rank == -1 || updates_rank == -1) {
    return errors::InvalidArgument(
        "Dynamic shaped inputs are not supported. Please file a feature "
        "request to TF DTensor: component id: 8333864");
  }

  llvm::SmallVector<int64_t, 4> global_shape;
  if (!ExtractConstVectorFromValue(scatter_op.getShape(), &global_shape).ok()) {
    return errors::InvalidArgument(
        "Failed in extracting constant vector from shape tensor. Please file "
        "a bug to TF DTensor: component id: 833864");
  }

  // Only do computation after replicating indices tensor.
  // The expansion will work as the following:
  // Let N be the rank of the output tensor.
  // Let K be the rank of each update tensor.
  // Then the size of each tensor of indices must be size N-K+1.
  // We will enforce the indices tensor to be replicated, which means
  // that also the first N-K+1 dimensions of the output must be replicated as
  // well.
  // We will shard the updates tensor however much we can, which also means
  // we will shard the last K dimension tensors of the output tensor to be
  // as sharded as the updates tensor.
  //
  TF_ASSIGN_OR_RETURN(
      mlir::Value new_indices,
      EmitRelayout(scatter_op.getIndices(), indices_layout,
                   Layout::ReplicatedOnMesh(mesh, indices_rank)));

  // Create intermediate layouts for tensors and updates. Since the layout of
  // tensor and the output of the local tensor-scatter are the same we can reuse
  // GetOutputLayout. This intermediate layout will be the layout that
  // both the output and the updates tensor will agree upon. The updates
  // intermediate layout will also be computed from the last K dimension of
  // this.
  TF_ASSIGN_OR_RETURN(Layout output_intermediate_layout,
                      GetOutputLayout(output_layout, output_rank,
                                      updates_layout, updates_rank, mesh));

  std::vector<std::string> updates_specs(updates_rank);
  if (updates_rank == 0) {
    return errors::InvalidArgument(
        "Expected updates_rank to be greater than zero, but got: ",
        updates_rank);
  }
  updates_specs[0] = Layout::kUnshardedDim;

  for (int i = 1; i < updates_rank; ++i) {
    updates_specs[updates_rank - i] =
        output_intermediate_layout.sharding_spec(output_rank - i);
  }

  TF_ASSIGN_OR_RETURN(Layout new_updates_layout,
                      Layout::GetLayout(updates_specs, mesh));

  TF_ASSIGN_OR_RETURN(mlir::Value new_updates,
                      EmitRelayout(scatter_op.getUpdates(), updates_layout,
                                   new_updates_layout));

  const std::vector<int64_t>& local_shape =
      output_layout.LocalShapeFromGlobalShape(global_shape);

  mlir::OpBuilder builder(op);
  mlir::Operation* new_scatter = builder.create<mlir::TF::ScatterNdOp>(
      op->getLoc(), op->getResult(0).getType(), new_indices, new_updates,
      /*shape=*/
      ::mlir::TF::collection_ops_util::GetR1Const(local_shape, builder,
                                                  op->getLoc()));

  TF_ASSIGN_OR_RETURN(mlir::Value new_output,
                      EmitRelayout(new_scatter->getResult(0),
                                   output_intermediate_layout, output_layout));

  op->getResult(0).replaceAllUsesWith(new_output);
  op->erase();

  return InferSPMDExpandedLocalShape(new_output.getDefiningOp());
}

StatusOr<llvm::DenseMap<int, Layout>>
ScatterNdOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto scatter_op = llvm::cast<mlir::TF::ScatterNdOp>(op);

  const int output_rank = ValueRank(scatter_op.getResult());
  const int updates_rank = ValueRank(scatter_op.getUpdates());
  if (output_rank == -1 || updates_rank == -1)
    return errors::InvalidArgument(
        "Dynamic shaped inputs are not supported. Please file a feature "
        "request to TF DTensor: component id: 8333864");

  std::optional<Layout> updates_layout;
  auto iter = input_layouts.find(1);
  if (iter == input_layouts.end()) {
    return llvm::DenseMap<int, Layout>();
  }
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      GetOutputLayout(std::nullopt, output_rank,
                                      iter->getSecond(), updates_rank, mesh));
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
ScatterNdOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto scatter_op = llvm::cast<mlir::TF::ScatterNdOp>(op);

  const int output_rank = ValueRank(scatter_op.getResult());
  const int indices_rank = ValueRank(scatter_op.getIndices());
  const int updates_rank = ValueRank(scatter_op.getUpdates());

  if (output_rank == -1 || indices_rank == -1 || updates_rank == -1)
    return errors::InvalidArgument(
        "Dynamic shaped inputs are not supported. Please file a feature "
        "request to TF DTensor: component id: 8333864");

  llvm::DenseMap<int, Layout> input_layouts(scatter_op.getNumOperands());

  // Always set `indices` tensor and 'shape' tensor to replicated.
  input_layouts[0] = Layout::ReplicatedOnMesh(mesh, /*rank=*/indices_rank);
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);

  auto iter = output_layouts.find(0);
  if (iter == output_layouts.end()) {
    return input_layouts;
  }
  // Compute the updates layout.
  const Layout& output_layout = iter->getSecond();
  std::vector<std::string> updates_sharding_specs(updates_rank);

  // Replicate the first dimension. This is the number of update tensors.
  // Set the rest of the dimensions equal to the output's corresponding
  // sharding.
  updates_sharding_specs[0] = Layout::kUnshardedDim;
  for (int i = 1; i < updates_rank; ++i) {
    updates_sharding_specs[i] = output_layout.sharding_spec(output_rank - i);
  }

  TF_ASSIGN_OR_RETURN(const Layout updates_layout,
                      Layout::GetLayout(updates_sharding_specs, mesh));
  input_layouts[1] = updates_layout;

  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
