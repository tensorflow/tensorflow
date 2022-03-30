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

#include <string>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
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
  std::vector<ShardingSpec> output_specs(tensor_rank);

  // The number of dimensions at the start of the tensor input that are used
  // for the index, also the size of the second dimension of the indices tensor.
  const int index_dimensions = tensor_rank - (updates_rank - 1);

  for (int i = 0; i < tensor_rank; ++i)
    output_specs[i].set_sharding_spec(Layout::kUnshardedDim);

  absl::flat_hash_set<std::string> used_mesh_dims;

  if (tensor_layout) {
    for (int i = index_dimensions; i < tensor_rank; ++i) {
      output_specs[i] = tensor_layout->dim(i);
      if (Layout::IsShardedSpec(output_specs[i]))
        used_mesh_dims.emplace(output_specs[i].sharding_spec());
    }
  }

  if (updates_layout) {
    for (int i = index_dimensions; i < tensor_rank; ++i) {
      const ShardingSpec& update_spec =
          updates_layout->dim(i - index_dimensions + 1);

      if (Layout::IsUnshardedSpec(output_specs[i]) &&
          Layout::IsShardedSpec(update_spec) &&
          !used_mesh_dims.contains(update_spec.sharding_spec()))
        output_specs[i] = update_spec;
    }
  }

  return Layout::GetLayout(output_specs, mesh);
}

template <typename OpType>
StatusOr<mlir::Operation*> TensorScatterOpExpand(mlir::Operation* op) {
  auto scatter_op = llvm::cast<OpType>(op);
  TF_ASSIGN_OR_RETURN(auto tensor_layout,
                      ExtractLayoutFromOperand(scatter_op.tensor()));
  TF_ASSIGN_OR_RETURN(auto indices_layout,
                      ExtractLayoutFromOperand(scatter_op.indices()));
  TF_ASSIGN_OR_RETURN(auto updates_layout,
                      ExtractLayoutFromOperand(scatter_op.updates()));
  TF_ASSIGN_OR_RETURN(auto output_layout,
                      ExtractSingleLayoutFromOp(scatter_op));

  const int tensor_rank = ValueRank(scatter_op.tensor());
  const int updates_rank = ValueRank(scatter_op.updates());

  if (tensor_rank == -1 || updates_rank == -1)
    return errors::InvalidArgument("all inputs must have valid rank.");

  // Get the global shape of all inputs as we need them for the Relayout
  // operations.
  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> tensor_shape,
      GetGlobalShapeOfValueFromDTensorLayout(scatter_op.tensor()));
  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> indices_shape,
      GetGlobalShapeOfValueFromDTensorLayout(scatter_op.indices()));
  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> updates_shape,
      GetGlobalShapeOfValueFromDTensorLayout(scatter_op.updates()));

  // Start by relaying out the inputs. Indices should replicated.
  TF_ASSIGN_OR_RETURN(
      mlir::Value new_indices,
      EmitRelayout(scatter_op.indices(), *indices_layout,
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

  std::vector<ShardingSpec> updates_specs(updates_rank);
  updates_specs[0].set_sharding_spec(Layout::kUnshardedDim);

  const int index_dimensions = tensor_rank - (updates_rank - 1);

  for (int i = 0; i < updates_rank - 1; ++i)
    updates_specs[i + 1] = pre_output_layout.dim(index_dimensions + i);

  TF_ASSIGN_OR_RETURN(Layout new_updates_layout,
                      Layout::GetLayout(updates_specs, updates_layout->mesh()));
  TF_ASSIGN_OR_RETURN(
      mlir::Value new_tensor,
      EmitRelayout(scatter_op.tensor(), *tensor_layout, pre_output_layout));
  TF_ASSIGN_OR_RETURN(
      mlir::Value new_updates,
      EmitRelayout(scatter_op.updates(), *updates_layout, new_updates_layout));

  mlir::OpBuilder builder(op);
  OpType new_scatter = builder.create<OpType>(
      op->getLoc(), new_tensor.getType(), new_tensor, new_indices, new_updates);

  TF_ASSIGN_OR_RETURN(
      mlir::Value new_output,
      EmitRelayout(new_scatter.output(), pre_output_layout, *output_layout));

  op->getResult(0).replaceAllUsesWith(new_output);
  op->erase();

  return new_output.getDefiningOp();
}

template <typename OpType>
StatusOr<llvm::DenseMap<int, Layout>> TensorScatterOpComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto scatter_op = llvm::cast<OpType>(op);

  const int tensor_rank = ValueRank(scatter_op.tensor());
  const int updates_rank = ValueRank(scatter_op.updates());
  if (tensor_rank == -1 || updates_rank == -1)
    return errors::InvalidArgument("all inputs must have valid rank.");

  absl::optional<Layout> tensor_layout;
  if (input_layouts.find(0) != input_layouts.end())
    tensor_layout.emplace(input_layouts.lookup(0));
  absl::optional<Layout> updates_layout;
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

  const int tensor_rank = ValueRank(scatter_op.tensor());
  const int indices_rank = ValueRank(scatter_op.indices());
  const int updates_rank = ValueRank(scatter_op.updates());
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

}  // namespace dtensor
}  // namespace tensorflow
