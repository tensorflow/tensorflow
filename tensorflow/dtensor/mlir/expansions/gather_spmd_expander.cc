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

#include "tensorflow/dtensor/mlir/expansions/gather_spmd_expander.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> GatherV2SPMDExpander::ExpandOp(mlir::Operation* op) {
  return ExpandOpHelper<mlir::TF::GatherV2Op>(op);
}

StatusOr<int64_t> GatherV2SPMDExpander::GetAxis(mlir::Operation* op) {
  return ExtractConstIntFromValue(
      llvm::cast<mlir::TF::GatherV2Op>(op).getAxis());
}

StatusOr<uint64_t> GatherV2SPMDExpander::GetBatchDim(mlir::Operation* op) {
  return llvm::cast<mlir::TF::GatherV2Op>(op).getBatchDims();
}

StatusOr<mlir::Operation*> ResourceGatherSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  return ExpandOpHelper<mlir::TF::ResourceGatherOp>(op);
}

StatusOr<int64_t> ResourceGatherSPMDExpander::GetAxis(mlir::Operation* op) {
  return 0;
}

StatusOr<uint64_t> ResourceGatherSPMDExpander::GetBatchDim(
    mlir::Operation* op) {
  return llvm::cast<mlir::TF::ResourceGatherOp>(op).getBatchDims();
}

StatusOr<llvm::DenseMap<int, Layout>>
GatherCommonSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(int64_t axis, GetAxis(op));
  TF_ASSIGN_OR_RETURN(uint64_t batch_dims, GetBatchDim(op));

  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  std::optional<Layout> params_layout;
  if (input_layouts.find(0) != input_layouts.end()) {
    params_layout.emplace(input_layouts.lookup(0));
  }
  std::optional<Layout> indices_layout;
  if (input_layouts.find(1) != input_layouts.end()) {
    indices_layout.emplace(input_layouts.lookup(1));
  }

  const int params_rank = ValueRank(op->getOperand(0));
  const int indices_rank = ValueRank(op->getOperand(1));
  if (params_rank == -1)
    return errors::InvalidArgument("missing rank for params input");
  if (indices_rank == -1)
    return errors::InvalidArgument("missing rank for indices input");

  // Handle the case of negative axis.
  if (axis < 0) axis += params_rank;
  if (batch_dims < 0) batch_dims += indices_rank;
  if (!params_layout && !indices_layout) {
    return llvm::DenseMap<int, Layout>();
  }
  std::vector<std::string> output_layout_specs;

  // Get a list of mesh dims that params uses, other than the dim for axis.
  llvm::DenseSet<llvm::StringRef> params_mesh_dims;
  if (params_layout) {
    for (int i = 0; i < params_rank; ++i)
      if (i != axis &&
          !Layout::IsUnshardedDimension(params_layout->sharding_spec(i)))
        params_mesh_dims.insert(params_layout->sharding_spec(i));
  }

  auto add_mesh_dim_if = [&](const absl::optional<Layout>& input_layout,
                             int64 dim, bool indices = false) {
    // Only add the mesh dimension to the output_layout if 1) the input layout
    // exists and 2) when the input is indices and the params dims don't
    // contain the mesh dim we are adding (to avoid two different tensor dims
    // being sharded over the same mesh dim).
    // Note that params->sharding_spec(axis) is specifically excluded from the
    // params_mesh_dims during construction above. This means that if we are
    // processing the indices layout and it contains
    // params->sharding_spec(axis) we will still add that sharding_spec to the
    // output layout.
    if (input_layout && (!indices || !params_mesh_dims.contains(
                                         input_layout->sharding_spec(dim))))
      output_layout_specs.push_back(input_layout->sharding_spec(dim));
    else
      output_layout_specs.push_back(Layout::kUnshardedDim);
  };

  for (int i = 0; i < axis; ++i) add_mesh_dim_if(params_layout, i);
  for (int i = batch_dims; i < indices_rank; ++i)
    add_mesh_dim_if(indices_layout, i, /*indices=*/true);
  for (int i = axis + 1; i < params_rank; ++i)
    add_mesh_dim_if(params_layout, i);

  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      Layout::GetLayout(output_layout_specs, mesh));
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
GatherCommonSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(int64_t axis, GetAxis(op));
  TF_ASSIGN_OR_RETURN(uint64_t batch_dims, GetBatchDim(op));
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(op->getNumOperands());
  // Axis is a constant so replicate it. ResourceGatherOp does not have axis.
  if (llvm::isa<mlir::TF::GatherV2Op>(op)) {
    input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/0);
  }

  auto it = output_layouts.find(0);
  if (it == output_layouts.end()) {
    return input_layouts;
  }

  // This will always exist since there is only one output.
  const Layout output_layout = it->getSecond();

  const int params_rank = ValueRank(op->getOperand(0));
  const int indices_rank = ValueRank(op->getOperand(1));
  if (params_rank == -1)
    return errors::InvalidArgument("missing rank for params input");
  if (indices_rank == -1)
    return errors::InvalidArgument("missing rank for indices input");

  // Handle the case of negative axis.
  if (axis < 0) axis += params_rank;
  if (batch_dims < 0) batch_dims += indices_rank;
  std::vector<std::string> params_layout_specs;
  std::vector<std::string> indices_layout_specs;
  params_layout_specs.reserve(params_rank);
  indices_layout_specs.reserve(indices_rank);

  // Extract the params layout. We will request that axis is replicated as
  // that gives the least issues with spmd expansion.
  // E.g. If we had axis = 1 and parmas layout [p0 p1 p2 p3]
  // input layout [i0 i1] then the output layout would have been
  // [p0 i0 i1 p2 p3] so to go backwards, we extract the ranges [0, axis)
  // and [axis + indices.rank(), output.rank()) for params (with a replicated
  // dim for the missing dimension inbetween). Indices layout is based on the
  // range [axis, axis+indices)/
  for (int i = 0; i < axis; ++i)
    params_layout_specs.push_back(output_layout.sharding_spec(i));
  params_layout_specs.push_back(Layout::kUnshardedDim);
  for (int i = axis + indices_rank - batch_dims; i < output_layout.rank(); ++i)
    params_layout_specs.push_back(output_layout.sharding_spec(i));

  // Extract the indices layout.
  for (int i = 0; i < batch_dims; ++i)
    indices_layout_specs.push_back(output_layout.sharding_spec(i));
  for (int i = axis; i < axis + indices_rank - batch_dims; ++i)
    indices_layout_specs.push_back(output_layout.sharding_spec(i));

  TF_ASSIGN_OR_RETURN(const Layout params_layout,
                      Layout::GetLayout(params_layout_specs, mesh));
  TF_ASSIGN_OR_RETURN(const Layout indices_layout,
                      Layout::GetLayout(indices_layout_specs, mesh));
  input_layouts[0] = params_layout;
  input_layouts[1] = indices_layout;

  return input_layouts;
}

namespace {

StatusOr<Layout> GatherNdGetOutputLayoutFromInput(
    const absl::optional<Layout>& params_layout, int params_rank,
    const absl::optional<Layout>& indices_layout, int indices_rank,
    int index_dimensions, const Mesh& mesh) {
  // The layout of the output should be the layout of the first rank-1
  // dimensions of the indices plus the layout of the last params.rank -
  // indices.dims[-1] dimensions of params. If one of the two is missing we
  // replace them with replicated.
  // If sharding dimension is used by both params and indices, the params
  // layout will be respected as generally params is larger than indices.
  std::vector<std::string> output_specs(params_rank - index_dimensions +
                                        indices_rank - 1);
  absl::flat_hash_set<std::string> used_dimensions;
  const int params_offset = -index_dimensions + indices_rank - 1;

  for (int i = index_dimensions; i < params_rank; ++i) {
    if (params_layout &&
        Layout::IsShardedDimension(params_layout->sharding_spec(i))) {
      const auto& params_spec = params_layout->sharding_spec(i);
      output_specs[i + params_offset] = params_spec;
      used_dimensions.emplace(params_spec);
    } else {
      output_specs[i + params_offset] = Layout::kUnshardedDim;
    }
  }
  for (int i = 0; i < indices_rank - 1; ++i) {
    if (indices_layout &&
        Layout::IsShardedDimension(indices_layout->sharding_spec(i)) &&
        !used_dimensions.contains(indices_layout->sharding_spec(i)))
      output_specs[i] = indices_layout->sharding_spec(i);
    else
      output_specs[i] = Layout::kUnshardedDim;
  }
  return Layout::GetLayout(output_specs, mesh);
}

absl::Status GatherNdGetInputLayoutFromOutput(
    const Layout& output_layout, Layout* params_layout, int params_rank,
    Layout* indices_layout, int indices_rank, int index_dimensions,
    const Mesh& mesh) {
  // We copy the first indices_rank - 1 dimensions of the output layout to
  // indices_layout (with the last dimensions replicated) and the remaining
  // dimensions to params_layout (with the first index_dimensions dimensions
  // replicated).
  std::vector<std::string> params_specs(params_rank);
  std::vector<std::string> indices_specs(indices_rank);

  for (int i = 0; i < index_dimensions; ++i)
    params_specs[i] = Layout::kUnshardedDim;

  const int params_offset = -index_dimensions + indices_rank - 1;
  for (int i = index_dimensions; i < params_rank; ++i)
    params_specs[i] = output_layout.sharding_spec(i + params_offset);

  for (int i = 0; i < indices_rank - 1; ++i)
    indices_specs[i] = output_layout.sharding_spec(i);

  indices_specs[indices_rank - 1] = Layout::kUnshardedDim;

  TF_ASSIGN_OR_RETURN(*params_layout, Layout::GetLayout(params_specs, mesh));
  TF_ASSIGN_OR_RETURN(*indices_layout, Layout::GetLayout(indices_specs, mesh));

  return absl::OkStatus();
}

}  // namespace

StatusOr<mlir::Operation*> GatherNdSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto gather_op = llvm::cast<mlir::TF::GatherNdOp>(op);
  TF_ASSIGN_OR_RETURN(Layout params_layout,
                      ExtractRequiredLayoutFromOperand(gather_op.getParams()));
  TF_ASSIGN_OR_RETURN(Layout indices_layout,
                      ExtractRequiredLayoutFromOperand(gather_op.getIndices()));
  TF_ASSIGN_OR_RETURN(Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(gather_op));

  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> indices_shape,
      GetGlobalShapeOfValueFromDTensorLayout(gather_op.getIndices()));
  const int index_dimensions = indices_shape.back();

  const auto params_rank = ValueRank(gather_op.getParams());
  const auto indices_rank = ValueRank(gather_op.getIndices());
  if (params_rank == -1)
    return errors::InvalidArgument("missing rank for params input");
  if (indices_rank == -1)
    return errors::InvalidArgument("missing rank for indices input");

  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Note that there may be conflicts between the two input layouts.
  // To resolve these we:
  // 1) Compute output layout from the current input layouts. This ignores any
  //    sharding on the 'lookup' axis in params and the last axis in indices.
  //    If params and indices are sharded on same dimension, the indices will
  //    be unsharded in that dimension.
  // 2) Merge any new sharding from the final output layout into this layout.
  //    E.g. if the two inputs are unsharded, but the output is, this will be
  //    more efficient as the local computations will be smaller since we will
  //    more finely shard the inputs.
  // 3) Finally, we use this new output layout to compute what input layouts we
  //    need to get the given output.

  // Step 1)
  TF_ASSIGN_OR_RETURN(const Layout pre_output_layout,
                      GatherNdGetOutputLayoutFromInput(
                          params_layout, params_rank, indices_layout,
                          indices_rank, index_dimensions, mesh));

  // Step 2)
  llvm::DenseSet<llvm::StringRef> used_dimensions;
  for (const auto& spec : pre_output_layout.sharding_spec_strs())
    if (Layout::IsShardedDimension(spec)) used_dimensions.insert(spec);

  std::vector<std::string> sharding_specs(output_layout.rank());
  for (int i = 0; i < sharding_specs.size(); ++i) {
    if (Layout::IsShardedDimension(pre_output_layout.sharding_spec(i)))
      sharding_specs[i] = pre_output_layout.sharding_spec(i);
    // Merge in sharded dimensions from the output which aren't already used
    // by the pre_output_layout.
    else if (Layout::IsShardedDimension(output_layout.sharding_spec(i)) &&
             !used_dimensions.contains(output_layout.sharding_spec(i)))
      sharding_specs[i] = output_layout.sharding_spec(i);
    else
      sharding_specs[i] = Layout::kUnshardedDim;
  }

  // Step 3)
  TF_ASSIGN_OR_RETURN(
      const Layout merged_output_layout,
      Layout::GetLayout(sharding_specs, pre_output_layout.mesh()));

  Layout new_params_layout;
  Layout new_indices_layout;
  TF_RETURN_IF_ERROR(GatherNdGetInputLayoutFromOutput(
      merged_output_layout, &new_params_layout, params_rank,
      &new_indices_layout, indices_rank, index_dimensions, mesh));

  TF_ASSIGN_OR_RETURN(
      mlir::Value params,
      EmitRelayout(gather_op.getParams(), params_layout, new_params_layout));
  TF_ASSIGN_OR_RETURN(
      mlir::Value indices,
      EmitRelayout(gather_op.getIndices(), indices_layout, new_indices_layout));

  mlir::OpBuilder builder(op);

  // mlir::TF::GatherNdOp does not have a builder that does shape inference
  // so we have to provided the output type. This output type is incorrect and
  // we manually run shape inference after.
  mlir::TF::GatherNdOp new_gather = builder.create<mlir::TF::GatherNdOp>(
      op->getLoc(), gather_op.getOutput().getType(), params, indices);
  InferSPMDExpandedLocalShape(new_gather);

  TF_ASSIGN_OR_RETURN(mlir::Value output,
                      EmitRelayout(new_gather.getOutput(), merged_output_layout,
                                   output_layout));

  gather_op.getOutput().replaceAllUsesWith(output);
  gather_op.erase();
  return output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
GatherNdSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto gather_op = llvm::cast<mlir::TF::GatherNdOp>(op);
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  std::optional<Layout> params_layout;
  if (input_layouts.find(0) != input_layouts.end())
    params_layout.emplace(input_layouts.lookup(0));
  std::optional<Layout> indices_layout;
  if (input_layouts.find(1) != input_layouts.end())
    indices_layout.emplace(input_layouts.lookup(1));

  TF_ASSIGN_OR_RETURN(llvm::ArrayRef<int64_t> indices_shape,
                      ExtractGlobalInputShape(op->getOpOperand(1)));
  const int index_dimensions = indices_shape.back();
  if (index_dimensions < 0)
    return errors::Unimplemented(
        "dynamic last dimension for index is not supported");

  const int params_rank = ValueRank(gather_op.getParams());
  const int indices_rank = ValueRank(gather_op.getIndices());
  if (params_rank == -1)
    return errors::InvalidArgument("missing rank for params input");
  if (indices_rank == -1)
    return errors::InvalidArgument("missing rank for indices input");

  if (params_layout || indices_layout) {
    TF_ASSIGN_OR_RETURN(const Layout output_layout,
                        GatherNdGetOutputLayoutFromInput(
                            params_layout, params_rank, indices_layout,
                            indices_rank, index_dimensions, mesh));
    return llvm::DenseMap<int, Layout>({{0, output_layout}});
  }

  return llvm::DenseMap<int, Layout>();
}

StatusOr<llvm::DenseMap<int, Layout>>
GatherNdSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If there is no output layout present then do not infer any operand layouts.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto gather_op = llvm::cast<mlir::TF::GatherNdOp>(op);
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  TF_ASSIGN_OR_RETURN(llvm::ArrayRef<int64_t> indices_shape,
                      ExtractGlobalInputShape(op->getOpOperand(1)));
  const int index_dimensions = indices_shape.back();
  if (index_dimensions < 0)
    return errors::Unimplemented(
        "dynamic last dimension for index is not supported");

  const int params_rank = ValueRank(gather_op.getParams());
  const int indices_rank = ValueRank(gather_op.getIndices());
  if (params_rank == -1)
    return errors::InvalidArgument("missing rank for params input");
  if (indices_rank == -1)
    return errors::InvalidArgument("missing rank for indices input");

  const Layout output_layout = output_layouts.lookup(0);
  Layout params_layout, indices_layout;
  TF_RETURN_IF_ERROR(GatherNdGetInputLayoutFromOutput(
      output_layout, &params_layout, params_rank, &indices_layout, indices_rank,
      index_dimensions, mesh));
  return llvm::DenseMap<int, Layout>({
      {0, params_layout},
      {1, indices_layout},
  });
}

}  // namespace dtensor
}  // namespace tensorflow
