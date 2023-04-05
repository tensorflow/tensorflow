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

#include "tensorflow/dtensor/mlir/collectives.h"

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives_common.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/sparse_expander_common.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {

namespace ops_util = ::mlir::TF::collection_ops_util;

}  // namespace

StatusOr<mlir::Value> EmitAllGather(
    mlir::OpBuilder& builder, mlir::Value input,
    const dtensor::Layout& src_layout, const dtensor::Layout& tgt_layout,
    llvm::SmallPtrSet<mlir::Operation*, 4>* newly_created_ops) {
  if (src_layout.IsEquivalent(tgt_layout)) return input;

  if (src_layout.rank() != tgt_layout.rank()) {
    return errors::InvalidArgument(
        "Expected source and target layout to have the same rank, got ",
        src_layout.rank(), " vs ", tgt_layout.rank());
  }

  // Check that the tgt_layout is less sharded then src_layout.
  for (int i = 0; i < src_layout.rank(); ++i) {
    if (src_layout.sharding_spec(i) != tgt_layout.sharding_spec(i) &&
        Layout::IsShardedDimension(tgt_layout.sharding_spec(i))) {
      return errors::InvalidArgument("source layout (", src_layout.ToString(),
                                     ") for all gather is not less sharded "
                                     "than the target layout (",
                                     tgt_layout.ToString());
    }
  }

  // For convenience, operate on explicit input shapes. This isn't necessary,
  // as we could instead generate operations on top of the dynamic shape.
  const mlir::TensorType input_type =
      input.getType().dyn_cast<mlir::TensorType>();
  if (!input_type) {
    return errors::Internal(
        llvm::formatv(
            "Cannot cast input_type : {0} to TensorType. Shape must be "
            " statically known before emitting AllGather. This should not "
            "happen as we already cast it when getting its shape.",
            input.getType())
            .str());
  }

  TF_ASSIGN_OR_RETURN(mlir::TensorType global_type,
                      GlobalTypeFromLocalType(src_layout, input_type));
  TF_ASSIGN_OR_RETURN(mlir::TensorType output_type,
                      LocalTypeFromGlobalType(tgt_layout, global_type));

  mlir::Location loc = DT_LOC2(input.getLoc(), "DTensorAllGatherOp");
  mlir::TF::DTensorAllGatherOp all_gather =
      builder.create<mlir::TF::DTensorAllGatherOp>(
          loc, output_type, input,
          mlir::dtensor::LayoutAttr::get(builder.getContext(), src_layout),
          mlir::dtensor::LayoutAttr::get(builder.getContext(), tgt_layout));
  SetSingleLayoutOnOp(all_gather, tgt_layout);

  if (newly_created_ops != nullptr) newly_created_ops->insert(all_gather);

  return all_gather.getOutput();
}

StatusOr<const mlir::Value> EmitAllScatter(
    mlir::OpBuilder& builder, const mlir::Value& original_value,
    const Layout& original_layout, const Layout& desired_layout,
    llvm::SmallPtrSet<mlir::Operation*, 4>* newly_created_ops) {
  if (original_layout.IsEquivalent(desired_layout)) return original_value;

  // Have an early return if desired layout is not more sharded then the
  // original_layout.
  if (original_layout.rank() != desired_layout.rank()) {
    return errors::InvalidArgument(absl::StrCat(
        "Rank mismatch for original layout (", original_layout.ToString(),
        ") and desired layout (", desired_layout.ToString(), ")"));
  }
  for (int i = 0; i < original_layout.rank(); ++i) {
    if (original_layout.sharding_spec(i) != desired_layout.sharding_spec(i) &&
        Layout::IsShardedDimension(original_layout.sharding_spec(i))) {
      return errors::InvalidArgument(
          "EmitAllScatter was passed a desired_layout ",
          desired_layout.ToString(),
          " which was not more sharded than the original_layout ",
          original_layout.ToString());
    }
  }

  const mlir::TensorType input_type =
      original_value.getType().dyn_cast<mlir::TensorType>();
  if (!input_type)
    return errors::InvalidArgument(
        "input to EmitAllScatter does not have a TensorType");

  TF_ASSIGN_OR_RETURN(const mlir::TensorType global_type,
                      GlobalTypeFromLocalType(original_layout, input_type));
  TF_ASSIGN_OR_RETURN(const mlir::TensorType output_type,
                      LocalTypeFromGlobalType(desired_layout, global_type));

  mlir::Location loc = DT_LOC2(original_value.getLoc(), "DTensorAllScatterOp");
  mlir::TF::DTensorAllScatterOp all_scatter =
      builder.create<mlir::TF::DTensorAllScatterOp>(
          loc, output_type, original_value,
          mlir::dtensor::LayoutAttr::get(builder.getContext(), original_layout),
          mlir::dtensor::LayoutAttr::get(builder.getContext(), desired_layout));
  SetSingleLayoutOnOp(all_scatter, desired_layout);

  if (newly_created_ops != nullptr) newly_created_ops->insert(all_scatter);

  return all_scatter.getOutput();
}

StatusOr<mlir::Value> EmitDenseToSparseToDense(
    mlir::OpBuilder& builder, mlir::Value input,
    llvm::SmallPtrSet<mlir::Operation*, 4>* newly_created_ops) {
  // First create a Dense To Sparse Op. Since there is no DenseToSparseOp,
  // we do it manually by creating the indices, values, and shapes tensor
  // through various ops.
  //
  // indices tensor = tf.where(tf.not_equal(input, tf.zeros_like(tensor)))
  // values tensor = tf.gather_nd(input, indices)
  // shape tensor = tf.shape(input)
  mlir::TF::ZerosLikeOp zeros_like =
      builder.create<mlir::TF::ZerosLikeOp>(input.getLoc(), input);
  mlir::TF::NotEqualOp not_equal = builder.create<mlir::TF::NotEqualOp>(
      zeros_like.getLoc(), input, zeros_like, builder.getBoolAttr(false));

  mlir::TF::WhereOp indices = builder.create<mlir::TF::WhereOp>(
      not_equal.getLoc(),
      mlir::RankedTensorType::get(GetShapeOfValue(not_equal).value(),
                                  builder.getI64Type()),
      not_equal);

  mlir::TF::GatherNdOp values = builder.create<mlir::TF::GatherNdOp>(
      input.getLoc(), input.getType(), input, indices);
  auto shape = builder.create<mlir::TF::ShapeOp>(input.getLoc(), input,
                                                 builder.getBoolAttr(false));

  // Emit a SparseToDenseOp and replace the SparseTensor with the result of
  // this new op.
  TF_ASSIGN_OR_RETURN(
      mlir::Value zero_scalar,
      CreateZeroScalarConst(
          builder, input.getLoc(),
          input.getType().cast<mlir::TensorType>().getElementType()));

  auto dense = builder.create<mlir::TF::SparseToDenseOp>(
      input.getLoc(), input.getType(),
      mlir::ValueRange({indices, shape, values, zero_scalar}));

  if (newly_created_ops != nullptr) {
    for (auto new_op : {dense.getOperation(), shape.getOperation(),
                        values.getOperation(), indices.getOperation(),
                        not_equal.getOperation(), zeros_like.getOperation()}) {
      newly_created_ops->insert(new_op);
    }
  }

  return dense.getResult();
}

StatusOr<mlir::Value> EmitRelayout(
    mlir::Value input, const dtensor::Layout& src_layout,
    const dtensor::Layout& tgt_layout,
    llvm::SmallPtrSet<mlir::Operation*, 4>* newly_created_ops) {
  // EmitRelayout is performed by doing a split, an AllGather and another split.
  // The first split oppertunistically splits input tensor dimension i on mesh
  // mesh axis x if:
  // 1.  tgt_layout contains x at position i
  // 2.  src_layout is unsharded at position i.
  // 3.  src_layout does not contain mesh axis x.
  // This produces intermediate layout 1.
  // Next an all concat is performed on any axis in the intermediate layout 1
  // that does not agree with the sharding on the output axis.
  // This produces intermediate layout 2.
  // A split is performed from intermediate layout 2 to the tgt layout.

  if (src_layout.IsEquivalent(tgt_layout)) return input;

  // Save whether the input is from a SparseToDenseOp. If it is, then we will
  // emit a DenseToSparse and a SparseToDense op.
  bool is_sparse = IsSparseValue(input);
  if (!input.getType().isa<mlir::RankedTensorType>())
    return errors::Internal(
        "attempting to relayout a tensor that does not "
        "have a rank");

  if (src_layout.mesh() != tgt_layout.mesh()) {
    return errors::Internal(
        absl::StrCat("Attempted to relayout to a different "
                     " mesh. Source Mesh = (",
                     src_layout.mesh().ToString(),
                     "). Target Mesh = ", tgt_layout.mesh().ToString(), ")."));
  }
  if (src_layout.rank() != tgt_layout.rank()) {
    return errors::Internal(
        "Attempted to relayout to a different global shape.");
  }

  absl::flat_hash_set<std::string> src_sharding_dims;
  for (int i = 0; i < src_layout.rank(); ++i)
    src_sharding_dims.emplace(src_layout.sharding_spec(i));

  std::vector<ShardingSpec> intermediate_specs_1(src_layout.rank());
  for (int i = 0; i < src_layout.rank(); ++i) {
    if (Layout::IsShardedSpec(tgt_layout.dim(i)) &&
        !Layout::IsShardedSpec(src_layout.dim(i)) &&
        !src_sharding_dims.contains(tgt_layout.sharding_spec(i)))
      intermediate_specs_1[i] = tgt_layout.dim(i);
    else
      intermediate_specs_1[i] = src_layout.dim(i);
  }
  TF_ASSIGN_OR_RETURN(
      Layout intermediate_layout_1,
      Layout::GetLayout(intermediate_specs_1, src_layout.mesh()));

  mlir::OpBuilder builder(input.getContext());
  TF_RETURN_IF_ERROR(SetBuilderInsertionAfterValue(input, builder));

  llvm::SmallPtrSet<mlir::Operation*, 4> local_newly_created_ops;
  TF_ASSIGN_OR_RETURN(mlir::Value split_result,
                      EmitAllScatter(builder, input, src_layout,
                                     intermediate_layout_1, newly_created_ops));

  std::vector<ShardingSpec> intermediate_specs_2(src_layout.rank());
  for (int i = 0; i < src_layout.rank(); ++i) {
    if (Layout::IsShardedSpec(intermediate_specs_1[i]) &&
        intermediate_specs_1[i].sharding_spec() != tgt_layout.sharding_spec(i))
      intermediate_specs_2[i].set_sharding_spec(Layout::kUnshardedDim);
    else
      intermediate_specs_2[i] = intermediate_specs_1[i];
  }
  TF_ASSIGN_OR_RETURN(
      Layout intermediate_layout_2,
      Layout::GetLayout(intermediate_specs_2, src_layout.mesh()));

  TF_ASSIGN_OR_RETURN(
      mlir::Value concat_result,
      EmitAllGather(builder, split_result, intermediate_layout_1,
                    intermediate_layout_2, newly_created_ops));

  auto all_scatter =
      EmitAllScatter(builder, concat_result, intermediate_layout_2, tgt_layout,
                     newly_created_ops);

  if (!is_sparse) return all_scatter;
  if (!all_scatter.ok()) return all_scatter;
  return EmitDenseToSparseToDense(builder, all_scatter.value(),
                                  newly_created_ops);
}

StatusOr<mlir::Operation*> EmitBarrierWithConstValue(mlir::OpBuilder& builder,
                                                     mlir::Location loc,
                                                     const Mesh& mesh,
                                                     int32 value) {
  absl::flat_hash_set<std::string> reduce_dims;
  for (const MeshDimension& mesh_dim : mesh.dims()) {
    reduce_dims.insert(mesh_dim.name);
  }
  return EmitAllReduce(
      builder, Layout::ReplicatedOnMesh(mesh, /*rank=*/1), reduce_dims,
      IntConst(builder, loc, std::vector<int32>{value}).getDefiningOp(),
      kReduceOpAdd);
}

StatusOr<mlir::Operation*> EmitAllReduce(
    mlir::OpBuilder& builder, const dtensor::Layout& output_layout,
    const absl::flat_hash_set<std::string>& reduced_dims,
    mlir::Operation* input, absl::string_view reduce_op) {
  TF_ASSIGN_OR_RETURN(auto partitions, GetAllReducePartitionsFromReducedDims(
                                           output_layout, reduced_dims));
  const int32 num_partitions = partitions.size();

  // If every device lives in its own partition, we don't need to emit a
  // collective.
  if (num_partitions == output_layout.num_devices()) {
    return InferSPMDExpandedLocalShape(input);
  }

  // Construct a flattened list of reduce partitions. This will be converted
  // into a 2-D const tensor for the DTensorAllReduce op.
  std::vector<int32> partitions_flat;
  for (auto& p : partitions) {
    if (p.second.size() != partitions.begin()->second.size()) {
      return errors::InvalidArgument(
          "AllReduce partitions had different sizes -- this is not supported "
          "in MLIR.");
    }
    partitions_flat.insert(partitions_flat.end(), p.second.begin(),
                           p.second.end());
  }

  int32 partition_size = partitions.begin()->second.size();
  auto shaped_type = mlir::RankedTensorType::get(
      {num_partitions, partition_size},
      mlir::IntegerType::get(builder.getContext(), 32));
  auto group_assignment =
      mlir::DenseIntElementsAttr::get(shaped_type, partitions_flat);

  TF_ASSIGN_OR_RETURN(std::string device_type,
                      DeviceTypeFromMesh(output_layout.mesh()));

  mlir::Location loc = DT_LOC2(input->getLoc(), "DTensorAllReduceOp");
  auto all_reduce = builder.create<mlir::TF::DTensorAllReduceOp>(
      loc, input->getResultTypes()[0], input->getOpResult(0),
      builder.create<mlir::TF::ConstOp>(DT_LOC2(loc, "group_assignment"),
                                        group_assignment),
      builder.getStringAttr(std::string(reduce_op)),
      builder.getStringAttr(device_type));
  SetSingleLayoutOnOp(all_reduce, output_layout);
  input->getOpResult(0).replaceAllUsesExcept(
      all_reduce.getResult(),
      llvm::SmallPtrSet<mlir::Operation*, 1>{all_reduce});
  return all_reduce.getOperation();
}

namespace {

// Returns a offset multiplier to calculate device id / mesh coordinate.
int GetMeshDimensionOffsetWithNeighbor(const Mesh& mesh,
                                       const std::string& mesh_dim) {
  const int index = mesh.GetMeshDimIndexWithName(mesh_dim);
  const std::vector<int64_t> mesh_dim_sizes = mesh.dim_sizes();
  int offset = 1;
  for (int i = index + 1; i < mesh_dim_sizes.size(); ++i) {
    offset = offset * mesh_dim_sizes[i];
  }
  return offset;
}

// Returns a mesh coordinate of mesh index with `mesh_dim_name` given
// `device_id`.
StatusOr<int> GetMeshCoordinateIndex(const Mesh& mesh,
                                     const std::string& mesh_dim_name,
                                     int device_id) {
  const int offset = GetMeshDimensionOffsetWithNeighbor(mesh, mesh_dim_name);
  TF_ASSIGN_OR_RETURN(int64_t mesh_dim_size, mesh.dim_size(mesh_dim_name));

  return (device_id / offset) % mesh_dim_size;
}

// Returns a 2D tensor array of size [N, 2] that specifies source target pair
// to be used for halo exchange.
StatusOr<mlir::Value> CreateConstSrcTargetPair(const Mesh& mesh,
                                               const std::string& mesh_dim_name,
                                               bool shift_left,
                                               mlir::Location location,
                                               mlir::OpBuilder& builder) {
  const int mesh_dim_index = mesh.GetMeshDimIndexWithName(mesh_dim_name);
  const std::vector<MeshDimension> mesh_dimensions = mesh.dims();

  llvm::SmallVector<int, 4> src_target_pair_flat;
  src_target_pair_flat.reserve(mesh.local_device_ids().size() * 2);
  for (const int local_device_id : mesh.local_device_ids()) {
    // Calculate the mesh coordinate of the current local device id.
    llvm::SmallVector<int, 4> mesh_coordinate_for_device_id;

    for (const MeshDimension& mesh_dim : mesh_dimensions) {
      TF_ASSIGN_OR_RETURN(
          const int coordinate,
          GetMeshCoordinateIndex(mesh, mesh_dim.name, local_device_id));

      mesh_coordinate_for_device_id.push_back(coordinate);
    }

    // If mesh coordinate is on the left/right edge, then we conduct halo
    // exchange with a processor which executes input block which represent
    // `wrapped around` block.
    const int mesh_coordinate = mesh_coordinate_for_device_id[mesh_dim_index];
    TF_ASSIGN_OR_RETURN(const int dim_size, mesh.dim_size(mesh_dim_name));

    // For tensor requiring halo exchange, we use collective permute.
    const int src_device_id = local_device_id;
    int target_device_id = 0;
    for (const auto& data : llvm::enumerate(mesh_dimensions)) {
      const MeshDimension& mesh_dim = data.value();
      const int index = data.index();

      int target_mesh_coordinate = 1;
      if (mesh_dim.name == mesh_dim_name) {
        target_mesh_coordinate =
            shift_left ? mesh_coordinate - 1 : mesh_coordinate + 1;

        // For processors executing input tensor on the left/right edges, target
        // processor is the processor that executes wrapped around input block.
        if (target_mesh_coordinate < 0 || target_mesh_coordinate >= dim_size)
          target_mesh_coordinate =
              (target_mesh_coordinate + dim_size) % dim_size;

      } else {
        target_mesh_coordinate = mesh_coordinate_for_device_id[index];
      }

      target_device_id +=
          target_mesh_coordinate *
          GetMeshDimensionOffsetWithNeighbor(mesh, mesh_dim.name);
    }
    src_target_pair_flat.push_back(src_device_id);
    src_target_pair_flat.push_back(target_device_id);
  }

  const int num_pairs = src_target_pair_flat.size() / 2;
  auto shaped_type = mlir::RankedTensorType::get(
      {num_pairs, 2}, mlir::IntegerType::get(builder.getContext(), 32));

  auto src_target_attr =
      mlir::DenseIntElementsAttr::get(shaped_type, src_target_pair_flat);
  mlir::Value src_target_pair_tensor =
      builder.create<mlir::TF::ConstOp>(location, src_target_attr);
  return src_target_pair_tensor;
}

}  // namespace

StatusOr<mlir::Value> EmitHaloExchange(mlir::OpBuilder& builder, int halo_size,
                                       const std::string& mesh_dim,
                                       const Layout& layout,
                                       mlir::Value mesh_coordinates,
                                       mlir::tf_device::ClusterOp cluster,
                                       mlir::Location location,
                                       mlir::Value tensor) {
  const Mesh& mesh = layout.mesh();

  // Check mesh dimension requirements for halo exchange.
  if (!mesh.IsMeshDim(mesh_dim))
    return errors::InvalidArgument(
        "Requested halo exchange on unknown mesh dim");

  // TODO(b/261485237): Add support for halo exchange for GPU/CPU.
  if (!mesh.is_tpu_mesh())
    return errors::InvalidArgument("Halo exchange is only supported on TPU.");

  auto input_tensor_type = tensor.getType().dyn_cast<mlir::RankedTensorType>();
  if (!input_tensor_type || !input_tensor_type.hasStaticShape())
    return errors::InvalidArgument(
        "Static shape of input tensor must be known for halo exchange.");

  llvm::ArrayRef<int64_t> input_tensor_shape = input_tensor_type.getShape();
  const std::vector<std::string> sharding_specs = layout.sharding_spec_strs();
  const int split_dim_index = std::distance(
      sharding_specs.begin(), llvm::find(sharding_specs, mesh_dim));

  if (input_tensor_shape[split_dim_index] < halo_size)
    return errors::InvalidArgument(
        "For halo exhange, input shard tensor size of each processor must be "
        "greater than halo size");

  TF_ASSIGN_OR_RETURN(const int mesh_dim_index, mesh.idx_for_dim(mesh_dim));

  TF_ASSIGN_OR_RETURN(mlir::Value scalar_mesh_coordinate,
                      SelectScalarValueFromArray(builder, mesh_dim_index,
                                                 location, mesh_coordinates));

  llvm::SmallVector<int64_t, 4> halo_exchange_tensor_shape;
  for (const auto& size_and_index : llvm::enumerate(input_tensor_shape)) {
    const int index = size_and_index.index();
    const int size = size_and_index.value();
    halo_exchange_tensor_shape.push_back(index == split_dim_index ? halo_size
                                                                  : size);
  }

  // Find the halo tensor value to pad on the `left` side. Note that halo
  // exchange can happen on top/bottom/left/right sides of a spatially
  // partitioned tensor. However, we use `left`/`right` as the
  // direction is implicit based on mesh dimension.
  //
  // For example, if mesh dimension splits the input tensor by its height
  // dimension, then `left` actually means tensor to pad on the top side.
  mlir::Value is_on_left_edge = builder.create<mlir::TF::EqualOp>(
      location, CreateIntScalarConst(0, builder, location, /*use_int64=*/false),
      scalar_mesh_coordinate, builder.getBoolAttr(true));

  TF_ASSIGN_OR_RETURN(const int mesh_dim_size, mesh.dim_size(mesh_dim));
  mlir::Value is_on_right_edge = builder.create<mlir::TF::EqualOp>(
      location,
      CreateIntScalarConst(mesh_dim_size - 1, builder, location,
                           /*use_int64=*/false),
      scalar_mesh_coordinate, builder.getBoolAttr(true));

  // Create zero ghost tensor to pad on left side.
  mlir::RankedTensorType halo_tensor_type = mlir::RankedTensorType::get(
      halo_exchange_tensor_shape, input_tensor_type.getElementType());
  auto halo_type = mlir::RankedTensorType::get(
      halo_tensor_type.getShape(), input_tensor_type.getElementType());

  mlir::Attribute const_attr;
  if (halo_type.getElementType().isIntOrIndex()) {
    const_attr =
        mlir::DenseIntElementsAttr::get(halo_type, llvm::SmallVector<int>{0});
  } else {
    const_attr =
        mlir::DenseFPElementsAttr::get(halo_type, llvm::SmallVector<float>{0});
  }

  mlir::Value ghost_tensor_left =
      builder.create<mlir::TF::ConstOp>(location, const_attr).getResult();

  // Get the right side slice of the input tensor to pad on left side.
  llvm::SmallVector<int64_t, 4> begin_left(layout.rank(), 0);
  begin_left[split_dim_index] = input_tensor_shape[split_dim_index] - halo_size;
  mlir::Value begin_tensor_left =
      ops_util::GetR1Const(begin_left, builder, location);

  llvm::SmallVector<int64_t, 4> size(input_tensor_shape.begin(),
                                     input_tensor_shape.end());
  size[split_dim_index] = halo_size;

  mlir::Value size_tensor_left = ops_util::GetR1Const(size, builder, location);
  mlir::Value sliced_tensor_left = builder.create<mlir::TF::SliceOp>(
      location, halo_type, tensor, begin_tensor_left, size_tensor_left);

  mlir::Value halo_tensor_left = builder.create<mlir::TF::SelectV2Op>(
      location, is_on_right_edge, ghost_tensor_left, sliced_tensor_left);

  // Invoke collective permute to receive the tensor from neighboring processor.
  // Halo slices from the left neighbor are received on each processor (they
  // are shifted right).
  TF_ASSIGN_OR_RETURN(
      mlir::Value src_target_pair_left,
      CreateConstSrcTargetPair(mesh, mesh_dim, /*shift_left=*/false, location,
                               builder));

  mlir::Value left_concat_value = builder.create<mlir::TF::CollectivePermuteOp>(
      location, sliced_tensor_left.getType(), halo_tensor_left,
      src_target_pair_left);

  mlir::Value ghost_tensor_right =
      builder.create<mlir::TF::ConstOp>(location, const_attr).getResult();

  // Else, values to pad is tensor from different processor. We use collective
  // permute to access tensor slice from another device.
  // Get the left side slice of the input tensor.
  llvm::SmallVector<int64_t, 4> begin_right(layout.rank(), 0);
  mlir::Value begin_tensor_right =
      ops_util::GetR1Const(begin_right, builder, location);
  mlir::Value size_tensor_right = ops_util::GetR1Const(size, builder, location);
  mlir::Value sliced_tensor_right = builder.create<mlir::TF::SliceOp>(
      location, halo_type, tensor, begin_tensor_right, size_tensor_right);

  // Find the halo tensor value to pad on the `right` side.
  // If input block is on the right edge, we use zero ghost tensor instead.
  mlir::Value halo_tensor_right = builder.create<mlir::TF::SelectV2Op>(
      location, is_on_left_edge, ghost_tensor_right, sliced_tensor_right);

  // Invoke collective permute to receive the tensor from neighboring processor.
  // Halo slices from the right neighbor are received on each processor (they
  // are shifted left).
  TF_ASSIGN_OR_RETURN(
      mlir::Value src_target_pair_right,
      CreateConstSrcTargetPair(mesh, mesh_dim, /*shift_left=*/true, location,
                               builder));
  mlir::Value right_concat_value =
      builder.create<mlir::TF::CollectivePermuteOp>(
          location, sliced_tensor_right.getType(), halo_tensor_right,
          src_target_pair_right);

  // Final halo exchanged value is concatenated value of left_concat_value,
  // tensor, and right_concat_value in the mesh_dimension.
  llvm::SmallVector<int64_t, 4> final_shape(input_tensor_shape.begin(),
                                            input_tensor_shape.end());
  final_shape[split_dim_index] = final_shape[split_dim_index] + 2 * halo_size;

  auto final_type = mlir::RankedTensorType::get(
      final_shape, input_tensor_type.getElementType());
  mlir::Value concat_axis =
      CreateIntScalarConst(split_dim_index, builder, location);
  mlir::Value final_value = builder.create<mlir::TF::ConcatV2Op>(
      location, final_type,
      llvm::SmallVector<mlir::Value, 4>{left_concat_value, tensor,
                                        right_concat_value},
      concat_axis);

  return final_value;
}

}  // namespace dtensor
}  // namespace tensorflow
