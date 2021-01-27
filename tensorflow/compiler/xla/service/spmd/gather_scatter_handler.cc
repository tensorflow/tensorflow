/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {
namespace spmd {

namespace {

// Returns whether partitioning in the operand only happens in dimensions with
// gather/scatter slice size 1.
bool GatherScatterOperandPartitionedOnlyOnTrivialSliceDims(
    const PartitionedHlo& operand, absl::Span<const int64> index_map,
    absl::Span<const int64> slice_size) {
  if (operand.sharding().IsTileMaximal()) {
    return false;
  }
  int64 trivial_slice_dims_partitions = 1;
  for (int64 dim : index_map) {
    if (slice_size[dim] == 1) {
      trivial_slice_dims_partitions *=
          operand.sharding().tile_assignment().dim(dim);
    }
  }
  return trivial_slice_dims_partitions == operand.sharding().NumTiles();
}

// Returns the min and max for the indices (replicated) in a scatter/gather
// which has the operand partitioned on trivial slice dimensions (slice size 1).
std::pair<HloInstruction*, HloInstruction*>
IndexBoundsForGatherScatterOperandPartitionedOnTrivialSliceDims(
    const PartitionedHlo& operand, const PartitionedHlo& replicated_indices,
    HloInstruction* partition_id, absl::Span<const int64> index_map,
    int64 index_vector_dim, SpmdBuilder* b) {
  auto operand_offsets = MakePartitionOffsets(
      operand.base_shape(), operand.sharding(), partition_id, b);
  // Find the per-dimension index bounds.
  std::vector<HloInstruction*> min_indices;
  std::vector<HloInstruction*> max_indices;
  for (int64 i = 0; i < index_map.size(); ++i) {
    int64 dim = index_map[i];
    int64 partitions = operand.sharding().tile_assignment().dim(dim);
    if (partitions == 1) {
      min_indices.push_back(CreateR0WithType<int32>(
          replicated_indices.base_shape().element_type(), 0, b));
      max_indices.push_back(CreateR0WithType<int32>(
          replicated_indices.base_shape().element_type(),
          operand.base_shape().dimensions(dim), b));
      continue;
    }
    auto offset = operand_offsets[dim];
    if (offset->shape().element_type() !=
        replicated_indices.base_shape().element_type()) {
      offset = b->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::MakeShape(replicated_indices.base_shape().element_type(),
                               {}),
          offset));
    }
    min_indices.push_back(offset);
    auto partition_size_minus_1 =
        CreateR0WithType<int32>(replicated_indices.base_shape().element_type(),
                                operand.hlo()->shape().dimensions(dim) - 1, b);
    max_indices.push_back(b->AddInstruction(HloInstruction::CreateBinary(
        offset->shape(), HloOpcode::kAdd, offset, partition_size_minus_1)));
  }
  // Broadcast the index bounds to the same shape as the indices.
  HloInstruction* broadcast_min;
  HloInstruction* broadcast_max;
  if (index_vector_dim < replicated_indices.base_shape().rank()) {
    // The index vector is an R1, we need to reshape individual bounds to
    // [1], and concat them if there are more than one.
    for (int64 i = 0; i < min_indices.size(); ++i) {
      min_indices[i] = b->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(min_indices[i]->shape().element_type(), {1}),
          min_indices[i]));
      max_indices[i] = b->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(max_indices[i]->shape().element_type(), {1}),
          max_indices[i]));
    }
    int64 slice_dims = max_indices.size();
    if (slice_dims > 1) {
      min_indices[0] = b->AddInstruction(HloInstruction::CreateConcatenate(
          ShapeUtil::MakeShape(min_indices[0]->shape().element_type(),
                               {slice_dims}),
          min_indices, 0));
      max_indices[0] = b->AddInstruction(HloInstruction::CreateConcatenate(
          min_indices[0]->shape(), max_indices, 0));
    }
    broadcast_min = b->AddInstruction(HloInstruction::CreateBroadcast(
        replicated_indices.base_shape(), min_indices[0], {index_vector_dim}));
    broadcast_max = b->AddInstruction(HloInstruction::CreateBroadcast(
        replicated_indices.base_shape(), max_indices[0], {index_vector_dim}));
  } else {
    CHECK_EQ(max_indices.size(), 1);
    broadcast_min = b->AddInstruction(HloInstruction::CreateBroadcast(
        replicated_indices.base_shape(), min_indices[0], {}));
    broadcast_max = b->AddInstruction(HloInstruction::CreateBroadcast(
        replicated_indices.base_shape(), max_indices[0], {}));
  }
  return {broadcast_min, broadcast_max};
}

}  // namespace

Status SpmdPartitioningVisitor::HandleScatter(HloInstruction* hlo) {
  auto scatter = Cast<HloScatterInstruction>(hlo);
  auto dnums = scatter->scatter_dimension_numbers();
  auto operand = GetPartitionedHlo(scatter->operand(0));
  auto indices = GetPartitionedHlo(scatter->operand(1));
  auto updates = GetPartitionedHlo(scatter->operand(2));
  std::vector<int64> slice_size(operand.base_shape().rank(), 1);
  int64 num_update_window_dims = 0;
  for (int64 i = 0; i < operand.base_shape().rank(); ++i) {
    if (absl::c_linear_search(dnums.inserted_window_dims(), i)) {
      continue;
    }
    slice_size[i] = updates.base_shape().dimensions(
        dnums.update_window_dims(num_update_window_dims++));
  }
  std::vector<int64> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  std::vector<int64> update_scatter_dims;
  for (int64 i = 0; i < updates.base_shape().rank(); ++i) {
    if (!absl::c_linear_search(dnums.update_window_dims(), i)) {
      update_scatter_dims.push_back(i);
    }
  }
  if (operand.sharding().IsTileMaximal()) {
    if (!indices.sharding().IsTileMaximal() &&
        (dnums.index_vector_dim() == indices.base_shape().rank() ||
         indices.sharding().tile_assignment().dim(dnums.index_vector_dim()) ==
             1)) {
      auto reduction_opcode = ParseReductionComputation(scatter->to_apply());
      if (!reduction_opcode.has_value()) {
        return DefaultAction(hlo);
      }
      HloInstruction* identity;
      switch (*reduction_opcode) {
        case HloOpcode::kAdd:
        case HloOpcode::kOr:
          identity = CreateZero(operand.hlo()->shape(), &b_);
          break;
        case HloOpcode::kMultiply:
        case HloOpcode::kAnd:
          identity = CreateOne(operand.hlo()->shape(), &b_);
          break;
        case HloOpcode::kMinimum:
          identity = CreateConstant(
              operand.hlo()->shape(),
              LiteralUtil::MaxValue(hlo->shape().element_type()), &b_);
          break;
        case HloOpcode::kMaximum:
          identity = CreateConstant(
              operand.hlo()->shape(),
              LiteralUtil::MinValue(hlo->shape().element_type()), &b_);
          break;
        default:
          return DefaultAction(hlo);
      }
      std::vector<int64> update_dim_to_index_dim(updates.base_shape().rank(),
                                                 -1);
      std::vector<int64> index_dim_to_update_dim(indices.base_shape().rank(),
                                                 -1);
      for (int64 i = 0; i < update_scatter_dims.size(); ++i) {
        int64 indices_scatter_dim = i < dnums.index_vector_dim() ? i : i + 1;
        update_dim_to_index_dim[update_scatter_dims[i]] = indices_scatter_dim;
        index_dim_to_update_dim[indices_scatter_dim] = update_scatter_dims[i];
      }
      auto new_updates_sharding =
          hlo_sharding_util::TransposeShardingWithCollapsedDims(
              indices.sharding(), index_dim_to_update_dim,
              update_dim_to_index_dim);
      CHECK(new_updates_sharding.has_value());
      updates = updates.Reshard(*new_updates_sharding);
      // Update collective_ops_creator and partition_id for partial replicate.
      auto collective_ops_creator = collective_ops_creator_;
      auto partition_id = partition_id_;
      if (indices.sharding().ReplicateOnLastTileDim()) {
        auto sharding_grouped = GroupShardingOnDims(
            indices.sharding(),
            {indices.sharding().tile_assignment().num_dimensions() - 1});
        auto per_group_partitioner_state = CreatePerGroupPartitioningState(
            indices.state(), sharding_grouped.device_groups, &b_);
        collective_ops_creator =
            per_group_partitioner_state.collective_ops_creator;
        partition_id = per_group_partitioner_state.partition_id;
      }
      // To avoid accumulating the initial operand multiple times during
      // all-reduce, we use identity operands for all non-zero partitions.
      auto not_partition_zero = b_.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::MakeScalarShape(PRED), partition_id));
      not_partition_zero = b_.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::ChangeElementType(identity->shape(), PRED),
          not_partition_zero, {}));
      auto select_operand =
          b_.AddInstruction(HloInstruction::HloInstruction::CreateTernary(
              identity->shape(), HloOpcode::kSelect, not_partition_zero,
              identity, operand.Replicate().hlo()));
      auto pscatter = b_.AddInstruction(scatter->CloneWithNewOperands(
          scatter->shape(), {select_operand, indices.hlo(), updates.hlo()}));
      auto all_reduce =
          collective_ops_creator.create_cross_partition_all_reduce(
              &b_, pscatter, scatter->to_apply(), {}, NewChannel());
      all_reduce->set_sharding(HloSharding::Replicate());
      SetPartitionedHlo(hlo, [&]() {
        return PartitionedHlo(all_reduce, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
  } else {
    auto maybe_passthrough = hlo_sharding_util::ScatterUpdateShardingFromOutput(
        operand.sharding(), *hlo);
    // Handle pass through cases if we can use compatible sharding for update.
    if (maybe_passthrough.has_value()) {
      indices = indices.Reshard(HloSharding::Replicate());
      updates = updates.Reshard(*maybe_passthrough);
      auto pscatter = b_.AddInstruction(HloInstruction::CreateScatter(
          operand.hlo()->shape(), operand.hlo(), indices.hlo(), updates.hlo(),
          scatter->to_apply(), dnums, scatter->indices_are_sorted(),
          scatter->unique_indices()));
      pscatter->set_sharding(*maybe_passthrough);
      SetPartitionedHlo(hlo, [&]() {
        return PartitionedHlo(pscatter, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
    if (GatherScatterOperandPartitionedOnlyOnTrivialSliceDims(
            operand, scatter_dims_to_operand_dims, slice_size) &&
        ShapeSizeInBytes(updates.base_shape()) <
            ShapeSizeInBytes(scatter->shape())) {
      // Operand is sharded on trivial slice dims (update slice size 1). We can
      // adjust the indices on each partition by subtracting the offsets. Then
      // we execute a scatter on full updated indices, and out-of-bound accesses
      // will have no effect on the result as guaranteed by the scatter
      // semantics.
      indices = indices.Reshard(HloSharding::Replicate());
      updates = updates.Reshard(HloSharding::Replicate());
      HloInstruction* indices_min;
      HloInstruction* indices_max_unused;
      std::tie(indices_min, indices_max_unused) =
          IndexBoundsForGatherScatterOperandPartitionedOnTrivialSliceDims(
              operand, indices, partition_id_, scatter_dims_to_operand_dims,
              dnums.index_vector_dim(), &b_);
      auto adjusted_indices = b_.AddInstruction(HloInstruction::CreateBinary(
          indices.hlo()->shape(), HloOpcode::kSubtract, indices.hlo(),
          indices_min));
      auto pscatter = b_.AddInstruction(HloInstruction::CreateScatter(
          operand.hlo()->shape(), operand.hlo(), adjusted_indices,
          updates.hlo(), scatter->to_apply(), dnums,
          scatter->indices_are_sorted(), scatter->unique_indices()));
      pscatter->set_sharding(operand.sharding());
      SetPartitionedHlo(hlo, [&]() {
        return PartitionedHlo(pscatter, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
  }
  return DefaultAction(hlo);
}

Status SpmdPartitioningVisitor::HandleGather(HloInstruction* hlo) {
  auto gather = Cast<HloGatherInstruction>(hlo);
  const auto& dnums = gather->gather_dimension_numbers();
  auto operand = GetPartitionedHlo(gather->operand(0));
  auto indices = GetPartitionedHlo(gather->operand(1));
  std::vector<int64> start_index_map(dnums.start_index_map().begin(),
                                     dnums.start_index_map().end());
  std::vector<int64> batch_dims;
  for (int64 i = 0; i < gather->shape().rank(); ++i) {
    if (!absl::c_linear_search(dnums.offset_dims(), i)) {
      batch_dims.push_back(i);
    }
  }
  // Check if we identify some of the dimensions of the gather as parallel and
  // if we have sharded the operand and indices across those dimensions.
  // If that's the case then we can partition the gather across such dimensions
  // by adjusting the offsets.
  if (absl::optional<hlo_sharding_util::GatherParallelDims> parallel_dims =
          hlo_sharding_util::GetGatherBatchParallelDims(*hlo)) {
    if (auto gather_sharding = GatherOperandsShardedAcrossParallelDims(
            *operand.hlo(), *indices.hlo(), *parallel_dims)) {
      auto indices_parallel_dims = parallel_dims->indices_parallel_dims;
      auto operand_parallel_dims = parallel_dims->operand_parallel_dims;
      auto output_parallel_dims =
          hlo_sharding_util::GatherParallelOutputDims(*hlo, *parallel_dims);
      HloSharding indices_sharding = gather_sharding->indices_sharding;
      HloSharding operand_sharding = gather_sharding->operand_sharding;
      if (operand_sharding.NumTiles() ==
              operand_sharding.NumTiles(operand_parallel_dims) &&
          indices_sharding.NumTiles() ==
              indices_sharding.NumTiles(indices_parallel_dims)) {
        int index_dim = dnums.index_vector_dim();
        // Construct the required sharding for the new gather we are gonna form.
        absl::InlinedVector<int64, 4> output_tiling(
            hlo->shape().dimensions_size(), 1);
        for (int i = 0, num_output_parallel_dims = output_parallel_dims.size();
             i < num_output_parallel_dims; ++i) {
          int output_idx = output_parallel_dims[i];
          int indices_idx = indices_parallel_dims[i];
          output_tiling[output_idx] =
              indices_sharding.tile_assignment().dim(indices_idx);
        }
        operand = operand.Reshard(operand_sharding);
        indices = indices.Reshard(indices_sharding);
        if (indices_sharding.ReplicateOnLastTileDim()) {
          output_tiling.push_back(
              indices_sharding.tile_assignment().dimensions().back());
        }
        Array<int64> output_tile_assignment =
            indices_sharding.tile_assignment();
        output_tile_assignment.Reshape(output_tiling);
        // New gather tiling.
        HloSharding output_sharding =
            indices_sharding.ReplicateOnLastTileDim()
                ? HloSharding::PartialTile(output_tile_assignment)
                : HloSharding::Tile(output_tile_assignment);
        // Shape of the partitioned gather
        Shape pshape = MakePartitionedShape(gather->shape(), output_sharding);
        // Construct the offsets for the operand sharding to be used to adjust
        // the indices. Because we know the only dimensions partitioned are the
        // parallel ones and because the partitioning is the same across indices
        // and operands we can apply the offsets on the operands on the indices.
        std::vector<HloInstruction*> operand_offsets = MakePartitionOffsets(
            operand.base_shape(), operand_sharding, partition_id_, &b_);
        absl::InlinedVector<HloInstruction*, 4> index_offsets;
        for (int start_idx = 0; start_idx < dnums.start_index_map_size();
             ++start_idx) {
          HloInstruction* index_offset =
              indices.base_shape().dimensions_size() > index_dim
                  ? b_.AddInstruction(HloInstruction::CreateReshape(
                        ShapeUtil::MakeShape(S32, {1}),
                        operand_offsets[dnums.start_index_map(start_idx)]))
                  : operand_offsets[dnums.start_index_map(start_idx)];
          index_offsets.push_back(index_offset);
        }
        HloInstruction* adjusted_indices = nullptr;
        if (indices.base_shape().dimensions_size() > index_dim) {
          // Concatenate the offsets for the parallel dimensions to subtract.
          adjusted_indices =
              b_.AddInstruction(HloInstruction::CreateConcatenate(
                  ShapeUtil::MakeShape(
                      S32, {indices.base_shape().dimensions(index_dim)}),
                  index_offsets, 0));
        } else {
          CHECK_EQ(index_offsets.size(), 1);
          adjusted_indices = index_offsets[0];
        }
        if (indices.hlo()->shape().element_type() != PrimitiveType::S32) {
          adjusted_indices = b_.AddInstruction(HloInstruction::CreateConvert(
              ShapeUtil::ChangeElementType(
                  adjusted_indices->shape(),
                  indices.hlo()->shape().element_type()),
              adjusted_indices));
        }
        if (adjusted_indices->shape().rank() == 0) {
          adjusted_indices = b_.AddInstruction(HloInstruction::CreateBroadcast(
              indices.hlo()->shape(), adjusted_indices, {}));
        } else {
          adjusted_indices = b_.AddInstruction(HloInstruction::CreateBroadcast(
              indices.hlo()->shape(), adjusted_indices, {index_dim}));
        }
        // Adjust indices by subtracting the offsets based on the partition id.
        adjusted_indices = b_.AddInstruction(HloInstruction::CreateBinary(
            indices.hlo()->shape(), HloOpcode::kSubtract, indices.hlo(),
            adjusted_indices));
        HloInstruction* pgather =
            b_.AddInstruction(HloInstruction::CreateGather(
                pshape, operand.hlo(), adjusted_indices, dnums,
                gather->gather_slice_sizes(), gather->indices_are_sorted()));
        pgather->set_sharding(output_sharding);
        SetPartitionedHlo(hlo, [&]() {
          return PartitionedHlo(pgather, hlo->shape(), MakePartitioningState())
              .Reshard(hlo->sharding())
              .hlo();
        });
        return Status::OK();
      }
    }
  }
  if (operand.sharding().IsTileMaximal()) {
    if (!indices.sharding().IsTileMaximal() &&
        (dnums.index_vector_dim() == indices.base_shape().rank() ||
         indices.sharding().tile_assignment().dim(dnums.index_vector_dim()) ==
             1)) {
      auto replicated_operand = operand.Replicate();
      TF_ASSIGN_OR_RETURN(
          Shape partitioned_output_shape,
          ShapeInference::InferGatherShape(replicated_operand.hlo()->shape(),
                                           indices.hlo()->shape(), dnums,
                                           gather->gather_slice_sizes()));
      auto pgather = b_.AddInstruction(gather->CloneWithNewOperands(
          partitioned_output_shape, {replicated_operand.hlo(), indices.hlo()}));
      std::vector<int64> output_dim_to_index_dim(pgather->shape().rank(), -1);
      std::vector<int64> index_dim_to_output_dim(indices.base_shape().rank(),
                                                 -1);
      for (int64 i = 0; i < batch_dims.size(); ++i) {
        int64 indices_batch_dim = i < dnums.index_vector_dim() ? i : i + 1;
        output_dim_to_index_dim[batch_dims[i]] = indices_batch_dim;
        index_dim_to_output_dim[indices_batch_dim] = batch_dims[i];
      }
      auto pgather_sharding =
          hlo_sharding_util::TransposeShardingWithCollapsedDims(
              indices.sharding(), index_dim_to_output_dim,
              output_dim_to_index_dim);
      CHECK(pgather_sharding.has_value());
      pgather->set_sharding(*pgather_sharding);
      SetPartitionedHlo(hlo, [&]() {
        return PartitionedHlo(pgather, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
  } else {
    auto maybe_passthrough =
        hlo_sharding_util::GatherOutputShardingFromDataOperand(
            operand.sharding(), *hlo);
    if (maybe_passthrough.has_value()) {
      indices = indices.Reshard(HloSharding::Replicate());
      auto pshape = MakePartitionedShape(gather->shape(), *maybe_passthrough);
      std::vector<int64> pslice_sizes(gather->gather_slice_sizes().begin(),
                                      gather->gather_slice_sizes().end());
      for (int64 i = 0; i < pslice_sizes.size(); ++i) {
        if (operand.sharding().tile_assignment().dim(i) > 1) {
          pslice_sizes[i] = operand.hlo()->shape().dimensions(i);
        }
      }
      auto pgather = b_.AddInstruction(HloInstruction::CreateGather(
          pshape, operand.hlo(), indices.hlo(), dnums, pslice_sizes,
          gather->indices_are_sorted()));
      pgather->set_sharding(*maybe_passthrough);
      SetPartitionedHlo(hlo, [&]() {
        return PartitionedHlo(pgather, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
    if (GatherScatterOperandPartitionedOnlyOnTrivialSliceDims(
            operand, start_index_map, gather->gather_slice_sizes()) &&
        ShapeSizeInBytes(gather->shape()) <
            ShapeSizeInBytes(gather->operand(0)->shape())) {
      indices = indices.Reshard(HloSharding::Replicate());
      // Now the operand is partitioned in trivial slice dimensions, and the
      // indices are replicated. We execute a gather on partitioned operand,
      // with full number of indices, where out-of-bounds indices are clamped,
      // and masked out with 0 in the result; then we use all-reduce to combine
      // results. Although gather will not get faster, we avoided the need to
      // replicate the operand.
      HloInstruction* indices_min;
      HloInstruction* indices_max;
      std::tie(indices_min, indices_max) =
          IndexBoundsForGatherScatterOperandPartitionedOnTrivialSliceDims(
              operand, indices, partition_id_, start_index_map,
              dnums.index_vector_dim(), &b_);
      // Clamp the indices.
      auto adjusted_indices = b_.AddInstruction(HloInstruction::CreateTernary(
          indices.base_shape(), HloOpcode::kClamp, indices_min, indices.hlo(),
          indices_max));
      // Adjust the indices by subtracting the offset.
      adjusted_indices = b_.AddInstruction(HloInstruction::CreateBinary(
          indices.base_shape(), HloOpcode::kSubtract, adjusted_indices,
          indices_min));
      // Gather on adjusted indices.
      auto pgather = b_.AddInstruction(HloInstruction::CreateGather(
          gather->shape(), operand.hlo(), adjusted_indices, dnums,
          gather->gather_slice_sizes(), gather->indices_are_sorted()));
      // Mask out invalid results.
      auto filter = b_.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::ChangeElementType(indices.base_shape(), PRED),
          indices.hlo(), indices_min, ComparisonDirection::kLt));
      filter = b_.AddInstruction(HloInstruction::CreateBinary(
          filter->shape(), HloOpcode::kOr, filter,
          b_.AddInstruction(HloInstruction::CreateCompare(
              ShapeUtil::ChangeElementType(indices.base_shape(), PRED),
              indices.hlo(), indices_max, ComparisonDirection::kGt))));
      if (dnums.index_vector_dim() < indices.base_shape().rank()) {
        std::vector<int64> reduced_filter_dims;
        for (int64 i = 0; i < filter->shape().rank(); ++i) {
          if (i != dnums.index_vector_dim()) {
            reduced_filter_dims.push_back(filter->shape().dimensions(i));
          }
        }
        filter = b_.AddInstruction(HloInstruction::CreateReduce(
            ShapeUtil::MakeShape(PRED, reduced_filter_dims), filter,
            CreateR0WithType(PRED, false, &b_), {dnums.index_vector_dim()},
            MakeBinaryAdd(PRED, module_)));
      }
      std::vector<int64> batch_dims;
      for (int64 i = 0; i < pgather->shape().rank(); ++i) {
        if (!absl::c_linear_search(dnums.offset_dims(), i)) {
          batch_dims.push_back(i);
        }
      }
      auto broadcast_filter = b_.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::ChangeElementType(pgather->shape(), PRED), filter,
          batch_dims));
      auto filtered = b_.AddInstruction(HloInstruction::CreateTernary(
          pgather->shape(), HloOpcode::kSelect, broadcast_filter,
          CreateZero(pgather->shape(), &b_), pgather));
      // Combine from different partitions.
      auto collective_ops_creator = collective_ops_creator_;
      if (operand.sharding().ReplicateOnLastTileDim()) {
        auto sharding_grouped = GroupShardingOnDims(
            operand.sharding(),
            {operand.sharding().tile_assignment().num_dimensions() - 1});
        auto per_group_partitioner_state = CreatePerGroupPartitioningState(
            operand.state(), sharding_grouped.device_groups, &b_);
        collective_ops_creator =
            per_group_partitioner_state.collective_ops_creator;
      }
      auto ar = collective_ops_creator.create_cross_partition_all_reduce(
          &b_, filtered,
          MakeBinaryAdd(filtered->shape().element_type(), module_), {},
          NewChannel());
      ar->set_sharding(HloSharding::Replicate());
      SetPartitionedHlo(hlo, [&]() {
        return PartitionedHlo(ar, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
  }
  return DefaultAction(hlo);
}

}  // namespace spmd
}  // namespace xla
