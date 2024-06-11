/* Copyright 2020 The OpenXLA Authors.

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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/literal_util.h"
#include "xla/service/shape_inference.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace spmd {

namespace {

using hlo_sharding_util::GroupedSharding;

// Generates per-group partitioned hlo based on given grouped sharding.
PartitionedHlo PerGroupPartitionedHlo(
    PartitionedHlo& phlo, const GroupedSharding& grouped_sharding,
    SpmdBuilder* b, absl::InlinedVector<std::function<void()>, 3>& clean_ups) {
  //  Make sure the shardings are in consistent state.
  phlo = phlo.Reshard(UngroupSharding(grouped_sharding));
  auto per_group_partitioner_state = CreatePerGroupPartitioningState(
      phlo.state(), grouped_sharding.device_groups, b);
  // Create per-group partitioned hlo, and restore to old sharding of the
  // underlying hlo after done using the per-group partitioned hlo, since it
  // might be used elsewhere.
  const HloSharding old_sharding = phlo.hlo()->sharding();
  HloInstruction* hlo = phlo.hlo();
  phlo.hlo()->set_sharding(grouped_sharding.sharding);
  clean_ups.push_back(
      [old_sharding, hlo]() { hlo->set_sharding(old_sharding); });
  return PartitionedHlo(
      phlo.hlo(), GetPerGroupBaseShape(grouped_sharding, phlo.base_shape()),
      per_group_partitioner_state);
}

// Helper to get multiple per-group partitioned hlos.
std::vector<PartitionedHlo> PerGroupPartitionedHlos(
    std::vector<PartitionedHlo>& phlos, const GroupedSharding& grouped_sharding,
    SpmdBuilder* b, absl::InlinedVector<std::function<void()>, 3>& clean_ups) {
  // Cache per-group partitioned hlos to avoid group-partitioning it more than
  // once.
  absl::flat_hash_map<HloInstruction*, PartitionedHlo> cached_per_group_hlos;
  std::vector<HloInstruction*> hlos;
  absl::c_transform(phlos, std::back_inserter(hlos),
                    [&](PartitionedHlo phlo) { return phlo.hlo(); });

  std::vector<PartitionedHlo> per_group_phlos;
  for (int i = 0; i != hlos.size(); ++i) {
    if (!cached_per_group_hlos.contains(hlos[i])) {
      cached_per_group_hlos.emplace(std::make_pair(
          hlos[i],
          PerGroupPartitionedHlo(phlos[i], grouped_sharding, b, clean_ups)));
    }
    per_group_phlos.push_back(cached_per_group_hlos.at(hlos[i]));
  }
  return per_group_phlos;
}

// Returns whether partitioning in the operand only happens in dimensions with
// gather/scatter slice size 1.
std::optional<std::vector<int64_t>>
GatherScatterOperandPartitionedOnTrivialSliceDims(
    const PartitionedHlo& operand, absl::Span<const int64_t> index_map,
    absl::Span<const int64_t> slice_size) {
  if (operand.sharding().IsTileMaximal()) {
    return std::nullopt;
  }
  std::vector<int64_t> slice_dims;
  int64_t trivial_slice_dims_partitions = 1;
  for (int64_t dim : index_map) {
    if (slice_size[dim] == 1) {
      trivial_slice_dims_partitions *=
          operand.sharding().tile_assignment().dim(dim);
      slice_dims.push_back(dim);
    }
  }
  if (trivial_slice_dims_partitions != 1) {
    return slice_dims;
  }
  return std::nullopt;
}

// Priority for operand dimensions in gather from lowest to highest, in case of
// needing to replicate a dimension, start from the lowest first.
std::vector<int64_t> GatherOperandDimsByPriority(
    const PartitionedHlo& operand, const HloGatherInstruction* gather,
    absl::Span<const int64_t> slice_sizes) {
  const GatherDimensionNumbers& dnums = gather->gather_dimension_numbers();
  std::vector<int64_t> priority_dims_for_operand;
  if (const std::optional<std::vector<int64_t>> trivial_slice_dims =
          GatherScatterOperandPartitionedOnTrivialSliceDims(
              operand, dnums.start_index_map(), slice_sizes)) {
    absl::c_copy(trivial_slice_dims.value(),
                 std::back_inserter(priority_dims_for_operand));
  }
  const auto operand_passthrough_dims =
      hlo_sharding_util::GetGatherOperandPassthroughOperandDims(
          operand.base_shape(), *gather, slice_sizes);
  absl::c_copy(operand_passthrough_dims,
               std::back_inserter(priority_dims_for_operand));
  return priority_dims_for_operand;
}

// Priority for index dimensions in gather from lowest to highest, in case of
// needing to replicate a dimension, start from the lowest first.
std::vector<int64_t> GatherIndexDimsByPriority(
    const PartitionedHlo& indices, const HloGatherInstruction* gather) {
  const GatherDimensionNumbers& dnums = gather->gather_dimension_numbers();

  std::vector<int64_t> priority_dims_for_indices;
  if (dnums.index_vector_dim() < indices.rank()) {
    priority_dims_for_indices.push_back(dnums.index_vector_dim());
  }
  absl::InlinedVector<int64_t, 1> index_passthrough_dims =
      hlo_sharding_util::GetGatherScatterIndexPassthroughIndexDims(
          indices.rank(), dnums.index_vector_dim());
  absl::c_copy(index_passthrough_dims,
               std::back_inserter(priority_dims_for_indices));
  return priority_dims_for_indices;
}

// Priority for output dimensions in gather from lowest to highest, in case of
// needing to replicate a dimension, start from the lowest first.
std::vector<int64_t> GatherOutputDimsByPriority(
    const Shape& output_shape, const PartitionedHlo& operand,
    const HloGatherInstruction* gather, absl::Span<const int64_t> slice_sizes) {
  // Try preserve operand passthrough dims as typically operand is larger in
  // size than indices, and preserving operand pass-through dims avoid more
  // costly reshard in operand passthrough case.
  // TODO(b/303295109): Use better cost-model based decision to determine the
  // priority ordering for scatter update.
  std::vector<int64_t> priority_dims_for_output;
  auto operand_passthrough_output_dims =
      hlo_sharding_util::GetGatherOperandPassthroughOutputDims(
          output_shape, operand.base_shape(), *gather, slice_sizes);
  for (int i = 0; i != output_shape.rank(); ++i) {
    if (!absl::c_linear_search(operand_passthrough_output_dims, i)) {
      priority_dims_for_output.push_back(i);
    }
  }
  absl::c_copy(operand_passthrough_output_dims,
               std::back_inserter(priority_dims_for_output));
  return priority_dims_for_output;
}

// Returns the min and max for the indices in a scatter/gather which has the
// operand partitioned on trivial slice dimensions (slice size 1).
std::pair<HloInstruction*, HloInstruction*>
IndexBoundsForGatherScatterOperandPartitionedOnTrivialSliceDims(
    const PartitionedHlo& operand, const PartitionedHlo& indices,
    HloInstruction* partition_id, absl::Span<const int64_t> index_map,
    absl::Span<const int64_t> trivial_slice_dims, int64_t index_vector_dim,
    SpmdBuilder* b) {
  auto operand_offsets = MakePartitionOffsets(
      operand.base_shape(), operand.sharding(), partition_id, b);
  const PrimitiveType indices_type = indices.hlo()->shape().element_type();
  // Find the per-dimension index bounds.
  std::vector<HloInstruction*> min_indices;
  std::vector<HloInstruction*> max_indices;
  for (int64_t i = 0; i < index_map.size(); ++i) {
    int64_t dim = index_map[i];
    int64_t partitions = operand.sharding().tile_assignment().dim(dim);
    if (partitions == 1 || !absl::c_linear_search(trivial_slice_dims, dim)) {
      min_indices.push_back(CreateR0WithType<int32_t>(indices_type, 0, b));
      max_indices.push_back(CreateR0WithType<int32_t>(
          indices_type, operand.base_shape().dimensions(dim), b));
      continue;
    }
    auto offset = operand_offsets[dim];
    if (offset->shape().element_type() != indices_type) {
      offset = b->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::MakeShape(indices_type, {}), offset));
    }
    min_indices.push_back(offset);
    auto partition_size_minus_1 = CreateR0WithType<int32_t>(
        indices_type, operand.hlo()->shape().dimensions(dim) - 1, b);
    max_indices.push_back(b->AddInstruction(HloInstruction::CreateBinary(
        offset->shape(), HloOpcode::kAdd, offset, partition_size_minus_1)));
  }
  // Broadcast the index bounds to the same shape as the indices.
  HloInstruction* broadcast_min;
  HloInstruction* broadcast_max;
  if (index_vector_dim < indices.rank()) {
    // The index vector is an R1, we need to reshape individual bounds to
    // [1], and concat them if there are more than one.
    for (int64_t i = 0; i < min_indices.size(); ++i) {
      min_indices[i] = b->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(indices_type, {1}), min_indices[i]));
      max_indices[i] = b->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(indices_type, {1}), max_indices[i]));
    }
    int64_t slice_dims = max_indices.size();
    if (slice_dims > 1) {
      min_indices[0] = b->AddInstruction(HloInstruction::CreateConcatenate(
          ShapeUtil::MakeShape(indices_type, {slice_dims}), min_indices, 0));
      max_indices[0] = b->AddInstruction(HloInstruction::CreateConcatenate(
          min_indices[0]->shape(), max_indices, 0));
    }
    broadcast_min = b->AddInstruction(HloInstruction::CreateBroadcast(
        indices.hlo()->shape(), min_indices[0], {index_vector_dim}));
    broadcast_max = b->AddInstruction(HloInstruction::CreateBroadcast(
        indices.hlo()->shape(), max_indices[0], {index_vector_dim}));
  } else {
    CHECK_EQ(max_indices.size(), 1);
    broadcast_min = b->AddInstruction(HloInstruction::CreateBroadcast(
        indices.hlo()->shape(), min_indices[0], {}));
    broadcast_max = b->AddInstruction(HloInstruction::CreateBroadcast(
        indices.hlo()->shape(), max_indices[0], {}));
  }
  return {broadcast_min, broadcast_max};
}

// Function that tries to perform recursive partitioning of Gather.
absl::StatusOr<HloInstruction*> PartitionGather(
    const HloGatherInstruction* gather, PartitionedHlo operand,
    PartitionedHlo indices, const Shape& output_shape,
    const HloSharding& output_sharding, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive = true);

// Perform partitioning of Gather when the indices are partitioned on the
// non-index vector dimension.
absl::StatusOr<HloInstruction*> PartitionGatherIndexPassthroughDimensions(
    const HloGatherInstruction* gather, PartitionedHlo operand,
    PartitionedHlo indices, const Shape& output_shape,
    const HloSharding& output_sharding, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive) {
  // Perform clean up actions upon exiting function scope.
  absl::InlinedVector<std::function<void()>, 3> clean_ups;
  absl::Cleanup cleaner = [&clean_ups] {
    for (auto& clean_up : clean_ups) {
      clean_up();
    }
  };

  GatherDimensionNumbers dnums = gather->gather_dimension_numbers();
  SpmdBuilder* b = visitor->builder();
  absl::InlinedVector<int64_t, 1> index_group_dims =
      hlo_sharding_util::GetGatherScatterIndexPassthroughIndexDims(
          indices.rank(), dnums.index_vector_dim());
  absl::InlinedVector<int64_t, 1> output_group_dims =
      hlo_sharding_util::GetGatherScatterIndexPassthroughOutputOrUpdateDims(
          output_shape.rank(), dnums.offset_dims());
  const int64_t num_groups = indices.sharding().NumTiles(index_group_dims);
  const int64_t num_tiles = indices.sharding().TotalNumTiles();
  // Compute output sharding.
  HloSharding passthrough_sharding = hlo_sharding_util::
      GatherOutputShardingFromIndexIndexPassthroughDimensions(
          indices.sharding(), gather);
  if (passthrough_sharding.IsTileMaximal()) {
    return nullptr;
  }
  hlo_sharding_util::MergeShardingIfCompatible(
      output_sharding, passthrough_sharding.NumTiles() + 1,
      &passthrough_sharding);
  // Group shardings on index pass-through dimensions.
  const GroupedSharding output_grouped = hlo_sharding_util::GroupShardingOnDims(
      passthrough_sharding, output_group_dims);
  const GroupedSharding indices_grouped =
      AlignGroupsWith(hlo_sharding_util::GroupShardingOnDims(indices.sharding(),
                                                             index_group_dims),
                      output_grouped);

  if (indices.sharding().IsTileMaximal()) {
    return nullptr;
  }
  // See if we can group partially replicated dimensions from the operand
  // otherwise replicate it.
  const GroupedSharding operand_grouped = AlignGroupsWith(
      hlo_sharding_util::GroupShardingOnReplicatedDim(
          operand.sharding(), num_groups, num_tiles, operand.rank(),
          GatherOperandDimsByPriority(operand, gather, slice_sizes)),
      output_grouped);
  PartitionedHlo per_group_operand =
      PerGroupPartitionedHlo(operand, operand_grouped, b, clean_ups);
  PartitionedHlo per_group_indices =
      PerGroupPartitionedHlo(indices, indices_grouped, b, clean_ups);
  const Shape pshape = GetPerGroupBaseShape(output_grouped, output_shape);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * pgather,
      PartitionGather(gather, per_group_operand, per_group_indices, pshape,
                      output_grouped.sharding, batch_dims, slice_sizes, visitor,
                      allow_recursive));
  pgather->set_sharding(passthrough_sharding);
  if (allow_recursive) {
    VLOG(5) << "[Gather partitioning]: Partitioned as index only";
  }
  return PartitionedHlo(pgather, gather->shape(), operand.state())
      .Reshard(output_sharding)
      .hlo();
}

// Perform partitioning of Gather when the operand is split in a offset
// dimension that is passed through (slice size is the same size of the operand
// dimension).
absl::StatusOr<HloInstruction*> PartitionGatherOperandPassthroughDimensions(
    const HloGatherInstruction* gather, PartitionedHlo operand,
    PartitionedHlo indices, const Shape& output_shape,
    const HloSharding& output_sharding, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive) {
  if (operand.sharding().IsTileMaximal()) {
    return nullptr;
  }
  // Perform clean up actions upon exiting function scope.
  absl::InlinedVector<std::function<void()>, 3> clean_ups;
  absl::Cleanup cleaner = [&clean_ups] {
    for (auto& clean_up : clean_ups) {
      clean_up();
    }
  };

  SpmdBuilder* b = visitor->builder();
  if (auto maybe_passthrough = hlo_sharding_util::
          GatherOutputShardingFromOperandOperandPassthroughDimensions(
              operand.base_shape(), operand.sharding(), *gather, slice_sizes)) {
    const auto operand_grouping_dims =
        hlo_sharding_util::GetGatherOperandPassthroughOperandDims(
            operand.base_shape(), *gather, slice_sizes);
    const int64_t num_groups =
        operand.sharding().NumTiles(operand_grouping_dims);
    const int64_t num_tiles = operand.sharding().TotalNumTiles();
    absl::InlinedVector<int64_t, 4> output_grouping_dims;
    for (int64_t i = 0; i < maybe_passthrough->TiledDataRank(); ++i) {
      if (maybe_passthrough->tile_assignment().dim(i) != 1) {
        output_grouping_dims.push_back(i);
      }
    }
    std::vector<int64_t> pslice_sizes(slice_sizes.begin(), slice_sizes.end());
    for (int64_t i = 0; i < pslice_sizes.size(); ++i) {
      if (absl::c_linear_search(operand_grouping_dims, i)) {
        pslice_sizes[i] = operand.hlo()->shape().dimensions(i);
      }
    }
    // Merge the sharding from the instruction with the sharding suggested from
    // the operand sharding.
    hlo_sharding_util::MergeShardingIfCompatible(
        output_sharding, maybe_passthrough->NumTiles() + 1,
        &*maybe_passthrough);
    // Group shardings on operand pass-through dimensions.
    const GroupedSharding output_grouped =
        hlo_sharding_util::GroupShardingOnDims(*maybe_passthrough,
                                               output_grouping_dims);
    const GroupedSharding operand_grouped =
        AlignGroupsWith(hlo_sharding_util::GroupShardingOnDims(
                            operand.sharding(), operand_grouping_dims),
                        output_grouped);
    // See if we can group partially replicated dimensions from the indices
    // otherwise replicate it.
    const GroupedSharding indices_grouped = AlignGroupsWith(
        hlo_sharding_util::GroupShardingOnReplicatedDim(
            indices.sharding(), num_groups, num_tiles, indices.rank(),
            GatherIndexDimsByPriority(indices, gather)),
        output_grouped);
    PartitionedHlo per_group_operand =
        PerGroupPartitionedHlo(operand, operand_grouped, b, clean_ups);
    PartitionedHlo per_group_indices =
        PerGroupPartitionedHlo(indices, indices_grouped, b, clean_ups);
    const Shape pshape = GetPerGroupBaseShape(output_grouped, output_shape);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * pgather,
        PartitionGather(gather, per_group_operand, per_group_indices, pshape,
                        output_grouped.sharding, batch_dims, pslice_sizes,
                        visitor, allow_recursive));
    pgather->set_sharding(*maybe_passthrough);
    if (allow_recursive) {
      VLOG(5) << "[Gather partitioning]: Partitioned as operand passthrough "
                 "offset_dim";
    }
    return PartitionedHlo(pgather, output_shape, operand.state())
        .Reshard(output_sharding)
        .hlo();
  }
  return nullptr;
}

// Partition a Gather when its sliced in a dimension in the operand that is
// trivially sliced (sliced with slice size of 1).
absl::StatusOr<HloInstruction*> PartitionGatherTrivialSlicedOperandDimensions(
    const HloGatherInstruction* gather, PartitionedHlo operand,
    PartitionedHlo indices, const Shape& output_shape,
    const HloSharding& output_sharding, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive) {
  // Perform clean up actions upon exiting function scope.
  absl::InlinedVector<std::function<void()>, 3> clean_ups;
  absl::Cleanup cleaner = [&clean_ups] {
    for (auto& clean_up : clean_ups) {
      clean_up();
    }
  };

  SpmdBuilder* b = visitor->builder();
  GatherDimensionNumbers dnums = gather->gather_dimension_numbers();
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  if (std::optional<std::vector<int64_t>> trivial_slice_dims =
          GatherScatterOperandPartitionedOnTrivialSliceDims(
              operand, start_index_map, slice_sizes)) {
    const HloSharding original_operand_sharding = operand.sharding();
    const int64_t num_groups = operand.sharding().NumTiles(*trivial_slice_dims);
    const int64_t num_tiles = operand.sharding().TotalNumTiles();
    const GroupedSharding operand_grouped =
        hlo_sharding_util::GroupShardingOnDims(operand.sharding(),
                                               *trivial_slice_dims);
    // See if we can group partially replicated dimensions from the indices
    // otherwise replicate it.
    GroupedSharding indices_grouped = AlignGroupsWith(
        hlo_sharding_util::GroupShardingOnReplicatedDim(
            indices.sharding(), num_groups, num_tiles, indices.rank(),
            GatherIndexDimsByPriority(indices, gather)),
        operand_grouped);
    GroupedSharding output_grouped = AlignGroupsWith(
        hlo_sharding_util::GroupShardingOnReplicatedDim(
            output_sharding, num_groups, num_tiles, output_shape.rank(),
            GatherOutputDimsByPriority(output_shape, operand, gather,
                                       slice_sizes)),
        operand_grouped);
    // For index and output sharding, if one is grouped partially but the other
    // is replicated, pass through the partially grouped sharding to the other
    // one.
    if (!indices_grouped.sharding.IsTileMaximal() &&
        output_grouped.sharding.IsTileMaximal()) {
      const HloSharding new_output_sharding = hlo_sharding_util::
          GatherOutputShardingFromIndexIndexPassthroughDimensions(
              indices.sharding(), gather);
      output_grouped = AlignGroupsWith(
          hlo_sharding_util::GroupShardingOnReplicatedDim(
              new_output_sharding, num_groups, num_tiles, output_shape.rank(),
              GatherOutputDimsByPriority(output_shape, operand, gather,
                                         slice_sizes)),
          operand_grouped);
    }
    if (indices_grouped.sharding.IsTileMaximal() &&
        !output_grouped.sharding.IsTileMaximal()) {
      const HloSharding new_indices_sharding = hlo_sharding_util::
          GatherIndexShardingFromOutputIndexPassthroughDimensions(
              output_sharding, gather);
      indices_grouped = AlignGroupsWith(
          hlo_sharding_util::GroupShardingOnReplicatedDim(
              new_indices_sharding, num_groups, num_tiles, indices.rank(),
              GatherIndexDimsByPriority(indices, gather)),
          operand_grouped);
    }
    // Reshard indices to its intended sharding before clamping and adjusting.
    indices =
        indices.Reshard(hlo_sharding_util::UngroupSharding(indices_grouped));
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
            operand, indices, operand.state().partition_id, start_index_map,
            *trivial_slice_dims, dnums.index_vector_dim(), b);
    // Clamp the indices.
    auto adjusted_indices = b->AddInstruction(
        HloInstruction::CreateTernary(indices.hlo()->shape(), HloOpcode::kClamp,
                                      indices_min, indices.hlo(), indices_max));
    // Adjust the indices by subtracting the offset.
    adjusted_indices = b->AddInstruction(HloInstruction::CreateBinary(
        indices.hlo()->shape(), HloOpcode::kSubtract, adjusted_indices,
        indices_min));
    PartitionedHlo new_indices = indices.CloneWithNewHlo(adjusted_indices);

    PartitionedHlo per_group_operand =
        PerGroupPartitionedHlo(operand, operand_grouped, b, clean_ups);
    PartitionedHlo per_group_new_indices =
        PerGroupPartitionedHlo(new_indices, indices_grouped, b, clean_ups);
    const Shape pshape = GetPerGroupBaseShape(output_grouped, output_shape);
    // Gather on adjusted indices.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * pgather,
        PartitionGather(gather, per_group_operand, per_group_new_indices,
                        pshape, output_grouped.sharding, batch_dims,
                        slice_sizes, visitor, allow_recursive));
    // Mask out invalid results.
    auto filter = b->AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::ChangeElementType(indices.hlo()->shape(), PRED),
        indices.hlo(), indices_min, ComparisonDirection::kLt));
    filter = b->AddInstruction(HloInstruction::CreateBinary(
        filter->shape(), HloOpcode::kOr, filter,
        b->AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::ChangeElementType(indices.hlo()->shape(), PRED),
            indices.hlo(), indices_max, ComparisonDirection::kGt))));
    // Make sure that filter is of the same shape on the index pass-through
    // dimensions as the partitioned gather output, since we will need to filter
    // the gather output later.
    PartitionedHlo pfilter = indices.CloneWithNewHlo(filter);
    pfilter = pfilter.Reshard(
        hlo_sharding_util::
            GatherIndexShardingFromOutputIndexPassthroughDimensions(
                hlo_sharding_util::UngroupSharding(output_grouped), gather));
    filter = pfilter.hlo();

    if (dnums.index_vector_dim() < indices.rank()) {
      std::vector<int64_t> reduced_filter_dims;
      for (int64_t i = 0; i < filter->shape().rank(); ++i) {
        if (i != dnums.index_vector_dim()) {
          reduced_filter_dims.push_back(filter->shape().dimensions(i));
        }
      }
      filter = b->AddInstruction(HloInstruction::CreateReduce(
          ShapeUtil::MakeShape(PRED, reduced_filter_dims), filter,
          CreateR0WithType(PRED, false, b), {dnums.index_vector_dim()},
          MakeBinaryAdd(PRED, indices.state().module)));
    }
    std::vector<int64_t> batch_dims;
    for (int64_t i = 0; i < pgather->shape().rank(); ++i) {
      if (!absl::c_linear_search(dnums.offset_dims(), i)) {
        batch_dims.push_back(i);
      }
    }
    auto broadcast_filter = b->AddInstruction(HloInstruction::CreateBroadcast(
        ShapeUtil::ChangeElementType(pgather->shape(), PRED), filter,
        batch_dims));

    auto filtered = b->AddInstruction(HloInstruction::CreateTernary(
        pgather->shape(), HloOpcode::kSelect, broadcast_filter,
        CreateZero(pgather->shape(), b), pgather));
    // All-reduce along trivially sliced dimensions.
    auto ar = operand.state().partitioner->AllReduceAlongShardingDims(
        b, filtered, original_operand_sharding, operand.state().next_channel_id,
        *trivial_slice_dims, operand.state().collective_ops_creator,
        MakeBinaryAdd(filtered->shape().element_type(),
                      operand.state().module));
    if (allow_recursive) {
      VLOG(5) << "[Gather partitioning]: Partitioned as trivial operand "
                 "batch_dim slice";
    }
    ar->set_sharding(hlo_sharding_util::UngroupSharding(output_grouped));
    return PartitionedHlo(ar, output_shape, operand.state())
        .Reshard(output_sharding)
        .hlo();
  }
  return nullptr;
}

// Partition a gather over a indices dimensions that are cosidered parallel
// (which means that the indices access the operand in a monotonically
// increasing way across the respective operand dimension referenced by the
// index).
absl::StatusOr<HloInstruction*> PartitionGatherIndexParallelDimensions(
    const HloGatherInstruction* gather, PartitionedHlo operand,
    PartitionedHlo indices, const Shape& output_shape,
    const HloSharding& output_sharding, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive) {
  // Perform clean up actions upon exiting function scope.
  absl::InlinedVector<std::function<void()>, 3> clean_ups;
  absl::Cleanup cleaner = [&clean_ups] {
    for (auto& clean_up : clean_ups) {
      clean_up();
    }
  };

  SpmdBuilder* b = visitor->builder();
  GatherDimensionNumbers dnums = gather->gather_dimension_numbers();
  const int64_t index_dim = dnums.index_vector_dim();
  // Handle the case where operand is tile maximal. In this case we check if
  // the index is not TileMaximal and in this case we use the index sharding
  // to drive the output sharding.
  if (std::optional<hlo_sharding_util::GatherScatterParallelDims>
          parallel_dims = hlo_sharding_util::GetGatherParallelBatchDims(
              *gather, visitor->call_graph())) {
    if (auto gather_sharding = GatherScatterOperandsShardedAcrossParallelDims(
            *operand.hlo(), *indices.hlo(), *parallel_dims)) {
      const auto indices_parallel_dims = parallel_dims->indices_parallel_dims;
      const auto operand_parallel_dims = parallel_dims->operand_parallel_dims;
      const auto output_parallel_dims =
          hlo_sharding_util::GetGatherParallelOutputDims(*gather,
                                                         *parallel_dims);
      operand = operand.Reshard(gather_sharding->operand_sharding);
      indices = indices.Reshard(gather_sharding->indices_sharding);
      HloSharding gather_output_sharding = hlo_sharding_util::
          GatherOutputOrScatterUpdateShardingFromIndicesParallelDimensions(
              indices.sharding(), output_shape.rank(), indices_parallel_dims,
              output_parallel_dims);

      // Refine output sharding from the operand. it should be inferred from
      // operand sharding, so that the partitioned gather can be either 1)
      // directly created on the partitioned operand, or 2) recursively created
      // without aligning the groups.
      if (auto maybe_passthrough = hlo_sharding_util::
              GatherOutputShardingFromOperandOperandPassthroughDimensions(
                  operand.base_shape(),
                  hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
                      operand.sharding(), operand_parallel_dims),
                  *gather, slice_sizes)) {
        hlo_sharding_util::MergeShardingIfCompatible(
            *maybe_passthrough,
            /*minimum_tiles=*/gather_output_sharding.NumTiles() + 1,
            &gather_output_sharding);
      }
      // Construct the offsets for the operand sharding to be used to adjust
      // the indices. Because we know the only dimensions partitioned are the
      // parallel ones and because the partitioning is the same across indices
      // and operands we can apply the offsets on the operands on the indices.
      std::vector<HloInstruction*> operand_offsets = MakePartitionOffsets(
          operand.base_shape(), operand.sharding(),
          operand.state().partition_id, b, operand_parallel_dims);
      absl::InlinedVector<HloInstruction*, 4> index_offsets;
      for (int start_idx = 0; start_idx < dnums.start_index_map_size();
           ++start_idx) {
        HloInstruction* index_offset =
            indices.rank() > index_dim
                ? b->AddInstruction(HloInstruction::CreateReshape(
                      ShapeUtil::MakeShape(S32, {1}),
                      operand_offsets[dnums.start_index_map(start_idx)]))
                : operand_offsets[dnums.start_index_map(start_idx)];
        index_offsets.push_back(index_offset);
      }
      HloInstruction* adjusted_indices = nullptr;
      if (indices.rank() > index_dim) {
        // Concatenate the offsets for the parallel dimensions to subtract.
        adjusted_indices = b->AddInstruction(HloInstruction::CreateConcatenate(
            ShapeUtil::MakeShape(S32,
                                 {indices.base_shape().dimensions(index_dim)}),
            index_offsets, 0));
      } else {
        CHECK_EQ(index_offsets.size(), 1);
        adjusted_indices = index_offsets[0];
      }
      if (indices.hlo()->shape().element_type() != PrimitiveType::S32) {
        adjusted_indices = b->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(adjusted_indices->shape(),
                                         indices.hlo()->shape().element_type()),
            adjusted_indices));
      }
      if (adjusted_indices->shape().rank() == 0) {
        adjusted_indices = b->AddInstruction(HloInstruction::CreateBroadcast(
            indices.hlo()->shape(), adjusted_indices, {}));
      } else {
        adjusted_indices = b->AddInstruction(HloInstruction::CreateBroadcast(
            indices.hlo()->shape(), adjusted_indices, {index_dim}));
      }
      // Adjust indices by subtracting the offsets based on the partition id.
      adjusted_indices = b->AddInstruction(HloInstruction::CreateBinary(
          indices.hlo()->shape(), HloOpcode::kSubtract, indices.hlo(),
          adjusted_indices));
      PartitionedHlo new_indices = indices.CloneWithNewHlo(adjusted_indices);
      const GroupedSharding new_indices_grouped =
          hlo_sharding_util::GroupShardingOnDims(new_indices.sharding(),
                                                 indices_parallel_dims);
      const GroupedSharding operand_grouped =
          hlo_sharding_util::GroupShardingOnDims(operand.sharding(),
                                                 operand_parallel_dims);
      const GroupedSharding output_grouped =
          hlo_sharding_util::GroupShardingOnDims(gather_output_sharding,
                                                 output_parallel_dims);
      PartitionedHlo per_group_operand =
          PerGroupPartitionedHlo(operand, operand_grouped, b, clean_ups);
      PartitionedHlo per_group_new_indices = PerGroupPartitionedHlo(
          new_indices, new_indices_grouped, b, clean_ups);
      const Shape pshape = GetPerGroupBaseShape(output_grouped, output_shape);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * pgather,
          PartitionGather(gather, per_group_operand, per_group_new_indices,
                          pshape, output_grouped.sharding, batch_dims,
                          slice_sizes, visitor, allow_recursive));
      if (allow_recursive) {
        VLOG(5) << "[Gather partitioning]: Partitioned as parallel batch_dim";
      }
      pgather->set_sharding(hlo_sharding_util::UngroupSharding(output_grouped));
      return PartitionedHlo(pgather, output_shape, operand.state())
          .Reshard(output_sharding)
          .hlo();
    }
  }
  return nullptr;
}

// Returns a full list of partitioning methods used for gather.
std::vector<std::pair<decltype(PartitionGather)*, absl::string_view>>
GatherPartitionMethods() {
  return {{PartitionGatherIndexParallelDimensions,
           "PartitionGatherIndexParallelDimensions"},
          {PartitionGatherOperandPassthroughDimensions,
           "PartitionGatherOperandPassthroughDimensions"},
          {PartitionGatherTrivialSlicedOperandDimensions,
           "PartitionGatherTrivialSlicedOperandDimensions"},
          {PartitionGatherIndexPassthroughDimensions,
           "PartitionGatherIndexPassthroughDimensions"}};
}

// Estimates the memory and communication cost for each partitioning methods for
// gather.
std::pair<int64_t, int64_t> GatherPartitionMethodCostModel(
    decltype(PartitionGather)* partition_method,
    const HloGatherInstruction* gather, const PartitionedHlo& operand,
    const PartitionedHlo& indices, const Shape& output_shape,
    const HloSharding& output_sharding, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor) {
  if (partition_method == PartitionGatherIndexParallelDimensions) {
    // Always prioritize index parallel partitioning, and assume it has zero
    // cost.
    return {0, 0};
  }
  return EvaluatePartitionCost(gather, partition_method, gather, operand,
                               indices, output_shape, output_sharding,
                               batch_dims, slice_sizes, visitor,
                               /*allow_recursive=*/false)
      .value();
}

// Returns a full list of partitioning methods for gather ordered by the
// estimated partitioning cost from low to high.
// TODO(b/245443033): Take recursion of gather/scatter partitioning into
// consideration of the cost model.
std::vector<decltype(PartitionGather)*> GatherPartitionMethodsOrderedByCost(
    const HloGatherInstruction* gather, const PartitionedHlo& operand,
    const PartitionedHlo& indices, const Shape& output_shape,
    const HloSharding& output_sharding, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor) {
  std::vector<decltype(PartitionGather)*> ordered_partition_methods;
  absl::flat_hash_map<decltype(PartitionGather)*, std::pair<int64_t, int64_t>>
      partition_method_costs;
  auto gather_partition_methods = GatherPartitionMethods();
  for (auto [partition_method, method_name] : gather_partition_methods) {
    auto [memory_cost, communication_cost] = GatherPartitionMethodCostModel(
        partition_method, gather, operand, indices, output_shape,
        output_sharding, batch_dims, slice_sizes, visitor);
    VLOG(5) << method_name << " has memory cost of " << memory_cost << " bytes"
            << " and communication cost of " << communication_cost << " bytes";
    partition_method_costs.emplace(
        partition_method, std::make_pair(memory_cost, communication_cost));
    ordered_partition_methods.push_back(partition_method);
  }
  absl::c_sort(ordered_partition_methods, [&](decltype(PartitionGather)* lhs,
                                              decltype(PartitionGather)* rhs) {
    auto [lhs_memory_cost, lhs_communication_cost] =
        partition_method_costs[lhs];
    auto [rhs_memory_cost, rhs_communication_cost] =
        partition_method_costs[rhs];
    return lhs_memory_cost != rhs_memory_cost
               ? lhs_memory_cost < rhs_memory_cost
               : lhs_communication_cost < rhs_communication_cost;
  });
  VLOG(5) << "Gather partitioning methods(ordered by cost):";
  for (auto partition_method : ordered_partition_methods) {
    VLOG(5) << "  "
            << absl::c_find_if(gather_partition_methods,
                               [&](const std::pair<decltype(PartitionGather)*,
                                                   absl::string_view>& p) {
                                 return p.first == partition_method;
                               })
                   ->second;
  }
  return ordered_partition_methods;
}

absl::StatusOr<HloInstruction*> PartitionGather(
    const HloGatherInstruction* gather, PartitionedHlo operand,
    PartitionedHlo indices, const Shape& output_shape,
    const HloSharding& output_sharding, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive) {
  if (allow_recursive) {
    HloInstruction* partitioned_gather;
    for (auto partition_method : GatherPartitionMethodsOrderedByCost(
             gather, operand, indices, output_shape, output_sharding,
             batch_dims, slice_sizes, visitor)) {
      TF_ASSIGN_OR_RETURN(
          partitioned_gather,
          partition_method(gather, operand, indices, output_shape,
                           output_sharding, batch_dims, slice_sizes, visitor,
                           allow_recursive));
      if (partitioned_gather) {
        return partitioned_gather;
      }
    }
  }
  HloInstruction* new_gather =
      visitor->builder()->AddInstruction(HloInstruction::CreateGather(
          output_shape, operand.Replicate().hlo(), indices.Replicate().hlo(),
          gather->gather_dimension_numbers(), slice_sizes,
          gather->indices_are_sorted()));
  new_gather->set_sharding(HloSharding::Replicate());
  new_gather = PartitionedHlo(new_gather, new_gather->shape(), operand.state())
                   .Reshard(output_sharding)
                   .hlo();
  return new_gather;
}

}  // namespace

absl::Status SpmdPartitioningVisitor::HandleGather(HloInstruction* hlo) {
  if (hlo->sharding().HasUniqueDevice()) {
    return DefaultAction(hlo);
  }
  auto gather = Cast<HloGatherInstruction>(hlo);
  const auto& dnums = gather->gather_dimension_numbers();
  auto operand = GetPartitionedHlo(gather->operand(0));
  auto indices = GetPartitionedHlo(gather->operand(1));
  std::vector<int64_t> batch_dims;
  for (int64_t i = 0; i < gather->shape().rank(); ++i) {
    if (!absl::c_linear_search(dnums.offset_dims(), i)) {
      batch_dims.push_back(i);
    }
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * pgather,
      PartitionGather(gather, operand, indices, gather->shape(),
                      gather->sharding(), absl::MakeConstSpan(batch_dims),
                      gather->gather_slice_sizes(), this));
  SetPartitionedHlo(gather, PartitionedHlo(pgather, gather->shape(),
                                           MakePartitioningState()));
  return absl::OkStatus();
}

namespace {

template <typename T, typename F>
int64_t ShapeSizeInBytesSum(absl::Span<const T> operands, F&& get_shape) {
  return absl::c_accumulate(operands, int64_t{0},
                            [&](int64_t sum, const T& operand) {
                              return sum + ShapeSizeInBytes(get_shape(operand));
                            });
}

int64_t BaseShapeSizeSum(absl::Span<const PartitionedHlo> phlos) {
  return ShapeSizeInBytesSum(
      phlos, [](const PartitionedHlo& phlo) { return phlo.base_shape(); });
}

int64_t BaseShapeSizeSum(absl::Span<const PartitionedHlo> phlos,
                         const HloSharding& sharding) {
  return ShapeSizeInBytesSum(phlos, [&sharding](const PartitionedHlo& phlo) {
    return MakePartitionedShape(phlo.base_shape(), sharding);
  });
}

int64_t ShapeSizeSum(absl::Span<const PartitionedHlo> phlos) {
  return ShapeSizeInBytesSum(
      phlos, [](const PartitionedHlo& phlo) { return phlo.hlo()->shape(); });
}

int64_t ShapeSizeSum(absl::Span<const Shape> shapes) {
  return ShapeSizeInBytesSum(shapes, [](const Shape& shape) { return shape; });
}

Shape MaybeMakeTupleShape(absl::Span<const HloInstruction* const> hlos) {
  if (hlos.size() == 1) {
    return hlos[0]->shape();
  }
  absl::InlinedVector<const Shape*, 2> element_shapes;
  element_shapes.reserve(hlos.size());
  for (const HloInstruction* hlo : hlos) {
    element_shapes.push_back(&hlo->shape());
  }
  return ShapeUtil::MakeTupleShapeWithPtrs(element_shapes);
}

Shape MaybeGetTuplePerGroupBaseShape(const GroupedSharding& grouped_sharding,
                                     const Shape& original_base_shape) {
  if (original_base_shape.IsArray()) {
    return GetPerGroupBaseShape(grouped_sharding, original_base_shape);
  }
  absl::InlinedVector<Shape, 2> element_shapes;
  element_shapes.reserve(original_base_shape.tuple_shapes_size());
  for (const Shape& shape : original_base_shape.tuple_shapes()) {
    element_shapes.push_back(GetPerGroupBaseShape(grouped_sharding, shape));
  }
  return ShapeUtil::MakeTupleShape(element_shapes);
}

// Returns the opcode if `reduction_comp` represents a simple binary elementwise
// computation on the two operands.
std::optional<HloOpcode> ParseReductionComputation(
    const HloComputation* reduction_comp) {
  if (reduction_comp->num_parameters() != 2) {
    return std::nullopt;
  }
  auto root = reduction_comp->root_instruction();
  if (!HloOpcodeIsBinaryCommutative(root->opcode())) {
    return std::nullopt;
  }
  if (!absl::c_linear_search(root->operands(),
                             reduction_comp->parameter_instruction(0)) ||
      !absl::c_linear_search(root->operands(),
                             reduction_comp->parameter_instruction(1))) {
    return std::nullopt;
  }
  return root->opcode();
}

// Priority for operand dimensions in scatter from lowest to highest, in case of
// needing to replicate a dimension, start from the lowest first.
std::vector<int64_t> ScatterOperandDimsByPriority(
    const PartitionedHlo& operand, const HloScatterInstruction* scatter,
    absl::Span<const int64_t> slice_sizes) {
  const ScatterDimensionNumbers& dnums = scatter->scatter_dimension_numbers();
  std::vector<int64_t> priority_dims_for_operand;
  if (const std::optional<std::vector<int64_t>> trivial_slice_dims =
          GatherScatterOperandPartitionedOnTrivialSliceDims(
              operand, dnums.scatter_dims_to_operand_dims(), slice_sizes)) {
    absl::c_copy(trivial_slice_dims.value(),
                 std::back_inserter(priority_dims_for_operand));
  }
  const auto operand_passthrough_dims =
      hlo_sharding_util::GetScatterOperandPassthroughOperandDims(
          operand.base_shape(), operand.sharding(), *scatter, slice_sizes);
  absl::c_copy(operand_passthrough_dims,
               std::back_inserter(priority_dims_for_operand));
  return priority_dims_for_operand;
}

// Priority for index dimensions in scatter from lowest to highest, in case of
// needing to replicate a dimension, start from the lowest first.
std::vector<int64_t> ScatterIndexDimsByPriority(
    const PartitionedHlo& indices, const HloScatterInstruction* scatter) {
  const ScatterDimensionNumbers& dnums = scatter->scatter_dimension_numbers();

  std::vector<int64_t> priority_dims_for_indices;
  if (dnums.index_vector_dim() < indices.rank()) {
    priority_dims_for_indices.push_back(dnums.index_vector_dim());
  }
  absl::InlinedVector<int64_t, 1> index_passthrough_dims =
      hlo_sharding_util::GetGatherScatterIndexPassthroughIndexDims(
          indices.rank(), dnums.index_vector_dim());
  absl::c_copy(index_passthrough_dims,
               std::back_inserter(priority_dims_for_indices));
  return priority_dims_for_indices;
}

// Priority for update dimensions in scatter from lowest to highest, in case of
// needing to replicate a dimension, start from the lowest first.
std::vector<int64_t> ScatterUpdateDimsByPriority(
    const Shape& update_shape, const PartitionedHlo& operand,
    const HloScatterInstruction* scatter,
    absl::Span<const int64_t> slice_sizes) {
  // Try preserve operand passthrough dims as typically operand is larger in
  // size than indices, and preserving operand pass-through dims avoid more
  // costly reshard in operand passthrough case.
  // TODO(b/303295109): Use better cost-model based decision to determine the
  // priority ordering for scatter update.
  std::vector<int64_t> priority_dims_for_output;
  auto operand_passthrough_update_dims =
      hlo_sharding_util::GetScatterOperandPassthroughUpdateDims(
          update_shape, operand.base_shape(), operand.sharding(), *scatter,
          slice_sizes);
  for (int i = 0; i != update_shape.rank(); ++i) {
    if (!absl::c_linear_search(operand_passthrough_update_dims, i)) {
      priority_dims_for_output.push_back(i);
    }
  }
  absl::c_copy(operand_passthrough_update_dims,
               std::back_inserter(priority_dims_for_output));
  return priority_dims_for_output;
}

absl::StatusOr<HloInstruction*> PartitionScatter(
    const HloScatterInstruction* scatter, std::vector<PartitionedHlo> operands,
    PartitionedHlo indices, std::vector<PartitionedHlo> updates,
    const Shape& output_shape, const HloSharding& output_sharding,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive = true);

// Partition a scatter over a indices dimensions that are cosidered parallel
// (which means that the indices access the operand in a monotonically
// increasing way across the respective operand dimension referenced by the
// index).
absl::StatusOr<HloInstruction*> PartitionScatterIndexParallelDimensions(
    const HloScatterInstruction* scatter, std::vector<PartitionedHlo> operands,
    PartitionedHlo indices, std::vector<PartitionedHlo> updates,
    const Shape& output_shape, const HloSharding& output_sharding,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive) {
  // Perform clean up actions upon exiting function scope.
  absl::InlinedVector<std::function<void()>, 3> clean_ups;
  absl::Cleanup cleaner = [&clean_ups] {
    for (auto& clean_up : clean_ups) {
      clean_up();
    }
  };

  SpmdBuilder* b = visitor->builder();
  auto dnums = scatter->scatter_dimension_numbers();
  const int64_t index_dim = dnums.index_vector_dim();
  // Handle the case where operand is tile maximal. In this case we check if
  // the index is not TileMaximal and in this case we use the index sharding
  // to drive the output sharding.
  if (std::optional<hlo_sharding_util::GatherScatterParallelDims>
          parallel_dims = hlo_sharding_util::GetScatterParallelBatchDims(
              *scatter, visitor->call_graph())) {
    if (auto scatter_sharding = GatherScatterOperandsShardedAcrossParallelDims(
            *operands[0].hlo(), *indices.hlo(), *parallel_dims)) {
      const auto operand_parallel_dims = parallel_dims->operand_parallel_dims;
      const auto indices_parallel_dims = parallel_dims->indices_parallel_dims;
      const auto update_parallel_dims =
          hlo_sharding_util::GetScatterParallelUpdateDims(*scatter,
                                                          *parallel_dims);
      for (auto& operand : operands) {
        operand = operand.Reshard(scatter_sharding->operand_sharding);
      }
      indices = indices.Reshard(scatter_sharding->indices_sharding);
      HloSharding update_sharding = hlo_sharding_util::
          GatherOutputOrScatterUpdateShardingFromIndicesParallelDimensions(
              indices.sharding(), updates[0].rank(), indices_parallel_dims,
              update_parallel_dims);
      // Refine update sharding from the operand. it should be inferred from
      // operand sharding, so that the partitioned scatter can be either 1)
      // directly created on the partitioned operand, or 2) recursively created
      // without aligning the groups.
      if (auto maybe_passthrough = hlo_sharding_util::
              ScatterUpdateShardingFromOutputOperandPassthroughDimensions(
                  operands[0].base_shape(),
                  hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
                      operands[0].sharding(), operand_parallel_dims),
                  *scatter, slice_sizes)) {
        hlo_sharding_util::MergeShardingIfCompatible(
            *maybe_passthrough,
            /*minimum_tiles=*/update_sharding.NumTiles() + 1, &update_sharding);
      }

      for (auto& update : updates) {
        update = update.Reshard(update_sharding);
      }

      // Construct the offsets for the operand sharding to be used to adjust
      // the indices. Because we know the only dimensions partitioned are the
      // parallel ones and because the partitioning is the same across indices
      // and operands we can apply the offsets on the operands on the indices.
      std::vector<HloInstruction*> operand_offsets = MakePartitionOffsets(
          operands[0].base_shape(), operands[0].sharding(),
          operands[0].state().partition_id, b, operand_parallel_dims);
      absl::InlinedVector<HloInstruction*, 4> index_offsets;
      for (int start_idx = 0;
           start_idx < dnums.scatter_dims_to_operand_dims_size(); ++start_idx) {
        HloInstruction* index_offset =
            indices.base_shape().dimensions_size() > index_dim
                ? b->AddInstruction(HloInstruction::CreateReshape(
                      ShapeUtil::MakeShape(S32, {1}),
                      operand_offsets[dnums.scatter_dims_to_operand_dims(
                          start_idx)]))
                : operand_offsets[dnums.scatter_dims_to_operand_dims(
                      start_idx)];
        index_offsets.push_back(index_offset);
      }
      HloInstruction* adjusted_indices = nullptr;
      if (indices.base_shape().dimensions_size() > index_dim) {
        // Concatenate the offsets for the parallel dimensions to subtract.
        adjusted_indices = b->AddInstruction(HloInstruction::CreateConcatenate(
            ShapeUtil::MakeShape(S32,
                                 {indices.base_shape().dimensions(index_dim)}),
            index_offsets, 0));
      } else {
        CHECK_EQ(index_offsets.size(), 1);
        adjusted_indices = index_offsets[0];
      }
      if (indices.hlo()->shape().element_type() != PrimitiveType::S32) {
        adjusted_indices = b->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(adjusted_indices->shape(),
                                         indices.hlo()->shape().element_type()),
            adjusted_indices));
      }
      if (adjusted_indices->shape().rank() == 0) {
        adjusted_indices = b->AddInstruction(HloInstruction::CreateBroadcast(
            indices.hlo()->shape(), adjusted_indices, {}));
      } else {
        adjusted_indices = b->AddInstruction(HloInstruction::CreateBroadcast(
            indices.hlo()->shape(), adjusted_indices, {index_dim}));
      }
      // Adjust indices by subtracting the offsets based on the partition id.
      adjusted_indices = b->AddInstruction(HloInstruction::CreateBinary(
          indices.hlo()->shape(), HloOpcode::kSubtract, indices.hlo(),
          adjusted_indices));
      PartitionedHlo new_indices = indices.CloneWithNewHlo(adjusted_indices);
      const GroupedSharding new_indices_grouped =
          hlo_sharding_util::GroupShardingOnDims(new_indices.sharding(),
                                                 indices_parallel_dims);
      const GroupedSharding operand_grouped =
          hlo_sharding_util::GroupShardingOnDims(operands[0].sharding(),
                                                 operand_parallel_dims);
      const GroupedSharding update_grouped =
          hlo_sharding_util::GroupShardingOnDims(updates[0].sharding(),
                                                 update_parallel_dims);
      const GroupedSharding& output_grouped = operand_grouped;
      std::vector<PartitionedHlo> per_group_operands =
          PerGroupPartitionedHlos(operands, operand_grouped, b, clean_ups);
      std::vector<PartitionedHlo> per_group_updates =
          PerGroupPartitionedHlos(updates, update_grouped, b, clean_ups);
      PartitionedHlo per_group_new_indices = PerGroupPartitionedHlo(
          new_indices, new_indices_grouped, b, clean_ups);
      auto pshape =
          MaybeGetTuplePerGroupBaseShape(output_grouped, output_shape);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * pscatter,
          PartitionScatter(
              scatter, per_group_operands, per_group_new_indices,
              per_group_updates, pshape,
              HloSharding::Single(scatter->shape(), output_grouped.sharding),
              slice_sizes, visitor, allow_recursive));
      pscatter->set_sharding(HloSharding::Single(
          pscatter->shape(),
          hlo_sharding_util::UngroupSharding(output_grouped)));
      if (allow_recursive) {
        VLOG(5) << "[Scatter partitioning]: Partitioned as index parallel";
      }
      return PartitionedHlo(pscatter, output_shape, operands[0].state())
          .Reshard(output_sharding)
          .hlo();
    }
  }
  return nullptr;
}
// Perform partitioning of Scatter when the operand is split in a update window
// dimension that is passed through (slice size is the same size of the operand
// dimension).
absl::StatusOr<HloInstruction*> PartitionScatterOperandPassthroughDimensions(
    const HloScatterInstruction* scatter, std::vector<PartitionedHlo> operands,
    PartitionedHlo indices, std::vector<PartitionedHlo> updates,
    const Shape& output_shape, const HloSharding& output_sharding,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive) {
  if (operands[0].sharding().IsTileMaximal()) {
    return nullptr;
  }
  // Perform clean up actions upon exiting function scope.
  absl::InlinedVector<std::function<void()>, 3> clean_ups;
  absl::Cleanup cleaner = [&clean_ups] {
    for (auto& clean_up : clean_ups) {
      clean_up();
    }
  };

  SpmdBuilder* b = visitor->builder();
  if (auto maybe_passthrough = hlo_sharding_util::
          ScatterUpdateShardingFromOutputOperandPassthroughDimensions(
              operands[0].base_shape(), operands[0].sharding(), *scatter,
              slice_sizes)) {
    const auto operand_grouping_dims =
        hlo_sharding_util::GetScatterOperandPassthroughOperandDims(
            operands[0].base_shape(), operands[0].sharding(), *scatter,
            slice_sizes);
    const int64_t num_groups =
        operands[0].sharding().NumTiles(operand_grouping_dims);
    const int64_t num_tiles = operands[0].sharding().TotalNumTiles();
    absl::InlinedVector<int64_t, 4> update_grouping_dims;
    for (int64_t i = 0; i < maybe_passthrough->TiledDataRank(); ++i) {
      if (maybe_passthrough->tile_assignment().dim(i) != 1) {
        update_grouping_dims.push_back(i);
      }
    }
    std::vector<int64_t> pslice_sizes(slice_sizes.begin(), slice_sizes.end());
    for (int64_t i = 0; i < pslice_sizes.size(); ++i) {
      if (absl::c_linear_search(operand_grouping_dims, i)) {
        pslice_sizes[i] = operands[0].hlo()->shape().dimensions(i);
      }
    }
    // Merge the sharding from update with the sharding suggested from the
    // operand sharding.
    hlo_sharding_util::MergeShardingIfCompatible(
        updates[0].sharding(), maybe_passthrough->NumTiles() + 1,
        &*maybe_passthrough);
    // Group shardings on operand pass-through dimensions.
    const GroupedSharding update_grouped =
        hlo_sharding_util::GroupShardingOnDims(*maybe_passthrough,
                                               update_grouping_dims);
    const GroupedSharding operand_grouped =
        AlignGroupsWith(hlo_sharding_util::GroupShardingOnDims(
                            operands[0].sharding(), operand_grouping_dims),
                        update_grouped);
    // See if we can group partially replicated dimensions from the operand
    // otherwise replicate it.
    const GroupedSharding indices_grouped = AlignGroupsWith(
        hlo_sharding_util::GroupShardingOnReplicatedDim(
            indices.sharding(), num_groups, num_tiles, indices.rank(),
            ScatterIndexDimsByPriority(indices, scatter)),
        update_grouped);
    const GroupedSharding& output_grouped = operand_grouped;
    std::vector<PartitionedHlo> per_group_operands =
        PerGroupPartitionedHlos(operands, operand_grouped, b, clean_ups);
    std::vector<PartitionedHlo> per_group_updates =
        PerGroupPartitionedHlos(updates, update_grouped, b, clean_ups);
    PartitionedHlo per_group_indices =
        PerGroupPartitionedHlo(indices, indices_grouped, b, clean_ups);
    auto pshape = MaybeGetTuplePerGroupBaseShape(output_grouped, output_shape);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * pscatter,
        PartitionScatter(
            scatter, per_group_operands, per_group_indices, per_group_updates,
            pshape,
            HloSharding::Single(scatter->shape(), output_grouped.sharding),
            pslice_sizes, visitor, allow_recursive));
    pscatter->set_sharding(HloSharding::Single(
        pscatter->shape(), hlo_sharding_util::UngroupSharding(output_grouped)));
    if (allow_recursive) {
      VLOG(5) << "[Scatter partitioning]: Partitioned as operand passthrough "
                 "update_window_dims";
    }
    return PartitionedHlo(pscatter, output_shape, operands[0].state())
        .Reshard(output_sharding)
        .hlo();
  }
  return nullptr;
}

// Perform partitioning of Scatter when the indices are partitioned on the
// non-index vector dimension.
absl::StatusOr<HloInstruction*> PartitionScatterIndexPassthroughDimensions(
    const HloScatterInstruction* scatter, std::vector<PartitionedHlo> operands,
    PartitionedHlo indices, std::vector<PartitionedHlo> updates,
    const Shape& output_shape, const HloSharding& output_sharding,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive) {
  // Perform clean up actions upon exiting function scope.
  absl::InlinedVector<std::function<void()>, 3> clean_ups;
  absl::Cleanup cleaner = [&clean_ups] {
    for (auto& clean_up : clean_ups) {
      clean_up();
    }
  };

  SpmdBuilder* b = visitor->builder();
  auto dnums = scatter->scatter_dimension_numbers();
  // Parse non-variadic computation only. Vardiadic case will be replicated.
  const HloSharding original_indices_sharding = indices.sharding();
  absl::InlinedVector<int64_t, 1> index_group_dims =
      hlo_sharding_util::GetGatherScatterIndexPassthroughIndexDims(
          indices.rank(), dnums.index_vector_dim());
  absl::InlinedVector<int64_t, 1> update_group_dims =
      hlo_sharding_util::GetGatherScatterIndexPassthroughOutputOrUpdateDims(
          updates[0].rank(), dnums.update_window_dims());
  const int64_t num_groups = indices.sharding().NumTiles(index_group_dims);
  const int64_t num_tiles = indices.sharding().TotalNumTiles();
  const HloSharding passthrough_sharding = hlo_sharding_util::
      ScatterUpdateShardingFromIndexIndexPassthroughDimensions(
          indices.sharding(), scatter);
  if (passthrough_sharding.IsTileMaximal()) {
    return nullptr;
  }
  const GroupedSharding update_grouped = hlo_sharding_util::GroupShardingOnDims(
      passthrough_sharding, update_group_dims);
  // See if we can group partially replicated dimensions from the operand
  // otherwise replicate it.
  const GroupedSharding operand_grouped = AlignGroupsWith(
      hlo_sharding_util::GroupShardingOnReplicatedDim(
          operands[0].sharding(), num_groups, num_tiles, operands[0].rank(),
          ScatterOperandDimsByPriority(operands[0], scatter, slice_sizes)),
      update_grouped);
  const GroupedSharding indices_grouped =
      AlignGroupsWith(hlo_sharding_util::GroupShardingOnDims(indices.sharding(),
                                                             index_group_dims),
                      update_grouped);
  const GroupedSharding& output_grouped = operand_grouped;
  PartitionedHlo per_group_operand =
      PerGroupPartitionedHlo(operands[0], operand_grouped, b, clean_ups);

  auto reduction_opcode = ParseReductionComputation(scatter->to_apply());
  if (!reduction_opcode.has_value() || *reduction_opcode == HloOpcode::kXor) {
    // XOR is not supported for now, as it will need to keep the operand
    // around in buffer after local scatter to XOR with the final all-reduced
    // results.
    return nullptr;
  }
    HloInstruction* identity;
    switch (*reduction_opcode) {
      case HloOpcode::kAdd:
      case HloOpcode::kOr:
      identity = CreateZero(per_group_operand.hlo()->shape(), b);
      break;
      case HloOpcode::kMultiply:
      case HloOpcode::kAnd:
      identity = CreateOne(per_group_operand.hlo()->shape(), b);
      break;
      case HloOpcode::kMinimum:
      identity = CreateConstant(
          per_group_operand.hlo()->shape(),
          LiteralUtil::MaxValue(scatter->shape().element_type()), b);
      break;
      case HloOpcode::kMaximum:
      identity = CreateConstant(
          per_group_operand.hlo()->shape(),
          LiteralUtil::MinValue(scatter->shape().element_type()), b);
      break;
      default:
        return nullptr;
    }
    // Update partition_id for partial replicate.
    auto partition_id = indices.state().partition_id;
    if (indices.sharding().ReplicateOnLastTileDim()) {
      auto sharding_grouped = hlo_sharding_util::GroupShardingOnDims(
          indices.sharding(),
          {indices.sharding().tile_assignment().num_dimensions() - 1});
      auto per_group_partitioner_state = CreatePerGroupPartitioningState(
          indices.state(), sharding_grouped.device_groups, b);
      partition_id = per_group_partitioner_state.partition_id;
    }
    // To avoid accumulating the initial operand multiple times during
    // all-reduce, we use identity operands for all non-zero partitions.
    auto not_partition_zero = b->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::MakeScalarShape(PRED), partition_id));
    not_partition_zero = b->AddInstruction(HloInstruction::CreateBroadcast(
        ShapeUtil::ChangeElementType(identity->shape(), PRED),
        not_partition_zero, {}));
    auto select_operand =
        b->AddInstruction(HloInstruction::HloInstruction::CreateTernary(
            identity->shape(), HloOpcode::kSelect, not_partition_zero, identity,
            per_group_operand.hlo()));
    PartitionedHlo new_operand =
        per_group_operand.CloneWithNewHlo(select_operand);
    std::vector<PartitionedHlo> per_group_new_operands = {new_operand};
    std::vector<PartitionedHlo> per_group_updates = {
        PerGroupPartitionedHlo(updates[0], update_grouped, b, clean_ups)};
    PartitionedHlo per_group_indices =
        PerGroupPartitionedHlo(indices, indices_grouped, b, clean_ups);
    auto pshape = MaybeGetTuplePerGroupBaseShape(output_grouped, output_shape);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * pscatter,
        PartitionScatter(
            scatter, per_group_new_operands, per_group_indices,
            per_group_updates, pshape,
            HloSharding::Single(scatter->shape(), output_grouped.sharding),
            slice_sizes, visitor, allow_recursive));
    // All-reduce along all dims in operand sharding -- this is OK because the
    // operand is not sharded on index_vector_dim.
    std::vector<int64_t> all_dims(indices.rank());
    absl::c_iota(all_dims, 0);
    auto all_reduce =
        operands[0].state().partitioner->AllReduceAlongShardingDims(
            b, pscatter, original_indices_sharding,
            indices.state().next_channel_id, all_dims,
            operands[0].state().collective_ops_creator, scatter->to_apply());
    all_reduce->set_sharding(
        hlo_sharding_util::UngroupSharding(output_grouped));
    if (allow_recursive) {
      VLOG(5) << "[Scatter partitioning]: Partitioned as index passthrough";
    }
    return PartitionedHlo(all_reduce, output_shape, operands[0].state())
        .Reshard(output_sharding)
        .hlo();
}

// Partition a Scatter when its sliced in a dimension in the operand that is
// trivially sliced (sliced with slice size of 1).
absl::StatusOr<HloInstruction*> PartitionScatterTrivialSlicedOperandDimensions(
    const HloScatterInstruction* scatter, std::vector<PartitionedHlo> operands,
    PartitionedHlo indices, std::vector<PartitionedHlo> updates,
    const Shape& output_shape, const HloSharding& output_sharding,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive) {
  // Perform clean up actions upon exiting function scope.
  absl::InlinedVector<std::function<void()>, 3> clean_ups;
  absl::Cleanup cleaner = [&clean_ups] {
    for (auto& clean_up : clean_ups) {
      clean_up();
    }
  };

  SpmdBuilder* b = visitor->builder();
  auto dnums = scatter->scatter_dimension_numbers();
  if (std::optional<std::vector<int64_t>> trivial_slice_dims =
          GatherScatterOperandPartitionedOnTrivialSliceDims(
              operands[0], dnums.scatter_dims_to_operand_dims(), slice_sizes)) {
    // Operand is sharded on trivial slice dims (update slice size 1). We can
    // adjust the indices on each partition by subtracting the offsets. Then
    // we execute a scatter on full updated indices, and out-of-bound accesses
    // will have no effect on the result as guaranteed by the scatter
    // semantics.
    const int64_t num_groups =
        operands[0].sharding().NumTiles(*trivial_slice_dims);
    const int64_t num_tiles = operands[0].sharding().TotalNumTiles();
    const GroupedSharding operand_grouped =
        hlo_sharding_util::GroupShardingOnDims(operands[0].sharding(),
                                               *trivial_slice_dims);
    // See if we can group partially replicated dimensions from the indices
    // otherwise replicate it.
    GroupedSharding indices_grouped = AlignGroupsWith(
        hlo_sharding_util::GroupShardingOnReplicatedDim(
            indices.sharding(), num_groups, num_tiles, indices.rank(),
            ScatterIndexDimsByPriority(indices, scatter)),
        operand_grouped);
    // See if we can group partially replicated dimensions from the updates
    // otherwise replicate it.
    GroupedSharding update_grouped = AlignGroupsWith(
        hlo_sharding_util::GroupShardingOnReplicatedDim(
            updates[0].sharding(), num_groups, num_tiles, updates[0].rank(),
            ScatterUpdateDimsByPriority(updates[0].base_shape(), operands[0],
                                        scatter, slice_sizes)),
        operand_grouped);
    // For index and update sharding, if one is grouped partially but the
    // other is replicated, pass through the partially grouped sharding to the
    // other one.
    if (!indices_grouped.sharding.IsTileMaximal() &&
        update_grouped.sharding.IsTileMaximal()) {
      const HloSharding new_update_sharding = hlo_sharding_util::
          ScatterUpdateShardingFromIndexIndexPassthroughDimensions(
              indices.sharding(), scatter);
      update_grouped = AlignGroupsWith(
          hlo_sharding_util::GroupShardingOnReplicatedDim(
              new_update_sharding, num_groups, num_tiles, output_shape.rank(),
              ScatterUpdateDimsByPriority(updates[0].base_shape(), operands[0],
                                          scatter, slice_sizes)),
          operand_grouped);
    }
    if (indices_grouped.sharding.IsTileMaximal() &&
        !update_grouped.sharding.IsTileMaximal()) {
      const HloSharding new_indices_sharding = hlo_sharding_util::
          ScatterIndexShardingFromUpdateIndexPassthroughDimensions(
              updates[0].sharding(), scatter);
      indices_grouped = AlignGroupsWith(
          hlo_sharding_util::GroupShardingOnReplicatedDim(
              new_indices_sharding, num_groups, num_tiles, indices.rank(),
              ScatterIndexDimsByPriority(indices, scatter)),
          operand_grouped);
    }
    const GroupedSharding& output_grouped = operand_grouped;
    // Reshard indices to its intended sharding before adjusting.
    indices =
        indices.Reshard(hlo_sharding_util::UngroupSharding(indices_grouped));
    HloInstruction* indices_min;
    std::tie(indices_min, std::ignore) =
        IndexBoundsForGatherScatterOperandPartitionedOnTrivialSliceDims(
            operands[0], indices, operands[0].state().partition_id,
            dnums.scatter_dims_to_operand_dims(), *trivial_slice_dims,
            dnums.index_vector_dim(), b);
    auto adjusted_indices = b->AddInstruction(HloInstruction::CreateBinary(
        indices.hlo()->shape(), HloOpcode::kSubtract, indices.hlo(),
        indices_min));
    PartitionedHlo new_indices = indices.CloneWithNewHlo(adjusted_indices);
    std::vector<PartitionedHlo> per_group_operands =
        PerGroupPartitionedHlos(operands, operand_grouped, b, clean_ups);
    std::vector<PartitionedHlo> per_group_updates =
        PerGroupPartitionedHlos(updates, update_grouped, b, clean_ups);
    PartitionedHlo per_group_new_indices =
        PerGroupPartitionedHlo(new_indices, indices_grouped, b, clean_ups);
    auto pshape = MaybeGetTuplePerGroupBaseShape(output_grouped, output_shape);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * pscatter,
        PartitionScatter(
            scatter, per_group_operands, per_group_new_indices,
            per_group_updates, pshape,
            HloSharding::Single(scatter->shape(), output_grouped.sharding),
            slice_sizes, visitor, allow_recursive));
    pscatter->set_sharding(HloSharding::Single(
        pscatter->shape(), hlo_sharding_util::UngroupSharding(output_grouped)));
    if (allow_recursive) {
      VLOG(5) << "[Scatter partitioning]: Partitioned as trivially sliced "
                 "operand";
    }
    return PartitionedHlo(pscatter, output_shape, operands[0].state())
        .Reshard(output_sharding)
        .hlo();
  }
  return nullptr;
}

// Returns a full list of partitioning methods used for scatter.
std::vector<std::pair<decltype(PartitionScatter)*, absl::string_view>>
ScatterPartitionMethods() {
    return {{PartitionScatterIndexParallelDimensions,
             "PartitionScatterIndexParallelDimensions"},
            {PartitionScatterOperandPassthroughDimensions,
             "PartitionScatterOperandPassthroughDimensions"},
            {PartitionScatterTrivialSlicedOperandDimensions,
             "PartitionScatterTrivialSlicedOperandDimensions"},
            {PartitionScatterIndexPassthroughDimensions,
             "PartitionScatterIndexPassthroughDimensions"}};
}

// Estimates the memory and communication for each partitioning methods for
// scatter.
std::pair<int64_t, int64_t> ScatterPartitionMethodCostModel(
    decltype(PartitionScatter)* partition_method,
    const HloScatterInstruction* scatter,
    const std::vector<PartitionedHlo>& operands, const PartitionedHlo& indices,
    const std::vector<PartitionedHlo>& updates, const Shape& output_shape,
    const HloSharding& output_sharding, absl::Span<const int64_t> slice_sizes,
    SpmdPartitioningVisitor* visitor) {
  if (partition_method == PartitionScatterIndexParallelDimensions) {
    // Always prioritize index parallel partitioning, and assume it has zero
    // cost.
    return {0, 0};
  }
  return EvaluatePartitionCost(scatter, partition_method, scatter, operands,
                               indices, updates, output_shape, output_sharding,
                               slice_sizes, visitor,
                               /*allow_recursive=*/false)
      .value();
}

// Returns a full list of partitioning methods for scatter ordered by the
// estimated partitioning cost from low to high.
// TODO(b/245443033): Take recursion of gather/scatter partitioning into
// consideration of the cost model.
std::vector<decltype(PartitionScatter)*> ScatterPartitionMethodsOrderedByCost(
    const HloScatterInstruction* scatter,
    const std::vector<PartitionedHlo>& operands, const PartitionedHlo& indices,
    const std::vector<PartitionedHlo>& updates, const Shape& output_shape,
    const HloSharding& output_sharding, absl::Span<const int64_t> slice_sizes,
    SpmdPartitioningVisitor* visitor) {
  std::vector<decltype(PartitionScatter)*> ordered_partition_methods;
  absl::flat_hash_map<decltype(PartitionScatter)*, std::pair<int64_t, int64_t>>
      partition_method_costs;
  auto scatter_partition_methods = ScatterPartitionMethods();
  for (auto [partition_method, method_name] : scatter_partition_methods) {
    auto [memory_cost, communication_cost] = ScatterPartitionMethodCostModel(
        partition_method, scatter, operands, indices, updates, output_shape,
        output_sharding, slice_sizes, visitor);

    VLOG(5) << method_name << " has memory cost of " << memory_cost
            << " bytes and communication cost of " << communication_cost
            << " bytes";
    partition_method_costs.emplace(
        partition_method, std::make_pair(memory_cost, communication_cost));
    ordered_partition_methods.push_back(partition_method);
  }
  absl::c_sort(ordered_partition_methods, [&](decltype(PartitionScatter)* lhs,
                                              decltype(PartitionScatter)* rhs) {
    auto [lhs_memory_cost, lhs_communication_cost] =
        partition_method_costs[lhs];
    auto [rhs_memory_cost, rhs_communication_cost] =
        partition_method_costs[rhs];
    return lhs_memory_cost != rhs_memory_cost
               ? lhs_memory_cost < rhs_memory_cost
               : lhs_communication_cost < rhs_communication_cost;
  });
  VLOG(5) << "Scatter partitioning methods(ordered by cost):";
  for (auto partition_method : ordered_partition_methods) {
    VLOG(5) << "  "
            << absl::c_find_if(scatter_partition_methods,
                               [&](const std::pair<decltype(PartitionScatter)*,
                                                   absl::string_view>& p) {
                                 return p.first == partition_method;
                               })
                   ->second;
  }
  return ordered_partition_methods;
}

absl::StatusOr<HloInstruction*> PartitionScatter(
    const HloScatterInstruction* scatter, std::vector<PartitionedHlo> operands,
    PartitionedHlo indices, std::vector<PartitionedHlo> updates,
    const Shape& output_shape, const HloSharding& output_sharding,
    absl::Span<const int64_t> slice_sizes, SpmdPartitioningVisitor* visitor,
    bool allow_recursive) {
  HloInstruction* partitioned_scatter;
  if (allow_recursive) {
    for (auto partition_method : ScatterPartitionMethodsOrderedByCost(
             scatter, operands, indices, updates, output_shape, output_sharding,
             slice_sizes, visitor)) {
      TF_ASSIGN_OR_RETURN(
          partitioned_scatter,
          partition_method(scatter, operands, indices, updates, output_shape,
                           output_sharding, slice_sizes, visitor,
                           allow_recursive));
      if (partitioned_scatter) {
        return partitioned_scatter;
      }
    }
  }
  std::vector<HloInstruction*> operand_hlos, update_hlos;
  absl::c_transform(operands, std::back_inserter(operand_hlos),
                    [](PartitionedHlo phlo) { return phlo.Replicate().hlo(); });
  absl::c_transform(updates, std::back_inserter(update_hlos),
                    [](PartitionedHlo phlo) { return phlo.Replicate().hlo(); });
  HloInstruction* new_scatter =
      visitor->builder()->AddInstruction(HloInstruction::CreateScatter(
          MaybeMakeTupleShape(operand_hlos), operand_hlos,
          indices.Replicate().hlo(), update_hlos, scatter->to_apply(),
          scatter->scatter_dimension_numbers(), scatter->indices_are_sorted(),
          scatter->unique_indices()));
  new_scatter->set_sharding(
      HloSharding::Replicate().NormalizeTupleSharding(new_scatter->shape()));
  new_scatter =
      PartitionedHlo(new_scatter, new_scatter->shape(), operands[0].state())
          .Reshard(output_sharding)
          .hlo();
  return new_scatter;
}

}  // namespace

absl::Status SpmdPartitioningVisitor::HandleScatter(HloInstruction* hlo) {
  if (hlo->sharding().HasUniqueDevice()) {
    return DefaultAction(hlo);
  }
  auto scatter = Cast<HloScatterInstruction>(hlo);
  auto dnums = scatter->scatter_dimension_numbers();
  // Check all operands have the same shapes and shardings, and all updates have
  // the same shapes and shardings, and live with this assumption during scatter
  // partitioning.
  std::vector<PartitionedHlo> operands, updates;
  absl::c_transform(
      scatter->scatter_operands(), std::back_inserter(operands),
      [this](HloInstruction* hlo) { return GetPartitionedHlo(hlo); });
  if (!absl::c_all_of(operands, [&](const PartitionedHlo& operand) {
        return operand.sharding() == operands[0].sharding() &&
               operand.base_shape() == operands[0].base_shape();
      })) {
    std::vector<HloSharding> shardings;
    absl::c_transform(operands, std::back_inserter(shardings),
                      [](const PartitionedHlo& instruction) {
                        return instruction.sharding();
                      });
    HloSharding common_sharding =
        hlo_sharding_util::FindCommonSharding(shardings);
    absl::c_for_each(operands, [&](PartitionedHlo& operand) {
      operand = operand.Reshard(common_sharding);
    });
  }
  absl::c_transform(
      scatter->scatter_updates(), std::back_inserter(updates),
      [this](HloInstruction* hlo) { return GetPartitionedHlo(hlo); });
  if (!absl::c_all_of(updates, [&](const PartitionedHlo& update) {
        return update.sharding() == updates[0].sharding() &&
               update.base_shape() == updates[0].base_shape();
      })) {
    std::vector<HloSharding> shardings;
    absl::c_transform(updates, std::back_inserter(shardings),
                      [](const PartitionedHlo& instruction) {
                        return instruction.sharding();
                      });
    HloSharding common_sharding =
        hlo_sharding_util::FindCommonSharding(shardings);
    absl::c_for_each(operands, [&](PartitionedHlo& operand) {
      operand = operand.Reshard(common_sharding);
    });
  }
  CHECK_EQ(operands.size(), updates.size());
  CHECK_EQ(operands.size() * 2,
           scatter->to_apply()->parameter_instructions().size());
  HloInstruction* scatter_reduction_root =
      scatter->to_apply()->root_instruction();
  CHECK_EQ(operands.size(),
           scatter_reduction_root->shape().IsTuple()
               ? scatter_reduction_root->shape().tuple_shapes_size()
               : 1);
  auto indices = GetPartitionedHlo(scatter->scatter_indices());
  auto indices_sharding = indices.sharding();
  // Reshard indices with -1 padding, which will have no effect on the result as
  // guaranteed by the scatter semantics.
  for (auto i = 0; i != indices.rank(); ++i) {
    if (indices.base_shape().dimensions(i) !=
        indices_sharding.tile_assignment().dim(i) *
            indices.hlo()->shape().dimensions(i)) {
      // Reshard only when we know that some dimension is padded.
      indices = indices.Replicate().Reshard(
          indices_sharding, /*pad_value=*/LiteralUtil::CreateR0<int32_t>(-1));
      break;
    }
  }
  std::vector<int64_t> slice_sizes = hlo_sharding_util::GetScatterSliceSize(
      operands[0].base_shape(), updates[0].base_shape(), dnums);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * pscatter,
      PartitionScatter(scatter, operands, indices, updates, scatter->shape(),
                       scatter->sharding(), slice_sizes, this));
  if (!pscatter) {
    return DefaultAction(hlo);
  }
  SetPartitionedHlo(scatter, PartitionedHlo(pscatter, scatter->shape(),
                                            MakePartitioningState()));
  return absl::OkStatus();
}

}  // namespace spmd
}  // namespace xla
