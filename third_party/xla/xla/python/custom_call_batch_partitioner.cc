/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/custom_call_batch_partitioner.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

// Parse the number of batch dimensions from the frontend attributes of the
// custom call.
std::optional<int64_t> GetNumBatchDimensions(
    const HloCustomCallInstruction* custom_call) {
  auto maybe_num_batch_dims =
      custom_call->get_frontend_attribute("num_batch_dims");
  int64_t num_batch_dims;
  if (!maybe_num_batch_dims.has_value() ||
      !absl::SimpleAtoi(maybe_num_batch_dims.value(), &num_batch_dims)) {
    return std::nullopt;
  }
  return num_batch_dims;
}

// Extract the sharding of the leading `num_batch_dims` dimensions from a given
// HLO sharding.
HloSharding GetBatchSharding(const HloSharding& sharding,
                             int64_t num_batch_dims) {
  if (!sharding.IsTiled() || num_batch_dims <= 0 ||
      num_batch_dims >= sharding.TiledDataRank()) {
    return HloSharding::Replicate(sharding.metadata());
  }

  const int64_t num_replicate_dims = sharding.TiledDataRank() - num_batch_dims;
  std::vector<int64_t> replicate_dims;
  replicate_dims.reserve(num_replicate_dims);
  for (int64_t i = 0; i < num_replicate_dims; ++i) {
    replicate_dims.push_back(sharding.TiledDataRank() - num_replicate_dims + i);
  }
  const HloSharding batch_sharding =
      hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(sharding,
                                                               replicate_dims);
  if (!batch_sharding.IsTiled()) {
    return batch_sharding;
  }

  std::vector<int64_t> dimensions(
      batch_sharding.tile_assignment().dimensions().begin(),
      batch_sharding.tile_assignment().dimensions().end());
  dimensions.erase(
      dimensions.begin() + batch_sharding.TiledDataRank() - num_replicate_dims,
      dimensions.begin() + batch_sharding.TiledDataRank());
  auto tile_assignment = batch_sharding.tile_assignment().Reshape(dimensions);
  return batch_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(tile_assignment,
                                        batch_sharding.metadata())
             : HloSharding::Subgroup(tile_assignment,
                                     batch_sharding.subgroup_types(),
                                     batch_sharding.metadata());
}

// Append `num_replicate_dims` replicated dimensions to the given HLO sharding.
HloSharding InsertNonBatchSharding(const HloSharding& sharding,
                                   int64_t num_replicate_dims) {
  if (!sharding.IsTiled() || num_replicate_dims < 0) {
    return HloSharding::Replicate(sharding.metadata());
  }
  if (num_replicate_dims == 0) {
    return sharding;
  }
  std::vector<int64_t> dimensions(
      sharding.tile_assignment().dimensions().begin(),
      sharding.tile_assignment().dimensions().end());
  for (int64_t i = 0; i < num_replicate_dims; ++i) {
    dimensions.insert(dimensions.begin() + sharding.TiledDataRank(), 1);
  }
  auto tile_assignment = sharding.tile_assignment().Reshape(dimensions);
  return sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(tile_assignment, sharding.metadata())
             : HloSharding::Subgroup(tile_assignment, sharding.subgroup_types(),
                                     sharding.metadata());
}

// Compute a common batch sharding from the leading batch dimensions of a set of
// HLO operands.
//
// Returns a pair specifying the batch sharding and the number of batch
// dimensions. When there are multiple operands, the
// `hlo_sharding_util::FindCommonSharding` helper function does the heavy
// lifting of finding a sharding across all the leading batch dimensions to
// minimize resharding.
std::optional<std::pair<HloSharding, int64_t>> ComputeBatchShardingFromOperands(
    const HloInstruction* hlo) {
  const HloCustomCallInstruction* custom_call =
      Cast<HloCustomCallInstruction>(hlo);
  auto maybe_num_batch_dims = GetNumBatchDimensions(custom_call);
  if (!maybe_num_batch_dims.has_value() || maybe_num_batch_dims.value() <= 0) {
    return std::nullopt;
  }
  const int64_t num_batch_dims = maybe_num_batch_dims.value();

  if (hlo->operand_count() == 1) {
    const HloInstruction* operand = hlo->operand(0);
    if (!operand->has_sharding()) {
      return std::nullopt;
    }
    return std::make_pair(GetBatchSharding(operand->sharding(), num_batch_dims),
                          num_batch_dims);
  }

  std::vector<HloSharding> target_shardings;
  target_shardings.reserve(hlo->operand_count());
  for (size_t i = 0; i < hlo->operand_count(); ++i) {
    const HloInstruction* operand = hlo->operand(i);
    if (operand->has_sharding()) {
      target_shardings.push_back(
          GetBatchSharding(operand->sharding(), num_batch_dims));
    }
  }
  if (target_shardings.empty()) {
    return std::nullopt;
  }
  return std::make_pair(hlo_sharding_util::FindCommonSharding(target_shardings),
                        num_batch_dims);
}

// Propagate the leading batch dimension sharding computed using
// `ComputeBatchShardingFromOperands` to the results of an HLO instruction by
// appending replicated dimensions.
std::pair<Shape, HloSharding> ComputeResultShapeAndSharding(
    const Shape& shape, const HloSharding& batch_sharding,
    int64_t num_batch_dims) {
  if (!shape.IsTuple()) {
    const int64_t num_replicate_dims =
        shape.dimensions().size() - num_batch_dims;
    auto result_sharding =
        InsertNonBatchSharding(batch_sharding, num_replicate_dims);
    auto result_shape = spmd::MakePartitionedShape(shape, result_sharding);
    return std::make_pair(result_shape, result_sharding);
  }
  int num_results = shape.tuple_shapes_size();
  std::vector<Shape> result_shapes;
  result_shapes.reserve(num_results);
  std::vector<HloSharding> result_shardings;
  result_shardings.reserve(num_results);
  for (size_t i = 0; i < num_results; ++i) {
    auto [sub_shape, sub_sharding] = ComputeResultShapeAndSharding(
        shape.tuple_shapes(i), batch_sharding, num_batch_dims);
    result_shapes.push_back(sub_shape);
    result_shardings.push_back(sub_sharding);
  }
  Shape result_shape = ShapeUtil::MakeTupleShape(result_shapes);
  return std::make_pair(result_shape,
                        HloSharding::Tuple(result_shape, result_shardings));
}

std::optional<HloSharding>
CustomCallBatchPartitioner::InferShardingFromOperands(
    const HloInstruction* hlo) const {
  auto maybe_batch_sharding = ComputeBatchShardingFromOperands(hlo);
  if (!maybe_batch_sharding.has_value()) {
    return std::nullopt;
  }
  const auto [batch_sharding, num_batch_dims] = maybe_batch_sharding.value();
  auto [_, result_sharding] = ComputeResultShapeAndSharding(
      hlo->shape(), batch_sharding, num_batch_dims);
  return result_sharding;
}

absl::Status CustomCallBatchPartitioner::Partition(
    spmd::SpmdPartitioningVisitor* partitioner, HloInstruction* hlo) const {
  if (!hlo->has_sharding()) {
    return partitioner->DefaultAction(hlo);
  }

  if (hlo->sharding().IsManual()) {
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(hlo->operands().size());
    for (HloInstruction* operand : hlo->operands()) {
      new_operands.push_back(partitioner->GetPartitionedHlo(operand).hlo());
    }
    HloInstruction* partitioned_hlo = partitioner->builder()->AddInstruction(
        hlo->CloneWithNewOperands(hlo->shape(), new_operands));
    if (hlo->shape().IsTuple()) {
      std::vector<HloSharding> subshardings(
          hlo->sharding().tuple_elements().size(),
          HloSharding::AssignDevice(0));
      partitioned_hlo->set_sharding(
          HloSharding::Tuple(hlo->shape(), subshardings));
    } else {
      partitioned_hlo->set_sharding(HloSharding::AssignDevice(0));
    }
    spmd::PartitionedHlo result_partitioned = spmd::PartitionedHlo(
        partitioned_hlo, hlo->shape(), partitioner->MakePartitioningState());
    partitioner->SetPartitionedHlo(hlo, result_partitioned);
    return absl::OkStatus();
  }

  if (hlo->operand_count() == 0) {
    return partitioner->DefaultAction(hlo);
  }
  auto maybe_batch_sharding = ComputeBatchShardingFromOperands(hlo);
  if (!maybe_batch_sharding.has_value()) {
    return partitioner->DefaultAction(hlo);
  }
  const auto [batch_sharding, num_batch_dims] = maybe_batch_sharding.value();
  auto [result_shape, result_sharding] = ComputeResultShapeAndSharding(
      hlo->shape(), batch_sharding, num_batch_dims);

  const HloCustomCallInstruction* custom_call =
      Cast<HloCustomCallInstruction>(hlo);
  const int64_t num_operands = hlo->operand_count();
  std::vector<HloInstruction*> partitioned_operands;
  partitioned_operands.reserve(num_operands);
  std::vector<Shape> partitioned_shapes_with_layout_constraints;
  partitioned_shapes_with_layout_constraints.reserve(num_operands);
  for (size_t i = 0; i < num_operands; ++i) {
    const int64_t num_replicate_dims =
        hlo->operand(i)->shape().dimensions().size() - num_batch_dims;
    HloSharding operand_sharding =
        InsertNonBatchSharding(batch_sharding, num_replicate_dims);
    spmd::PartitionedHlo partitioned_operand =
        partitioner->GetPartitionedHlo(hlo->operand(i))
            .Reshard(operand_sharding);
    partitioned_operands.push_back(partitioned_operand.hlo());
    Shape partitioned_shape_with_layout_constraint =
        partitioned_operand.hlo()->shape();
    (*partitioned_shape_with_layout_constraint.mutable_layout()) =
        custom_call->operand_shapes_with_layout()[i].layout();
    partitioned_shapes_with_layout_constraints.push_back(
        partitioned_shape_with_layout_constraint);
  }

  HloInstruction* partitioned_hlo =
      partitioner->builder()->AddInstruction(HloInstruction::CreateCustomCall(
          result_shape, partitioned_operands, custom_call->custom_call_target(),
          partitioned_shapes_with_layout_constraints, custom_call->opaque(),
          custom_call->api_version()));
  partitioned_hlo->set_sharding(result_sharding);

  spmd::PartitionedHlo result_partitioned =
      spmd::PartitionedHlo(partitioned_hlo, hlo->shape(),
                           partitioner->MakePartitioningState())
          .Reshard(hlo->sharding());
  partitioner->SetPartitionedHlo(hlo, result_partitioned);

  return absl::OkStatus();
}

}  // namespace xla
