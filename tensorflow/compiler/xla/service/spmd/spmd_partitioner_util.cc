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

#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"

#include <algorithm>
#include <memory>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace spmd {

bool HasReplicatedSharding(const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    return absl::c_any_of(sharding.tuple_elements(), HasReplicatedSharding);
  }
  return sharding.IsReplicated();
}

HloInstruction* CreateConstant(const Shape& shape, Literal value,
                               SpmdBuilder* b) {
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> elements;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      elements.push_back(CreateConstant(
          ShapeUtil::GetTupleElementShape(shape, i), value.Clone(), b));
    }
    return b->AddInstruction(HloInstruction::CreateTuple(elements));
  }

  CHECK(
      ShapeUtil::IsScalarWithElementType(value.shape(), shape.element_type()));
  auto c = b->AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  return b->AddInstruction(HloInstruction::CreateBroadcast(shape, c, {}));
}

HloInstruction* CreateZero(const Shape& shape, SpmdBuilder* b) {
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> elements;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      elements.push_back(
          CreateZero(ShapeUtil::GetTupleElementShape(shape, i), b));
    }
    return b->AddInstruction(HloInstruction::CreateTuple(elements));
  }

  if (shape.IsToken()) {
    return b->AddInstruction(HloInstruction::CreateToken());
  }
  auto zero = b->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(shape.element_type())));
  if (shape.rank() == 0) {
    return zero;
  }
  return b->AddInstruction(HloInstruction::CreateBroadcast(shape, zero, {}));
}

HloInstruction* CreateOne(const Shape& shape, SpmdBuilder* b) {
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> elements;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      elements.push_back(
          CreateOne(ShapeUtil::GetTupleElementShape(shape, i), b));
    }
    return b->AddInstruction(HloInstruction::CreateTuple(elements));
  }

  if (shape.IsToken()) {
    return b->AddInstruction(HloInstruction::CreateToken());
  }
  auto one = b->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::One(shape.element_type())));
  return b->AddInstruction(HloInstruction::CreateBroadcast(shape, one, {}));
}

HloComputation* MakeBinaryAdd(PrimitiveType type, HloModule* module) {
  HloComputation::Builder sum_b("add");
  auto x = sum_b.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(type, {}), "x"));
  auto y = sum_b.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(type, {}), "y"));
  if (type == PRED) {
    sum_b.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(type, {}), HloOpcode::kOr, x, y));
  } else {
    sum_b.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(type, {}), HloOpcode::kAdd, x, y));
  }
  HloComputation* reduction = module->AddEmbeddedComputation(sum_b.Build());
  return reduction;
}

bool EvenlyPartitions(const Shape& shape, const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      if (!EvenlyPartitions(ShapeUtil::GetTupleElementShape(shape, i),
                            sharding.GetSubSharding(shape, {i}))) {
        return false;
      }
    }
  }

  if (sharding.IsTileMaximal()) {
    return sharding.IsReplicated();
  }
  for (int64_t i = 0; i < shape.dimensions_size(); ++i) {
    if (shape.dimensions(i) % sharding.tile_assignment().dim(i) != 0) {
      return false;
    }
  }
  return true;
}

Shape MakePartitionedShape(const Shape& shape, const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    std::vector<Shape> subshapes;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      subshapes.push_back(
          MakePartitionedShape(ShapeUtil::GetTupleElementShape(shape, i),
                               sharding.GetSubSharding(shape, {i})));
    }
    return ShapeUtil::MakeTupleShape(subshapes);
  }
  return sharding.TileShape(shape);
}

int64 ShapeSizeInBytes(const Shape& shape) {
  return ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type()) *
         ShapeUtil::ElementsIn(shape);
}

Shape MakeNonPaddedShapeForGivenPartition(const Shape& shape,
                                          const HloSharding& sharding,
                                          int64_t partition_id) {
  if (sharding.IsTuple()) {
    std::vector<Shape> subshapes;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      subshapes.push_back(MakeNonPaddedShapeForGivenPartition(
          ShapeUtil::GetTupleElementShape(shape, i),
          sharding.GetSubSharding(shape, {i}), partition_id));
    }
    return ShapeUtil::MakeTupleShape(subshapes);
  }

  if (sharding.IsReplicated()) {
    return shape;
  }
  if (sharding.IsTileMaximal()) {
    if (partition_id == *sharding.UniqueDevice()) {
      return shape;
    }
    return ShapeUtil::MakeTupleShape({});
  }

  auto partition_shape = shape;
  std::vector<int64> tile_offset =
      sharding.TileOffsetForDevice(shape, partition_id);
  std::vector<int64> tile_limit =
      sharding.TileLimitForDevice(shape, partition_id);
  for (int64_t i = 0; i < tile_offset.size(); ++i) {
    if (sharding.UsesDevice(partition_id)) {
      partition_shape.set_dimensions(i, tile_limit[i] - tile_offset[i]);
    } else {
      partition_shape.set_dimensions(i, 0);
    }
  }
  return partition_shape;
}

std::vector<HloInstruction*> MakePartitionOffsets(
    const Shape& shape, const HloSharding& sharding,
    HloInstruction* partition_id, SpmdBuilder* b,
    absl::Span<const int64> dims) {
  CHECK(!shape.IsTuple());

  std::vector<std::vector<int32>> offset_arrays(shape.rank());
  for (int64_t i = 0; i < shape.rank(); ++i) {
    offset_arrays[i].resize(sharding.tile_assignment().num_elements());
  }
  auto shard_shape = MakePartitionedShape(shape, sharding);
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64> indices, int64_t device) {
        for (int64_t i = 0; i < shape.rank(); ++i) {
          offset_arrays[i][device] = indices[i] * shard_shape.dimensions(i);
        }
      });
  std::vector<HloInstruction*> offsets;
  for (int64_t i = 0; i < shape.rank(); ++i) {
    if (sharding.tile_assignment().dim(i) == 1 ||
        (!dims.empty() && !absl::c_linear_search(dims, i))) {
      offsets.push_back(b->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(S32))));
    } else {
      auto offset_table = b->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<int32>(offset_arrays[i])));
      auto index = b->AddInstruction(HloInstruction::CreateDynamicSlice(
          ShapeUtil::MakeShape(S32, {1}), offset_table, {partition_id}, {1}));
      offsets.push_back(b->AddInstruction(
          HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {}), index)));
    }
  }
  return offsets;
}

std::vector<HloInstruction*> MakeTiledPartitionOrdinals(
    const HloSharding& sharding, HloInstruction* partition_id, SpmdBuilder* b) {
  CHECK(!sharding.IsTileMaximal());
  auto dimensions = sharding.tile_assignment().dimensions();
  if (sharding.ReplicateOnLastTileDim()) {
    dimensions.pop_back();
  }
  auto table_shape = ShapeUtil::MakeShape(S32, dimensions);
  return MakePartitionOffsets(table_shape, sharding, partition_id, b);
}

HloInstruction* PadToShape(HloInstruction* hlo, const Shape& padded_shape,
                           SpmdBuilder* b, HloComputation* computation) {
  CHECK(b == nullptr || computation == nullptr);
  if (ShapeUtil::Compatible(hlo->shape(), padded_shape)) {
    return hlo;
  }
  PaddingConfig padding_config;
  for (int64_t i = 0; i < padded_shape.rank(); ++i) {
    auto padding_config_dim = padding_config.add_dimensions();
    padding_config_dim->set_edge_padding_low(0);
    padding_config_dim->set_interior_padding(0);
    padding_config_dim->set_edge_padding_high(padded_shape.dimensions(i) -
                                              hlo->shape().dimensions(i));
  }
  auto add_hlo = [&](std::unique_ptr<HloInstruction> to_add) {
    if (b == nullptr) {
      return computation->AddInstruction(std::move(to_add));
    }
    return b->AddInstruction(std::move(to_add));
  };
  auto zero = add_hlo(HloInstruction::CreateConstant(
      LiteralUtil::Zero(hlo->shape().element_type())));
  return add_hlo(
      HloInstruction::CreatePad(padded_shape, hlo, zero, padding_config));
}

Shape GetPaddedShapeForUnevenPartitioning(const Shape& base_shape,
                                          const HloSharding& sharding) {
  if (sharding.IsTileMaximal()) {
    return base_shape;
  }
  if (EvenlyPartitions(base_shape, sharding)) {
    return base_shape;
  }
  auto shard_shape = MakePartitionedShape(base_shape, sharding);
  Shape padded_base_shape = base_shape;
  for (int64_t i = 0; i < padded_base_shape.rank(); ++i) {
    padded_base_shape.set_dimensions(
        i, shard_shape.dimensions(i) * sharding.tile_assignment().dim(i));
  }
  return padded_base_shape;
}

HloInstruction* PadBaseShapeBeforeUnevenTiledSharding(
    HloInstruction* hlo, const HloSharding& sharding, SpmdBuilder* b) {
  auto padded_base_shape =
      GetPaddedShapeForUnevenPartitioning(hlo->shape(), sharding);
  if (ShapeUtil::Compatible(padded_base_shape, hlo->shape())) {
    return hlo;
  }
  return PadToShape(hlo, padded_base_shape, b);
}

absl::optional<HloSharding> PartialReplicateReshardCompatibleSharding(
    const HloSharding& partial_sharding, const HloSharding& target_sharding) {
  if (!partial_sharding.ReplicateOnLastTileDim()) {
    return absl::nullopt;
  }
  int64_t rank = partial_sharding.tile_assignment().num_dimensions() - 1;
  int64_t target_rank = target_sharding.tile_assignment().num_dimensions() -
                        (target_sharding.ReplicateOnLastTileDim() ? 1 : 0);
  if (target_rank != rank) {
    return absl::nullopt;
  }

  absl::flat_hash_map<int64, int64> device_to_replication_group;
  partial_sharding.tile_assignment().Each(
      [&](absl::Span<const int64> indices, int64_t device) {
        int64_t gid = 0;
        for (int64_t i = 0; i < rank; ++i) {
          gid *= partial_sharding.tile_assignment().dim(i);
          gid += indices[i];
        }
        device_to_replication_group[device] = gid;
      });

  // A dimension is expanded when target_tile_size > partial_tile_size and
  // target_tile_size % partial_tile_size == 0.
  // expand_tile_dims_positions is the index of the expand_dim.
  std::vector<int64> expand_tile_dims_indices(rank, -1);
  // expand_tile_size = target_tile_size / partial_tile_size.
  std::vector<int64> expand_tile_sizes;
  int num_expand_dims = 0;
  for (int64_t dim = 0; dim < rank; dim++) {
    int64_t partial_tile_size = partial_sharding.tile_assignment().dim(dim);
    int64_t target_tile_size = target_sharding.tile_assignment().dim(dim);
    if (target_tile_size % partial_tile_size != 0 ||
        target_tile_size < partial_tile_size) {
      return absl::nullopt;
    }

    if (target_tile_size > partial_tile_size) {
      expand_tile_dims_indices[dim] = num_expand_dims++;
      expand_tile_sizes.emplace_back(target_tile_size / partial_tile_size);
    }
  }

  // Reshape the partial replicate tile_dimensions.
  int64_t num_target_replication = 1;
  if (target_sharding.ReplicateOnLastTileDim()) {
    num_target_replication =
        target_sharding.tile_assignment().dimensions().back();
  }
  auto reshape_dimensions = partial_sharding.tile_assignment().dimensions();
  int64_t num_replication = reshape_dimensions.back();
  if (num_replication / num_target_replication != Product(expand_tile_sizes) ||
      num_replication % num_target_replication != 0) {
    return absl::nullopt;
  }

  reshape_dimensions.pop_back();
  reshape_dimensions.insert(reshape_dimensions.end(), expand_tile_sizes.begin(),
                            expand_tile_sizes.end());

  if (target_sharding.ReplicateOnLastTileDim()) {
    reshape_dimensions.push_back(num_target_replication);
  }

  auto reshape_tile_assignment = partial_sharding.tile_assignment();
  reshape_tile_assignment.Reshape(reshape_dimensions);

  // Transpose.
  std::vector<int64> perm;
  perm.reserve(rank + expand_tile_sizes.size());
  for (int64_t dim = 0; dim < rank; dim++) {
    perm.emplace_back(dim);
    if (expand_tile_dims_indices[dim] > -1) {
      perm.emplace_back(expand_tile_dims_indices[dim] + rank);
    }
  }
  auto transpose_sharding = hlo_sharding_util::TransposeSharding(
      target_sharding.ReplicateOnLastTileDim()
          ? HloSharding::PartialTile(reshape_tile_assignment)
          : HloSharding::Tile(reshape_tile_assignment),
      perm);

  // Reshape to target shape
  auto transpose_tile_assignment = transpose_sharding.tile_assignment();
  transpose_tile_assignment.Reshape(
      target_sharding.tile_assignment().dimensions());

  bool groups_matching = true;
  target_sharding.tile_assignment().Each(
      [&](absl::Span<const int64> indices, int64_t device) {
        if (device_to_replication_group[device] !=
            device_to_replication_group[transpose_tile_assignment(indices)]) {
          groups_matching = false;
        }
      });

  if (groups_matching) {
    return target_sharding;
  }
  return target_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(transpose_tile_assignment)
             : HloSharding::Tile(transpose_tile_assignment);
}

absl::optional<HloInstruction*> TileToPartialReplicateHaloExchange(
    HloInstruction* hlo, const Shape& base_shape,
    const HloSharding& src_sharding, const HloSharding& dst_sharding,
    const std::vector<int64>& replicate_dims,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, HloInstruction* partition_id, SpmdBuilder* b) {
  // Source is tile sharding.
  auto padded_src_shape =
      GetPaddedShapeForUnevenPartitioning(base_shape, src_sharding);
  // Target is partial replicate.
  auto padded_dst_shape =
      GetPaddedShapeForUnevenPartitioning(base_shape, dst_sharding);
  if (ShapeUtil::Compatible(padded_dst_shape, hlo->shape())) {
    return hlo;
  }

  auto partition_ordinals =
      MakeTiledPartitionOrdinals(dst_sharding, partition_id, b);

  auto result = hlo;
  auto hlo_shape = hlo->shape();
  for (auto dim : replicate_dims) {
    int64_t dst_shard_count = dst_sharding.tile_assignment().dim(dim);
    int64_t src_per_shard_size =
        padded_src_shape.dimensions(dim) / dst_shard_count;
    // Calculate per shard size using the sharding to compare if dst_sharding
    // needs more padding at the end.
    int64_t dst_per_shard_size =
        padded_dst_shape.dimensions(dim) / dst_shard_count;

    // If src per shard doesn't have redundant data.
    if (src_per_shard_size <= dst_per_shard_size || dst_shard_count == 1) {
      continue;
    }

    // If src_per_shard * replicate_factor > dst_per_shard , need to
    // re-distribute the data between each shard using collective permute. For
    // example, if dimension size is 6 and shard 4 ways in the src but needs to
    // shard 2 ways in the dst. 4 way sharding has 2 element in each shard,
    // while 2 way sharding has 3 elements, the last element in the first shard
    // will be sliced out. re-distribution is needed.
    //
    // 1. Calculate left_halo size.
    // left-halo size is
    //   (src_per_shard_size - dst_per_shard_size) * i / replicate_factor
    int64_t replicate_factor = src_sharding.tile_assignment().dim(dim) /
                               dst_sharding.tile_assignment().dim(dim);
    OffsetCalculation left_halo_size_function =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            src_per_shard_size - dst_per_shard_size, 0, replicate_factor));

    // 2. Calculate right_halo size.
    // right-halo size is 0
    OffsetCalculation right_halo_size_function =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(0, 0, 1));

    auto concat = result;
    // 3. Halo exchange.
    auto halo_exchange_result = ExchangeHalo(
        result, left_halo_size_function, right_halo_size_function, dim,
        src_sharding, collective_ops_creator, next_channel_id, b);

    if (halo_exchange_result.has_value()) {
      concat = halo_exchange_result.value();
    } else {
      return absl::nullopt;
    }

    // 4. Slice the valid result.
    // Slice offset is
    // (dst_shard_count - i - 1) *
    // (src_per_shard_size - dst_per_shard_size)
    // i is the index in dst_sharindg.
    auto zero_s32 = b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
    OffsetCalculation start_offset_on_padded_concat_calculation =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            dst_per_shard_size - src_per_shard_size,
            (src_per_shard_size - dst_per_shard_size) * (dst_shard_count - 1),
            1));
    auto slice_shape = concat->shape();
    slice_shape.set_dimensions(dim,
                               padded_src_shape.dimensions(dim) /
                                   src_sharding.tile_assignment().dim(dim));
    std::vector<HloInstruction*> slice_offsets(concat->shape().rank(),
                                               zero_s32);
    slice_offsets[dim] = start_offset_on_padded_concat_calculation.Calculate(
        partition_ordinals[dim], b);
    result = b->AddInstruction(HloInstruction::CreateDynamicSlice(
        slice_shape, concat, slice_offsets, slice_shape.dimensions()));
  }
  return result;
}

absl::optional<HloInstruction*> PadFromPartialReplicateShape(
    HloInstruction* hlo, const Shape& base_shape,
    const HloSharding& src_sharding, const HloSharding& dst_sharding,
    const std::vector<int64>& expand_tile_dims,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, HloInstruction* partition_id, SpmdBuilder* b) {
  auto padded_src_shape =
      GetPaddedShapeForUnevenPartitioning(base_shape, src_sharding);
  auto padded_dst_shape =
      GetPaddedShapeForUnevenPartitioning(base_shape, dst_sharding);
  if (ShapeUtil::Compatible(padded_dst_shape, hlo->shape())) {
    return hlo;
  }

  auto partition_ordinals =
      MakeTiledPartitionOrdinals(src_sharding, partition_id, b);

  HloInstruction* result = hlo;
  auto zero = b->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(hlo->shape().element_type())));
  std::vector<int64> expand_dims_without_halo_exchange;
  // Pad the dimensions needs halo exchange and record the padded dims that
  // won't need halo exchange.
  for (auto dim : expand_tile_dims) {
    int64_t src_shard_count = src_sharding.tile_assignment().dim(dim);
    int64_t src_per_shard_size =
        padded_src_shape.dimensions(dim) / src_shard_count;
    // Calculate per shard size using the sharding to compare if dst_sharding
    // needs more padding at the end.
    int64_t dst_per_shard_size =
        padded_dst_shape.dimensions(dim) / src_shard_count;

    // If dst_sharding doesn't need more padding at the end.
    if (src_per_shard_size >= dst_per_shard_size) {
      continue;
    }
    // If src sharding at this dimension is not partitoned, simply pad to
    // the desired shape.
    if (src_shard_count == 1) {
      expand_dims_without_halo_exchange.emplace_back(dim);
      continue;
    }

    // If dst_padding needs more padding at the end, need to re-distribute the
    // data between each shard using collective permute.
    // For example, if dimension size is 6 and shard 2 ways in the src but
    // needs to shard 4 ways in the dst. 4 ways needs padding 2 0s at the end
    // and has 2 elements at each shard, while 2 way sharding has 3 elements
    // in each shard, re-distribution is needed.
    //
    // 1. Calculate left_halo size.
    // left-halo size is 0
    OffsetCalculation left_halo_size_function =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(0, 0, 1));

    // 2. Calculate right_halo size.
    // right-halo size is D * (i + 1) - S * (i + 1) = (D - S) * i + (D - S)
    OffsetCalculation right_halo_size_function =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            dst_per_shard_size - src_per_shard_size,
            dst_per_shard_size - src_per_shard_size, 1));

    auto concat = result;
    // 3. Halo exchange.
    auto halo_exchange_result = ExchangeHalo(
        result, left_halo_size_function, right_halo_size_function, dim,
        src_sharding, collective_ops_creator, next_channel_id, b);

    if (halo_exchange_result.has_value()) {
      concat = halo_exchange_result.value();
    } else {
      return absl::nullopt;
    }

    // 4. Pad.
    std::vector<int64> zero_padding(concat->shape().rank());
    PaddingConfig pad_config = window_util::MakeSymmetricPadding(zero_padding);
    pad_config.mutable_dimensions(dim)->set_edge_padding_low(0);
    int64_t max_right_halo_size =
        right_halo_size_function.MaxInRange(0, src_shard_count - 1);
    pad_config.mutable_dimensions(dim)->set_edge_padding_high(std::max(
        int64{0}, padded_dst_shape.dimensions(dim) -
                      padded_src_shape.dimensions(dim) - max_right_halo_size));
    auto padded_concat_shape = ShapeInference::InferPadShape(
                                   concat->shape(), zero->shape(), pad_config)
                                   .ValueOrDie();
    concat = b->AddInstruction(HloInstruction::CreatePad(
        padded_concat_shape, concat, zero, pad_config));

    // 5. Slice the valid result.
    // Slice offset is (D-S) * i
    auto zero_s32 = b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
    OffsetCalculation start_offset_on_padded_concat_calculation =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            dst_per_shard_size - src_per_shard_size, 0, 1));
    auto slice_shape = concat->shape();
    slice_shape.set_dimensions(dim, dst_per_shard_size);
    std::vector<HloInstruction*> slice_offsets(concat->shape().rank(),
                                               zero_s32);
    slice_offsets[dim] = start_offset_on_padded_concat_calculation.Calculate(
        partition_ordinals[dim], b);
    result = b->AddInstruction(HloInstruction::CreateDynamicSlice(
        slice_shape, concat, slice_offsets, slice_shape.dimensions()));
  }

  // Pad other dimensions that won't need halo exchange with a single pad.
  if (!expand_dims_without_halo_exchange.empty()) {
    std::vector<int64> zero_padding(result->shape().rank());
    PaddingConfig pad_config = window_util::MakeSymmetricPadding(zero_padding);

    auto padded_shape = result->shape();
    for (auto dim : expand_dims_without_halo_exchange) {
      pad_config.mutable_dimensions(dim)->set_edge_padding_low(0);
      pad_config.mutable_dimensions(dim)->set_edge_padding_high(
          padded_dst_shape.dimensions(dim) - padded_src_shape.dimensions(dim));
      padded_shape.set_dimensions(dim, result->shape().dimensions(dim) +
                                           padded_dst_shape.dimensions(dim) -
                                           padded_src_shape.dimensions(dim));
    }
    result = b->AddInstruction(
        HloInstruction::CreatePad(padded_shape, result, zero, pad_config));
  }

  return result;
}

absl::optional<int64> UniqueTiledDim(const HloSharding& sharding) {
  if (sharding.IsTileMaximal()) {
    return absl::nullopt;
  }
  int64_t dim = -1;
  int64_t rank = sharding.ReplicateOnLastTileDim()
                     ? sharding.tile_assignment().num_dimensions() - 1
                     : sharding.tile_assignment().num_dimensions();
  for (int64_t i = 0; i < rank; ++i) {
    if (sharding.tile_assignment().dim(i) > 1) {
      if (dim != -1) {
        return absl::nullopt;
      }
      dim = i;
    }
  }
  CHECK_NE(dim, -1);
  return dim;
}

MultiplyAddDivideOffsetCalculation::MultiplyAddDivideOffsetCalculation(
    int64_t multiplier, int64_t offset, int64_t divisor)
    : multiplier_(multiplier), offset_(offset), divisor_(divisor) {
  CHECK_GT(divisor_, 0);
  Simplify();
}

OffsetCalculation MultiplyAddDivideOffsetCalculation::operator-(
    const MultiplyAddDivideOffsetCalculation& other) const {
  if (divisor_ == 1 && other.divisor_ == 1) {
    return OffsetCalculation(MultiplyAddDivideOffsetCalculation(
        multiplier_ - other.multiplier_, offset_ - other.offset_, 1));
  }
  return OffsetCalculation(HloOpcode::kSubtract, *this, other);
}

void MultiplyAddDivideOffsetCalculation::Simplify() {
  // We could simplify the calculation when multiplier is a multiple of
  // divisor_. However, when offset_ is not a multiple of divisor_, we must
  // make sure that offset_ and multiplier_ are both non-negative or both
  // non-positive. E.g., (3 * i  - 1) / 3 is not equivalent to i or i - 1.
  if (divisor_ != 1 && multiplier_ % divisor_ == 0 &&
      (offset_ % divisor_ == 0 || offset_ * multiplier_ > 0)) {
    multiplier_ /= divisor_;
    offset_ /= divisor_;
    divisor_ = 1;
  }
}

int64 MultiplyAddDivideOffsetCalculation::Calculate(
    int64_t shard_ordinal) const {
  return (shard_ordinal * multiplier_ + offset_) / divisor_;
}

HloInstruction* MultiplyAddDivideOffsetCalculation::Calculate(
    HloInstruction* shard_ordinal, SpmdBuilder* b) const {
  auto scalar_shape = ShapeUtil::MakeShape(S32, {});
  if (multiplier_ == 0) {
    return b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32>(offset_ / divisor_)));
  }
  HloInstruction* result = shard_ordinal;
  if (multiplier_ != 1) {
    result = b->AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kMultiply, shard_ordinal,
        b->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32>(multiplier_)))));
  }
  if (offset_ != 0) {
    auto offset = b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(offset_)));
    result = b->AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kAdd, result, offset));
  }
  if (divisor_ != 1) {
    auto divisor = b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(divisor_)));
    result = b->AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kDivide, result, divisor));
  }
  return result;
}

int64 MultiplyAddDivideOffsetCalculation::MaxInRange(
    int64_t start_ordinal, int64_t limit_ordinal) const {
  int64_t max = Calculate(start_ordinal);
  for (int64_t i = start_ordinal + 1; i < limit_ordinal; ++i) {
    max = std::max(max, Calculate(i));
  }
  return max;
}

OffsetCalculation& OffsetCalculation::operator=(
    const OffsetCalculation& other) {
  opcode_ = other.opcode_;
  copy_from_ = other.copy_from_;
  if (opcode_ != HloOpcode::kCopy) {
    lhs_ = absl::make_unique<OffsetCalculation>(*other.lhs_);
    rhs_ = absl::make_unique<OffsetCalculation>(*other.rhs_);
  }
  return *this;
}

bool OffsetCalculation::IsConstant() const {
  if (opcode_ == HloOpcode::kCopy) {
    return copy_from_.IsConstant();
  }
  if (opcode_ == HloOpcode::kSubtract && *lhs_ == *rhs_) {
    return true;
  }
  return lhs_->IsConstant() && rhs_->IsConstant();
}

OffsetCalculation OffsetCalculation::operator-(
    const OffsetCalculation& other) const {
  if (opcode_ == HloOpcode::kCopy && other.opcode_ == HloOpcode::kCopy) {
    return copy_from_ - other.copy_from_;
  }
  return OffsetCalculation(HloOpcode::kSubtract, *this, other);
}

bool OffsetCalculation::operator==(const OffsetCalculation& other) const {
  if (opcode_ != other.opcode_) {
    return false;
  }
  if (opcode_ == HloOpcode::kCopy) {
    return copy_from_ == other.copy_from_;
  }
  return *lhs_ == *other.lhs_ && *rhs_ == *other.rhs_;
}

int64 OffsetCalculation::Calculate(int64_t shard_ordinal) const {
  switch (opcode_) {
    case HloOpcode::kCopy:
      return copy_from_.Calculate(shard_ordinal);
    case HloOpcode::kSubtract:
      return lhs_->Calculate(shard_ordinal) - rhs_->Calculate(shard_ordinal);
    case HloOpcode::kMultiply:
      return lhs_->Calculate(shard_ordinal) * rhs_->Calculate(shard_ordinal);
    default:
      LOG(FATAL) << "Should not happen";
  }
}

HloInstruction* OffsetCalculation::Calculate(HloInstruction* shard_ordinal,
                                             SpmdBuilder* b) const {
  if (opcode_ == HloOpcode::kCopy) {
    return copy_from_.Calculate(shard_ordinal, b);
  }
  auto lhs = lhs_->Calculate(shard_ordinal, b);
  auto rhs = rhs_->Calculate(shard_ordinal, b);
  return b->AddInstruction(
      HloInstruction::CreateBinary(lhs->shape(), opcode_, lhs, rhs));
}

int64 OffsetCalculation::MaxInRange(int64_t start_ordinal,
                                    int64_t limit_ordinal) const {
  if (IsConstant()) {
    return Calculate(start_ordinal);
  }
  if (opcode_ == HloOpcode::kCopy) {
    return std::max(Calculate(start_ordinal), Calculate(limit_ordinal - 1));
  }
  int64_t max = Calculate(start_ordinal);
  for (int64_t i = start_ordinal + 1; i < limit_ordinal; ++i) {
    max = std::max(max, Calculate(i));
  }
  return max;
}

absl::optional<HloInstruction*> ExchangeHalo(
    HloInstruction* hlo, const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function, int64_t dim,
    const HloSharding& target,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, SpmdBuilder* b) {
  int64_t input_shard_size = hlo->shape().dimensions(dim);
  int64_t shard_count = target.tile_assignment().dim(dim);

  std::vector<HloInstruction*> concat_pieces;

  int64_t max_left_halo_size =
      left_halo_size_function.MaxInRange(1, shard_count);
  int64_t max_right_halo_size =
      right_halo_size_function.MaxInRange(0, shard_count - 1);
  if (max_left_halo_size + max_right_halo_size + input_shard_size >=
          input_shard_size * shard_count &&
      (max_left_halo_size > input_shard_size ||
       max_right_halo_size > input_shard_size)) {
    return absl::nullopt;
  }
  // Since max halo sizes could be negative, we only need to include data within
  // certain bounds. Useful region is [left_bound, right_bound).
  const int64_t left_bound =
      -left_halo_size_function.MaxInRange(0, shard_count);
  const int64_t right_bound =
      input_shard_size + right_halo_size_function.MaxInRange(0, shard_count);
  if (left_bound >= right_bound) {
    return absl::nullopt;
  }
  // Left halo.
  for (int64_t i = CeilOfRatio(max_left_halo_size, input_shard_size) - 1;
       i >= 0 && (-i - 1) * input_shard_size < right_bound; --i) {
    std::vector<std::pair<int64, int64>> source_target_pairs;
    target.tile_assignment().Each(
        [&](absl::Span<const int64> indices, int64_t device) {
          if (indices[dim] > i) {
            std::vector<int64> source_indices(indices.begin(), indices.end());
            source_indices[dim] -= i + 1;
            source_target_pairs.emplace_back(
                target.tile_assignment()(source_indices), device);
          }
        });
    int64_t halo_size_including_skips =
        std::min(max_left_halo_size - input_shard_size * i, input_shard_size);
    int64_t halo_right_skips =
        std::max<int64>(-i * input_shard_size - right_bound, 0);
    int64_t halo_size = halo_size_including_skips - halo_right_skips;
    auto halo_shape = hlo->shape();
    auto source_halo_slice = hlo;
    if (halo_size != hlo->shape().dimensions(dim)) {
      halo_shape.set_dimensions(dim, halo_size);
      std::vector<int64> halo_start_indices(halo_shape.rank(), 0);
      halo_start_indices[dim] =
          hlo->shape().dimensions(dim) - halo_size_including_skips;
      std::vector<int64> halo_limit_indices(hlo->shape().dimensions().begin(),
                                            hlo->shape().dimensions().end());
      halo_limit_indices[dim] -= halo_right_skips;
      std::vector<int64> halo_slice_strides(halo_shape.rank(), 1);
      source_halo_slice = b->AddInstruction(
          HloInstruction::CreateSlice(halo_shape, hlo, halo_start_indices,
                                      halo_limit_indices, halo_slice_strides));
    }
    auto left_halo =
        collective_ops_creator.create_cross_partition_collective_permute(
            b, source_halo_slice, source_target_pairs, (*next_channel_id)++);
    concat_pieces.push_back(left_halo);
  }

  if (left_bound < input_shard_size && right_bound > 0) {
    int64_t self_start = std::max<int64>(0, left_bound);
    int64_t self_limit = std::min<int64>(input_shard_size, right_bound);
    if (self_start == 0 && self_limit == input_shard_size) {
      concat_pieces.push_back(hlo);
    } else {
      auto self_shape = hlo->shape();
      self_shape.set_dimensions(dim, self_limit - self_start);
      std::vector<int64> start_indices(self_shape.rank(), 0);
      start_indices[dim] = self_start;
      std::vector<int64> limit_indices(hlo->shape().dimensions().begin(),
                                       hlo->shape().dimensions().end());
      limit_indices[dim] = self_limit;
      std::vector<int64> slice_strides(self_shape.rank(), 1);
      concat_pieces.push_back(b->AddInstruction(HloInstruction::CreateSlice(
          self_shape, hlo, start_indices, limit_indices, slice_strides)));
    }
  }

  int64_t skipped_right_halos =
      std::min<int64>(std::max<int64>(left_bound - input_shard_size, 0),
                      std::max<int64>(max_right_halo_size, 0)) /
      input_shard_size;
  // Right halo.
  for (int64_t i = skipped_right_halos;
       i < CeilOfRatio(max_right_halo_size, input_shard_size); ++i) {
    std::vector<std::pair<int64, int64>> source_target_pairs;
    target.tile_assignment().Each(
        [&](absl::Span<const int64> indices, int64_t device) {
          if (indices[dim] > i) {
            std::vector<int64> target_indices(indices.begin(), indices.end());
            target_indices[dim] -= i + 1;
            source_target_pairs.emplace_back(
                device, target.tile_assignment()(target_indices));
          }
        });
    int64_t halo_size_including_skips =
        std::min(max_right_halo_size - input_shard_size * i, input_shard_size);
    int64_t halo_left_skips =
        std::max<int64>(left_bound - (i + 1) * input_shard_size, 0);
    int64_t halo_size = halo_size_including_skips - halo_left_skips;
    auto halo_shape = hlo->shape();
    HloInstruction* source_halo_slice = hlo;
    if (halo_size != halo_shape.dimensions(dim)) {
      halo_shape.set_dimensions(dim, halo_size);
      std::vector<int64> halo_start_indices(halo_shape.rank(), 0);
      halo_start_indices[dim] = halo_left_skips;
      std::vector<int64> halo_limit_indices(halo_shape.dimensions().begin(),
                                            halo_shape.dimensions().end());
      halo_limit_indices[dim] += halo_left_skips;
      std::vector<int64> halo_slice_strides(halo_shape.rank(), 1);
      source_halo_slice = b->AddInstruction(
          HloInstruction::CreateSlice(halo_shape, hlo, halo_start_indices,
                                      halo_limit_indices, halo_slice_strides));
    }
    auto right_halo =
        collective_ops_creator.create_cross_partition_collective_permute(
            b, source_halo_slice, source_target_pairs, (*next_channel_id)++);
    concat_pieces.push_back(right_halo);
  }

  auto concat = concat_pieces[0];
  // Concat with halos/padding.
  if (concat_pieces.size() > 1) {
    auto concat_shape = hlo->shape();
    int64_t concat_dim_size = 0;
    for (auto piece : concat_pieces) {
      concat_dim_size += piece->shape().dimensions(dim);
    }
    concat_shape.set_dimensions(dim, concat_dim_size);
    concat = b->AddInstruction(
        HloInstruction::CreateConcatenate(concat_shape, concat_pieces, dim));
  }

  return concat;
}

absl::optional<HloInstruction*> ExchangeHalo(
    HloInstruction* hlo,
    std::vector<OffsetCalculation> left_halo_size_functions,
    std::vector<OffsetCalculation> right_halo_size_functions,
    const HloSharding& target,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, SpmdBuilder* b) {
  CHECK(left_halo_size_functions.size() == hlo->shape().rank());
  CHECK(right_halo_size_functions.size() == hlo->shape().rank());

  HloInstruction* visiting_hlo = hlo;
  for (int dim = 0; dim < hlo->shape().rank(); ++dim) {
    auto concat = ExchangeHalo(visiting_hlo, left_halo_size_functions[dim],
                               right_halo_size_functions[dim], dim, target,
                               collective_ops_creator, next_channel_id, b);
    if (!concat) {
      return absl::nullopt;
    }
    visiting_hlo = *concat;
  }
  return visiting_hlo;
}

absl::optional<HloInstruction*> ExchangeHaloAndGetValidData(
    HloInstruction* hlo, const Shape& base_shape,
    const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function,
    int64_t explicit_left_padding_on_full_shape, int64_t padded_full_shape_size,
    int64_t shard_size_with_halo, int64_t dim, const HloSharding& target,
    HloInstruction* offset_on_padded_shape, HloInstruction* pad_value,
    HloInstruction* partition_ordinal,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, SpmdBuilder* b, bool mask_invalid_region) {
  auto halo_exchange_result =
      ExchangeHalo(hlo, left_halo_size_function, right_halo_size_function, dim,
                   target, collective_ops_creator, next_channel_id, b);
  if (!halo_exchange_result) {
    return absl::nullopt;
  }
  auto concat = *halo_exchange_result;
  int64_t shard_count = target.tile_assignment().dim(dim);
  int64_t max_left_halo_size =
      left_halo_size_function.MaxInRange(1, shard_count);

  // Now we determine if we need extra padding after the concat.
  //
  // The max of halo size or the first shard's explicit left padding.
  int64_t max_left_halo_or_padding_size =
      std::max(max_left_halo_size, explicit_left_padding_on_full_shape);
  // The calculation that returns the dynamic slice index for a shard on the
  // padded concat, which is the difference between
  // max_left_halo_or_padding_size and its left halo size.
  auto start_offset_on_padded_concat_calculation =
      OffsetCalculation(MultiplyAddDivideOffsetCalculation(
          0, max_left_halo_or_padding_size, 1)) -
      left_halo_size_function;

  // See if we need to pad the concat before dynamic slice.
  int64_t extra_left_padding =
      std::max(int64{0}, max_left_halo_or_padding_size -
                             std::max(int64{0}, max_left_halo_size));
  int64_t extra_right_padding =
      start_offset_on_padded_concat_calculation.MaxInRange(0, shard_count) +
      shard_size_with_halo - concat->shape().dimensions(dim) -
      extra_left_padding;
  extra_right_padding = std::max(int64{0}, extra_right_padding);
  if (extra_left_padding > 0 || extra_right_padding > 0) {
    PaddingConfig padding_config;
    auto padded_concat_shape = concat->shape();
    for (int64_t i = 0; i < base_shape.rank(); ++i) {
      auto padding_config_dim = padding_config.add_dimensions();
      padding_config_dim->set_interior_padding(0);
      padding_config_dim->set_edge_padding_low(0);
      padding_config_dim->set_edge_padding_high(0);
      if (i != dim) {
        continue;
      }
      padding_config_dim->set_edge_padding_low(extra_left_padding);
      padding_config_dim->set_edge_padding_high(extra_right_padding);
      padded_concat_shape.set_dimensions(dim, concat->shape().dimensions(dim) +
                                                  extra_left_padding +
                                                  extra_right_padding);
    }
    concat = b->AddInstruction(HloInstruction::CreatePad(
        padded_concat_shape, concat, pad_value, padding_config));
  }

  auto valid_slice = concat;
  if (shard_size_with_halo != concat->shape().dimensions(dim)) {
    // Concat is bigger than the shard shape, so we need a dynamic slice.
    CHECK_LT(shard_size_with_halo, concat->shape().dimensions(dim));
    auto slice_shape = concat->shape();
    slice_shape.set_dimensions(dim, shard_size_with_halo);

    if (left_halo_size_function.IsConstant() &&
        left_halo_size_function.Calculate(0) ==
            explicit_left_padding_on_full_shape) {
      std::vector<int64> start_indices(slice_shape.rank(), 0);
      std::vector<int64> strides(slice_shape.rank(), 1);
      valid_slice = b->AddInstruction(
          HloInstruction::CreateSlice(slice_shape, concat, start_indices,
                                      slice_shape.dimensions(), strides));
    } else {
      auto zero = b->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
      std::vector<HloInstruction*> slice_offsets(base_shape.rank(), zero);
      slice_offsets[dim] = start_offset_on_padded_concat_calculation.Calculate(
          partition_ordinal, b);
      valid_slice = b->AddInstruction(HloInstruction::CreateDynamicSlice(
          slice_shape, concat, slice_offsets, slice_shape.dimensions()));
    }
  }

  if (!mask_invalid_region) {
    return valid_slice;
  }

  int64_t total_right_padding = padded_full_shape_size -
                                base_shape.dimensions(dim) -
                                explicit_left_padding_on_full_shape;
  // Mask off garbage data due to uneven partition or low/high padding.
  if (explicit_left_padding_on_full_shape > 0 || total_right_padding > 0) {
    auto index_shape = ShapeUtil::ChangeElementType(valid_slice->shape(), S32);
    auto iota = b->AddInstruction(HloInstruction::CreateIota(index_shape, dim));
    auto broadcast_start_index_in_padded_shape =
        b->AddInstruction(HloInstruction::CreateBroadcast(
            index_shape, offset_on_padded_shape, {}));
    auto index_in_padded_shape = b->AddInstruction(
        HloInstruction::CreateBinary(index_shape, HloOpcode::kAdd, iota,
                                     broadcast_start_index_in_padded_shape));
    auto mask_shape = ShapeUtil::ChangeElementType(index_shape, PRED);
    std::vector<HloInstruction*> predicates;
    if (explicit_left_padding_on_full_shape > 0) {
      auto valid_index_start =
          b->AddInstruction(HloInstruction::CreateBroadcast(
              index_shape,
              b->AddInstruction(
                  HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(
                      explicit_left_padding_on_full_shape))),
              {}));
      predicates.push_back(b->AddInstruction(HloInstruction::CreateCompare(
          mask_shape, index_in_padded_shape, valid_index_start,
          ComparisonDirection::kGe)));
    }
    if (total_right_padding > 0) {
      auto valid_index_limit =
          b->AddInstruction(HloInstruction::CreateBroadcast(
              index_shape,
              b->AddInstruction(
                  HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(
                      base_shape.dimensions(dim) +
                      explicit_left_padding_on_full_shape))),
              {}));
      predicates.push_back(b->AddInstruction(HloInstruction::CreateCompare(
          mask_shape, index_in_padded_shape, valid_index_limit,
          ComparisonDirection::kLt)));
    }
    CHECK(!predicates.empty());
    auto is_valid =
        predicates.size() == 2
            ? b->AddInstruction(HloInstruction::CreateBinary(
                  mask_shape, HloOpcode::kAnd, predicates[0], predicates[1]))
            : predicates[0];
    auto masking_value = b->AddInstruction(
        HloInstruction::CreateBroadcast(valid_slice->shape(), pad_value, {}));
    valid_slice = b->AddInstruction(
        HloInstruction::CreateTernary(valid_slice->shape(), HloOpcode::kSelect,
                                      is_valid, valid_slice, masking_value));
  }
  return valid_slice;
}

HloInstruction* HaloExchangeToPadOnLeft(PartitionedHlo& original,
                                        absl::Span<const int64> dims) {
  if (original.sharding().IsTileMaximal()) {
    return original.hlo();
  }
  // Create a window config to halo exchange for unevenly partitioned reverse
  // dimensions.
  Window window;
  for (int64_t i = 0; i < original.base_shape().rank(); ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_stride(1);
    dim->set_window_dilation(1);
    dim->set_window_reversal(false);
    int64_t low_padding = 0;
    if (absl::c_linear_search(dims, i)) {
      low_padding =
          RoundUpToNearest(original.base_shape().dimensions(i),
                           original.sharding().tile_assignment().dim(i)) -
          original.base_shape().dimensions(i);
    }
    dim->set_padding_low(low_padding);
    dim->set_padding_high(0);
    dim->set_base_dilation(1);
  }

  auto reshard_window = original.ReshardAsWindowedInput(
      window, original.sharding(),
      CreateZero(ShapeUtil::MakeShape(original.base_shape().element_type(), {}),
                 original.state().b),
      /*mask_invalid_region=*/false);
  if (!reshard_window.has_value()) {
    return nullptr;
  }
  CHECK(!reshard_window->dynamic_slice_index_on_output.has_value());
  return reshard_window->sharded_input;
}

bool IsNanSafeGt(HloComputation* comp) {
  namespace m = match;
  auto match_bitcast_f32 = [](int64_t parameter_number) {
    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    return m::Select(
        m::Lt(param_s32, m::ConstantScalar(0)),
        m::BitcastConvert(
            m::Subtract(m::ConstantScalar(std::numeric_limits<int32>::max()),
                        param_u32))
            .WithShape(m::Shape().WithElementType(S32)),
        param_s32);
  };
  auto match_bitcast_bf16 = [](int64_t parameter_number) {
    auto param = m::Convert(m::Parameter(parameter_number)
                                .WithShape(m::Shape().WithElementType(BF16)))
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    return m::Select(
        m::Lt(param_s32, m::ConstantScalar(0)),
        m::BitcastConvert(
            m::Subtract(m::ConstantScalar(std::numeric_limits<int32>::max()),
                        param_u32))
            .WithShape(m::Shape().WithElementType(S32)),
        param_s32);
  };
  // If root instruction is kSelect and compares indices if values are equal.
  if (comp->root_instruction()->opcode() == HloOpcode::kSelect) {
    return Match(comp->root_instruction()->operand(2),
                 m::Gt(match_bitcast_f32(0), match_bitcast_f32(1))) ||
           Match(comp->root_instruction()->operand(2),
                 m::Gt(match_bitcast_bf16(0), match_bitcast_bf16(1)));
  }
  return Match(comp->root_instruction(),
               m::Gt(match_bitcast_f32(0), match_bitcast_f32(1))) ||
         Match(comp->root_instruction(),
               m::Gt(match_bitcast_bf16(0), match_bitcast_bf16(1)));
}

absl::optional<int64> GetKValueInTopKWhenPartitionSortDim(HloInstruction* hlo) {
  HloSortInstruction* sort = DynCast<HloSortInstruction>(hlo);
  if (sort == nullptr || sort->operand_count() != 2) {
    return absl::nullopt;
  }
  if (!IsNanSafeGt(sort->to_apply())) {
    return absl::nullopt;
  }
  HloInstruction* data = sort->mutable_operand(0);
  HloIotaInstruction* iota =
      DynCast<HloIotaInstruction>(sort->mutable_operand(1));
  const PrimitiveType element_type = data->shape().element_type();
  if (iota == nullptr || iota->shape().element_type() != S32 ||
      iota->opcode() != HloOpcode::kIota ||
      iota->iota_dimension() != sort->sort_dimension()) {
    return absl::nullopt;
  }

  const int64_t sort_dim = sort->sort_dimension();

  if (element_type != F32 && element_type != BF16 && element_type != S32 &&
      element_type != U32) {
    return absl::nullopt;
  }

  bool supported = true;
  absl::optional<int64> k;
  for (HloInstruction* gte : sort->users()) {
    if (gte->opcode() != HloOpcode::kGetTupleElement) {
      supported = false;
      break;
    }

    const HloInstruction* slice = gte->users()[0];
    if (slice->opcode() != HloOpcode::kSlice) {
      // Non-slice user means we are not doing a TopK
      supported = false;
      break;
    }
    if (absl::c_any_of(slice->slice_starts(), [](int x) { return x != 0; }) ||
        absl::c_any_of(slice->slice_strides(), [](int x) { return x != 1; })) {
      // Strided slice or slicing at the beginning isn't supported.
      supported = false;
      break;
    }
    for (int64_t dim = 0; dim < data->shape().dimensions_size(); dim++) {
      if (dim == sort_dim) {
        continue;
      }
      if (slice->slice_limits(dim) !=
          slice->operand(0)->shape().dimensions(dim)) {
        // Slicing along the other dimension isn't supported.
        supported = false;
        break;
      }
    }
    if (!k.has_value()) {
      k = slice->slice_limits(sort_dim);
    } else if (k != slice->slice_limits(sort_dim)) {
      // Different k for the different operands isn't supported.
      supported = false;
      break;
    }
  }
  if (k == absl::nullopt || !supported) {
    return absl::nullopt;
  }

  // Only support when sort dim is sharded.
  if (!data->has_sharding()) {
    return absl::nullopt;
  }
  const HloSharding& sharding = sort->operand(0)->sharding();

  if (sharding.IsTileMaximal()) {
    return absl::nullopt;
  }

  // Check if partitioned at sort dimension.
  for (int64_t dim = 0; dim < sort->shape().tuple_shapes(0).dimensions_size();
       ++dim) {
    if (sharding.tile_assignment().dim(dim) > 1) {
      if (dim != sort_dim) {
        return absl::nullopt;
      }
    }
  }

  // Checks if partition size is smaller than k.
  const int64_t shard_count = sharding.tile_assignment().dim(sort_dim);

  if (shard_count <= 1) {
    return absl::nullopt;
  }

  const int64_t input_size = hlo->operand(0)->shape().dimensions(sort_dim);
  const int64_t per_partition_size = CeilOfRatio(input_size, shard_count);

  if (k.value() >= per_partition_size) {
    return absl::nullopt;
  }

  return k;
}

// Slice first k elements from sort_dim.
HloInstruction* SliceFirstK(HloInstruction* hlo, SpmdBuilder* builder,
                            int64_t slice_dim, int64_t k) {
  const Shape& hlo_shape = hlo->shape();
  auto hlo_dims = hlo_shape.dimensions();
  std::vector<int64> start_indices(hlo_shape.dimensions_size(), 0);
  std::vector<int64> limit_indices(hlo_dims.begin(), hlo_dims.end());
  std::vector<int64> strides(hlo_shape.dimensions_size(), 1);
  limit_indices[slice_dim] = k;
  auto output_shape = hlo_shape;
  output_shape.set_dimensions(slice_dim, k);
  return builder->AddInstruction(HloInstruction::CreateSlice(
      output_shape, hlo, start_indices, limit_indices, strides));
}

// Check if a dimension is sharded.
int64 ShardCountAtDim(const HloSharding& sharding, int64_t dim) {
  if (sharding.IsTileMaximal()) {
    return 1;
  }
  return sharding.tile_assignment().dim(dim);
}

absl::optional<std::vector<std::pair<int64, int64>>>
GetReshardAllToAllSourceTargetDims(const HloSharding& source,
                                   const HloSharding& target) {
  if (source.IsTileMaximal() || target.IsTileMaximal() ||
      source.tile_assignment().num_dimensions() !=
          target.tile_assignment().num_dimensions() ||
      source.NumTiles() != target.NumTiles()) {
    return absl::nullopt;
  }
  // Record partition count to index for indices that have different partition
  // counts on source and target.
  std::map<int64, std::vector<int64>> source_size_to_dim;
  std::map<int64, std::vector<int64>> target_size_to_dim;
  for (int64_t i = 0; i < source.tile_assignment().num_dimensions(); ++i) {
    if (source.tile_assignment().dim(i) == target.tile_assignment().dim(i)) {
      continue;
    }
    source_size_to_dim[source.tile_assignment().dim(i)].push_back(i);
    target_size_to_dim[target.tile_assignment().dim(i)].push_back(i);
  }
  // In order to shard via AllToAll, source_size_to_dim and target_size_to_dim
  // must have the same distribution.
  if (source_size_to_dim.empty() ||
      source_size_to_dim.size() != target_size_to_dim.size()) {
    return absl::nullopt;
  }
  for (const auto& entry : source_size_to_dim) {
    auto target_it = target_size_to_dim.find(entry.first);
    if (target_it == target_size_to_dim.end() ||
        target_it->second.size() != entry.second.size()) {
      return absl::nullopt;
    }
  }
  std::vector<std::pair<int64, int64>> result;
  auto remove_entry = [](int64_t size, int64_t dim,
                         std::map<int64, std::vector<int64>>& size_to_dim) {
    size_to_dim[size].erase(
        std::remove_if(size_to_dim[size].begin(), size_to_dim[size].end(),
                       [dim](int64_t a) { return a == dim; }),
        size_to_dim[size].end());
    if (size_to_dim[size].empty()) {
      size_to_dim.erase(size);
    }
  };
  // Find one pair of dimensions to swap at a time.
  while (!source_size_to_dim.empty()) {
    int64_t source_size = source_size_to_dim.begin()->first;
    int64_t i = source_size_to_dim.begin()->second.back();
    int64_t target_i_size = target.tile_assignment().dim(i);
    if (target_i_size == source_size) {
      remove_entry(source_size, i, source_size_to_dim);
      remove_entry(source_size, i, target_size_to_dim);
      continue;
    }
    auto j_it = source_size_to_dim[target_i_size].begin();
    int64_t j = *j_it;
    if (source_size == 1) {
      // If possible, find a j where the target partition count is not one, so
      // that when we swap, the resulting size-1 dimension will still be useful
      // to other dimensions.
      while (target.tile_assignment().dim(j) == 1) {
        if (++j_it == source_size_to_dim[target_i_size].end()) {
          break;
        }
        j = *j_it;
      }
    } else if (target_i_size % source_size == 0) {
      // If possible, find a j where the target partition count is source_size,
      // so that we can do a single swap.
      while (target.tile_assignment().dim(j) != source_size) {
        if (++j_it == source_size_to_dim[target_i_size].end()) {
          break;
        }
        j = *j_it;
      }
    } else {
      return absl::nullopt;
    }
    result.emplace_back(j, i);
    remove_entry(target_i_size, i, target_size_to_dim);
    source_size_to_dim.begin()->second.back() = j;
    remove_entry(target_i_size, j, source_size_to_dim);
  }
  return result;
}

bool CanReshardWithCollectivePermute(const HloSharding& source,
                                     const HloSharding& target) {
  return !source.IsTileMaximal() && !target.IsTileMaximal() &&
         source.tile_assignment().dimensions() ==
             target.tile_assignment().dimensions() &&
         source.ReplicateOnLastTileDim() == target.ReplicateOnLastTileDim() &&
         source.tile_assignment() != target.tile_assignment();
}

GroupedSharding GroupShardingOnDims(const HloSharding& sharding,
                                    absl::Span<const int64> group_dims) {
  std::vector<int64> group_dim_shards(group_dims.size(), 1);
  return GroupShardingOnDims(sharding, group_dims, group_dim_shards);
}

GroupedSharding GroupShardingOnDims(const HloSharding& sharding,
                                    absl::Span<const int64> group_dims,
                                    absl::Span<const int64> group_dim_shards) {
  CHECK(!sharding.IsTileMaximal());
  std::vector<int64> grouped_tiling_dims =
      sharding.tile_assignment().dimensions();
  std::vector<int64> group_dim_sizes(group_dims.size());
  for (int64_t i = 0; i < group_dims.size(); ++i) {
    CHECK_EQ(grouped_tiling_dims[group_dims[i]] % group_dim_shards[i], 0);
    group_dim_sizes[i] =
        grouped_tiling_dims[group_dims[i]] / group_dim_shards[i];
    grouped_tiling_dims[group_dims[i]] = group_dim_shards[i];
  }

  std::vector<std::vector<int64>> device_groups(Product(group_dim_sizes));
  sharding.tile_assignment().Each([&](absl::Span<const int64> indices,
                                      int64_t device) {
    int64_t group_id = 0;
    for (int64_t i = 0; i < group_dims.size(); ++i) {
      group_id *=
          sharding.tile_assignment().dim(group_dims[i]) / group_dim_shards[i];
      group_id += indices[group_dims[i]] / group_dim_shards[i];
    }
    device_groups[group_id].push_back(device);
  });
  auto grouped = GroupedSharding(
      std::move(device_groups),
      std::vector<int64>(group_dims.begin(), group_dims.end()),
      std::move(group_dim_sizes), sharding.tile_assignment().num_dimensions(),
      HloSharding::Replicate());
  if (sharding.ReplicateOnLastTileDim()) {
    grouped.data_rank--;
  }
  if (Product(grouped_tiling_dims) == 1 ||
      (sharding.ReplicateOnLastTileDim() &&
       Product(grouped_tiling_dims) == grouped_tiling_dims.back())) {
    return grouped;
  }
  if (sharding.ReplicateOnLastTileDim() && grouped_tiling_dims.back() == 1) {
    grouped_tiling_dims.pop_back();
  }
  Array<int64> grouped_tiling(grouped_tiling_dims);
  grouped_tiling.FillIota(0);
  grouped.sharding = sharding.ReplicateOnLastTileDim() &&
                             grouped_tiling_dims.size() ==
                                 sharding.tile_assignment().num_dimensions()
                         ? HloSharding::PartialTile(grouped_tiling)
                         : HloSharding::Tile(grouped_tiling);
  return grouped;
}

HloSharding UngroupSharding(const GroupedSharding& grouped_sharding) {
  std::vector<int64> tiling_dims;
  bool partial_sharding = false;
  auto grouped_tiling = grouped_sharding.sharding.tile_assignment();
  if (grouped_sharding.sharding.IsTileMaximal()) {
    tiling_dims = std::vector<int64>(grouped_sharding.data_rank, 1);
    if (grouped_sharding.device_groups[0].size() != 1) {
      // This is partial sharding.
      tiling_dims.push_back(grouped_sharding.device_groups[0].size());
      partial_sharding = true;
    }
    grouped_tiling = Array<int64>(tiling_dims);
    grouped_tiling.FillIota(0);
  } else {
    partial_sharding = grouped_sharding.sharding.ReplicateOnLastTileDim();
    tiling_dims = grouped_sharding.sharding.tile_assignment().dimensions();
    if (absl::c_linear_search(grouped_sharding.group_dims,
                              tiling_dims.size())) {
      tiling_dims.push_back(1);
      grouped_tiling.Reshape(tiling_dims);
      partial_sharding = true;
    }
  }
  for (int64_t i = 0; i < grouped_sharding.group_dims.size(); ++i) {
    int64_t dim = grouped_sharding.group_dims[i];
    tiling_dims[dim] *= grouped_sharding.group_dim_sizes[i];
  }
  Array<int64> tiling(tiling_dims);
  grouped_tiling.Each([&](absl::Span<const int64> indices, int64_t device) {
    std::vector<int64> ungrouped_inds(indices.begin(), indices.end());
    for (int64_t g = 0; g < grouped_sharding.device_groups.size(); ++g) {
      int64_t remaining_group_index = g;
      for (int64_t i = grouped_sharding.group_dims.size() - 1; i >= 0; --i) {
        int64_t dim = grouped_sharding.group_dims[i];
        int64_t groups_in_this_dim = grouped_sharding.group_dim_sizes[i];
        ungrouped_inds[dim] = (remaining_group_index % groups_in_this_dim) *
                                  grouped_tiling.dim(dim) +
                              indices[dim];
        remaining_group_index /= groups_in_this_dim;
      }
      tiling(ungrouped_inds) = grouped_sharding.device_groups[g][device];
    }
  });
  return partial_sharding ? HloSharding::PartialTile(tiling)
                          : HloSharding::Tile(tiling);
}

GroupedSharding AlignGroupsWith(GroupedSharding grouped_sharding,
                                const GroupedSharding& reference,
                                bool ignore_group_order) {
  // Returns src -> dst index mapping.
  auto get_permutation = [](absl::Span<const int64> src,
                            absl::Span<const int64> dst) {
    CHECK_EQ(src.size(), dst.size());
    absl::flat_hash_map<int64, int64> dst_reverse_map;
    for (int64_t i = 0; i < dst.size(); ++i) {
      dst_reverse_map[dst[i]] = i;
    }
    std::vector<int64> permutation(src.size());
    for (int64_t i = 0; i < src.size(); ++i) {
      auto it = dst_reverse_map.find(src[i]);
      CHECK(it != dst_reverse_map.end());
      permutation[i] = it->second;
    }
    return permutation;
  };
  CHECK_EQ(grouped_sharding.device_groups.size(),
           reference.device_groups.size());
  absl::flat_hash_map<int64, int64> device_to_ref_group;
  for (int64_t g = 0; g < reference.device_groups.size(); ++g) {
    for (int64_t device : reference.device_groups[g]) {
      device_to_ref_group[device] = g;
    }
  }
  auto unique_ref_dev_group = [&](absl::Span<const int64> devices) -> int64 {
    int64_t ref_g = -1;
    for (int64_t device : devices) {
      if (ref_g == -1) {
        ref_g = device_to_ref_group[device];
      } else if (ref_g != device_to_ref_group[device]) {
        return -1;
      }
    }
    return ref_g;
  };
  bool matching_groups = true;
  std::vector<int64> original_src_to_ref_permutation;
  for (int64_t g = 0; g < grouped_sharding.device_groups.size(); ++g) {
    int64_t ref_g = unique_ref_dev_group(grouped_sharding.device_groups[g]);
    if (ref_g < 0 || (!ignore_group_order && g != ref_g)) {
      matching_groups = false;
      break;
    }
    if (g == 0) {
      original_src_to_ref_permutation = get_permutation(
          grouped_sharding.device_groups[g], reference.device_groups[ref_g]);
    }
  }
  if (matching_groups && !grouped_sharding.sharding.IsTileMaximal()) {
    auto tiles = grouped_sharding.sharding.tile_assignment();
    tiles.Each([&](absl::Span<const int64> indices, int64* device) {
      *device = original_src_to_ref_permutation[*device];
    });
    grouped_sharding.sharding =
        grouped_sharding.sharding.ReplicateOnLastTileDim()
            ? HloSharding::PartialTile(tiles)
            : HloSharding::Tile(tiles);
  }
  grouped_sharding.device_groups = std::move(reference.device_groups);
  return grouped_sharding;
}

HloSharding AlignShardingOnDims(const HloSharding& sharding,
                                absl::Span<const int64> sharding_dims,
                                const HloSharding& reference,
                                absl::Span<const int64> reference_dims) {
  auto sharding_grouped = GroupShardingOnDims(sharding, sharding_dims);
  auto reference_grouped = GroupShardingOnDims(reference, reference_dims);
  return UngroupSharding(AlignGroupsWith(sharding_grouped, reference_grouped));
}

Shape GetPerGroupBaseShape(const GroupedSharding& grouped_sharding,
                           const Shape& original_base_shape) {
  auto result = original_base_shape;
  for (int64_t i = 0; i < grouped_sharding.group_dims.size(); ++i) {
    int64_t dim = grouped_sharding.group_dims[i];
    if (dim >= original_base_shape.rank()) {
      continue;
    }
    int64_t groups = grouped_sharding.group_dim_sizes[i];
    result.set_dimensions(dim, CeilOfRatio(result.dimensions(dim), groups));
  }
  return result;
}

namespace {

HloInstruction* GetInGroupPartitionId(
    HloInstruction* partition_id,
    const std::vector<std::vector<int64>>& device_groups, SpmdBuilder* b) {
  int64_t total_devices = device_groups.size() * device_groups[0].size();
  std::vector<uint32> in_group_ids(total_devices);
  for (uint32 i = 0; i < device_groups.size(); ++i) {
    for (uint32 j = 0; j < device_groups[i].size(); ++j) {
      in_group_ids[device_groups[i][j]] = j;
    }
  }
  auto id_table = b->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<uint32>(in_group_ids)));
  return b->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeScalarShape(U32),
      b->AddInstruction(HloInstruction::CreateDynamicSlice(
          ShapeUtil::MakeShape(U32, {1}), id_table, {partition_id}, {1}))));
}

SPMDCollectiveOpsCreator GetPerGroupCollectiveOpsCreator(
    const SPMDCollectiveOpsCreator& creator,
    const std::vector<std::vector<int64>>& device_groups) {
  SPMDCollectiveOpsCreator result;
  result.create_partition_id = [creator, device_groups](SpmdBuilder* b) {
    return GetInGroupPartitionId(creator.create_partition_id(b), device_groups,
                                 b);
  };
  auto expand_partition_groups =
      [device_groups](
          const std::vector<std::vector<int64>>& partition_subgroups) {
        if (partition_subgroups.empty()) {
          return device_groups;
        }
        std::vector<std::vector<int64>> result(partition_subgroups.size() *
                                               device_groups.size());
        for (int64_t g = 0; g < device_groups.size(); ++g) {
          for (int64_t i = 0; i < partition_subgroups.size(); ++i) {
            result[g * partition_subgroups.size() + i].resize(
                partition_subgroups[i].size());
            for (int64_t j = 0; j < partition_subgroups[i].size(); ++j) {
              result[g * partition_subgroups.size() + i][j] =
                  device_groups[g][partition_subgroups[i][j]];
            }
          }
        }
        return result;
      };
  result.create_cross_partition_all_reduce =
      [creator, expand_partition_groups](
          SpmdBuilder* b, HloInstruction* operand, HloComputation* reduction,
          const std::vector<std::vector<int64>>& partition_subgroups,
          int64_t channel_id) {
        return creator.create_cross_partition_all_reduce(
            b, operand, reduction, expand_partition_groups(partition_subgroups),
            channel_id);
      };
  result.create_cross_partition_collective_permute =
      [creator, device_groups](
          SpmdBuilder* b, HloInstruction* operand,
          std::vector<std::pair<int64, int64>>& src_dst_pairs,
          int64_t next_channel_id) {
        std::vector<std::pair<int64, int64>> expanded_pairs(
            src_dst_pairs.size() * device_groups.size());
        for (int64_t g = 0; g < device_groups.size(); ++g) {
          for (int64_t i = 0; i < src_dst_pairs.size(); ++i) {
            expanded_pairs[g * src_dst_pairs.size() + i] =
                std::pair<int64, int64>{
                    device_groups[g][src_dst_pairs[i].first],
                    device_groups[g][src_dst_pairs[i].second]};
          }
        }
        return creator.create_cross_partition_collective_permute(
            b, operand, expanded_pairs, next_channel_id);
      };
  result.create_cross_partition_all_to_all =
      [creator, expand_partition_groups](
          SpmdBuilder* b, absl::Span<HloInstruction* const> operands,
          const std::vector<std::vector<int64>>& partition_subgroups,
          int64_t channel_id, absl::optional<int64> split_dimension) {
        return creator.create_cross_partition_all_to_all(
            b, operands, expand_partition_groups(partition_subgroups),
            channel_id, split_dimension);
      };
  if (creator.create_cross_partition_all_gather) {
    result.create_cross_partition_all_gather =
        [creator, expand_partition_groups](
            SpmdBuilder* b, HloInstruction* operand, const Shape& ag_shape,
            const std::vector<std::vector<int64>>& partition_subgroups,
            int64_t channel_id, int64_t all_gather_dimension) {
          return creator.create_cross_partition_all_gather(
              b, operand, ag_shape,
              expand_partition_groups(partition_subgroups), channel_id,
              all_gather_dimension);
        };
  }
  return result;
}

}  // namespace

PartitionedHlo::PartitioningState CreatePerGroupPartitioningState(
    const PartitionedHlo::PartitioningState& state,
    const std::vector<std::vector<int64>>& device_groups, SpmdBuilder* b) {
  auto result = state;
  result.collective_ops_creator = GetPerGroupCollectiveOpsCreator(
      state.collective_ops_creator, device_groups);
  result.partition_id =
      GetInGroupPartitionId(state.partition_id, device_groups, b);
  // Create a string key for the groups.
  std::vector<std::string> per_group_strings(device_groups.size());
  for (int64_t i = 0; i < per_group_strings.size(); ++i) {
    per_group_strings[i] = absl::StrJoin(device_groups[i], ",");
  }
  auto& grouped_cache =
      state.reshard_cache->groupd_caches[absl::StrJoin(per_group_strings, ";")];
  if (!grouped_cache) {
    grouped_cache = absl::make_unique<PartitionedHlo::ReshardCache>();
  }
  result.reshard_cache = grouped_cache.get();
  return result;
}

HloInstruction* PerGroupSliceFromReplicated(
    HloInstruction* replicated, HloInstruction* partition_id,
    const std::vector<std::vector<int64>>& device_groups,
    absl::Span<const int64> group_dims, absl::Span<const int64> group_dim_sizes,
    SpmdBuilder* b) {
  std::vector<uint32> group_ids(device_groups.size() * device_groups[0].size());
  for (int64_t g = 0; g < device_groups.size(); ++g) {
    for (int64_t device : device_groups[g]) {
      group_ids[device] = g;
    }
  }
  auto group_id_table = b->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint32>(group_ids)));
  auto group_id = b->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeScalarShape(U32),
      b->AddInstruction(HloInstruction::CreateDynamicSlice(
          ShapeUtil::MakeShape(U32, {1}), group_id_table, {partition_id},
          {1}))));
  std::vector<int64> group_level_tile_dims(replicated->shape().rank(), 1);
  for (int64_t i = 0; i < group_dims.size(); ++i) {
    group_level_tile_dims[group_dims[i]] = group_dim_sizes[i];
  }
  Array<int64> group_level_tile(group_level_tile_dims);
  group_level_tile.Each([&](absl::Span<const int64> indices, int64* group) {
    *group = 0;
    for (int64_t dim : group_dims) {
      *group *= group_level_tile.dim(dim);
      *group += indices[dim];
    }
  });
  auto group_level_sharding = HloSharding::Tile(group_level_tile);
  auto padded_hlo = PadBaseShapeBeforeUnevenTiledSharding(
      replicated, group_level_sharding, b);
  auto shard_shape =
      MakePartitionedShape(replicated->shape(), group_level_sharding);
  return b->AddInstruction(HloInstruction::CreateDynamicSlice(
      shard_shape, padded_hlo,
      MakePartitionOffsets(replicated->shape(), group_level_sharding, group_id,
                           b),
      shard_shape.dimensions()));
}

absl::optional<HloOpcode> ParseReductionComputation(
    const HloComputation* reduction_comp) {
  if (reduction_comp->num_parameters() != 2) {
    return absl::nullopt;
  }
  auto root = reduction_comp->root_instruction();
  if (!root->IsElementwiseBinary()) {
    return absl::nullopt;
  }
  if (!absl::c_linear_search(root->operands(),
                             reduction_comp->parameter_instruction(0)) ||
      !absl::c_linear_search(root->operands(),
                             reduction_comp->parameter_instruction(1))) {
    return absl::nullopt;
  }
  return root->opcode();
}

absl::optional<std::vector<int64>> FindMatchingPartitionedDimsForGrouping(
    const HloSharding& sharding,
    const std::vector<std::vector<int64>>& device_groups) {
  if (sharding.NumTiles() < device_groups.size() || device_groups.size() < 2 ||
      device_groups[0].size() < 2) {
    return absl::nullopt;
  }
  int64_t rank = sharding.tile_assignment().num_dimensions();
  if (sharding.ReplicateOnLastTileDim()) {
    rank--;
  }
  absl::flat_hash_map<int64, std::vector<int64>> device_to_index;
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64> index, int64_t device) {
        device_to_index[device] =
            std::vector<int64>(index.begin(), index.begin() + rank);
      });
  std::vector<int64> dims;
  int64_t group_count = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (device_to_index[device_groups[0][0]][i] ==
        device_to_index[device_groups[0][1]][i]) {
      dims.push_back(i);
      group_count *= sharding.tile_assignment().dim(i);
    }
  }
  if (group_count != device_groups.size()) {
    return absl::nullopt;
  }
  for (const auto& group : device_groups) {
    for (int64_t i = 1; i < group.size(); ++i) {
      if (absl::c_any_of(dims, [&](const int64_t dim) {
            return device_to_index[group[i]][dim] !=
                   device_to_index[group[0]][dim];
          })) {
        return absl::nullopt;
      }
    }
  }
  return dims;
}

HloSharding CreateMatchingShardingOnDims(const Shape& target_shape,
                                         const HloSharding& source_sharding,
                                         absl::Span<const int64> target_dims,
                                         absl::Span<const int64> source_dims) {
  CHECK(target_dims.size() == source_dims.size())
      << "Expected 1:1 match between parallel dimensions";
  if (source_sharding.IsReplicated()) {
    return HloSharding::Replicate();
  }
  absl::InlinedVector<int64, 4> tile_dims(target_shape.dimensions_size(), 1);
  int num_tiles = 1;
  for (int i = 0, end = target_dims.size(); i < end; ++i) {
    num_tiles *= source_sharding.tile_assignment().dim(source_dims[i]);
    tile_dims[target_dims[i]] =
        source_sharding.tile_assignment().dim(source_dims[i]);
  }
  // If there is some partition across non-parallel dimensions in the
  // other operand then partially replicate for the new
  bool to_be_partially_replicated = false;
  if (num_tiles != source_sharding.tile_assignment().num_elements()) {
    CHECK_EQ(source_sharding.tile_assignment().num_elements() % num_tiles, 0);
    to_be_partially_replicated = true;
    tile_dims.push_back(source_sharding.tile_assignment().num_elements() /
                        num_tiles);
  }
  auto tgt_tile_assignment = source_sharding.tile_assignment();
  tgt_tile_assignment.Reshape(tile_dims);
  if (to_be_partially_replicated) {
    return AlignShardingOnDims(HloSharding::PartialTile(tgt_tile_assignment),
                               target_dims, source_sharding, source_dims);
  } else {
    return AlignShardingOnDims(HloSharding::Tile(tgt_tile_assignment),
                               target_dims, source_sharding, source_dims);
  }
}

absl::optional<GatherParallelDimSharding>
GatherOperandsShardedAcrossParallelDims(
    const HloInstruction& operand, const HloInstruction& indices,
    const hlo_sharding_util::GatherParallelDims& parallel_dims) {
  auto& indices_parallel_dims = parallel_dims.indices_parallel_dims;
  auto& operand_parallel_dims = parallel_dims.operand_parallel_dims;
  if (indices_parallel_dims.size() != operand_parallel_dims.size()) {
    return absl::nullopt;
  }
  auto new_index_shard = indices.sharding();
  auto new_operand_shard = operand.sharding();
  int idx_parallel_tiles_num = new_index_shard.NumTiles(indices_parallel_dims);
  int op_parallel_tiles_num = new_operand_shard.NumTiles(operand_parallel_dims);
  if (idx_parallel_tiles_num == 1 && op_parallel_tiles_num == 1) {
    return absl::nullopt;
  }
  absl::InlinedVector<int64, 1> indices_parallel_dims_ordered_as_operand;
  for (int idx : parallel_dims.index_parallel_in_dim) {
    if (idx != -1) {
      indices_parallel_dims_ordered_as_operand.push_back(idx);
    }
  }
  if (new_index_shard.IsReplicated()) {
    return GatherParallelDimSharding{
        CreateMatchingShardingOnDims(indices.shape(), new_operand_shard,
                                     indices_parallel_dims_ordered_as_operand,
                                     operand_parallel_dims),
        new_operand_shard};
  }
  if (new_operand_shard.IsReplicated()) {
    return GatherParallelDimSharding{
        new_index_shard,
        CreateMatchingShardingOnDims(operand.shape(), new_index_shard,
                                     operand_parallel_dims,
                                     indices_parallel_dims_ordered_as_operand)};
  }

  // Parallel dimension distribution needs to be the same, so try to steal
  // sharding from partial replication to compensate.
  if (idx_parallel_tiles_num != op_parallel_tiles_num) {
    auto to_adjust_dims = operand_parallel_dims;
    auto target_dims = indices_parallel_dims_ordered_as_operand;
    HloSharding* target = &new_index_shard;
    HloSharding* to_adjust = &new_operand_shard;
    if (idx_parallel_tiles_num < op_parallel_tiles_num) {
      std::swap(to_adjust_dims, target_dims);
      std::swap(to_adjust, target);
    }
    if (!to_adjust->ReplicateOnLastTileDim()) {
      return absl::nullopt;
    }
    auto new_tile_assignment_dims = to_adjust->tile_assignment().dimensions();
    for (int i = 0; i < to_adjust_dims.size(); ++i) {
      int64_t target_dim = target->tile_assignment().dim(target_dims[i]);
      int64_t to_adjust_dim =
          to_adjust->tile_assignment().dim(to_adjust_dims[i]);
      if (target_dim < to_adjust_dim) {
        return absl::nullopt;
      }
      if (target_dim == to_adjust_dim) {
        continue;
      }
      int64_t ratio = target_dim / to_adjust_dim;
      if (target_dim % to_adjust_dim != 0 ||
          new_tile_assignment_dims.back() % ratio != 0) {
        return absl::nullopt;
      }
      new_tile_assignment_dims[to_adjust_dims[i]] *= ratio;
      new_tile_assignment_dims.back() /= ratio;
    }
    CHECK_GE(new_tile_assignment_dims.back(), 1);
    bool to_partially_replicate = true;
    if (new_tile_assignment_dims.back() == 1) {
      new_tile_assignment_dims.pop_back();
      to_partially_replicate = false;
    }
    auto new_tile_assignment = to_adjust->tile_assignment();
    new_tile_assignment.Reshape(new_tile_assignment_dims);
    if (to_partially_replicate) {
      *to_adjust =
          AlignShardingOnDims(HloSharding::PartialTile(new_tile_assignment),
                              to_adjust_dims, *target, target_dims);
    } else {
      *to_adjust = AlignShardingOnDims(HloSharding::Tile(new_tile_assignment),
                                       to_adjust_dims, *target, target_dims);
    }
  }
  // Make sure that the parallel dimensions are aligned.
  auto operand_shard_tile_dims =
      new_operand_shard.tile_assignment().dimensions();
  for (int i = 0; i < indices_parallel_dims_ordered_as_operand.size(); ++i) {
    operand_shard_tile_dims[operand_parallel_dims[i]] =
        new_index_shard.tile_assignment().dim(
            indices_parallel_dims_ordered_as_operand[i]);
  }
  auto operand_shard_tiles = new_operand_shard.tile_assignment();
  operand_shard_tiles.Reshape(operand_shard_tile_dims);
  new_operand_shard =
      AlignShardingOnDims(new_operand_shard.ReplicateOnLastTileDim()
                              ? HloSharding::PartialTile(operand_shard_tiles)
                              : HloSharding::Tile(operand_shard_tiles),
                          operand_parallel_dims, new_index_shard,
                          indices_parallel_dims_ordered_as_operand);
  return GatherParallelDimSharding{new_index_shard, new_operand_shard};
}

int64 FindRotateRightPattern(const HloInstruction* concat,
                             const HloInstruction* lhs,
                             const HloInstruction* rhs) {
  if (lhs->opcode() != HloOpcode::kSlice ||
      rhs->opcode() != HloOpcode::kSlice ||
      lhs->operand(0) != rhs->operand(0)) {
    return -1;
  }
  const HloInstruction* to_rotate = lhs->operand(0);
  if (!ShapeUtil::Compatible(to_rotate->shape(), concat->shape()) ||
      concat->sharding() != to_rotate->sharding()) {
    return -1;
  }
  const int64_t dim = concat->concatenate_dimension();
  if (lhs->slice_strides(dim) != 1 || rhs->slice_strides(dim) != 1 ||
      lhs->slice_starts(dim) != rhs->slice_limits(dim)) {
    return -1;
  }
  return lhs->shape().dimensions(dim);
}

absl::optional<PadWithWrapPattern> FindPadWithWrapPattern(
    const HloInstruction* concat, const HloInstruction* lhs,
    const HloInstruction* mid, const HloInstruction* rhs) {
  if (!lhs || !mid || !rhs) {
    return absl::nullopt;
  }

  // Skip elementwise unary operations applied to inst, returning
  // a list of applied operations that were skipped.
  auto skip_elementwise_ops = [&](const HloInstruction* inst) {
    std::vector<const HloInstruction*> modifiers;
    while (inst->IsElementwise() && inst->operand_count() == 1 &&
           inst->user_count() == 1) {
      if (inst->opcode() != HloOpcode::kCopy) {
        modifiers.push_back(inst);
      }
      inst = inst->operand(0);
    }
    return std::make_pair(modifiers, inst);
  };

  PadWithWrapPattern pad_pattern;
  auto skip_result = skip_elementwise_ops(lhs);
  pad_pattern.lhs_modifiers = std::move(skip_result.first);
  lhs = skip_result.second;

  skip_result = skip_elementwise_ops(rhs);
  pad_pattern.rhs_modifiers = std::move(skip_result.first);
  rhs = skip_result.second;

  const int64_t dim = concat->concatenate_dimension();
  if (lhs->opcode() != HloOpcode::kSlice ||
      rhs->opcode() != HloOpcode::kSlice || lhs->operand(0) != mid ||
      rhs->operand(0) != mid || lhs->slice_strides(dim) != 1 ||
      rhs->slice_strides(dim) != 1 || lhs->sharding() != mid->sharding() ||
      rhs->sharding() != mid->sharding() ||
      lhs->sharding() != concat->sharding()) {
    return absl::nullopt;
  }
  pad_pattern.lhs_slice_start = lhs->slice_starts(dim);
  pad_pattern.rhs_slice_start = rhs->slice_starts(dim);
  return pad_pattern;
}

}  // namespace spmd
}  // namespace xla
