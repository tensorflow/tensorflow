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

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace spmd {

bool HasReplicatedSharding(const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    return absl::c_any_of(sharding.tuple_elements(), HasReplicatedSharding);
  }
  return sharding.IsReplicated();
}

HloInstruction* CreateZero(const Shape& shape, SpmdBuilder* b) {
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> elements;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
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
  return b->AddInstruction(HloInstruction::CreateBroadcast(shape, zero, {}));
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
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      if (!EvenlyPartitions(ShapeUtil::GetTupleElementShape(shape, i),
                            sharding.GetSubSharding(shape, {i}))) {
        return false;
      }
    }
  }

  if (sharding.IsTileMaximal()) {
    return sharding.IsReplicated();
  }
  for (int64 i = 0; i < shape.dimensions_size(); ++i) {
    if (shape.dimensions(i) % sharding.tile_assignment().dim(i) != 0) {
      return false;
    }
  }
  return true;
}

Shape MakePartitionedShape(const Shape& shape, const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    std::vector<Shape> subshapes;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
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
                                          int64 partition_id) {
  if (sharding.IsTuple()) {
    std::vector<Shape> subshapes;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      subshapes.push_back(MakeNonPaddedShapeForGivenPartition(
          ShapeUtil::GetTupleElementShape(shape, i),
          sharding.GetSubSharding(shape, {i}), partition_id));
    }
    return ShapeUtil::MakeTupleShape(subshapes);
  }

  auto partition_shape = shape;
  std::vector<int64> tile_offset =
      sharding.TileOffsetForDevice(shape, partition_id);
  std::vector<int64> tile_limit =
      sharding.TileLimitForDevice(shape, partition_id);
  for (int64 i = 0; i < tile_offset.size(); ++i) {
    if (sharding.UsesDevice(partition_id)) {
      partition_shape.set_dimensions(i, tile_limit[i] - tile_offset[i]);
    } else {
      partition_shape.set_dimensions(i, 0);
    }
  }
  return partition_shape;
}

std::vector<HloInstruction*> MakePartitionOffsets(const Shape& shape,
                                                  const HloSharding& sharding,
                                                  HloInstruction* partition_id,
                                                  SpmdBuilder* b) {
  CHECK(!shape.IsTuple());

  Array2D<int32> offset_array(
      {sharding.tile_assignment().num_elements(), shape.rank()});
  offset_array.Each([&](int64 i, int64 j, int32* value) {
    *value = sharding.TileOffsetForDevice(shape, i)[j];
  });
  auto offset_table = b->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2FromArray2D(offset_array)));
  std::vector<HloInstruction*> offsets;
  for (int64 i = 0; i < shape.rank(); ++i) {
    if (sharding.tile_assignment().dim(i) == 1) {
      offsets.push_back(b->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(S32))));
    } else {
      auto index = b->AddInstruction(HloInstruction::CreateDynamicSlice(
          ShapeUtil::MakeShape(S32, {1, 1}), offset_table,
          {partition_id, b->AddInstruction(HloInstruction::CreateConstant(
                             LiteralUtil::CreateR0<uint32>(i)))},
          {1, 1}));
      offsets.push_back(b->AddInstruction(
          HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {}), index)));
    }
  }
  return offsets;
}

std::vector<HloInstruction*> MakeTiledPartitionOrdinals(
    const HloSharding& sharding, HloInstruction* partition_id, SpmdBuilder* b) {
  CHECK(!sharding.IsTileMaximal());
  auto table_shape =
      ShapeUtil::MakeShape(S32, sharding.tile_assignment().dimensions());
  return MakePartitionOffsets(table_shape, sharding, partition_id, b);
}

HloInstruction* PadToShape(HloInstruction* hlo, const Shape& padded_shape,
                           SpmdBuilder* b, HloComputation* computation) {
  CHECK(b == nullptr || computation == nullptr);
  if (ShapeUtil::Compatible(hlo->shape(), padded_shape)) {
    return hlo;
  }
  PaddingConfig padding_config;
  for (int64 i = 0; i < padded_shape.rank(); ++i) {
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
  for (int64 i = 0; i < padded_base_shape.rank(); ++i) {
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

absl::optional<int64> UniqueTiledDim(const HloSharding& sharding) {
  if (sharding.IsTileMaximal()) {
    return absl::nullopt;
  }
  int64 dim = -1;
  for (int64 i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
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
    int64 multiplier, int64 offset, int64 divisor)
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

int64 MultiplyAddDivideOffsetCalculation::Calculate(int64 shard_ordinal) const {
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
    int64 start_ordinal, int64 limit_ordinal) const {
  int64 max = Calculate(start_ordinal);
  for (int64 i = start_ordinal + 1; i < limit_ordinal; ++i) {
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

int64 OffsetCalculation::Calculate(int64 shard_ordinal) const {
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

int64 OffsetCalculation::MaxInRange(int64 start_ordinal,
                                    int64 limit_ordinal) const {
  if (IsConstant()) {
    return Calculate(start_ordinal);
  }
  if (opcode_ == HloOpcode::kCopy) {
    return std::max(Calculate(start_ordinal), Calculate(limit_ordinal - 1));
  }
  int64 max = Calculate(start_ordinal);
  for (int64 i = start_ordinal + 1; i < limit_ordinal; ++i) {
    max = std::max(max, Calculate(i));
  }
  return max;
}

absl::optional<HloInstruction*> ExchangeHalo(
    HloInstruction* hlo, const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function, int64 dim,
    const HloSharding& target,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, SpmdBuilder* b) {
  int64 input_shard_size = hlo->shape().dimensions(dim);
  int64 shard_count = target.tile_assignment().dim(dim);

  std::vector<HloInstruction*> concat_pieces;

  int64 max_left_halo_size = left_halo_size_function.MaxInRange(1, shard_count);
  for (int64 i = CeilOfRatio(max_left_halo_size, input_shard_size) - 1; i >= 0;
       --i) {
    std::vector<std::pair<int64, int64>> source_target_pairs;
    target.tile_assignment().Each(
        [&](absl::Span<const int64> indices, int64 device) {
          if (indices[dim] > i) {
            std::vector<int64> source_indices(indices.begin(), indices.end());
            source_indices[dim] -= i + 1;
            source_target_pairs.emplace_back(
                target.tile_assignment()(source_indices), device);
          }
        });
    int64 halo_size =
        std::min(max_left_halo_size - input_shard_size * i, input_shard_size);
    auto halo_shape = hlo->shape();
    auto source_halo_slice = hlo;
    if (halo_size != hlo->shape().dimensions(dim)) {
      halo_shape.set_dimensions(dim, halo_size);
      std::vector<int64> halo_start_indices(halo_shape.rank(), 0);
      halo_start_indices[dim] = hlo->shape().dimensions(dim) - halo_size;
      std::vector<int64> halo_slice_strides(halo_shape.rank(), 1);
      source_halo_slice = b->AddInstruction(HloInstruction::CreateSlice(
          halo_shape, hlo, halo_start_indices, hlo->shape().dimensions(),
          halo_slice_strides));
    }
    auto left_halo =
        collective_ops_creator.create_cross_partition_collective_permute(
            b, source_halo_slice, source_target_pairs, (*next_channel_id)++);
    concat_pieces.push_back(left_halo);
  }

  concat_pieces.push_back(hlo);

  // Right halo.
  int64 max_right_halo_size =
      right_halo_size_function.MaxInRange(0, shard_count - 1);
  for (int64 i = 0; i < CeilOfRatio(max_right_halo_size, input_shard_size);
       ++i) {
    std::vector<std::pair<int64, int64>> source_target_pairs;
    target.tile_assignment().Each(
        [&](absl::Span<const int64> indices, int64 device) {
          if (indices[dim] > i) {
            std::vector<int64> target_indices(indices.begin(), indices.end());
            target_indices[dim] -= i + 1;
            source_target_pairs.emplace_back(
                device, target.tile_assignment()(target_indices));
          }
        });
    int64 halo_size =
        std::min(max_right_halo_size - input_shard_size * i, input_shard_size);
    auto halo_shape = hlo->shape();
    HloInstruction* source_halo_slice = hlo;
    if (halo_size != halo_shape.dimensions(dim)) {
      halo_shape.set_dimensions(dim, halo_size);
      std::vector<int64> halo_start_indices(halo_shape.rank(), 0);
      std::vector<int64> halo_slice_strides(halo_shape.rank(), 1);
      source_halo_slice = b->AddInstruction(HloInstruction::CreateSlice(
          halo_shape, hlo, halo_start_indices, halo_shape.dimensions(),
          halo_slice_strides));
    }
    auto right_halo =
        collective_ops_creator.create_cross_partition_collective_permute(
            b, source_halo_slice, source_target_pairs, (*next_channel_id)++);
    concat_pieces.push_back(right_halo);
  }

  auto concat = hlo;
  // Concat with halos/padding.
  if (concat_pieces.size() > 1) {
    auto concat_shape = hlo->shape();
    int64 concat_dim_size = 0;
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
    int64 explicit_left_padding_on_full_shape, int64 padded_full_shape_size,
    int64 shard_size_with_halo, int64 dim, const HloSharding& target,
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
  int64 shard_count = target.tile_assignment().dim(dim);
  int64 max_left_halo_size = left_halo_size_function.MaxInRange(1, shard_count);

  // Now we determine if we need extra padding after the concat.
  //
  // The max of halo size or the first shard's explicit left padding.
  int64 max_left_halo_or_padding_size =
      std::max(std::max(int64{0}, max_left_halo_size),
               explicit_left_padding_on_full_shape);
  // The calculation that returns the dynamic slice index for a shard on the
  // padded concat, which is the difference between
  // max_left_halo_or_padding_size and its left halo size.
  auto start_offset_on_padded_concat_calculation =
      OffsetCalculation(MultiplyAddDivideOffsetCalculation(
          0, max_left_halo_or_padding_size, 1)) -
      left_halo_size_function;

  // See if we need to pad the concat before dynamic slice.
  int64 extra_left_padding =
      std::max(int64{0}, max_left_halo_or_padding_size -
                             std::max(int64{0}, max_left_halo_size));
  int64 extra_right_padding =
      start_offset_on_padded_concat_calculation.MaxInRange(0, shard_count) +
      shard_size_with_halo - concat->shape().dimensions(dim) -
      extra_left_padding;
  extra_right_padding = std::max(int64{0}, extra_right_padding);
  if (extra_left_padding > 0 || extra_right_padding > 0) {
    PaddingConfig padding_config;
    auto padded_concat_shape = concat->shape();
    for (int64 i = 0; i < base_shape.rank(); ++i) {
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

  int64 total_right_padding = padded_full_shape_size -
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

}  // namespace spmd
}  // namespace xla
