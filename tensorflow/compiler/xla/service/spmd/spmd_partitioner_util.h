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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SPMD_PARTITIONER_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SPMD_PARTITIONER_UTIL_H_

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace spmd {

template <typename T>
using IsCompOrCompBuilder =
    typename std::enable_if_t<std::is_same<HloComputation, T>::value ||
                              std::is_same<HloComputation::Builder, T>::value ||
                              std::is_same<SpmdBuilder, T>::value>;

struct GatherParallelDimSharding {
  HloSharding indices_sharding;
  HloSharding operand_sharding;
};

// Returns true if the given sharding contains any replicated sharding.
bool HasReplicatedSharding(const HloSharding& sharding);

// Base for creating constants.
template <typename T, typename = IsCompOrCompBuilder<T>>
HloInstruction* CreateConstantBase(const Shape& shape, Literal value, T* b,
                                   Literal (*literal_creator)(Literal,
                                                              PrimitiveType)) {
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> elements;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      elements.push_back(
          CreateConstantBase(ShapeUtil::GetTupleElementShape(shape, i),
                             value.Clone(), b, literal_creator));
    }
    return b->AddInstruction(HloInstruction::CreateTuple(elements));
  }

  if (shape.IsToken()) {
    return b->AddInstruction(HloInstruction::CreateToken());
  }
  auto c = b->AddInstruction(HloInstruction::CreateConstant(
      literal_creator(std::move(value), shape.element_type())));
  if (shape.rank() == 0) {
    return c;
  }
  return b->AddInstruction(HloInstruction::CreateBroadcast(shape, c, {}));
}

// Creates constant value instructions of the given shape. The literal must be a
// scalar shape and is broadcast to the given shape.
template <typename T, typename = IsCompOrCompBuilder<T>>
HloInstruction* CreateConstant(const Shape& shape, Literal value, T* b) {
  auto identity = [](Literal value, PrimitiveType primitive_type) {
    CHECK(ShapeUtil::IsScalarWithElementType(value.shape(), primitive_type));
    return value;
  };
  return CreateConstantBase(shape, std::move(value), b, identity);
}

// Creates zero value instructions of the given shape.
template <typename T, typename = IsCompOrCompBuilder<T>>
HloInstruction* CreateZero(const Shape& shape, T* b) {
  auto zero = [](Literal /*unused*/, PrimitiveType primitive_type) {
    return LiteralUtil::Zero(primitive_type);
  };
  return CreateConstantBase(shape, /*unused*/ Literal(), b, zero);
}

// Creates one value instructions of the given shape.
template <typename T, typename = IsCompOrCompBuilder<T>>
HloInstruction* CreateOne(const Shape& shape, T* b) {
  auto one = [](Literal /*unused*/, PrimitiveType primitive_type) {
    return LiteralUtil::One(primitive_type);
  };
  return CreateConstantBase(shape, /*unused*/ Literal(), b, one);
}

template <typename NativeT, typename T, typename = IsCompOrCompBuilder<T>>
HloInstruction* CreateR0WithType(PrimitiveType type, NativeT value, T* b) {
  auto literal = LiteralUtil::CreateR0(value)
                     .ConvertToShape(ShapeUtil::MakeShape(type, {}))
                     .value();
  return b->AddInstruction(HloInstruction::CreateConstant(std::move(literal)));
}

template <typename T, typename = IsCompOrCompBuilder<T>>
inline HloInstruction* CreateFirstWithType(PrimitiveType type, T* b) {
  if (type == F32) {
    auto float_pad_value = std::numeric_limits<float>::quiet_NaN();
    return CreateR0WithType(type, -float_pad_value, b);
  }
  auto literal = LiteralUtil::MinValue(type);
  return b->AddInstruction(HloInstruction::CreateConstant(std::move(literal)));
}

template <typename T, typename = IsCompOrCompBuilder<T>>
inline HloInstruction* CreateLastWithType(PrimitiveType type, T* b) {
  if (type == F32) {
    auto float_pad_value = std::numeric_limits<float>::quiet_NaN();
    return CreateR0WithType(type, float_pad_value, b);
  }
  auto literal = LiteralUtil::MaxValue(type);
  return b->AddInstruction(HloInstruction::CreateConstant(std::move(literal)));
}

// Create a binary add computation of the given type and add to the module.
HloComputation* MakeBinaryAdd(PrimitiveType type, HloModule* module);

// Returns true if the shape can be evenly partitioned for the given sharding.
// All tile sharded dimensions should be evenly divisible and there should be no
// single-device sharding. Replicate sharding is considered even partition.
bool EvenlyPartitions(const Shape& shape, const HloSharding& sharding);

// Returns the shard shape of the given shape when it is partitioned for the
// target sharding.
Shape MakePartitionedShape(const Shape& shape, const HloSharding& sharding);

// Similar to ShapeUtil::ByteSizeOf(), but does not check it has dense layout
// since this can be before layout assignment.
int64_t ShapeSizeInBytes(const Shape& shape);

// Returns the shard shape for a partition without padding due to uneven
// sharding.
Shape MakeNonPaddedShapeForGivenPartition(const Shape& shape,
                                          const HloSharding& sharding,
                                          int64_t partition_id);

// Generates the HLO instructions that represent the dimension offsets on any
// device. The size of the returned vector is the rank of the given shape.
// If `dims` is non-empty, the generated offsets will only be non-zero for those
// dimensions.
std::vector<HloInstruction*> MakePartitionOffsets(
    const Shape& shape, const HloSharding& sharding,
    HloInstruction* partition_id, SpmdBuilder* b,
    absl::Span<const int64_t> dims = {});

// Returns the offsets of the partition in the tile assignment.
std::vector<HloInstruction*> MakeTiledPartitionOrdinals(
    const HloSharding& sharding, HloInstruction* partition_id, SpmdBuilder* b);

// Pads hlo to the desired shape using high padding. Either a builder or a
// computation needs to be supplied, but not both.
template <typename T, typename = IsCompOrCompBuilder<T>>
HloInstruction* PadToShape(HloInstruction* hlo, const Shape& padded_shape, T* b,
                           std::optional<Literal> value = std::nullopt) {
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
  const Shape padding_shape =
      ShapeUtil::MakeScalarShape(hlo->shape().element_type());
  HloInstruction* padding =
      value.has_value() ? CreateConstant(padding_shape, std::move(*value), b)
                        : CreateZero(padding_shape, b);
  return b->AddInstruction(
      HloInstruction::CreatePad(padded_shape, hlo, padding, padding_config));
}

// Returns the padded shape when combining all partitions.
Shape GetPaddedShapeForUnevenPartitioning(const Shape& base_shape,
                                          const HloSharding& sharding);

// Pads the HLO (with base shape) for uneven tiled partition to make it evenly
// partitionable.
template <typename T, typename = IsCompOrCompBuilder<T>>
HloInstruction* PadBaseShapeBeforeUnevenTiledSharding(
    HloInstruction* hlo, const HloSharding& sharding, T* b,
    std::optional<Literal> value = std::nullopt) {
  auto padded_base_shape =
      GetPaddedShapeForUnevenPartitioning(hlo->shape(), sharding);
  if (ShapeUtil::Compatible(padded_base_shape, hlo->shape())) {
    return hlo;
  }
  return PadToShape(hlo, padded_base_shape, b, std::move(value));
}

// Returns the index of the unique tile dimension. Returns std::nullopt if the
// given sharding is not tiled or tiled along multiple dimensions.
std::optional<int64_t> UniqueTiledDim(const HloSharding& sharding);

// Utilities for symbolic offset calculation and halo exchange.
class OffsetCalculation;

// Represents a calculation over integers:
//   (shard_ordinal * multiplier + offset) / divisor
class MultiplyAddDivideOffsetCalculation {
 public:
  MultiplyAddDivideOffsetCalculation()
      : multiplier_(0), offset_(0), divisor_(1) {}
  MultiplyAddDivideOffsetCalculation(int64_t multiplier, int64_t offset,
                                     int64_t divisor);

  OffsetCalculation operator-(
      const MultiplyAddDivideOffsetCalculation& other) const;

  bool operator==(const MultiplyAddDivideOffsetCalculation& other) const {
    return multiplier_ == other.multiplier_ && offset_ == other.offset_ &&
           divisor_ == other.divisor_;
  }

  bool IsConstant() const { return multiplier_ == 0; }
  void Simplify();
  int64_t Calculate(int64_t shard_ordinal) const;
  HloInstruction* Calculate(HloInstruction* shard_ordinal,
                            SpmdBuilder* b) const;

  // Returns the maximum result for shard ordinals in the range
  // [start_ordinal, limit_ordinal).
  int64_t MaxInRange(int64_t start_ordinal, int64_t limit_ordinal) const;

 private:
  int64_t multiplier_;
  int64_t offset_;
  int64_t divisor_;
};

// Represents a calculation over integers based on results of other calculations
// defined by an opcode. If the opcode is kCopy, it simply wraps an
// MultiplyAddDivideOffsetCalculation.
class OffsetCalculation {
 public:
  OffsetCalculation() : opcode_(HloOpcode::kCopy), copy_from_() {}
  explicit OffsetCalculation(
      const MultiplyAddDivideOffsetCalculation& copy_from)
      : opcode_(HloOpcode::kCopy), copy_from_(copy_from) {}
  OffsetCalculation(const OffsetCalculation& copy_from) { *this = copy_from; }
  OffsetCalculation(HloOpcode opcode,
                    const MultiplyAddDivideOffsetCalculation& lhs,
                    const MultiplyAddDivideOffsetCalculation& rhs)
      : opcode_(opcode),
        lhs_(std::make_unique<OffsetCalculation>(lhs)),
        rhs_(std::make_unique<OffsetCalculation>(rhs)) {}
  OffsetCalculation(HloOpcode opcode, const OffsetCalculation& lhs,
                    const OffsetCalculation& rhs)
      : opcode_(opcode),
        lhs_(std::make_unique<OffsetCalculation>(lhs)),
        rhs_(std::make_unique<OffsetCalculation>(rhs)) {}

  OffsetCalculation& operator=(const OffsetCalculation& other);

  // Returns whether the calculation returns the same value for all shards. This
  // is conservative and could return false even if it is actually constant.
  bool IsConstant() const;

  OffsetCalculation operator-(const OffsetCalculation& other) const;
  bool operator==(const OffsetCalculation& other) const;
  int64_t Calculate(int64_t shard_ordinal) const;
  HloInstruction* Calculate(HloInstruction* shard_ordinal,
                            SpmdBuilder* b) const;

  // Returns the maximum result for shard ordinals in the range
  // [start_ordinal, limit_ordinal).
  int64_t MaxInRange(int64_t start_ordinal, int64_t limit_ordinal) const;

 private:
  HloOpcode opcode_;
  std::unique_ptr<OffsetCalculation> lhs_;
  std::unique_ptr<OffsetCalculation> rhs_;
  MultiplyAddDivideOffsetCalculation copy_from_;
};

// Performs halo exchange on the given dimension based on the provided
// left/right halo size functions. Returns nullopt if the halo is beyond the
// direct neighbor of the shard.
std::optional<HloInstruction*> ExchangeHalo(
    HloInstruction* hlo, const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function, int64_t dim,
    const HloSharding& target,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdBuilder* b);

// Exchange halo on all dimensions of the HLO. Returns nullopt if any one of the
// dimensions fails to exchange halo (halo is beyond the neighbor shard).
std::optional<HloInstruction*> ExchangeHalo(
    HloInstruction* hlo,
    std::vector<OffsetCalculation> left_halo_size_functions,
    std::vector<OffsetCalculation> right_halo_size_functions,
    const HloSharding& target,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdBuilder* b);

// Exchanges halos and performs pad/dynamic-slice on the concatenated data such
// that the result starts with the first needed element on each shard. It also
// masks off invalid data due to padding.
// Arguments:
//  hlo: the HLO op before halo exchange
//  explicit_left_padding_on_full_shape: the amount of left padding to be added
//   explicitly by this function on the base shape before partitioning. Without
//   base dilation, this is usually set to the window's padding_low so that the
//   sharded op do not need to add padding_low on the window; however, with base
//   dilation, this could only be set to a custom size.
//  padded_full_shape_size: the size of the padded full shape on the given
//   dimension, which includes explicit_left_padding_on_full_shape and required
//   right padding to make the shape evenly shardable.
//  shard_size_with_halo: the shard size on the dimension after halo exchange.
//   If different shards have different sizes, use the maximum size.
//  offset_on_padded_shape: the offset HLO (S32) that represents the start of
//   each shard on the padded full shape.
//  pad_value: the padding value used on the full shape.
std::optional<HloInstruction*> ExchangeHaloAndGetValidData(
    HloInstruction* hlo, const Shape& base_shape,
    const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function,
    int64_t explicit_left_padding_on_full_shape, int64_t padded_full_shape_size,
    int64_t shard_size_with_halo, int64_t dim, const HloSharding& target,
    HloInstruction* offset_on_padded_shape, HloInstruction* pad_value,
    HloInstruction* partition_ordinal,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdBuilder* b, bool mask_invalid_region = true);

// Uses halo exchange to change from right-padding to left-padding for uneven
// tiled sharding on the given dimensions. Tiled sharding always pads uneven
// partitioned data on the right, but we need to swap it to the left for
// kReverse or kConvolution with window reversal.
HloInstruction* HaloExchangeToPadOnLeft(PartitionedHlo& original,
                                        absl::Span<const int64_t> dims);

// Check if the computation is GT comparison and safe for NaNs.
bool IsNanSafeGt(HloComputation* computation);

// Return k in TopK when input value is parttioned in the sort dimension.
std::optional<int64_t> GetKValueInTopKWhenPartitionSortDim(HloInstruction* hlo);

// Slices the first k elements at slice dimension.
HloInstruction* SliceFirstK(HloInstruction* hlo, SpmdBuilder* builder,
                            int64_t slice_dim, int64_t k);

// Check if a dimension is sharded.
int64_t ShardCountAtDim(const HloSharding& sharding, int64_t dim);

// Returns the list of source-target pairs of dimensions to swap during
// resharding via all-to-all. Reshard can be done by swapping each pair at a
// time.
std::optional<std::vector<std::pair<int64_t, int64_t>>>
GetReshardAllToAllSourceTargetDims(const HloSharding& source,
                                   const HloSharding& target);

// Returns whether the resharding can be done via collective-permute.
bool CanReshardWithCollectivePermute(const HloSharding& source,
                                     const HloSharding& target);

// Returns a new GroupedSharding that has the same group definition of
// `reference`.
hlo_sharding_util::GroupedSharding AlignGroupsWith(
    hlo_sharding_util::GroupedSharding grouped_sharding,
    const hlo_sharding_util::GroupedSharding& reference,
    bool ignore_group_order = false);

// Align device groups between the two ahrdings. Equivalent in calling
// GroupShardingOnDims on the two sharding AlignGroupsWith and then
// UngroupSharding
HloSharding AlignShardingOnDims(const HloSharding& sharding,
                                absl::Span<const int64_t> sharding_dims,
                                const HloSharding& reference,
                                absl::Span<const int64_t> reference_dims);

// AlignShardingOnDims only if it doesn't change the sharding when ungrouped.
std::optional<hlo_sharding_util::GroupedSharding> AlignGroupsWithIfCompatible(
    hlo_sharding_util::GroupedSharding grouped_sharding,
    const hlo_sharding_util::GroupedSharding& reference);

// Returns the per-group base shape, i.e., before applying the in-group
// sharding.
Shape GetPerGroupBaseShape(
    const hlo_sharding_util::GroupedSharding& grouped_sharding,
    const Shape& original_base_shape);

// Creates the nested partitioner state for in-group patitioning.
PartitionedHlo::PartitioningState CreatePerGroupPartitioningState(
    const PartitionedHlo::PartitioningState& state,
    const std::vector<std::vector<int64_t>>& device_groups, SpmdBuilder* b);

// Partially shards a replicated HLO into groups along the group dimensions, and
// within each group data is still replicated.
HloInstruction* PerGroupSliceFromReplicated(
    HloInstruction* replicated, HloInstruction* partition_id,
    const std::vector<std::vector<int64_t>>& device_groups,
    absl::Span<const int64_t> group_dims,
    absl::Span<const int64_t> group_dim_sizes, SpmdBuilder* b);

// Pad the shape from partial replicate shape for `dst_sharding`.
// If dst_sharding needs more padding and per_shard_size increased in
// dst_sharding, halo exchange on the right side is needed.
std::optional<HloInstruction*> PadFromPartialReplicateShape(
    HloInstruction* hlo, const Shape& base_shape,
    const HloSharding& src_sharding, const HloSharding& dst_sharding,
    const std::vector<int64_t>& expand_tile_dims,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, HloInstruction* partition_id, SpmdBuilder* b);

// Get the compatible sharding from a partial replicate sharding to a desired
// target tiled sharding.
// Compatible means replicate sharding can transform to the target tile
// dimensions by dynamic slice.
// For example, if partial_sharding is
// {devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
// Target sharding is {devices=[2,2]0,1,2,3}, the returned compatible sharding
// will be sharding={devices=[2,2]0,2,1,3}.
// If patial replicate sharding is not partial replicate or can't reshard to
// target_tile_dims by dynamic slice, return std::nullopt.
// If target_sharding is already compatible, returns it.
std::optional<HloSharding> PartialReplicateReshardCompatibleSharding(
    const HloSharding& partial_sharding, const HloSharding& target_sharding);

// Do left halo exchange if all-reduce directly from tile sharding to partial
// replicate sharding will remove useful data from the source.
std::optional<HloInstruction*> TileToPartialReplicateHaloExchange(
    HloInstruction* hlo, const Shape& base_shape,
    const HloSharding& src_sharding, const HloSharding& dst_sharding,
    const std::vector<int64_t>& replicate_dims,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, HloInstruction* partition_id, SpmdBuilder* b);

// Finds a list of dimensions that can be grouped on such that it will have the
// specified device groups. Group order and dimension order are ignored.
std::optional<std::vector<int64_t>> FindMatchingPartitionedDimsForGrouping(
    const HloSharding& sharding,
    const std::vector<std::vector<int64_t>>& device_groups);

// Create a sharding that matches the provided source sharding on the
// specified dimensions. 'target_dims' and 'source_dims' represent the
// dimensions for which the sharding should match in their respective shape.
// If some devices from the source sharding are left over (because not all the
// devices are allocated to 'source_dims' dimensions) then partial replication
// is employed to make sure the number of devices for the two sharding match.
HloSharding CreateMatchingShardingOnDims(const Shape& target_shape,
                                         const HloSharding& source_sharding,
                                         absl::Span<const int64_t> target_dims,
                                         absl::Span<const int64_t> source_dims);

// Returns if the sharding across operand and indices of a gather is across
// parallel dimensions and matches what SPMD partitioner supports.
std::optional<GatherParallelDimSharding>
GatherOperandsShardedAcrossParallelDims(
    const HloInstruction& operand, const HloInstruction& indices,
    const hlo_sharding_util::GatherParallelDims& parallel_dims);

// Pattern rewrite preprocessing utilities.

// Returns rotate_amount if the concat(lhs, rhs) is equivalent to rotating the
// elements along the concat dimension to the right by rotate_amount, where the
// input of rotation is the shard operand of lhs and rhs. Returns -1 if the
// pattern is not found.
int64_t FindRotateRightPattern(const HloInstruction* concat,
                               const HloInstruction* lhs,
                               const HloInstruction* rhs);

// Describes the pad with wrap pattern.
struct PadWithWrapPattern {
  int64_t lhs_slice_start;
  int64_t rhs_slice_start;
  std::vector<const HloInstruction*> lhs_modifiers;
  std::vector<const HloInstruction*> rhs_modifiers;
};

// Returns the `PadWithWrapPattern` if the concat(lhs,mid,rhs) is equivalent to
// padding mid with wrapping (i.e., padding mid with slices of itself). Return
// std::nullopt if the pattern is not found.
std::optional<PadWithWrapPattern> FindPadWithWrapPattern(
    const HloInstruction* concat, const HloInstruction* lhs,
    const HloInstruction* mid, const HloInstruction* rhs);

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SPMD_PARTITIONER_UTIL_H_
