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

#ifndef XLA_SERVICE_SPMD_SPMD_PARTITIONER_UTIL_H_
#define XLA_SERVICE_SPMD_SPMD_PARTITIONER_UTIL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "absl/utility/utility.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {

Window GenNewWindow(const HloInstruction* original_dot,
                    const HloInstruction* dot_lhs,
                    const HloInstruction* dot_rhs, int64_t lhs_concat_dim,
                    int64_t rhs_concat_dim, bool windowed_at_contracting_dims,
                    bool windowed_at_batch_dims);

ConvolutionDimensionNumbers GenNewConvDNums(
    const HloInstruction* original_dot, const HloInstruction* dot_lhs,
    const HloInstruction* dot_rhs, int64_t lhs_concat_dim,
    int64_t rhs_concat_dim, bool windowed_at_contracting_dims,
    bool windowed_at_batch_dims,
    const std::vector<int64_t>& lhs_to_output_indices,
    const std::vector<int64_t>& rhs_to_output_indices,
    const Shape& new_dot_shape);

template <typename T>
using IsCompOrCompBuilder =
    typename std::enable_if_t<std::is_same<HloComputation, T>::value ||
                              std::is_same<HloComputation::Builder, T>::value ||
                              std::is_same<SpmdBuilder, T>::value>;

struct GatherScatterParallelDimSharding {
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
    elements.reserve(ShapeUtil::TupleElementCount(shape));
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
  if (shape.dimensions().size() == 0) {
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

// Creates a table lookup HLO using the ordinal as the offset.
template <typename NativeT>
HloInstruction* TableLookup(absl::Span<const NativeT> table, PrimitiveType type,
                            HloInstruction* ordinal, SpmdBuilder* b) {
  HloInstruction* table_hlo = b->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<NativeT>(table)));
  HloInstruction* value = b->AddInstruction(HloInstruction::CreateDynamicSlice(
      ShapeUtil::MakeShape(type, {1}), table_hlo, {ordinal}, {1}));
  return b->AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(type, {}), value));
}

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
  for (int64_t i = 0; i < padded_shape.dimensions().size(); ++i) {
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
  OffsetCalculation operator+(
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
  OffsetCalculation operator+(const OffsetCalculation& other) const;
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

// A compact version of halo exchange, which generates fewer collective permutes
// when the halo ranges are far from the current shard while the final result
// size is small. It tries to reuse the same collective permute to do as many
// disjoint communications as possible. It also includes data masking. pad_value
// can be nullptr, which means the value in padding regions doesn't matter.
HloInstruction* ExchangeHaloCompact(
    HloInstruction* hlo, const Shape& base_shape,
    const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function,
    HloInstruction* pad_value, int64_t dim, const HloSharding& sharding,
    HloInstruction* shard_ordinal,
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
//  force_mask_in_compact: If true, masking is always applied if it uses
//   ExchangeHaloCompact. An example is that certain cases in pad can skip
//   masking in non-compact halo exchange, but not in compact ones.
std::optional<HloInstruction*> ExchangeHaloAndGetValidData(
    HloInstruction* hlo, const Shape& base_shape,
    const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function,
    int64_t explicit_left_padding_on_full_shape, int64_t padded_full_shape_size,
    int64_t shard_size_with_halo, int64_t dim, const HloSharding& target,
    HloInstruction* offset_on_padded_shape, HloInstruction* pad_value,
    HloInstruction* partition_ordinal,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdBuilder* b, bool mask_invalid_region = true,
    bool force_mask_in_compact = false);

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

// Align device groups between the two shardings. Equivalent in calling
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

// Returns the partition id within a group.
HloInstruction* GetInGroupPartitionId(
    HloInstruction* partition_id,
    const hlo_sharding_util::DeviceGroupTileAssignment& device_groups,
    SpmdBuilder* b);

// Creates the nested partitioner state for in-group partitioning.
PartitionedHlo::PartitioningState CreatePerGroupPartitioningState(
    const PartitionedHlo::PartitioningState& state,
    const hlo_sharding_util::DeviceGroupTileAssignment& device_groups,
    SpmdBuilder* b);

// Partially shards a replicated HLO into groups along the group dimensions, and
// within each group data is still replicated.
HloInstruction* PerGroupSliceFromReplicated(
    HloInstruction* replicated, HloInstruction* partition_id,
    const hlo_sharding_util::DeviceGroupTileAssignment& device_groups,
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
// If partial_sharding is not partial replicate or can't reshard to
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
    const hlo_sharding_util::DeviceGroupTileAssignment& device_groups);

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

// Returns if the sharding across operand and indices of a gather/scatter is
// across parallel dimensions and matches what SPMD partitioner supports.
std::optional<GatherScatterParallelDimSharding>
GatherScatterOperandsShardedAcrossParallelDims(
    const HloInstruction& operand, const HloInstruction& indices,
    const hlo_sharding_util::GatherScatterDims& parallel_dims);

// Pattern rewrite preprocessing utilities.

// Returns rotate_amount if the concat(lhs, rhs) is equivalent to rotating the
// elements along the concat dimension to the right by rotate_amount, where the
// input of rotation is the shard operand of lhs and rhs. Returns std::nullopt
// if the pattern is not found.
std::optional<int64_t> FindRotateRightPattern(const HloInstruction* concat);

// Describes the pad with wrap pattern.
struct PadWithWrapPattern {
  int64_t lhs_slice_start;
  int64_t rhs_slice_start;
  std::vector<const HloInstruction*> lhs_modifiers;
  std::vector<const HloInstruction*> rhs_modifiers;
};

// Returns the `PadWithWrapPattern` if the concat(lhs, mid, rhs) is equivalent
// to padding mid with wrapping (i.e., padding mid with slices of itself).
// Returns std::nullopt if the pattern is not found.
std::optional<PadWithWrapPattern> FindPadWithWrapPattern(
    const HloInstruction* concat);

// Reshards data for a slice to be happening on such data with the passed
// parameters.
std::optional<PartitionedHlo::WindowedInputShardReturnValue>
ReshardDataForSlicing(absl::Span<const int64_t> strides,
                      absl::Span<const int64_t> starts,
                      absl::Span<const int64_t> limits,
                      PartitionedHlo to_reshard,
                      const HloSharding& target_sharding, SpmdBuilder* b);

// Performs slicing of data based on the windowed sharding passed as input.
HloInstruction* SliceDataFromWindowReshard(
    const PartitionedHlo::WindowedInputShardReturnValue& reshard_operand,
    absl::Span<const int64_t> strides, const Shape& base_shape,
    const HloSharding& target_sharding, SpmdBuilder* b);

// Reshards data for a pad to be happening on such data with the passed
// parameters.
std::optional<PartitionedHlo::WindowedInputShardReturnValue> ReshardDataForPad(
    HloInstruction* pad_value, PaddingConfig pc, PartitionedHlo to_reshard,
    const HloSharding& target_sharding, SpmdBuilder* b);

// Performs padding of data based on the windowed sharding passed as input.
HloInstruction* PadDataFromWindowReshard(
    const PartitionedHlo::WindowedInputShardReturnValue& reshard_operand,
    HloInstruction* pad_value, SpmdBuilder* b);

// Generates partition groups (groups of devices that will communicate via a
// collective) from sharding and provided replication_dims.
std::vector<std::vector<int64_t>> GetPartitionGroupsForReplication(
    const HloSharding& sharding, absl::Span<const int64_t> replication_dims);

// Generates partition groups (groups of devices that will communicate via a
// collective) across provided target dims with provided group sizes in vector
// of vector format (legacy format).
std::vector<std::vector<int64_t>> GetPartitionGroupsAcrossTargetDims(
    const HloSharding& sharding, std::vector<int64_t> target_dims,
    std::vector<int64_t> group_sizes);

// Generates partition groups (groups of devices that will communicate via a
// collective) across provided target dims with provided group sizes in iota
// format from sharding.
std::optional<IotaReplicaGroupList> GetIotaPartitionGroupsAcrossTargetDims(
    const HloSharding& sharding, std::vector<int64_t> target_dims,
    std::vector<int64_t> group_sizes, int64_t num_partitions);

// Generates partition groups (groups of devices that will communicate via a
// collective) in iota format from sharding and provided replication_dims.
// NOTE: If provided sharding does not utilize all the partitions, we skip
// generating a compressed format. This is because this device ids
// (IotaReplicaGroupList) generated by this method are partition ids, but later
// they have to be expanded across replicas into global device ids (see
// ExpandPartitionGroupListAcrossReplicas) before they are inserted into a
// collective. The expansion to global device ids while retaining the compressed
// format is only possible if the device list generated covers all partitions.
// The generated device list can cover all partitions if the provided
// sharding covers all partitions.
std::optional<IotaReplicaGroupList> GetIotaPartitionGroupsForReplication(
    const HloSharding& sharding, absl::Span<const int64_t> replication_dims,
    int64_t num_partitions);

// Expands partition group list across all replicas. Expects that provided
// partition_group_list utilizes all the partitions.
CollectiveDeviceList ExpandPartitionGroupListAcrossReplicas(
    IotaReplicaGroupList partition_group_list, int num_replicas,
    int num_partitions);

namespace detail {

// Check if a type is SpmdPartitioningVisitor* type.
template <typename T, typename = void>
struct IsSpmdPartitioningVisitorPointerType : std::false_type {};

template <typename T>
struct IsSpmdPartitioningVisitorPointerType<
    T, std::enable_if_t<std::is_same_v<std::remove_reference_t<T>,
                                       SpmdPartitioningVisitor*>>>
    : std::true_type {};

template <typename T>
constexpr bool IsSpmdPartitioningVisitorPointerType_v =
    IsSpmdPartitioningVisitorPointerType<T>::value;

template <typename T>
using IsSpmdPartitioningVisitorPointer =
    std::enable_if_t<IsSpmdPartitioningVisitorPointerType_v<T>, int>;

template <typename T>
using IsNotSpmdPartitioningVisitorPointer =
    std::enable_if_t<!IsSpmdPartitioningVisitorPointerType_v<T>, int>;

// Check if a type is SpmdBuilder* type.
template <typename T, typename = void>
struct IsSpmdBuilderPointerType : std::false_type {};

template <typename T>
struct IsSpmdBuilderPointerType<
    T,
    std::enable_if_t<std::is_same_v<std::remove_reference_t<T>, SpmdBuilder*>>>
    : std::true_type {};

template <typename T>
constexpr bool IsSpmdBuilderPointerType_v = IsSpmdBuilderPointerType<T>::value;

template <typename T>
using IsSpmdBuilderPointer =
    std::enable_if_t<IsSpmdBuilderPointerType_v<T>, int>;

template <typename T>
using IsNotSpmdBuilderPointer =
    std::enable_if_t<!IsSpmdBuilderPointerType_v<T>, int>;

// Check if a type is HloModule* type.
template <typename T, typename = void>
struct IsHloModulePointerType : std::false_type {};

template <typename T>
struct IsHloModulePointerType<
    T, std::enable_if_t<std::is_same_v<std::remove_reference_t<T>, HloModule*>>>
    : std::true_type {};

template <typename T>
constexpr bool IsHloModulePointerType_v = IsHloModulePointerType<T>::value;

template <typename T>
using IsHloModulePointer = std::enable_if_t<IsHloModulePointerType_v<T>, int>;

template <typename T>
using IsNotHloModulePointer =
    std::enable_if_t<!IsHloModulePointerType_v<T>, int>;

// Check if a type is PartitionedHlo type.
template <typename T, typename = void>
struct IsPartitionedHloType : std::false_type {};

template <typename T>
struct IsPartitionedHloType<
    T, std::enable_if_t<std::is_same_v<std::decay_t<T>, PartitionedHlo>>>
    : std::true_type {};

template <typename T>
constexpr bool IsPartitionedHloType_v = IsPartitionedHloType<T>::value;

template <typename T>
using IsPartitionedHlo = std::enable_if_t<IsPartitionedHloType_v<T>, int>;

template <typename T>
using IsNotPartitionedHlo = std::enable_if_t<!IsPartitionedHloType_v<T>, int>;

// Check if a type is iterable type.
template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::declval<T>().begin()),
                                  decltype(std::declval<T>().end())>>
    : std::true_type {};

template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

template <typename T>
using iterable_element_type =
    std::decay_t<decltype(*std::declval<T>().begin())>;

// Check if a type is iterable container type of PartitionedHlo.
template <typename T, typename = void>
struct IsIterablePartitionedHloContainerType : std::false_type {};

template <typename T>
struct IsIterablePartitionedHloContainerType<
    T,
    std::enable_if_t<is_iterable_v<T> &&
                     std::is_same_v<iterable_element_type<T>, PartitionedHlo>>>
    : std::true_type {};

template <typename T>
constexpr bool IsIterablePartitionedHloContainerType_v =
    IsIterablePartitionedHloContainerType<T>::value;

template <typename T>
using IsIterablePartitionedHloContainer =
    std::enable_if_t<IsIterablePartitionedHloContainerType_v<T>, int>;

template <typename T>
using IsNotIterablePartitionedHloContainer =
    std::enable_if_t<!IsIterablePartitionedHloContainerType_v<T>, int>;

// Create a fake PartitionedHlo object in a fake builder/module as a new
// parameter.
template <typename Arg, IsPartitionedHlo<Arg> = 0>
std::decay_t<Arg> FakePartitionedHlo(Arg&& phlo, HloModule* module,
                                     int* parameter_count,
                                     SpmdPartitioningVisitor* fake_visitor) {
  HloInstruction* param =
      fake_visitor->builder()
          ->AddParameter(HloInstruction::CreateParameter(
              *parameter_count, phlo.hlo()->shape(),
              "fake_parameter." + std::to_string(*parameter_count)))
          .value();
  *parameter_count = *parameter_count + 1;
  PartitionedHlo fake_phlo = phlo.CloneWithNewHlo(param);
  PartitionedHlo::PartitioningState fake_state =
      fake_visitor->MakePartitioningState();
  fake_state.module = module;
  fake_phlo.set_state(fake_state);
  return fake_phlo;
}

// Create a fake PartitionedHlo container object in a fake builder/module as a
// number new parameters.
template <typename Arg, IsIterablePartitionedHloContainer<Arg> = 0>
std::decay_t<Arg> FakeIterablePartitionedHloContainer(
    Arg&& phlo_container, HloModule* module, int* parameter_count,
    SpmdPartitioningVisitor* fake_visitor) {
  std::vector<iterable_element_type<Arg>> phlos;
  phlos.reserve(phlo_container.size());
  for (const PartitionedHlo& phlo : phlo_container) {
    phlos.push_back(std::move(
        FakePartitionedHlo(phlo, module, parameter_count, fake_visitor)));
  }
  bool is_constructible_from_iterators =
      std::is_constructible_v<std::decay_t<Arg>, decltype(phlos.begin()),
                              decltype(phlos.end())>;
  CHECK(is_constructible_from_iterators);
  return std::decay_t<Arg>(phlos.begin(), phlos.end());
}

// Create a fake SpmdPartitioningVisitor*.
template <typename Arg, IsSpmdPartitioningVisitorPointer<Arg> = 0>
std::decay_t<Arg> FakeSpmdPartitioningVisitor(
    Arg&& visitor, SpmdPartitioningVisitor* fake_visitor) {
  return fake_visitor;
}

// Create a fake SpmdBuilder*.
template <typename Arg, IsSpmdBuilderPointer<Arg> = 0>
std::decay_t<Arg> FakeSpmdBuilder(Arg&& builder,
                                  SpmdPartitioningVisitor* fake_visitor) {
  return fake_visitor->builder();
}
// Create a fake HloModule*.
template <typename Arg, IsHloModulePointer<Arg> = 0>
std::decay_t<Arg> FakeHloModule(Arg&& module, HloModule* fake_module) {
  return fake_module;
}
template <class T>
using decay_rvalue_reference_t =
    std::conditional_t<std::is_rvalue_reference<T>::value, std::decay_t<T>, T>;

// Modifies SpmdPartitioningVisitor* type objects.
template <typename Arg, IsSpmdPartitioningVisitorPointer<Arg> = 0>
std::decay_t<Arg> ArgModifier(Arg&& arg, HloModule* module,
                              int* parameter_count,
                              SpmdPartitioningVisitor* fake_visitor) {
  VLOG(5) << "Faking argument type: " << typeid(arg).name();
  return FakeSpmdPartitioningVisitor(std::forward<Arg>(arg), fake_visitor);
}

// Modifies SpmdBuilder* type objects.
template <typename Arg, IsSpmdBuilderPointer<Arg> = 0>
std::decay_t<Arg> ArgModifier(Arg&& arg, HloModule* module,
                              int* parameter_count,
                              SpmdPartitioningVisitor* fake_visitor) {
  VLOG(5) << "Faking argument type: " << typeid(arg).name();
  return FakeSpmdBuilder(std::forward<Arg>(arg), fake_visitor);
}

// Modifies SpmdPartitioningVisitor* type objects.
template <typename Arg, IsHloModulePointer<Arg> = 0>
std::decay_t<Arg> ArgModifier(Arg&& arg, HloModule* module,
                              int* parameter_count,
                              SpmdPartitioningVisitor* fake_visitor) {
  VLOG(5) << "Faking argument type: " << typeid(arg).name();
  return FakeHloModule(std::forward<Arg>(arg), module);
}

// Modifies PartitionedHlo type objects.
template <typename Arg, IsPartitionedHlo<Arg> = 0>
std::decay_t<Arg> ArgModifier(Arg&& arg, HloModule* module,
                              int* parameter_count,
                              SpmdPartitioningVisitor* fake_visitor) {
  VLOG(5) << "Faking argument type: " << typeid(arg).name();
  return FakePartitionedHlo(std::forward<Arg>(arg), module, parameter_count,
                            fake_visitor);
}

// Modifies PartitionedHlo container type objects.
template <typename Arg, IsIterablePartitionedHloContainer<Arg> = 0>
std::decay_t<Arg> ArgModifier(Arg&& arg, HloModule* module,
                              int* parameter_count,
                              SpmdPartitioningVisitor* fake_visitor) {
  VLOG(5) << "Faking argument type: " << typeid(arg).name();
  return FakeIterablePartitionedHloContainer(std::forward<Arg>(arg), module,
                                             parameter_count, fake_visitor);
}

// Modifies nothing, equivalent to no-op.
template <typename Arg, IsNotSpmdPartitioningVisitorPointer<Arg> = 0,
          IsNotSpmdBuilderPointer<Arg> = 0, IsNotHloModulePointer<Arg> = 0,
          IsNotIterablePartitionedHloContainer<Arg> = 0,
          IsNotPartitionedHlo<Arg> = 0>
std::decay_t<Arg> ArgModifier(Arg&& arg, HloModule* module,
                              int* parameter_count,
                              SpmdPartitioningVisitor* fake_visitor) {
  VLOG(5) << "Passing through argument type: " << typeid(arg).name();
  return arg;
}

// Finds SpmdPartitioningVisitor* object in an arg list.
template <typename Arg, IsSpmdPartitioningVisitorPointer<Arg> = 0>
absl::StatusOr<SpmdPartitioningVisitor*> FindSpmdPartitioningVisitor(
    Arg&& arg) {
  return arg;
}

template <typename Arg, typename... Args,
          IsSpmdPartitioningVisitorPointer<Arg> = 0>
absl::StatusOr<SpmdPartitioningVisitor*> FindSpmdPartitioningVisitor(
    Arg&& arg, Args&&... args) {
  return arg;
}

template <typename Arg, typename... Args,
          IsNotSpmdPartitioningVisitorPointer<Arg> = 0>
absl::StatusOr<SpmdPartitioningVisitor*> FindSpmdPartitioningVisitor(
    Arg&& arg, Args&&... args) {
  return FindSpmdPartitioningVisitor(std::forward<Args>(args)...);
}

}  // namespace detail

// Evaluate the memory and communication cost for any arbitrary partitioning
// methods.
template <typename F, typename... Args>
absl::StatusOr<std::pair<int64_t, int64_t>> EvaluatePartitionCost(
    const HloInstruction* original_hlo, F partition_method,
    Args&&... partition_method_args) {
  HloModule* module = original_hlo->GetModule();
  auto comp_env =
      std::make_unique<CompilationEnvironments>(module->comp_envs());
  // Create a fake module and run partitioning with this fake module later.
  HloModule fake_module("fake_module", module->config(), std::move(comp_env));
  auto temp_b = HloComputation::Builder("temp_entry");
  auto temp_p = temp_b.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "input"));
  HloComputation* temp_entry = fake_module.AddEntryComputation(temp_b.Build());

  TF_ASSIGN_OR_RETURN(SpmdPartitioningVisitor * visitor,
                      detail::FindSpmdPartitioningVisitor(
                          std::forward<Args>(partition_method_args)...));
  SpmdPartitioner* partitioner = visitor->partitioner();
  std::unique_ptr<SpmdPartitioningVisitor> fake_visitor = visitor->Clone();
  fake_visitor->set_module(&fake_module);
  auto* fake_b = fake_visitor->builder();
  fake_b->set_visiting_hlo(temp_p);
  auto parameter_count = std::make_unique<int>(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_hlo,
      partition_method(detail::ArgModifier(
          std::forward<Args>(partition_method_args), &fake_module,
          parameter_count.get(), fake_visitor.get())...));

  if (new_hlo == nullptr) {
    return std::make_pair(INT64_MAX, INT64_MAX);
  }
  auto new_entry = fake_module.AddEmbeddedComputation(fake_b->Build(new_hlo));
  // Replace the original computation with the new SPMD computation.
  absl::flat_hash_map<HloComputation*, HloComputation*> replacement;
  replacement[temp_entry] = new_entry;
  for (HloInstruction* hlo : new_entry->instructions()) {
    for (HloComputation* comp : hlo->called_computations()) {
      if (comp->parent() != &fake_module) {
        replacement[comp] = fake_module.AddEmbeddedComputation(comp->Clone());
      }
    }
  }
  fake_module.ReplaceComputations(replacement);

  HloDCE hlo_dce;
  TF_ASSIGN_OR_RETURN(
      auto _, hlo_dce.Run(&fake_module, partitioner->execution_threads()));
  (void)_;  // Suppress unused variable warning in OSS
  VLOG(5) << "Dry-run partitioning for op: " << original_hlo->ToString() << "\n"
          << fake_module.ToString();

  int64_t max_memory = 0;
  int64_t total_communication = 0;
  for (HloComputation* computation : fake_module.computations()) {
    for (HloInstruction* hlo : computation->instructions()) {
      // Check the memory cost for the partitioned hlo op, as well as the
      // memory cost for collectives for potential overhead from full remat.
      if (hlo->opcode() == original_hlo->opcode() || IsCollective(hlo)) {
        int64_t memory_cost = partitioner->MemoryCostInBytes(hlo);
        if (memory_cost > max_memory) {
          VLOG(5) << hlo->ToString() << " has memory cost of " << memory_cost;
          max_memory = memory_cost;
        }
      }
      if (IsCollective(hlo)) {
        total_communication += partitioner->CommunicationCostInBytes(hlo);
      }
    }
  }
  if (max_memory != 0) {
    return std::make_pair(max_memory, total_communication);
  }
  return std::make_pair(INT64_MAX, INT64_MAX);
}

// Creates a copy for the HloInstruction in the PartitionedHlo and returns a
// new PartitionedHlo for the copy.
PartitionedHlo MakeACopyAndReturnItsPartitionedHlo(const PartitionedHlo& phlo,
                                                   SpmdBuilder* b);

// For dynamic-update-slice, we focus on the partitioned slice dimensions,
// ignoring batch dimensions and replicated slice dimensions. We have three
// methods to handle the partitioned slice dimensions.
//
// 1. **Default.** Replicate all tensors along the slice dimensions.
// 2. **Single Partition Update.** The update is entirely contained within a
//    single partition. All partitioned slice dimensions satisfy
//    2.1 The slice size is 1, OR
//    2.2 The update indices are compile-time constants, and the start and end
//        indices reside in the same partition.
// 3. **Constant Indices.** All partitioned slice dimensions have compile-time
//    constant indices.
//
// If both optimizations (2 and 3) are feasible, we prioritize (2) over (3).
// Refer to go/dus-spmd for more details.
enum class DynamicUpdateSliceMethod {
  // Replicate all tensors along the slice dimensions.
  kDefault,

  // The update is fully contained in a single partition.
  kUpdateOnASinglePartition,

  // All partitioned slice dimensions have compile-time constant indices.
  kAllPartitionedSliceDimsHaveConstantIndices,
};

struct DynamicUpdateSliceAnalysis {
  DynamicUpdateSliceMethod method;
  // All slice dimensions of the dynamic update slice instruction.
  std::vector<int64_t> slice_dims;
  // The slice dimensions that are partitioned.
  std::vector<int64_t> partitioned_slice_dims;
};

DynamicUpdateSliceAnalysis AnalyzeDynamicUpdateSlice(const HloInstruction* hlo);

}  // namespace spmd
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SPMD_PARTITIONER_UTIL_H_
