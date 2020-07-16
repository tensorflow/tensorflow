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

#include <memory>
#include <string>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"

namespace xla {
namespace spmd {

// Returns true if the given sharding contains any replicated sharding.
bool HasReplicatedSharding(const HloSharding& sharding);

// Creates zero value instructions of the given shape.
HloInstruction* CreateZero(const Shape& shape, SpmdBuilder* b);

template <typename NativeT>
HloInstruction* CreateR0WithType(PrimitiveType type, NativeT value,
                                 SpmdBuilder* b) {
  auto literal = LiteralUtil::CreateR0(value)
                     .ConvertToShape(ShapeUtil::MakeShape(type, {}))
                     .ValueOrDie();
  return b->AddInstruction(HloInstruction::CreateConstant(std::move(literal)));
}

inline HloInstruction* CreateFirstWithType(PrimitiveType type, SpmdBuilder* b) {
  if (type == F32) {
    auto float_pad_value = std::numeric_limits<float>::quiet_NaN();
    return CreateR0WithType(type, -float_pad_value, b);
  }
  auto literal = LiteralUtil::MinValue(type);
  return b->AddInstruction(HloInstruction::CreateConstant(std::move(literal)));
}

inline HloInstruction* CreateLastWithType(PrimitiveType type, SpmdBuilder* b) {
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
int64 ShapeSizeInBytes(const Shape& shape);

// Returns the shard shape for a partition without padding due to uneven
// sharding.
Shape MakeNonPaddedShapeForGivenPartition(const Shape& shape,
                                          const HloSharding& sharding,
                                          int64 partition_id);

// Generates the HLO instructions that represent the dimension offsets on any
// device. The size of the returned vector is the rank of the given shape.
std::vector<HloInstruction*> MakePartitionOffsets(const Shape& shape,
                                                  const HloSharding& sharding,
                                                  HloInstruction* partition_id,
                                                  SpmdBuilder* b);

// Returns the offsets of the partition in the tile assignment.
std::vector<HloInstruction*> MakeTiledPartitionOrdinals(
    const HloSharding& sharding, HloInstruction* partition_id, SpmdBuilder* b);

// Pads hlo to the desired shape using high padding. Either a builder or a
// computation needs to be supplied, but not both.
HloInstruction* PadToShape(HloInstruction* hlo, const Shape& padded_shape,
                           SpmdBuilder* b,
                           HloComputation* computation = nullptr);

// Returns the padded shape when combining all partitions.
Shape GetPaddedShapeForUnevenPartitioning(const Shape& base_shape,
                                          const HloSharding& sharding);

// Pads the HLO (with base shape) for uneven tiled partition to make it evenly
// partitionable.
HloInstruction* PadBaseShapeBeforeUnevenTiledSharding(
    HloInstruction* hlo, const HloSharding& sharding, SpmdBuilder* b);

// Returns the index of the unique tile dimension. Returns absl::nullopt if the
// given sharding is not tiled or tiled along multiple dimensions.
absl::optional<int64> UniqueTiledDim(const HloSharding& sharding);

// Utilities for symbolic offset calculation and halo exchange.
class OffsetCalculation;

// Represents a calculation over integers:
//   (shard_ordinal * multiplier + offset) / divisor
class MultiplyAddDivideOffsetCalculation {
 public:
  MultiplyAddDivideOffsetCalculation()
      : multiplier_(0), offset_(0), divisor_(1) {}
  MultiplyAddDivideOffsetCalculation(int64 multiplier, int64 offset,
                                     int64 divisor);

  OffsetCalculation operator-(
      const MultiplyAddDivideOffsetCalculation& other) const;

  bool operator==(const MultiplyAddDivideOffsetCalculation& other) const {
    return multiplier_ == other.multiplier_ && offset_ == other.offset_ &&
           divisor_ == other.divisor_;
  }

  bool IsConstant() const { return multiplier_ == 0; }
  void Simplify();
  int64 Calculate(int64 shard_ordinal) const;
  HloInstruction* Calculate(HloInstruction* shard_ordinal,
                            SpmdBuilder* b) const;

  // Returns the maximum result for shard ordinals in the range
  // [start_ordinal, limit_ordinal).
  int64 MaxInRange(int64 start_ordinal, int64 limit_ordinal) const;

 private:
  int64 multiplier_;
  int64 offset_;
  int64 divisor_;
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
        lhs_(absl::make_unique<OffsetCalculation>(lhs)),
        rhs_(absl::make_unique<OffsetCalculation>(rhs)) {}
  OffsetCalculation(HloOpcode opcode, const OffsetCalculation& lhs,
                    const OffsetCalculation& rhs)
      : opcode_(opcode),
        lhs_(absl::make_unique<OffsetCalculation>(lhs)),
        rhs_(absl::make_unique<OffsetCalculation>(rhs)) {}

  OffsetCalculation& operator=(const OffsetCalculation& other);

  // Returns whether the calculation returns the same value for all shards. This
  // is conservative and could return false even if it is actually constant.
  bool IsConstant() const;

  OffsetCalculation operator-(const OffsetCalculation& other) const;
  bool operator==(const OffsetCalculation& other) const;
  int64 Calculate(int64 shard_ordinal) const;
  HloInstruction* Calculate(HloInstruction* shard_ordinal,
                            SpmdBuilder* b) const;

  // Returns the maximum result for shard ordinals in the range
  // [start_ordinal, limit_ordinal).
  int64 MaxInRange(int64 start_ordinal, int64 limit_ordinal) const;

 private:
  HloOpcode opcode_;
  std::unique_ptr<OffsetCalculation> lhs_;
  std::unique_ptr<OffsetCalculation> rhs_;
  MultiplyAddDivideOffsetCalculation copy_from_;
};

// Performs halo exchange on the given dimension based on the provided
// left/right halo size functions. Returns nullopt if the halo is beyond the
// direct neighbor of the shard.
absl::optional<HloInstruction*> ExchangeHalo(
    HloInstruction* hlo, const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function, int64 dim,
    const HloSharding& target,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, SpmdBuilder* b);

// Exchange halo on all dimensions of the HLO. Returns nullopt if any one of the
// dimensions fails to exchange halo (halo is beyond the neighbor shard).
absl::optional<HloInstruction*> ExchangeHalo(
    HloInstruction* hlo,
    std::vector<OffsetCalculation> left_halo_size_functions,
    std::vector<OffsetCalculation> right_halo_size_functions,
    const HloSharding& target,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, SpmdBuilder* b);

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
absl::optional<HloInstruction*> ExchangeHaloAndGetValidData(
    HloInstruction* hlo, const Shape& base_shape,
    const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function,
    int64 explicit_left_padding_on_full_shape, int64 padded_full_shape_size,
    int64 shard_size_with_halo, int64 dim, const HloSharding& target,
    HloInstruction* offset_on_padded_shape, HloInstruction* pad_value,
    HloInstruction* partition_ordinal,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, SpmdBuilder* b, bool mask_invalid_region = true);

// Uses halo exchange to change from right-padding to left-padding for uneven
// tiled sharding on the given dimensions. Tiled sharding always pads uneven
// partitioned data on the right, but we need to swap it to the left for
// kReverse or kConvolution with window reversal.
HloInstruction* HaloExchangeToPadOnLeft(PartitionedHlo& original,
                                        absl::Span<const int64> dims);

// Check if the computation is GT comparison and safe for NaNs.
bool IsNanSafeGt(HloComputation* computation);

// Return k in TopK when input value is parttioned in the sort dimension.
absl::optional<int64> GetKValueInTopKWhenPartitionSortDim(HloInstruction* hlo);

// Slices the first k elements at slice dimension.
HloInstruction* SliceFirstK(HloInstruction* hlo, SpmdBuilder* builder,
                            int64 slice_dim, int64 k);

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SPMD_PARTITIONER_UTIL_H_
