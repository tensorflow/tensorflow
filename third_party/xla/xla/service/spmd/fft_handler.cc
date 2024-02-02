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

#include <float.h>

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "xla/client/lib/comparators.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/literal_util.h"
#include "xla/protobuf_util.h"
#include "xla/service/shape_inference.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {

namespace {

// Pad each partition to have size that is multiplication of num_partitions.
// For example, if input is {0, 1, 2, 3, 4, 5} and num_partitions = 2,
// after padding, it becomes {0, 1, 2, 3} in partition 0 and {4, 5, 0, 0} in
// partition 1.
std::optional<HloInstruction*> PadEachPartitionWithHaloExchange(
    HloInstruction* hlo, int64_t num_partitions, const HloSharding& sharding,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, HloInstruction* partition_id, SpmdBuilder* b) {
  int64_t size_per_partition = hlo->shape().dimensions().back();
  int64_t size_padded_per_partition =
      CeilOfRatio(size_per_partition, num_partitions) * num_partitions;
  if (size_per_partition == size_padded_per_partition) {
    return hlo;
  }
  // 1. Calculate left_halo size.
  // left-halo size is 0
  OffsetCalculation left_halo_size_function =
      OffsetCalculation(MultiplyAddDivideOffsetCalculation(0, 0, 1));

  // 2. Calculate right_halo size.
  // D = size_padded_per_partition
  // S = size_per_partition
  // i = shard_ordinal
  // right-halo size is D * (i + 2) - S * (i + 2) = (D - S) * i + 2 * (D - S)
  OffsetCalculation right_halo_size_function =
      OffsetCalculation(MultiplyAddDivideOffsetCalculation(
          size_padded_per_partition - size_per_partition,
          2 * (size_padded_per_partition - size_per_partition), 1));

  auto concat = hlo;
  // 3. Halo exchange.
  auto halo_exchange_result =
      ExchangeHalo(hlo, left_halo_size_function, right_halo_size_function,
                   hlo->shape().rank() - 1, sharding, collective_ops_creator,
                   next_channel_id, b);

  if (halo_exchange_result.has_value()) {
    concat = halo_exchange_result.value();
  } else {
    return std::nullopt;
  }

  // 4. Slice the valid result.
  // Slice offset is (D - S) * i
  OffsetCalculation start_offset_on_padded_concat_calculation =
      OffsetCalculation(MultiplyAddDivideOffsetCalculation(
          size_padded_per_partition - size_per_partition, 0, 1));
  auto slice_shape = concat->shape();
  slice_shape.set_dimensions(concat->shape().rank() - 1,
                             size_padded_per_partition);
  auto zero_s32 =
      b->AddInstruction(HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  std::vector<HloInstruction*> slice_offsets(concat->shape().rank(), zero_s32);
  auto partition_ordinals =
      MakeTiledPartitionOrdinals(sharding, partition_id, b);
  slice_offsets[concat->shape().rank() - 1] =
      start_offset_on_padded_concat_calculation.Calculate(
          partition_ordinals[concat->shape().rank() - 1], b);
  return b->AddInstruction(HloInstruction::CreateDynamicSlice(
      slice_shape, concat, slice_offsets, slice_shape.dimensions()));
}

// If partition 0 has {0, 1, 2, 3} and num partitions is 2, after shuffling,
// the data becomes {0, 2, 1, 3}.
HloInstruction* ShuffleWithinEachPartitionUsingOneHot(HloInstruction* hlo,
                                                      int64_t num_partitions,
                                                      SpmdBuilder* b) {
  int64_t size_per_partition = hlo->shape().dimensions().back();
  CHECK_EQ(size_per_partition % num_partitions, 0);
  auto indices_iota = b->AddInstruction(HloInstruction::CreateIota(
      ShapeUtil::MakeShape(S32, {size_per_partition}), 0));
  auto reshape_indices_iota = b->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(
          S32, {size_per_partition / num_partitions, num_partitions}),
      indices_iota));
  auto transpoe_indices_iota =
      b->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(
              S32, {num_partitions, size_per_partition / num_partitions}),
          reshape_indices_iota, {1, 0}));
  auto one_hot_indices = b->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(S32, {size_per_partition, size_per_partition}),
      b->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(S32, {size_per_partition}),
          transpoe_indices_iota)),
      /*broadcast_dimensions=*/{1}));

  auto partition_indices = b->AddInstruction(HloInstruction::CreateIota(
      ShapeUtil::MakeShape(S32, {size_per_partition, size_per_partition}), 0));

  auto shuffle_one_hot = b->AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::ChangeElementType(partition_indices->shape(),
                                   hlo->shape().element_type()),
      b->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::ChangeElementType(partition_indices->shape(), PRED),
          one_hot_indices, partition_indices, ComparisonDirection::kEq))));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(hlo->shape().rank() - 1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  HloInstruction* dot = b->AddInstruction(HloInstruction::CreateDot(
      hlo->shape(), hlo, shuffle_one_hot, dot_dnums, precision_config));
  return dot;
}

// If partition 0 has {0, 2, 1, 3}, partition 1 has {4, 0, 5, 0} and
// num partitions is 2, after all-to-all, partition 0 will have {0, 2, 4, 0}
// and partition 1 will have {1, 3, 5, 0}.
HloInstruction* ShuffleDataWithAllToAll(
    HloInstruction* hlo, int64_t num_partitions,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdBuilder* b) {
  std::vector<std::vector<int64_t>> groups(1);
  std::vector<int64_t> partition_subgroups(num_partitions);
  std::iota(partition_subgroups.begin(), partition_subgroups.end(), 0);
  groups[0] = partition_subgroups;
  auto all_to_all = collective_ops_creator.create_cross_partition_all_to_all(
      b, {hlo}, groups, (*next_channel_id)++, hlo->shape().rank() - 1);
  return all_to_all;
}

HloInstruction* GetCorrectionFactor(HloInstruction* hlo, int64_t num_partitions,
                                    HloInstruction* partition_id,
                                    SpmdBuilder* b) {
  /* n = size_per_replica
     m = num_partitions
  factor = tf.exp(-2.0j * np.pi * tf.cast(position_index, tf.complex64) *
                    * tf.cast(tf.range(n), dtype=tf.complex64) /
                    (n * m))

  */
  auto add_hlo = [&](std::unique_ptr<HloInstruction> to_add) {
    return b->AddInstruction(std::move(to_add));
  };
  int64_t per_replica_size = hlo->shape().dimensions().back();
  auto constant_factor =
      add_hlo(HloInstruction::CreateConstant(LiteralUtil::CreateR0(
          complex64(0, -2.0 * M_PI / (num_partitions * per_replica_size)))));
  constant_factor = add_hlo(HloInstruction::CreateBroadcast(
      hlo->shape(), constant_factor, /*broadcast_dimensions=*/{}));
  auto converted_partition_id = add_hlo(HloInstruction::CreateConvert(
      ShapeUtil::ChangeElementType(partition_id->shape(),
                                   hlo->shape().element_type()),
      partition_id));
  // TODO(wangtao): multipy before broadcast.
  auto broadcast_partition_id = add_hlo(HloInstruction::CreateBroadcast(
      hlo->shape(), converted_partition_id, /*broadcast_dimensions=*/{}));
  auto exp_operand = add_hlo(
      HloInstruction::CreateBinary(hlo->shape(), HloOpcode::kMultiply,
                                   constant_factor, broadcast_partition_id));
  auto iota = add_hlo(
      HloInstruction::CreateIota(hlo->shape(), hlo->shape().rank() - 1));
  exp_operand = add_hlo(HloInstruction::CreateBinary(
      hlo->shape(), HloOpcode::kMultiply, exp_operand, iota));
  return add_hlo(
      HloInstruction::CreateUnary(hlo->shape(), HloOpcode::kExp, exp_operand));
}

// Sudo code for the while loop:
// def body(dest_transform, dest_core_position, source_transform,
//             source_core_position, i):
//      factor = tf.exp(-2.0j * np.pi  *
//                      tf.cast(dest_core_position, tf.complex64) *
//                tf.cast(source_core_position, tf.complex64) / num_partitions)
//      dest_transform += factor * source_transform
//      source_core_position = tf.raw_ops.CollectivePermute(
//          input=source_core_position,
//          source_target_pairs=source_target_pairs,
//          name='source_core_position_permute')
//      source_transform = tf.raw_ops.CollectivePermute(
//          input=source_transform,
//          source_target_pairs=source_target_pairs,
//          name='source_transform_permute')
//      i += 1
//      return (dest_transform, dest_core_position, source_transform,
//              source_core_position, i)
HloInstruction* GetFinalFftUsingCollectivePermute(
    HloInstruction* hlo, const HloSharding& sharding,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t num_partitions, HloInstruction* partition_id,
    int64_t* next_channel_id, HloModule* module, SpmdBuilder* b) {
  auto iteration = b->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(0)));
  auto converted_partition_id = b->AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::ChangeElementType(partition_id->shape(),
                                   hlo->shape().element_type()),
      partition_id));
  // Buid while loop body.
  SpmdBuilder body_b("fft_collective_permute_body", hlo);
  auto param = body_b.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0,
      ShapeUtil::MakeTupleShape(
          {hlo->shape(), hlo->shape(), converted_partition_id->shape(),
           converted_partition_id->shape(), iteration->shape()}),
      "param"));
  auto dest_transform = body_b.AddInstruction(
      HloInstruction::CreateGetTupleElement(hlo->shape(), param, 0));
  auto source_transform = body_b.AddInstruction(
      HloInstruction::CreateGetTupleElement(hlo->shape(), param, 1));
  auto dest_partition_id =
      body_b.AddInstruction(HloInstruction::CreateGetTupleElement(
          converted_partition_id->shape(), param, 2));
  auto source_partition_id =
      body_b.AddInstruction(HloInstruction::CreateGetTupleElement(
          converted_partition_id->shape(), param, 3));
  auto i = body_b.AddInstruction(
      HloInstruction::CreateGetTupleElement(iteration->shape(), param, 4));
  /*
    factor = tf.exp(-2.0j * np.pi  *
                      tf.cast(dest_partiton_id, tf.complex64) *
                      tf.cast(source_partition_id, tf.complex64) /
    num_partitions) dest_transform += factor * source_transform
  */
  auto constant_factor = body_b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR0(complex64(0, -2.0 * M_PI / num_partitions))));

  constant_factor = body_b.AddInstruction(HloInstruction::CreateBinary(
      constant_factor->shape(), HloOpcode::kMultiply, constant_factor,
      dest_partition_id));
  constant_factor = body_b.AddInstruction(HloInstruction::CreateBinary(
      constant_factor->shape(), HloOpcode::kMultiply, constant_factor,
      source_partition_id));
  auto phase_factor = body_b.AddInstruction(HloInstruction::CreateUnary(
      constant_factor->shape(), HloOpcode::kExp, constant_factor));
  phase_factor = body_b.AddInstruction(
      HloInstruction::CreateBroadcast(hlo->shape(), phase_factor, {}));
  auto phase_adjust_source_transform =
      body_b.AddInstruction(HloInstruction::CreateBinary(
          hlo->shape(), HloOpcode::kMultiply, phase_factor, source_transform));
  dest_transform = body_b.AddInstruction(HloInstruction::CreateBinary(
      hlo->shape(), HloOpcode::kAdd, phase_adjust_source_transform,
      dest_transform));
  // collective permute for source partition_id and source_transfrom.
  std::vector<std::pair<int64_t, int64_t>> src_dst_pairs;
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64_t> indices, int64_t src_device) {
        std::vector<int64_t> target_indices(indices.begin(), indices.end());
        target_indices.back() = (indices.back() + 1) % num_partitions;
        int64_t dst_device = sharding.tile_assignment()(target_indices);
        src_dst_pairs.emplace_back(src_device, dst_device);
      });

  source_partition_id =
      collective_ops_creator.create_cross_partition_collective_permute(
          &body_b, source_partition_id, src_dst_pairs, (*next_channel_id)++);

  source_transform =
      collective_ops_creator.create_cross_partition_collective_permute(
          &body_b, source_transform, src_dst_pairs, (*next_channel_id)++);

  // ++i
  i = body_b.AddInstruction(HloInstruction::CreateBinary(
      i->shape(), HloOpcode::kAdd, i,
      body_b.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(1)))));
  body_b.AddInstruction(
      HloInstruction::CreateTuple({dest_transform, source_transform,
                                   dest_partition_id, source_partition_id, i}));

  // Build while loop conditions.
  auto zero = CreateZero(hlo->shape(), b);
  SpmdBuilder cond_b("fft_collective_permute_condition", hlo);
  auto cond_param = cond_b.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0,
      ShapeUtil::MakeTupleShape(
          {hlo->shape(), hlo->shape(), converted_partition_id->shape(),
           converted_partition_id->shape(), iteration->shape()}),
      "param"));
  auto cond_i = cond_b.AddInstruction(
      HloInstruction::CreateGetTupleElement(iteration->shape(), cond_param, 4));
  cond_b.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}), cond_i,
      cond_b.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<uint32_t>(num_partitions))),
      ComparisonDirection::kLt));

  // Build while loop.
  auto while_loop = b->AddInstruction(HloInstruction::CreateWhile(
      cond_param->shape(), module->AddEmbeddedComputation(cond_b.Build()),
      module->AddEmbeddedComputation(body_b.Build()),
      b->AddInstruction(
          HloInstruction::CreateTuple({zero, hlo, converted_partition_id,
                                       converted_partition_id, iteration}))));

  return b->AddInstruction(
      HloInstruction::CreateGetTupleElement(hlo->shape(), while_loop, 0));
}

// Slice valid data in each partition.
HloInstruction* SliceValidData(HloInstruction* hlo, const Shape& target_shape,
                               SpmdBuilder* b) {
  std::vector<int64_t> start_indices(target_shape.rank(), 0);
  std::vector<int64_t> strides(target_shape.rank(), 1);
  return b->AddInstruction(HloInstruction::CreateSlice(
      target_shape, hlo, start_indices, target_shape.dimensions(), strides));
}

}  // namespace

// Distributed FFT using the algorithm described in go/tpu-spmd-fft.
Status SpmdPartitioningVisitor::HandleFft(HloInstruction* hlo) {
  if (hlo->operand(0)->shape().rank() < 3 || hlo->fft_type() != FftType::FFT) {
    return DefaultAction(hlo);
  }

  // Only support input_length equals fft_length's case.
  int64_t input_length = hlo->operand(0)->shape().dimensions().back();
  int64_t fft_length = hlo->fft_length().back();
  if (input_length != fft_length || input_length % num_partitions_ != 0) {
    return DefaultAction(hlo);
  }

  // Support partition at the last dimension only.
  if (!hlo->has_sharding() ||
      hlo->sharding().tile_assignment().dimensions().back() !=
          num_partitions_) {
    return DefaultAction(hlo);
  }

  auto partitioned_input =
      GetPartitionedHlo(hlo->operand(0))
          .PadWithValue(CreateR0WithType(hlo->shape().element_type(), 0, &b_));

  // 1.a. Use right halo exchange to shuffle data first and slice with
  // valid data. Data shuffling ensures an in-order transform that the sequences
  // of data before and after the transform are the same. The data shuffling
  // requires the size of data per partition is divisible by the number of
  // partitions. For example, If input is {0, 1, 2, 3, 4, 5} and
  // num partitions is 2, after halo exchange partition 0 has {0, 1, 2, 3} and
  // partition 1 has {4, 5, 0, 0}, where 0s in the partition 1 are padding data.
  // Zeros paddings append zeros to the end of the full data.
  auto result = partitioned_input.hlo();
  auto padded_hlo = PadEachPartitionWithHaloExchange(
      partitioned_input.hlo(), num_partitions_, hlo->sharding(),
      partitioned_input.state().collective_ops_creator,
      partitioned_input.state().next_channel_id,
      partitioned_input.state().partition_id, partitioned_input.state().b);

  if (padded_hlo.has_value()) {
    result = padded_hlo.value();
  }

  // 1.b Shuffle data within each partition using one hot and matmul.
  // If partition 0 has {0, 1, 2, 3} and num partitions is 2, after shuffling,
  // the data becomes {0, 2, 1, 3}.
  result = ShuffleWithinEachPartitionUsingOneHot(result, num_partitions_,
                                                 partitioned_input.state().b);
  // 1.c all-to-all
  // If partition 0 has {0, 2, 1, 3}, partition 1 has {4, 0, 5, 0} and
  // num partitions is 2, after all-to-all, partition 0 will have {0, 2, 4, 0}
  // and partition 1 will have {1, 3, 5, 0}.
  result = ShuffleDataWithAllToAll(
      result, num_partitions_, partitioned_input.state().collective_ops_creator,
      partitioned_input.state().next_channel_id, partitioned_input.state().b);
  // 1.d Slice valid data in each partition.
  result = SliceValidData(result, partitioned_input.hlo()->shape(), &b_);

  // 2. Do local fft transform.
  auto partitioned_fft_length = hlo->fft_length();
  partitioned_fft_length.back() /= num_partitions_;
  result = b_.AddInstruction(HloInstruction::CreateFft(
      result->shape(), result, hlo->fft_type(), partitioned_fft_length));

  // Multiply by correct factor for local phase ajustment.
  auto correction_factor = GetCorrectionFactor(
      result, num_partitions_, partitioned_input.state().partition_id,
      partitioned_input.state().b);
  result = b_.AddInstruction(HloInstruction::CreateBinary(
      result->shape(), HloOpcode::kMultiply, result, correction_factor));

  // 3. Second phase FFT with collective permute. fft_length = num_partitions.
  result = GetFinalFftUsingCollectivePermute(
      result, hlo->sharding(), partitioned_input.state().collective_ops_creator,
      num_partitions_, partitioned_input.state().partition_id,
      partitioned_input.state().next_channel_id, module_,
      partitioned_input.state().b);

  result->set_sharding(hlo->sharding());
  auto partitioned_fft =
      PartitionedHlo(result, hlo->shape(), partitioned_input.state());
  SetPartitionedHlo(hlo, partitioned_fft);
  return OkStatus();
}

}  // namespace spmd
}  // namespace xla
