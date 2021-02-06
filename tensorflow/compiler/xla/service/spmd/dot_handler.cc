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

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/spmd/convolution_handler.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/numbers.h"

namespace xla {
namespace spmd {

Status SpmdPartitioningVisitor::HandleDot(HloInstruction* hlo) {
  DotConvDimsMapping mapping;
  const auto& dnums = hlo->dot_dimension_numbers();
  int64 next_output_dim = 0;
  for (int64 i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
    mapping.batch_dims.emplace_back();
    mapping.batch_dims.back().lhs = dnums.lhs_batch_dimensions(i);
    mapping.batch_dims.back().rhs = dnums.rhs_batch_dimensions(i);
    mapping.batch_dims.back().output = next_output_dim++;
  }
  for (int64 i = 0; i < dnums.lhs_contracting_dimensions_size(); ++i) {
    mapping.contracting_dims.emplace_back();
    mapping.contracting_dims.back().lhs = dnums.lhs_contracting_dimensions(i);
    mapping.contracting_dims.back().rhs = dnums.rhs_contracting_dimensions(i);
    mapping.contracting_dims.back().output = -1;
  }
  for (int64 i = 0; i < hlo->operand(0)->shape().rank(); ++i) {
    if (absl::c_linear_search(dnums.lhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.lhs_contracting_dimensions(), i)) {
      continue;
    }
    mapping.lhs_non_contracting_dims.emplace_back();
    mapping.lhs_non_contracting_dims.back().lhs = i;
    mapping.lhs_non_contracting_dims.back().rhs = -1;
    mapping.lhs_non_contracting_dims.back().output = next_output_dim++;
  }
  for (int64 i = 0; i < hlo->operand(1)->shape().rank(); ++i) {
    if (absl::c_linear_search(dnums.rhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.rhs_contracting_dimensions(), i)) {
      continue;
    }
    mapping.rhs_non_contracting_dims.emplace_back();
    mapping.rhs_non_contracting_dims.back().lhs = -1;
    mapping.rhs_non_contracting_dims.back().rhs = i;
    mapping.rhs_non_contracting_dims.back().output = next_output_dim++;
  }
  auto create_sharded_dot =
      [&](HloInstruction* l, HloInstruction* r, SpmdBuilder* b,
          const Window& conv_window) -> StatusOr<HloInstruction*> {
    TF_ASSIGN_OR_RETURN(
        auto sharded_dot_shape,
        ShapeInference::InferDotOpShape(
            l->shape(), r->shape(), hlo->dot_dimension_numbers(),
            /*preferred_element_type=*/hlo->shape().element_type()));
    return b->AddInstruction(HloInstruction::CreateDot(
        sharded_dot_shape, l, r, hlo->dot_dimension_numbers(),
        hlo->precision_config()));
  };
  return HandleDotHelper(hlo, mapping, create_sharded_dot);
}

namespace {

std::vector<int64> GetAllDevicesInOrder(const HloSharding& sharding) {
  CHECK(!sharding.IsTileMaximal());
  std::vector<int64> results;
  results.reserve(sharding.tile_assignment().num_elements());
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64> /* indices */, int64 device) {
        results.push_back(device);
      });
  return results;
}

StatusOr<HloInstruction*> PartitionBaseCase(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64 num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    int64 lhs_batch_partitions, int64 rhs_batch_partitions,
    int64 output_batch_partitions, int64 lhs_contracting_partitions,
    int64 rhs_contracting_partitions, int64 lhs_non_contracting_partitions,
    int64 rhs_non_contracting_partitions,
    int64 output_lhs_non_contracting_partitions,
    int64 output_rhs_non_contracting_partitions,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops,
    bool may_reshard_without_detecting_match) {
  const HloSharding& lhs_sharding = lhs.sharding();
  const HloSharding& rhs_sharding = rhs.sharding();
  if (lhs_sharding.ReplicateOnLastTileDim() ||
      rhs_sharding.ReplicateOnLastTileDim() ||
      output_sharding.ReplicateOnLastTileDim()) {
    return nullptr;
  }
  std::vector<int64> lhs_to_rhs_indices(lhs.base_shape().rank(), -1);
  std::vector<int64> lhs_to_output_indices(lhs.base_shape().rank(), -1);
  std::vector<int64> rhs_to_lhs_indices(rhs.base_shape().rank(), -1);
  std::vector<int64> rhs_to_output_indices(rhs.base_shape().rank(), -1);
  std::vector<int64> output_to_lhs_indices(output_base_shape.rank(), -1);
  std::vector<int64> output_to_rhs_indices(output_base_shape.rank(), -1);
  auto populate_indices_mapping =
      [&](const DotConvDimsMapping::DimsMapping& mapping) {
        if (mapping.lhs >= 0) {
          lhs_to_rhs_indices[mapping.lhs] = mapping.rhs;
          lhs_to_output_indices[mapping.lhs] = mapping.output;
        }
        if (mapping.rhs >= 0) {
          rhs_to_lhs_indices[mapping.rhs] = mapping.lhs;
          rhs_to_output_indices[mapping.rhs] = mapping.output;
        }
        if (mapping.output >= 0) {
          output_to_lhs_indices[mapping.output] = mapping.lhs;
          output_to_rhs_indices[mapping.output] = mapping.rhs;
        }
      };
  for (const auto& mapping : dims_mapping.batch_dims) {
    populate_indices_mapping(mapping);
  }
  for (const auto& mapping : dims_mapping.contracting_dims) {
    populate_indices_mapping(mapping);
  }
  for (const auto& mapping : dims_mapping.lhs_non_contracting_dims) {
    populate_indices_mapping(mapping);
  }
  for (const auto& mapping : dims_mapping.rhs_non_contracting_dims) {
    populate_indices_mapping(mapping);
  }
  for (const auto& mapping : dims_mapping.conv_spatial_dims) {
    populate_indices_mapping(mapping);
  }
  auto lhs_sharding_transposed_to_match_rhs =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          lhs_sharding, lhs_to_rhs_indices, rhs_to_lhs_indices);
  auto rhs_sharding_transposed_to_match_lhs =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          rhs_sharding, rhs_to_lhs_indices, lhs_to_rhs_indices);
  auto lhs_sharding_transposed_to_match_output =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          lhs_sharding, lhs_to_output_indices, output_to_lhs_indices);
  auto rhs_sharding_transposed_to_match_output =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          rhs_sharding, rhs_to_output_indices, output_to_rhs_indices);
  auto output_sharding_transposed_to_match_lhs =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          output_sharding, output_to_lhs_indices, lhs_to_output_indices);
  auto output_sharding_transposed_to_match_rhs =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          output_sharding, output_to_rhs_indices, rhs_to_output_indices);

  // LHS and RHS are partitioned the same way and only partitioned in batch
  // dimensions.
  if (lhs_batch_partitions == rhs_batch_partitions &&
      rhs_batch_partitions == num_partitions &&
      lhs_sharding_transposed_to_match_rhs == rhs_sharding) {
    TF_ASSIGN_OR_RETURN(
        auto dot, create_sharded_dot(lhs.hlo(), rhs.hlo(), b, conv_window));
    dot->set_sharding(*lhs_sharding_transposed_to_match_output);
    return PartitionedHlo(dot, output_base_shape, lhs.state())
        .Reshard(output_sharding)
        .hlo();
  }

  // Try emit batch-partitioned einsum with one operand resharded. Returns
  // partitioned HLO or nullptr if the attempt fails. If
  // may_reshard_with_allreduce is false, reshard must be done using
  // all-to-all/collective-permute; otherwise this attempt fails.
  auto try_emit_output_batch_partitioned_einsum_with_reshard =
      [&](bool may_reshard_with_allreduce) -> StatusOr<HloInstruction*> {
    // LHS and output are batch partitioned in the same way.
    if (lhs_batch_partitions == num_partitions &&
        output_batch_partitions == num_partitions &&
        lhs_sharding_transposed_to_match_output == output_sharding) {
      if (!may_reshard_with_allreduce &&
          !CanReshardWithCollectivePermute(
              rhs.sharding(), *lhs_sharding_transposed_to_match_rhs) &&
          !GetReshardAllToAllSourceTargetDims(
              rhs.sharding(), *lhs_sharding_transposed_to_match_rhs)) {
        return nullptr;
      }
      auto resharded_rhs = rhs.Reshard(*lhs_sharding_transposed_to_match_rhs);
      TF_ASSIGN_OR_RETURN(
          auto dot,
          create_sharded_dot(lhs.hlo(), resharded_rhs.hlo(), b, conv_window));
      return dot;
    }
    // RHS and output are batch partitioned in the same way.
    if (rhs_batch_partitions == num_partitions &&
        output_batch_partitions == num_partitions &&
        rhs_sharding_transposed_to_match_output == output_sharding) {
      if (!may_reshard_with_allreduce &&
          !CanReshardWithCollectivePermute(
              lhs.sharding(), *rhs_sharding_transposed_to_match_lhs) &&
          !GetReshardAllToAllSourceTargetDims(
              lhs.sharding(), *rhs_sharding_transposed_to_match_lhs)) {
        return nullptr;
      }
      auto resharded_lhs = lhs.Reshard(*rhs_sharding_transposed_to_match_lhs);
      TF_ASSIGN_OR_RETURN(
          auto dot,
          create_sharded_dot(resharded_lhs.hlo(), rhs.hlo(), b, conv_window));
      return dot;
    }
    return nullptr;
  };

  {
    // Try batch-parallel by resharding one operand, and not using all-reduce.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * partitioned_dot,
        try_emit_output_batch_partitioned_einsum_with_reshard(false));
    if (partitioned_dot) {
      return partitioned_dot;
    }
  }

  int64 output_sharding_dim = -1;
  for (int64 i = 0; i < output_sharding.tile_assignment().num_dimensions();
       ++i) {
    if (output_sharding.tile_assignment().dim(i) == num_partitions) {
      output_sharding_dim = i;
      break;
    }
  }
  // Try to emit windowed DotGeneral when one operand is partitioned in the same
  // way as the output along non-contracting dimensions, but the other operand
  // is tiled in other dimensions. Or both operands are partitioned in the same
  // way along contracting dimensions, but the output is partitioned along
  // non-contracting dimensions.
  auto emit_windowed_dot_general =
      [&](int64 matching_operand, int64 windowing_operand,
          bool windowed_at_contracting_dims, bool windowed_at_batch_dims,
          bool operands_sharded_at_contracting_dims)
      -> StatusOr<HloInstruction*> {
    CHECK_EQ(matching_operand + windowing_operand, 1);
    CHECK(!windowed_at_batch_dims || !windowed_at_contracting_dims);
    auto unpadded_result_buffer_shape =
        MakePartitionedShape(output_base_shape, output_sharding);
    auto padded_result_buffer_shape = unpadded_result_buffer_shape;
    // For windowing at batch/non-contracting dims, we produce the result one
    // partition at a time, so we need to pad the shape in case of uneven
    // partitioning in order to make dynamic-update-slice in-bound.
    if (!windowed_at_contracting_dims &&
        !operands_sharded_at_contracting_dims) {
      padded_result_buffer_shape = GetPaddedShapeForUnevenPartitioning(
          padded_result_buffer_shape,
          windowing_operand == 0 ? *lhs_sharding_transposed_to_match_output
                                 : *rhs_sharding_transposed_to_match_output);
    }
    // Mask the padding area of the windowed operand with zero if there is
    // uneven partitioning.
    if (windowed_at_contracting_dims) {
      auto& to_mask = windowing_operand == 0 ? lhs : rhs;
      to_mask =
          to_mask.PadWithValue(b->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(output_base_shape.element_type()))));
    }
    if (operands_sharded_at_contracting_dims) {
      auto zero = b->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(output_base_shape.element_type())));
      lhs = lhs.PadWithValue(zero);
      rhs = rhs.PadWithValue(zero);
    }
    auto result_buffer = CreateZero(padded_result_buffer_shape, b);
    auto extra_result_buffer = CreateZero(padded_result_buffer_shape, b);
    auto iteration = b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32>(0)));

    // Create a while loop that computes one window per iteration. During each
    // iteration, each partition sends its input window to its neighbor using
    // collective-permute for the next iteration.
    SpmdBuilder body_b("windowed_dot_general_body", original_hlo);

    auto get_partial_result =
        [&](HloInstruction* l, HloInstruction* r, HloInstruction* o,
            HloInstruction* i) -> StatusOr<HloInstruction*> {
      auto partition_id =
          lhs.state().collective_ops_creator.create_partition_id(&body_b);
      auto data_partition_id =
          body_b.AddInstruction(HloInstruction::CreateBinary(
              i->shape(), HloOpcode::kAdd, i, partition_id));
      auto partition_count =
          body_b.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<uint32>(num_partitions)));
      data_partition_id = body_b.AddInstruction(
          HloInstruction::CreateBinary(i->shape(), HloOpcode::kRemainder,
                                       data_partition_id, partition_count));
      auto dot_lhs = l;
      auto dot_rhs = r;
      if (windowed_at_contracting_dims || windowed_at_batch_dims ||
          operands_sharded_at_contracting_dims) {
        // Slice the matching operand according to the partitioned dimensions on
        // the windowed operand or the output.
        auto slice_operand = matching_operand == 0 ? l : r;
        // We do this by treating the matching operand as replicated, and
        // resharding it to match the windowed operand or the output.
        slice_operand->set_sharding(HloSharding::Replicate());
        auto state = lhs.state();
        state.b = &body_b;
        state.partition_id = data_partition_id;
        state.reshard_cache->per_hlo_cache.erase(slice_operand);
        const HloSharding* slice_sharding;
        if (operands_sharded_at_contracting_dims) {
          slice_sharding = windowing_operand == 0
                               ? &*output_sharding_transposed_to_match_rhs
                               : &*output_sharding_transposed_to_match_lhs;
        } else {
          slice_sharding = windowing_operand == 0
                               ? &*lhs_sharding_transposed_to_match_rhs
                               : &*rhs_sharding_transposed_to_match_lhs;
        }
        auto slice =
            PartitionedHlo(slice_operand, slice_operand->shape(), state)
                .Reshard(*slice_sharding)
                .hlo();
        slice_operand->clear_sharding();
        if (matching_operand == 0) {
          dot_lhs = slice;
        } else {
          dot_rhs = slice;
        }
      }
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(dot_lhs, dot_rhs, &body_b, conv_window));
      if (windowed_at_contracting_dims ||
          operands_sharded_at_contracting_dims) {
        // Accumulate the partial output to the result buffer.
        o = body_b.AddInstruction(
            HloInstruction::CreateBinary(o->shape(), HloOpcode::kAdd, o, dot));
      } else {
        // The windowing operand is partitioned along batch/non-contracting
        // dimensions, so we need a dynamic-update-slice to save the partial
        // output in the result buffer.
        auto offsets = MakePartitionOffsets(
            o->shape(),
            windowing_operand == 0 ? *lhs_sharding_transposed_to_match_output
                                   : *rhs_sharding_transposed_to_match_output,
            data_partition_id, &body_b);
        o = body_b.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            o->shape(), o, dot, offsets));
      }
      return o;
    };

    auto param = body_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0,
        ShapeUtil::MakeTupleShape(
            {lhs.hlo()->shape(), rhs.hlo()->shape(), result_buffer->shape(),
             extra_result_buffer->shape(), iteration->shape()}),
        "param"));
    auto l = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(lhs.hlo()->shape(), param, 0));
    auto r = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(rhs.hlo()->shape(), param, 1));
    auto o = body_b.AddInstruction(HloInstruction::CreateGetTupleElement(
        result_buffer->shape(), param, 2));
    auto extra_o = body_b.AddInstruction(HloInstruction::CreateGetTupleElement(
        extra_result_buffer->shape(), param, 3));
    auto i = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(iteration->shape(), param, 4));

    if (options.unroll_windowed_einsum && num_partitions % 2 == 0) {
      if (operands_sharded_at_contracting_dims) {
        std::vector<std::pair<int64, int64>> output_sd_pairs(num_partitions);
        for (int64 source = 0; source < num_partitions; ++source) {
          // 0 -> n-2, 1 -> n-1, 2 -> 0, ...
          output_sd_pairs[source] = {
              source, (source - 2 + num_partitions) % num_partitions};
        }

        o = lhs.state()
                .collective_ops_creator
                .create_cross_partition_collective_permute(
                    &body_b, o, output_sd_pairs,
                    (*lhs.state().next_channel_id)++);

        TF_ASSIGN_OR_RETURN(extra_o, get_partial_result(l, r, extra_o, i));

        extra_o = lhs.state()
                      .collective_ops_creator
                      .create_cross_partition_collective_permute(
                          &body_b, extra_o, output_sd_pairs,
                          (*lhs.state().next_channel_id)++);

        // i+2
        i = body_b.AddInstruction(HloInstruction::CreateBinary(
            i->shape(), HloOpcode::kAdd, i,
            body_b.AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<uint32>(2)))));
        auto real_i = body_b.AddInstruction(HloInstruction::CreateBinary(
            i->shape(), HloOpcode::kAdd, i,
            body_b.AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<uint32>(1)))));

        TF_ASSIGN_OR_RETURN(o, get_partial_result(l, r, o, real_i));
        body_b.AddInstruction(
            HloInstruction::CreateTuple({l, r, o, extra_o, i}));
      } else {
        std::vector<std::pair<int64, int64>> sd_pairs(num_partitions);
        for (int64 source = 0; source < num_partitions; ++source) {
          // 0 -> n-1, 1 -> 0, 2 -> 1, ...
          sd_pairs[source] = {source,
                              (source - 1 + num_partitions) % num_partitions};
        }

        // Even number iteration.
        auto next_l = l;
        auto next_r = r;
        auto cp_input = windowing_operand == 0 ? l : r;
        auto cp_output = lhs.state()
                             .collective_ops_creator
                             .create_cross_partition_collective_permute(
                                 &body_b, cp_input, sd_pairs,
                                 (*lhs.state().next_channel_id)++);
        if (windowing_operand == 0) {
          next_l = cp_output;
        } else {
          next_r = cp_output;
        }
        TF_ASSIGN_OR_RETURN(o, get_partial_result(l, r, o, i));

        // ++i
        i = body_b.AddInstruction(HloInstruction::CreateBinary(
            i->shape(), HloOpcode::kAdd, i,
            body_b.AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<uint32>(1)))));

        // Odd number iteration.
        auto second_next_l = next_l;
        auto second_next_r = next_r;
        cp_input = windowing_operand == 0 ? next_l : next_r;
        cp_output = lhs.state()
                        .collective_ops_creator
                        .create_cross_partition_collective_permute(
                            &body_b, cp_input, sd_pairs,
                            (*lhs.state().next_channel_id)++);
        if (windowing_operand == 0) {
          second_next_l = cp_output;
        } else {
          second_next_r = cp_output;
        }
        TF_ASSIGN_OR_RETURN(o, get_partial_result(next_l, next_r, o, i));

        // ++i
        i = body_b.AddInstruction(HloInstruction::CreateBinary(
            i->shape(), HloOpcode::kAdd, i,
            body_b.AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<uint32>(1)))));

        body_b.AddInstruction(HloInstruction::CreateTuple(
            {second_next_l, second_next_r, o, extra_o, i}));
      }
    } else {
      auto real_i = i;
      if (operands_sharded_at_contracting_dims) {
        // For reduce-scatter case, start from the data_partition_id + 1 to make
        // the data_partition_id of the final data shard in each partition the
        // same as the corresponding partition_id.
        real_i = body_b.AddInstruction(HloInstruction::CreateBinary(
            real_i->shape(), HloOpcode::kAdd, real_i,
            CreateOne(real_i->shape(), &body_b)));
      }
      TF_ASSIGN_OR_RETURN(o, get_partial_result(l, r, o, real_i));

      // ++i
      i = body_b.AddInstruction(HloInstruction::CreateBinary(
          i->shape(), HloOpcode::kAdd, i,
          body_b.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<uint32>(1)))));
      auto has_more = body_b.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), i,
          body_b.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<uint32>(num_partitions))),
          ComparisonDirection::kLt));
      // Collective-permute for the next window. We don't need it for the last
      // iteration, so we use a conditional around the collective-permute.
      HloInstruction* conditional;
      {
        SpmdBuilder cp_b("window_collective_permute", original_hlo);
        {
          auto p = cp_b.AddInstruction(HloInstruction::CreateParameter(
              0,
              operands_sharded_at_contracting_dims ? o->shape()
              : windowing_operand == 0             ? l->shape()
                                                   : r->shape(),
              "window"));
          std::vector<std::pair<int64, int64>> sd_pairs(num_partitions);
          for (int64 source = 0; source < num_partitions; ++source) {
            // 0 -> n-1, 1 -> 0, 2 -> 1, ...
            sd_pairs[source] = {source,
                                (source - 1 + num_partitions) % num_partitions};
          }
          lhs.state()
              .collective_ops_creator.create_cross_partition_collective_permute(
                  &cp_b, p, sd_pairs, (*lhs.state().next_channel_id)++);
        }
        SpmdBuilder ncp_b("last_iteration_noop", original_hlo);
        {
          ncp_b.AddInstruction(HloInstruction::CreateParameter(
              0,
              operands_sharded_at_contracting_dims ? o->shape()
              : windowing_operand == 0             ? l->shape()
                                                   : r->shape(),
              "window"));
        }
        conditional = body_b.AddInstruction(HloInstruction::CreateConditional(
            operands_sharded_at_contracting_dims ? o->shape()
            : windowing_operand == 0             ? l->shape()
                                                 : r->shape(),
            has_more,
            operands_sharded_at_contracting_dims ? o
            : windowing_operand == 0             ? l
                                                 : r,
            module->AddEmbeddedComputation(cp_b.Build()),
            operands_sharded_at_contracting_dims ? o
            : windowing_operand == 0             ? l
                                                 : r,
            module->AddEmbeddedComputation(ncp_b.Build())));
      }
      if (operands_sharded_at_contracting_dims) {
        o = conditional;
      } else if (windowing_operand == 0) {
        l = conditional;
      } else {
        r = conditional;
      }
      body_b.AddInstruction(HloInstruction::CreateTuple({l, r, o, extra_o, i}));
    }

    SpmdBuilder cond_b("windowed_dot_general_cond", original_hlo);
    auto cond_param = cond_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0,
        ShapeUtil::MakeTupleShape(
            {lhs.hlo()->shape(), rhs.hlo()->shape(), result_buffer->shape(),
             extra_result_buffer->shape(), iteration->shape()}),
        "param"));
    auto cond_i = cond_b.AddInstruction(HloInstruction::CreateGetTupleElement(
        iteration->shape(), cond_param, 4));
    cond_b.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), cond_i,
        cond_b.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<uint32>(num_partitions))),
        ComparisonDirection::kLt));
    auto while_loop = b->AddInstruction(HloInstruction::CreateWhile(
        cond_param->shape(), module->AddEmbeddedComputation(cond_b.Build()),
        module->AddEmbeddedComputation(body_b.Build()),
        b->AddInstruction(
            HloInstruction::CreateTuple({lhs.hlo(), rhs.hlo(), result_buffer,
                                         extra_result_buffer, iteration}))));
    windowed_dot_general_loops->push_back(
        {while_loop, windowing_operand, windowed_at_contracting_dims,
         windowed_at_batch_dims, operands_sharded_at_contracting_dims});
    auto result = b->AddInstruction(HloInstruction::CreateGetTupleElement(
        result_buffer->shape(), while_loop, 2));
    if (options.unroll_windowed_einsum && num_partitions % 2 == 0 &&
        operands_sharded_at_contracting_dims) {
      std::vector<std::pair<int64, int64>> extra_sd_pairs(num_partitions);
      for (int64 source = 0; source < num_partitions; ++source) {
        // 0 -> 1, 1 -> 2, 2 -> 3, ...
        extra_sd_pairs[source] = {source, (source + 1) % num_partitions};
      }
      result =
          lhs.state()
              .collective_ops_creator.create_cross_partition_collective_permute(
                  b, result, extra_sd_pairs, (*lhs.state().next_channel_id)++);
      auto extra_result =
          b->AddInstruction(HloInstruction::CreateGetTupleElement(
              extra_result_buffer->shape(), while_loop, 3));
      result = b->AddInstruction(HloInstruction::CreateBinary(
          result->shape(), HloOpcode::kAdd, result, extra_result));
    }
    if (!ShapeUtil::Compatible(padded_result_buffer_shape,
                               unpadded_result_buffer_shape)) {
      result = b->AddInstruction(HloInstruction::CreateSlice(
          unpadded_result_buffer_shape, result,
          std::vector<int64>(padded_result_buffer_shape.rank(), 0),
          unpadded_result_buffer_shape.dimensions(),
          std::vector<int64>(padded_result_buffer_shape.rank(), 1)));
    }
    return result;
  };
  if (output_lhs_non_contracting_partitions == num_partitions &&
      output_sharding_transposed_to_match_lhs == lhs_sharding &&
      ShapeSizeInBytes(rhs.base_shape()) >=
          options.threshold_for_windowed_einsum_mib * 1024 * 1024) {
    if (rhs_contracting_partitions == num_partitions) {
      return emit_windowed_dot_general(0, 1, true, false, false);
    }
    if (rhs_non_contracting_partitions == num_partitions) {
      return emit_windowed_dot_general(0, 1, false, false, false);
    }
    if (rhs_batch_partitions == num_partitions) {
      return emit_windowed_dot_general(0, 1, false, true, false);
    }
  }
  if (output_rhs_non_contracting_partitions == num_partitions &&
      output_sharding_transposed_to_match_rhs == rhs_sharding &&
      ShapeSizeInBytes(lhs.base_shape()) >=
          options.threshold_for_windowed_einsum_mib * 1024 * 1024) {
    if (lhs_contracting_partitions == num_partitions) {
      return emit_windowed_dot_general(1, 0, true, false, false);
    }
    if (lhs_non_contracting_partitions == num_partitions) {
      return emit_windowed_dot_general(1, 0, false, false, false);
    }
    if (lhs_batch_partitions == num_partitions) {
      return emit_windowed_dot_general(1, 0, false, true, false);
    }
  }
  if (lhs_contracting_partitions == rhs_contracting_partitions &&
      lhs_contracting_partitions == num_partitions &&
      output_sharding_dim > -1 &&
      ShapeSizeInBytes(output_base_shape) >=
          options.threshold_for_windowed_einsum_mib * 1024 * 1024) {
    if (output_lhs_non_contracting_partitions == num_partitions) {
      return emit_windowed_dot_general(0, 1, false, false, true);
    }
    if (output_rhs_non_contracting_partitions == num_partitions) {
      return emit_windowed_dot_general(1, 0, false, false, true);
    }
  }

  {
    // Try batch-parallel by resharding one operand, and allowing all-reduce.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * partitioned_dot,
        try_emit_output_batch_partitioned_einsum_with_reshard(true));
    if (partitioned_dot) {
      return partitioned_dot;
    }
  }

  // LHS and RHS have the same partitioned contracting dimensions.
  if (lhs_contracting_partitions == rhs_contracting_partitions &&
      lhs_contracting_partitions == num_partitions) {
    auto zero = b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(output_base_shape.element_type())));
    // Pad both sides with zero, since NaN at one side cannot be masked by zero
    // on the other side.
    if (ShapeSizeInBytes(lhs.base_shape()) <
        ShapeSizeInBytes(rhs.base_shape())) {
      lhs =
          lhs.Reshard(*rhs_sharding_transposed_to_match_lhs).PadWithValue(zero);
      rhs = rhs.PadWithValue(zero);
    } else {
      lhs = lhs.PadWithValue(zero);
      rhs =
          rhs.Reshard(*lhs_sharding_transposed_to_match_rhs).PadWithValue(zero);
    }
    TF_ASSIGN_OR_RETURN(
        auto dot, create_sharded_dot(lhs.hlo(), rhs.hlo(), b, conv_window));
    auto ar =
        lhs.state().collective_ops_creator.create_cross_partition_all_reduce(
            b, dot, MakeBinaryAdd(output_base_shape.element_type(), module),
            {GetAllDevicesInOrder(lhs.sharding())},
            (*lhs.state().next_channel_id)++);
    ar->set_sharding(HloSharding::Replicate());
    return PartitionedHlo(ar, output_base_shape, lhs.state())
        .Reshard(output_sharding)
        .hlo();
  }

  // LHS and output have the same partitioned non-contracting dimensions.
  if (lhs_non_contracting_partitions == num_partitions &&
      output_lhs_non_contracting_partitions == num_partitions &&
      lhs_sharding_transposed_to_match_output == output_sharding) {
    auto rhs_replicated = rhs.Reshard(HloSharding::Replicate()).hlo();
    TF_ASSIGN_OR_RETURN(auto dot, create_sharded_dot(lhs.hlo(), rhs_replicated,
                                                     b, conv_window));
    return dot;
  }

  // RHS and output have the same partitioned non-contracting dimensions.
  if (rhs_non_contracting_partitions == num_partitions &&
      output_rhs_non_contracting_partitions == num_partitions &&
      rhs_sharding_transposed_to_match_output == output_sharding) {
    auto lhs_replicated = lhs.Reshard(HloSharding::Replicate()).hlo();
    TF_ASSIGN_OR_RETURN(auto dot, create_sharded_dot(lhs_replicated, rhs.hlo(),
                                                     b, conv_window));
    return dot;
  }

  if (may_reshard_without_detecting_match) {
    // Output is batch partitioned.
    if (output_batch_partitions == num_partitions) {
      auto resharded_lhs =
          lhs.Reshard(*output_sharding_transposed_to_match_lhs);
      auto resharded_rhs =
          rhs.Reshard(*output_sharding_transposed_to_match_rhs);
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(resharded_lhs.hlo(), resharded_rhs.hlo(),
                                       b, conv_window));
      return dot;
    }
    // Output is partitioned along LHS non-contracting dimensions.
    if (output_lhs_non_contracting_partitions == num_partitions) {
      auto resharded_lhs =
          lhs.Reshard(*output_sharding_transposed_to_match_lhs);
      auto replicated_rhs = rhs.Reshard(HloSharding::Replicate());
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(resharded_lhs.hlo(),
                                       replicated_rhs.hlo(), b, conv_window));
      return dot;
    }
    // Output is partitioned along RHS non-contracting dimensions.
    if (output_rhs_non_contracting_partitions == num_partitions) {
      auto replicated_lhs = lhs.Reshard(HloSharding::Replicate());
      auto resharded_rhs =
          rhs.Reshard(*output_sharding_transposed_to_match_rhs);
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(replicated_lhs.hlo(),
                                       resharded_rhs.hlo(), b, conv_window));
      return dot;
    }
  }

  // Returns true if it is beneficial to reshard the operand at `operand_idx`
  // across the contracting dimension.
  const auto should_partition_contracting_dim = [&](int64 operand_idx) {
    if (!output_sharding.IsReplicated()) {
      return false;
    }

    if (operand_idx == 0) {
      // If LHS and output are replicated, we compare the cost of all-gather
      // on RHS vs all-reduce on the output.
      return (rhs_contracting_partitions == num_partitions) &&
             lhs.sharding().IsReplicated() &&
             ShapeUtil::ElementsIn(rhs.base_shape()) >
                 ShapeUtil::ElementsIn(output_base_shape);
    } else {
      return (lhs_contracting_partitions == num_partitions) &&
             rhs.sharding().IsReplicated() &&
             ShapeUtil::ElementsIn(lhs.base_shape()) >
                 ShapeUtil::ElementsIn(output_base_shape);
    }
  };

  // When the output is replicated and one of the operands is partitioned along
  // contracting dimension, align the other operand to be partitioned along
  // the contracting dimensions.
  if (output_sharding.IsReplicated() && (should_partition_contracting_dim(0) ||
                                         should_partition_contracting_dim(1))) {
    auto zero = b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(output_base_shape.element_type())));
    if (should_partition_contracting_dim(0)) {
      lhs =
          lhs.Reshard(*rhs_sharding_transposed_to_match_lhs).PadWithValue(zero);
      rhs = rhs.PadWithValue(zero);
    } else {
      lhs = lhs.PadWithValue(zero);
      rhs =
          rhs.Reshard(*lhs_sharding_transposed_to_match_rhs).PadWithValue(zero);
    }
    TF_ASSIGN_OR_RETURN(
        auto dot, create_sharded_dot(lhs.hlo(), rhs.hlo(), b, conv_window));
    return lhs.state().collective_ops_creator.create_cross_partition_all_reduce(
        b, dot, MakeBinaryAdd(output_base_shape.element_type(), module),
        {GetAllDevicesInOrder(lhs.sharding())},
        (*lhs.state().next_channel_id)++);
  }
  return nullptr;
}

StatusOr<HloInstruction*> PartitionDot(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64 num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops);

StatusOr<HloInstruction*> PartitionDotGroupOnBatch(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64 num_partitions, int64 lhs_contracting_partitions,
    int64 rhs_contracting_partitions, int64 lhs_non_contracting_partitions,
    int64 rhs_non_contracting_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    bool require_matching_devices_to_group,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops) {
  std::vector<std::pair<HloInstruction*, HloSharding>>
      top_level_sharding_to_reset;
  auto cleaner = tensorflow::gtl::MakeCleanup([&] {
    for (auto& to_reset : top_level_sharding_to_reset) {
      to_reset.first->set_sharding(to_reset.second);
    }
  });
  std::vector<int64> lhs_dims;
  std::vector<int64> rhs_dims;
  std::vector<int64> output_dims;
  auto lhs_sharding_dims_adjusted_to_output =
      lhs.sharding().IsReplicated()
          ? std::vector<int64>(lhs.base_shape().rank(), 1)
          : lhs.sharding().tile_assignment().dimensions();
  auto rhs_sharding_dims_adjusted_to_output =
      rhs.sharding().IsReplicated()
          ? std::vector<int64>(rhs.base_shape().rank(), 1)
          : rhs.sharding().tile_assignment().dimensions();
  auto output_sharding_dims_adjusted_to_lhs =
      output_sharding.tile_assignment().dimensions();
  bool lhs_rhs_dims_matching = true;
  for (const auto& dim : dims_mapping.batch_dims) {
    lhs_dims.push_back(dim.lhs);
    rhs_dims.push_back(dim.rhs);
    output_dims.push_back(dim.output);
    if (lhs_sharding_dims_adjusted_to_output[dim.lhs] !=
        rhs_sharding_dims_adjusted_to_output[dim.rhs]) {
      lhs_rhs_dims_matching = false;
    }
    lhs_sharding_dims_adjusted_to_output[dim.lhs] =
        output_sharding.tile_assignment().dim(dim.output);
    rhs_sharding_dims_adjusted_to_output[dim.rhs] =
        output_sharding.tile_assignment().dim(dim.output);
    output_sharding_dims_adjusted_to_lhs[dim.output] =
        lhs.sharding().tile_assignment().dim(dim.lhs);
  }
  if (require_matching_devices_to_group && lhs_rhs_dims_matching) {
    lhs_rhs_dims_matching =
        rhs.sharding() == UngroupSharding(AlignGroupsWith(
                              GroupShardingOnDims(rhs.sharding(), rhs_dims),
                              GroupShardingOnDims(lhs.sharding(), lhs_dims)));
  }
  auto output_grouped = GroupShardingOnDims(output_sharding, output_dims);
  PartitionedHlo per_group_lhs = lhs;
  PartitionedHlo per_group_rhs = rhs;
  if (lhs_rhs_dims_matching) {
    auto lhs_grouped = GroupShardingOnDims(lhs.sharding(), lhs_dims);
    auto rhs_grouped = GroupShardingOnDims(rhs.sharding(), rhs_dims);
    if (ShapeUtil::ByteSizeOf(lhs.hlo()->shape()) >
        ShapeUtil::ByteSizeOf(rhs.hlo()->shape())) {
      rhs_grouped = AlignGroupsWith(std::move(rhs_grouped), lhs_grouped);
      rhs = rhs.Reshard(UngroupSharding(rhs_grouped));
    } else {
      lhs_grouped = AlignGroupsWith(std::move(lhs_grouped), rhs_grouped);
      lhs = lhs.Reshard(UngroupSharding(lhs_grouped));
    }
    auto reshaped_output_tiling = output_sharding.tile_assignment();
    reshaped_output_tiling.Reshape(output_sharding_dims_adjusted_to_lhs);
    output_grouped = AlignGroupsWith(
        GroupShardingOnDims(
            output_sharding.ReplicateOnLastTileDim()
                ? HloSharding::PartialTile(reshaped_output_tiling)
                : HloSharding::Tile(reshaped_output_tiling),
            output_dims),
        lhs_grouped);
    auto per_group_partitioner_state = CreatePerGroupPartitioningState(
        lhs.state(), lhs_grouped.device_groups, b);
    top_level_sharding_to_reset.emplace_back(lhs.hlo(), lhs.sharding());
    lhs.hlo()->set_sharding(lhs_grouped.sharding);
    top_level_sharding_to_reset.emplace_back(rhs.hlo(), rhs.sharding());
    rhs.hlo()->set_sharding(rhs_grouped.sharding);
    CHECK(lhs.hlo() != rhs.hlo() ||
          lhs_grouped.sharding == rhs_grouped.sharding);
    per_group_lhs = PartitionedHlo(
        lhs.hlo(), GetPerGroupBaseShape(lhs_grouped, lhs.base_shape()),
        per_group_partitioner_state);
    per_group_rhs = PartitionedHlo(
        rhs.hlo(), GetPerGroupBaseShape(rhs_grouped, rhs.base_shape()),
        per_group_partitioner_state);
  } else {
    auto per_group_partitioner_state = CreatePerGroupPartitioningState(
        lhs.state(), output_grouped.device_groups, b);
    auto reshard_to_output_batch =
        [&](PartitionedHlo operand, absl::Span<const int64> batch_dims,
            absl::Span<const int64> contracting_dims,
            absl::Span<const int64> non_contracting_dims,
            int64 contracting_dim_partitions,
            int64 non_contracting_dim_partitions,
            int64 other_contracting_dim_partitions,
            std::vector<int64>* sharding_dims_adjusted_to_output)
        -> absl::optional<PartitionedHlo> {
      if (operand.sharding().IsTileMaximal()) {
        auto partially_sharded = PerGroupSliceFromReplicated(
            operand.Replicate().hlo(), operand.state().partition_id,
            output_grouped.device_groups, batch_dims,
            output_grouped.group_dim_sizes, b);
        partially_sharded->set_sharding(HloSharding::Replicate());
        return PartitionedHlo(partially_sharded, partially_sharded->shape(),
                              per_group_partitioner_state);
      }
      auto reshaped_tiling = operand.sharding().tile_assignment();
      // It's possible that the operand is not initially sharded on batch
      // dimensions in the same way as the output, although being tiled. In that
      // case, the current sharding_dims_adjusted_to_output may contain more
      // partitions than available devices. We remove partitioning on other
      // dimensions.
      if (Product(*sharding_dims_adjusted_to_output) >
          reshaped_tiling.num_elements()) {
        if (Product(*sharding_dims_adjusted_to_output) %
                reshaped_tiling.num_elements() !=
            0) {
          return absl::nullopt;
        }
        int64 ratio = Product(*sharding_dims_adjusted_to_output) /
                      reshaped_tiling.num_elements();
        if (operand.sharding().ReplicateOnLastTileDim() &&
            reshaped_tiling.dimensions().back() % ratio == 0) {
          sharding_dims_adjusted_to_output->back() /= ratio;
          if (sharding_dims_adjusted_to_output->back() == 1) {
            sharding_dims_adjusted_to_output->pop_back();
          }
        } else if (ratio == non_contracting_dim_partitions &&
                   (ratio != contracting_dim_partitions ||
                    contracting_dim_partitions ==
                        other_contracting_dim_partitions)) {
          for (int64 dim : non_contracting_dims) {
            (*sharding_dims_adjusted_to_output)[dim] = 1;
          }
        } else if (ratio == contracting_dim_partitions) {
          for (int64 dim : contracting_dims) {
            (*sharding_dims_adjusted_to_output)[dim] = 1;
          }
        } else {
          return absl::nullopt;
        }
      }
      // If the operand is initially sharded more ways than the output in the
      // batch dimensions, sharding_dims_adjusted_to_output currently contains
      // fewer partitions than available devices. We do not handle this case.
      if (Product(*sharding_dims_adjusted_to_output) <
          reshaped_tiling.num_elements()) {
        return absl::nullopt;
      }
      reshaped_tiling.Reshape(*sharding_dims_adjusted_to_output);
      auto grouped = AlignGroupsWith(
          GroupShardingOnDims(operand.base_shape().rank() <
                                      sharding_dims_adjusted_to_output->size()
                                  ? HloSharding::PartialTile(reshaped_tiling)
                                  : HloSharding::Tile(reshaped_tiling),
                              batch_dims),
          output_grouped);
      if (require_matching_devices_to_group &&
          operand.sharding() != UngroupSharding(grouped)) {
        return absl::nullopt;
      }
      auto resharded = operand.Reshard(UngroupSharding(grouped));
      top_level_sharding_to_reset.emplace_back(resharded.hlo(),
                                               resharded.sharding());
      resharded.hlo()->set_sharding(grouped.sharding);
      return PartitionedHlo(resharded.hlo(),
                            GetPerGroupBaseShape(grouped, operand.base_shape()),
                            per_group_partitioner_state);
    };
    std::vector<int64> lhs_contracting_dims;
    std::vector<int64> rhs_contracting_dims;
    lhs_contracting_dims.reserve(dims_mapping.contracting_dims.size());
    rhs_contracting_dims.reserve(dims_mapping.contracting_dims.size());
    for (const auto& dim : dims_mapping.contracting_dims) {
      lhs_contracting_dims.push_back(dim.lhs);
      rhs_contracting_dims.push_back(dim.rhs);
    }
    std::vector<int64> lhs_non_contracting_dims;
    std::vector<int64> rhs_non_contracting_dims;
    lhs_non_contracting_dims.reserve(
        dims_mapping.lhs_non_contracting_dims.size());
    rhs_non_contracting_dims.reserve(
        dims_mapping.rhs_non_contracting_dims.size());
    for (const auto& dim : dims_mapping.lhs_non_contracting_dims) {
      lhs_non_contracting_dims.push_back(dim.lhs);
    }
    for (const auto& dim : dims_mapping.rhs_non_contracting_dims) {
      rhs_non_contracting_dims.push_back(dim.rhs);
    }
    if (auto resharded = reshard_to_output_batch(
            lhs, lhs_dims, lhs_contracting_dims, lhs_non_contracting_dims,
            lhs_contracting_partitions, lhs_non_contracting_partitions,
            rhs_contracting_partitions,
            &lhs_sharding_dims_adjusted_to_output)) {
      per_group_lhs = *resharded;
    } else {
      return nullptr;
    }
    if (auto resharded = reshard_to_output_batch(
            rhs, rhs_dims, rhs_contracting_dims, rhs_non_contracting_dims,
            rhs_contracting_partitions, rhs_non_contracting_partitions,
            lhs_contracting_partitions,
            &rhs_sharding_dims_adjusted_to_output)) {
      per_group_rhs = *resharded;
    } else {
      return nullptr;
    }
    CHECK(lhs.hlo() != rhs.hlo() ||
          per_group_lhs.sharding() == per_group_rhs.sharding());
  }
  TF_ASSIGN_OR_RETURN(
      auto dot,
      PartitionDot(per_group_lhs, per_group_rhs,
                   GetPerGroupBaseShape(output_grouped, output_base_shape),
                   output_grouped.sharding, dims_mapping,
                   num_partitions / output_grouped.device_groups.size(),
                   create_sharded_dot, conv_window, module, original_hlo,
                   options, b, windowed_dot_general_loops));
  dot->set_sharding(UngroupSharding(output_grouped));
  return PartitionedHlo(dot, output_base_shape, lhs.state())
      .Reshard(output_sharding)
      .hlo();
}

StatusOr<HloInstruction*> PartitionDotGroupOnNonContracting(
    bool lhs_matching, PartitionedHlo matching, PartitionedHlo other,
    int64 matching_contracting_partitions, int64 other_contracting_partitions,
    absl::Span<const DotConvDimsMapping::DimsMapping>
        partitioned_non_contractin_dims,
    int64 other_non_contracting_partitions,
    int64 output_other_non_contracting_partitions,
    const Shape& output_base_shape, const HloSharding& output_sharding,
    const DotConvDimsMapping& dims_mapping, int64 num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    bool require_matching_devices_to_group,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops) {
  std::vector<std::pair<HloInstruction*, HloSharding>>
      top_level_sharding_to_reset;
  auto cleaner = tensorflow::gtl::MakeCleanup([&] {
    for (auto& to_reset : top_level_sharding_to_reset) {
      to_reset.first->set_sharding(to_reset.second);
    }
  });

  auto matching_sharding_dims =
      matching.sharding().tile_assignment().dimensions();
  std::vector<int64> matching_dims;
  std::vector<int64> output_dims;
  int64 group_count = 1;
  // Make sure the partitioning on matching's non-contracting dimensions
  // defines the same device groups for both matching and output.
  for (const auto& dim : partitioned_non_contractin_dims) {
    int64 md = lhs_matching ? dim.lhs : dim.rhs;
    matching_sharding_dims[md] =
        output_sharding.tile_assignment().dim(dim.output);
    matching_dims.push_back(md);
    output_dims.push_back(dim.output);
    group_count *= output_sharding.tile_assignment().dim(dim.output);
  }
  auto output_grouped = GroupShardingOnDims(output_sharding, output_dims);
  auto reshaped_matching_tiling = matching.sharding().tile_assignment();
  reshaped_matching_tiling.Reshape(matching_sharding_dims);
  auto matching_grouped = AlignGroupsWith(
      GroupShardingOnDims(
          matching.sharding().ReplicateOnLastTileDim()
              ? HloSharding::PartialTile(reshaped_matching_tiling)
              : HloSharding::Tile(reshaped_matching_tiling),
          matching_dims),
      output_grouped);
  if (require_matching_devices_to_group &&
      matching.sharding() != UngroupSharding(matching_grouped)) {
    return nullptr;
  }

  std::vector<int64> other_group_dims;
  if (other.sharding().ReplicateOnLastTileDim() &&
      other.sharding().tile_assignment().dimensions().back() % group_count ==
          0) {
    other_group_dims.push_back(other.base_shape().rank());
  } else {
    const bool may_replicate_other_contracting_dims =
        (other_contracting_partitions == group_count &&
         other_non_contracting_partitions ==
             output_other_non_contracting_partitions);
    const bool may_replicate_other_non_contracting_dims =
        group_count == other_non_contracting_partitions &&
        matching_contracting_partitions == other_contracting_partitions;
    if (auto found_dims = FindMatchingPartitionedDimsForGrouping(
            other.sharding(), output_grouped.device_groups)) {
      other_group_dims = std::move(*found_dims);
    } else if (may_replicate_other_contracting_dims &&
               (!may_replicate_other_non_contracting_dims ||
                ShapeUtil::ByteSizeOf(other.hlo()->shape()) <=
                    ShapeUtil::ByteSizeOf(MakePartitionedShape(
                        output_base_shape, output_sharding)))) {
      for (const auto& dim : dims_mapping.contracting_dims) {
        other_group_dims.push_back(lhs_matching ? dim.rhs : dim.lhs);
      }
    } else if (may_replicate_other_non_contracting_dims) {
      for (const auto& dim : lhs_matching
                                 ? dims_mapping.rhs_non_contracting_dims
                                 : dims_mapping.lhs_non_contracting_dims) {
        other_group_dims.push_back(lhs_matching ? dim.rhs : dim.lhs);
      }
    } else {
      other = other.Replicate();
    }
  }

  matching = matching.Reshard(UngroupSharding(matching_grouped));
  auto per_group_partitioner_state = CreatePerGroupPartitioningState(
      matching.state(), matching_grouped.device_groups, b);
  top_level_sharding_to_reset.emplace_back(matching.hlo(), matching.sharding());
  matching.hlo()->set_sharding(matching_grouped.sharding);
  auto matching_p = PartitionedHlo(
      matching.hlo(),
      GetPerGroupBaseShape(matching_grouped, matching.base_shape()),
      per_group_partitioner_state);

  auto partially_replicated_other = other.hlo();
  if (other_group_dims.size() == 1 &&
      other_group_dims[0] == other.base_shape().rank()) {
    // Group on replication dim.
    auto grouped = AlignGroupsWith(
        GroupShardingOnDims(
            other.sharding(), {other_group_dims[0]},
            {other.sharding().tile_assignment().dimensions().back() /
             group_count}),
        output_grouped, /*ignore_group_order=*/true);
    other = other.Reshard(UngroupSharding(grouped));
    partially_replicated_other = other.hlo();
    top_level_sharding_to_reset.emplace_back(other.hlo(), other.sharding());
    partially_replicated_other->set_sharding(grouped.sharding);
  } else if (!other.sharding().IsReplicated()) {
    auto other_grouped =
        AlignGroupsWith(GroupShardingOnDims(other.sharding(), other_group_dims),
                        output_grouped, /*ignore_group_order=*/true);
    other = other.Reshard(UngroupSharding(other_grouped));
    partially_replicated_other =
        other
            .Reshard(hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
                other.sharding(), other_grouped.group_dims))
            .hlo();
    top_level_sharding_to_reset.emplace_back(
        partially_replicated_other, partially_replicated_other->sharding());
    partially_replicated_other->set_sharding(other_grouped.sharding);
  }
  auto other_p = PartitionedHlo(partially_replicated_other, other.base_shape(),
                                per_group_partitioner_state);
  TF_ASSIGN_OR_RETURN(
      auto dot,
      PartitionDot(lhs_matching ? matching_p : other_p,
                   lhs_matching ? other_p : matching_p,
                   GetPerGroupBaseShape(output_grouped, output_base_shape),
                   output_grouped.sharding, dims_mapping,
                   num_partitions / matching_grouped.device_groups.size(),
                   create_sharded_dot, conv_window, module, original_hlo,
                   options, b, windowed_dot_general_loops));
  return dot;
}

StatusOr<HloInstruction*> PartitionDotGroupOnContracting(
    PartitionedHlo lhs, PartitionedHlo rhs,
    absl::Span<const DotConvDimsMapping::DimsMapping>
        partitioned_contractin_dims,
    int64 output_batch_partitions, int64 output_lhs_non_contracting_partitions,
    int64 output_rhs_non_contracting_partitions, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64 num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    bool require_matching_devices_to_group,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops) {
  std::vector<std::pair<HloInstruction*, HloSharding>>
      top_level_sharding_to_reset;
  auto cleaner = tensorflow::gtl::MakeCleanup([&] {
    for (auto& to_reset : top_level_sharding_to_reset) {
      to_reset.first->set_sharding(to_reset.second);
    }
  });
  auto lhs_sharding = lhs.sharding();
  auto rhs_sharding = rhs.sharding();
  auto lhs_tile_shape = lhs_sharding.tile_assignment().dimensions();
  auto rhs_tile_shape = rhs_sharding.tile_assignment().dimensions();
  std::vector<int64> lhs_dims;
  std::vector<int64> rhs_dims;
  int64 group_count = 1;
  for (const auto& dim : partitioned_contractin_dims) {
    lhs_dims.push_back(dim.lhs);
    rhs_dims.push_back(dim.rhs);
    group_count *= lhs_sharding.tile_assignment().dim(dim.lhs);
  }
  if (ShapeUtil::ByteSizeOf(lhs.hlo()->shape()) >
      ShapeUtil::ByteSizeOf(rhs.hlo()->shape())) {
    for (const auto& dim : partitioned_contractin_dims) {
      rhs_tile_shape[dim.rhs] = lhs_tile_shape[dim.lhs];
    }
    auto new_tile = rhs.sharding().tile_assignment();
    new_tile.Reshape(rhs_tile_shape);
    rhs_sharding = rhs_sharding.ReplicateOnLastTileDim()
                       ? HloSharding::PartialTile(new_tile)
                       : HloSharding::Tile(new_tile);
  } else {
    for (const auto& dim : partitioned_contractin_dims) {
      lhs_tile_shape[dim.lhs] = rhs_tile_shape[dim.rhs];
    }
    auto new_tile = lhs.sharding().tile_assignment();
    new_tile.Reshape(lhs_tile_shape);
    lhs_sharding = lhs_sharding.ReplicateOnLastTileDim()
                       ? HloSharding::PartialTile(new_tile)
                       : HloSharding::Tile(new_tile);
  }
  auto lhs_grouped = GroupShardingOnDims(lhs_sharding, lhs_dims);
  auto rhs_grouped = GroupShardingOnDims(rhs_sharding, rhs_dims);
  if (ShapeUtil::ByteSizeOf(lhs.hlo()->shape()) >
      ShapeUtil::ByteSizeOf(rhs.hlo()->shape())) {
    rhs_grouped = AlignGroupsWith(rhs_grouped, lhs_grouped);
    rhs_sharding = UngroupSharding(rhs_grouped);
    if (require_matching_devices_to_group && rhs.sharding() != rhs_sharding) {
      return nullptr;
    }
    rhs = rhs.Reshard(rhs_sharding);
  } else {
    lhs_grouped = AlignGroupsWith(lhs_grouped, rhs_grouped);
    lhs_sharding = UngroupSharding(lhs_grouped);
    if (require_matching_devices_to_group && lhs.sharding() != lhs_sharding) {
      return nullptr;
    }
    lhs = lhs.Reshard(lhs_sharding);
  }
  // Mask out invalid data.
  std::vector<int64> lhs_skipped_dims;
  for (int64 i = 0; i < lhs.base_shape().rank(); ++i) {
    if (absl::c_linear_search(lhs_dims, i)) {
      continue;
    }
    lhs_skipped_dims.push_back(i);
  }
  lhs = lhs.PadWithValue(
      CreateZero(ShapeUtil::MakeShape(lhs.base_shape().element_type(), {}), b),
      /*left_padded_dims=*/{}, lhs_skipped_dims);
  std::vector<int64> rhs_skipped_dims;
  for (int64 i = 0; i < rhs.base_shape().rank(); ++i) {
    if (absl::c_linear_search(rhs_dims, i)) {
      continue;
    }
    rhs_skipped_dims.push_back(i);
  }
  rhs = rhs.PadWithValue(
      CreateZero(ShapeUtil::MakeShape(rhs.base_shape().element_type(), {}), b),
      /*left_padded_dims=*/{}, rhs_skipped_dims);
  top_level_sharding_to_reset.emplace_back(lhs.hlo(), lhs_sharding);
  lhs.hlo()->set_sharding(lhs_grouped.sharding);
  top_level_sharding_to_reset.emplace_back(rhs.hlo(), rhs_sharding);
  rhs.hlo()->set_sharding(rhs_grouped.sharding);

  HloSharding inner_output_sharding = HloSharding::Replicate();
  HloSharding outer_output_tmp_sharding = HloSharding::Replicate();
  if (output_sharding.ReplicateOnLastTileDim() &&
      output_sharding.tile_assignment().dimensions().back() % group_count ==
          0) {
    auto grouped = AlignGroupsWith(
        GroupShardingOnDims(
            output_sharding,
            {output_sharding.tile_assignment().num_dimensions() - 1},
            {output_sharding.tile_assignment().dimensions().back() /
             group_count}),
        lhs_grouped,
        /*ignore_group_order=*/true);
    outer_output_tmp_sharding = UngroupSharding(grouped);
    inner_output_sharding = std::move(grouped.sharding);
  } else {
    std::vector<int64> group_dims;
    if (auto found_dims = FindMatchingPartitionedDimsForGrouping(
            output_sharding, lhs_grouped.device_groups)) {
      group_dims = std::move(*found_dims);
    } else if (output_lhs_non_contracting_partitions == group_count ||
               output_rhs_non_contracting_partitions == group_count ||
               output_batch_partitions == group_count) {
      if (output_lhs_non_contracting_partitions == group_count) {
        for (const auto& dim : dims_mapping.lhs_non_contracting_dims) {
          group_dims.push_back(dim.output);
        }
      } else if (output_rhs_non_contracting_partitions == group_count) {
        for (const auto& dim : dims_mapping.rhs_non_contracting_dims) {
          group_dims.push_back(dim.output);
        }
      } else {
        for (const auto& dim : dims_mapping.batch_dims) {
          group_dims.push_back(dim.output);
        }
      }
    }
    if (!group_dims.empty()) {
      auto grouped = AlignGroupsWith(
          GroupShardingOnDims(output_sharding, group_dims), lhs_grouped);
      inner_output_sharding = grouped.sharding;
      outer_output_tmp_sharding =
          hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
              UngroupSharding(grouped), group_dims);
    }
  }
  auto inner_state = CreatePerGroupPartitioningState(
      lhs.state(), lhs_grouped.device_groups, b);
  TF_ASSIGN_OR_RETURN(
      auto dot,
      PartitionDot(
          PartitionedHlo(lhs.hlo(),
                         GetPerGroupBaseShape(lhs_grouped, lhs.base_shape()),
                         inner_state),
          PartitionedHlo(rhs.hlo(),
                         GetPerGroupBaseShape(rhs_grouped, rhs.base_shape()),
                         inner_state),
          output_base_shape, inner_output_sharding, dims_mapping,
          num_partitions / group_count, create_sharded_dot, conv_window, module,
          original_hlo, options, b, windowed_dot_general_loops));
  if (!dot) {
    return nullptr;
  }
  std::vector<int64> other_lhs_dims;
  for (int64 i = 0; i < lhs_sharding.tile_assignment().num_dimensions(); ++i) {
    if (!absl::c_linear_search(lhs_dims, i)) {
      other_lhs_dims.push_back(i);
    }
  }
  auto inverse_grouped = GroupShardingOnDims(lhs_sharding, other_lhs_dims);
  auto ar =
      CreatePerGroupPartitioningState(lhs.state(),
                                      inverse_grouped.device_groups, b)
          .collective_ops_creator.create_cross_partition_all_reduce(
              b, dot, MakeBinaryAdd(output_base_shape.element_type(), module),
              {GetAllDevicesInOrder(inverse_grouped.sharding)},
              (*lhs.state().next_channel_id)++);
  ar->set_sharding(outer_output_tmp_sharding);
  return PartitionedHlo(ar, output_base_shape, lhs.state())
      .Reshard(output_sharding)
      .hlo();
}

DotConvDimsMapping ConvertDimsMappingWithFeatureGroupCount(
    const DotConvDimsMapping& dims_mapping, HloInstruction* original_hlo) {
  const auto& dnums = original_hlo->convolution_dimension_numbers();
  DotConvDimsMapping new_dims_mapping;
  new_dims_mapping.batch_dims = dims_mapping.batch_dims;
  new_dims_mapping.conv_spatial_dims = dims_mapping.conv_spatial_dims;
  // Append batch dims.
  new_dims_mapping.batch_dims.emplace_back();
  new_dims_mapping.batch_dims.back().lhs = dnums.input_feature_dimension();
  new_dims_mapping.batch_dims.back().rhs =
      dnums.kernel_output_feature_dimension();
  new_dims_mapping.batch_dims.back().output = dnums.output_feature_dimension();
  new_dims_mapping.batch_dims.back().spatial = -1;
  // Setup non contracting dims.
  new_dims_mapping.lhs_non_contracting_dims.emplace_back();
  new_dims_mapping.lhs_non_contracting_dims.back().lhs =
      dnums.input_batch_dimension();
  new_dims_mapping.rhs_non_contracting_dims.emplace_back();
  new_dims_mapping.rhs_non_contracting_dims.back().rhs =
      dnums.kernel_input_feature_dimension();
  return new_dims_mapping;
}

DotConvDimsMapping ConvertDimsMappingWithBatchGroupCount(
    const DotConvDimsMapping& dims_mapping, HloInstruction* original_hlo) {
  const auto& dnums = original_hlo->convolution_dimension_numbers();
  DotConvDimsMapping new_dims_mapping;
  new_dims_mapping.batch_dims = dims_mapping.batch_dims;
  new_dims_mapping.conv_spatial_dims = dims_mapping.conv_spatial_dims;
  new_dims_mapping.contracting_dims = dims_mapping.contracting_dims;
  // Append batch dims.
  new_dims_mapping.batch_dims.emplace_back();
  new_dims_mapping.batch_dims.back().lhs = dnums.input_batch_dimension();
  new_dims_mapping.batch_dims.back().rhs =
      dnums.kernel_output_feature_dimension();
  new_dims_mapping.batch_dims.back().output = dnums.output_feature_dimension();
  new_dims_mapping.batch_dims.back().spatial = -1;
  return new_dims_mapping;
}

// Recursive partitioning function. If there are partial dimensions matching in
// the operands and output, group the devices and recursively partition the
// in-group dot.
StatusOr<HloInstruction*> PartitionDot(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64 num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    bool require_matching_devices_to_group,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops) {
  // If lhs‘ hlo and rhs' hlo are identical, make a copy for rhs.
  if (lhs.hlo() == rhs.hlo()) {
    auto copy_hlo = b->AddInstruction(HloInstruction::CreateUnary(
        rhs.hlo()->shape(), HloOpcode::kCopy, rhs.hlo()));
    copy_hlo->set_sharding(rhs.sharding());
    rhs = PartitionedHlo(copy_hlo, rhs.base_shape(), rhs.state());
  }

  // lhs_rhs_or_output: 0 lhs, 1 rhs, 2 output.
  auto get_partitions_for_dims =
      [&](const HloSharding& sharding,
          absl::Span<const DotConvDimsMapping::DimsMapping> dims,
          int lhs_rhs_or_output) {
        int64 partitions = 1;
        if (sharding.IsTileMaximal()) {
          return partitions;
        }
        for (const auto& dim : dims) {
          if (lhs_rhs_or_output == 0) {
            partitions *= sharding.tile_assignment().dim(dim.lhs);
          } else if (lhs_rhs_or_output == 1) {
            partitions *= sharding.tile_assignment().dim(dim.rhs);
          } else {
            CHECK_EQ(lhs_rhs_or_output, 2);
            partitions *= sharding.tile_assignment().dim(dim.output);
          }
        }
        return partitions;
      };
  const int64 lhs_batch_partitions =
      get_partitions_for_dims(lhs.sharding(), dims_mapping.batch_dims, 0);
  const int64 rhs_batch_partitions =
      get_partitions_for_dims(rhs.sharding(), dims_mapping.batch_dims, 1);
  const int64 output_batch_partitions =
      get_partitions_for_dims(output_sharding, dims_mapping.batch_dims, 2);
  const int64 lhs_contracting_partitions =
      get_partitions_for_dims(lhs.sharding(), dims_mapping.contracting_dims, 0);
  const int64 rhs_contracting_partitions =
      get_partitions_for_dims(rhs.sharding(), dims_mapping.contracting_dims, 1);
  const int64 lhs_non_contracting_partitions = get_partitions_for_dims(
      lhs.sharding(), dims_mapping.lhs_non_contracting_dims, 0);
  const int64 rhs_non_contracting_partitions = get_partitions_for_dims(
      rhs.sharding(), dims_mapping.rhs_non_contracting_dims, 1);
  const int64 output_lhs_non_contracting_partitions = get_partitions_for_dims(
      output_sharding, dims_mapping.lhs_non_contracting_dims, 2);
  const int64 output_rhs_non_contracting_partitions = get_partitions_for_dims(
      output_sharding, dims_mapping.rhs_non_contracting_dims, 2);
  const int64 lhs_conv_spatial_partitions = get_partitions_for_dims(
      lhs.sharding(), dims_mapping.conv_spatial_dims, 0);
  const int64 rhs_conv_spatial_partitions = get_partitions_for_dims(
      rhs.sharding(), dims_mapping.conv_spatial_dims, 1);
  const int64 output_conv_spatial_partitions = get_partitions_for_dims(
      output_sharding, dims_mapping.conv_spatial_dims, 2);
  // Before we find partial matches along the dimensions, invoke base case again
  // without may_reshard_without_detecting_match.

  // Try partition the purely spatially-partitioned convolution with convolution
  // spatial dimension partitioned or depthwise parallel dimension partitioned.
  bool is_conv_spatial_dim_partitioned =
      (lhs_conv_spatial_partitions > 1 || rhs_conv_spatial_partitions > 1 ||
       output_conv_spatial_partitions > 1);
  bool is_conv_batch_or_contracting_dim_partitioned =
      (lhs_batch_partitions > 1 || rhs_batch_partitions > 1 ||
       output_batch_partitions > 1 ||
       (lhs_contracting_partitions > 1 && rhs_contracting_partitions > 1));
  if ((!dims_mapping.conv_spatial_dims.empty() &&
       is_conv_spatial_dim_partitioned &&
       !is_conv_batch_or_contracting_dim_partitioned) ||
      (original_hlo->opcode() == HloOpcode::kConvolution &&
       (original_hlo->batch_group_count() > 1 ||
        original_hlo->feature_group_count() > 1))) {
    // Partition with kernel_input_feature_dim > 1 and feature_group_count > 1
    // is not supported.
    const auto& dnums = original_hlo->convolution_dimension_numbers();
    if (original_hlo->feature_group_count() > 1 &&
        rhs.hlo()->shape().dimensions(dnums.kernel_input_feature_dimension()) >
            1) {
      return nullptr;
    }

    TF_ASSIGN_OR_RETURN(
        auto partitioned_conv,
        PartitionConvolution(lhs, rhs, output_base_shape, output_sharding,
                             dims_mapping, create_sharded_dot, conv_window,
                             original_hlo, num_partitions, options,
                             lhs.state().partition_id, module, b));

    if (partitioned_conv) {
      return partitioned_conv;
    }

    // Recursively partition on different types of dimensions for convolution.
    // Case 0.a: Group partitions by feature group count.
    if (original_hlo->feature_group_count() > 1 ||
        original_hlo->batch_group_count() > 1) {
      DotConvDimsMapping new_dims_mapping;
      if (original_hlo->feature_group_count() > 1) {
        new_dims_mapping =
            ConvertDimsMappingWithFeatureGroupCount(dims_mapping, original_hlo);
      }

      if (original_hlo->batch_group_count() > 1) {
        new_dims_mapping =
            ConvertDimsMappingWithBatchGroupCount(dims_mapping, original_hlo);
      }

      const int64 conv_lhs_contracting_partitions = get_partitions_for_dims(
          lhs.sharding(), new_dims_mapping.contracting_dims, 0);
      const int64 conv_rhs_contracting_partitions = get_partitions_for_dims(
          rhs.sharding(), new_dims_mapping.contracting_dims, 1);
      const int64 conv_lhs_non_contracting_partitions = get_partitions_for_dims(
          lhs.sharding(), new_dims_mapping.lhs_non_contracting_dims, 0);
      const int64 conv_rhs_non_contracting_partitions = get_partitions_for_dims(
          rhs.sharding(), new_dims_mapping.rhs_non_contracting_dims, 1);
      const int64 conv_lhs_batch_partitions = get_partitions_for_dims(
          lhs.sharding(), new_dims_mapping.batch_dims, 0);
      const int64 conv_rhs_batch_partitions = get_partitions_for_dims(
          rhs.sharding(), new_dims_mapping.batch_dims, 1);
      const int64 conv_output_batch_partitions = get_partitions_for_dims(
          output_sharding, new_dims_mapping.batch_dims, 2);
      if ((conv_lhs_batch_partitions == conv_output_batch_partitions ||
           conv_rhs_batch_partitions == conv_output_batch_partitions) &&
          conv_output_batch_partitions > 1) {
        TF_ASSIGN_OR_RETURN(
            auto try_partitioned_conv,
            PartitionDotGroupOnBatch(
                lhs, rhs, output_base_shape, output_sharding, new_dims_mapping,
                num_partitions, conv_lhs_contracting_partitions,
                conv_rhs_contracting_partitions,
                conv_lhs_non_contracting_partitions,
                conv_rhs_non_contracting_partitions, create_sharded_dot,
                conv_window, module, original_hlo,
                require_matching_devices_to_group, options, b,
                windowed_dot_general_loops));
        if (try_partitioned_conv) {
          return try_partitioned_conv;
        }
      }
      return nullptr;
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto try_partitioned_dot,
      PartitionBaseCase(
          lhs, rhs, output_base_shape, output_sharding, dims_mapping,
          num_partitions, create_sharded_dot, conv_window, module, original_hlo,
          lhs_batch_partitions, rhs_batch_partitions, output_batch_partitions,
          lhs_contracting_partitions, rhs_contracting_partitions,
          lhs_non_contracting_partitions, rhs_non_contracting_partitions,
          output_lhs_non_contracting_partitions,
          output_rhs_non_contracting_partitions, options, b,
          windowed_dot_general_loops,
          /*may_reshard_without_detecting_match=*/false));
  if (try_partitioned_dot) {
    return try_partitioned_dot;
  }

  // Recursively partition on different types of dimensions.
  //
  // Case 1: Group partitions by batch.
  if ((lhs_batch_partitions == output_batch_partitions ||
       rhs_batch_partitions == output_batch_partitions) &&
      output_batch_partitions > 1) {
    TF_ASSIGN_OR_RETURN(
        auto dot,
        PartitionDotGroupOnBatch(
            lhs, rhs, output_base_shape, output_sharding, dims_mapping,
            num_partitions, lhs_contracting_partitions,
            rhs_contracting_partitions, lhs_non_contracting_partitions,
            rhs_non_contracting_partitions, create_sharded_dot, conv_window,
            module, original_hlo, require_matching_devices_to_group, options, b,
            windowed_dot_general_loops));
    if (dot) {
      return dot;
    }
  }

  // Case 2: Group partitions by non-contracting dimensions.
  const bool may_group_on_lhs_non_contracting =
      lhs_non_contracting_partitions == output_lhs_non_contracting_partitions &&
      lhs_non_contracting_partitions > 1;
  const bool may_group_on_rhs_non_contracting =
      rhs_non_contracting_partitions == output_rhs_non_contracting_partitions &&
      rhs_non_contracting_partitions > 1;
  if (may_group_on_lhs_non_contracting || may_group_on_rhs_non_contracting) {
    // If both match output non-contracting dimensions, choose the one which
    // will result in smaller replication of the other operand.
    const bool lhs_matching =
        may_group_on_lhs_non_contracting &&
        (!may_group_on_rhs_non_contracting ||
         lhs_non_contracting_partitions *
                 ShapeUtil::ByteSizeOf(rhs.hlo()->shape()) <=
             rhs_non_contracting_partitions *
                 ShapeUtil::ByteSizeOf(lhs.hlo()->shape()));
    TF_ASSIGN_OR_RETURN(
        auto dot,
        PartitionDotGroupOnNonContracting(
            lhs_matching, lhs_matching ? lhs : rhs, lhs_matching ? rhs : lhs,
            lhs_matching ? lhs_contracting_partitions
                         : rhs_contracting_partitions,
            lhs_matching ? rhs_contracting_partitions
                         : lhs_contracting_partitions,
            lhs_matching ? dims_mapping.lhs_non_contracting_dims
                         : dims_mapping.rhs_non_contracting_dims,
            lhs_matching ? rhs_non_contracting_partitions
                         : lhs_non_contracting_partitions,
            lhs_matching ? output_rhs_non_contracting_partitions
                         : output_lhs_non_contracting_partitions,
            output_base_shape, output_sharding, dims_mapping, num_partitions,
            create_sharded_dot, conv_window, module, original_hlo,
            require_matching_devices_to_group, options, b,
            windowed_dot_general_loops));
    if (dot) {
      return dot;
    }
  }
  if (lhs_non_contracting_partitions > 1 &&
      output_lhs_non_contracting_partitions > 1) {
    // If part of LHS non-contracting dims match output, try them.
    std::vector<DotConvDimsMapping::DimsMapping> matching_dims;
    for (const auto& dim : dims_mapping.lhs_non_contracting_dims) {
      int64 lhs_partitions = lhs.sharding().tile_assignment().dim(dim.lhs);
      if (lhs_partitions > 1 &&
          lhs_partitions == output_sharding.tile_assignment().dim(dim.output)) {
        matching_dims.push_back(dim);
      }
    }
    if (!matching_dims.empty()) {
      TF_ASSIGN_OR_RETURN(
          auto dot, PartitionDotGroupOnNonContracting(
                        /*lhs_matching=*/true, lhs, rhs,
                        lhs_contracting_partitions, rhs_contracting_partitions,
                        matching_dims, rhs_non_contracting_partitions,
                        output_rhs_non_contracting_partitions,
                        output_base_shape, output_sharding, dims_mapping,
                        num_partitions, create_sharded_dot, conv_window, module,
                        original_hlo, require_matching_devices_to_group,
                        options, b, windowed_dot_general_loops));
      if (dot) {
        return dot;
      }
    }
  }
  if (rhs_non_contracting_partitions > 1 &&
      output_rhs_non_contracting_partitions > 1) {
    // If part of RHS non-contracting dims match output, try them.
    std::vector<DotConvDimsMapping::DimsMapping> matching_dims;
    for (const auto& dim : dims_mapping.rhs_non_contracting_dims) {
      int64 rhs_partitions = rhs.sharding().tile_assignment().dim(dim.rhs);
      if (rhs_partitions > 1 &&
          rhs_partitions == output_sharding.tile_assignment().dim(dim.output)) {
        matching_dims.push_back(dim);
      }
    }
    if (!matching_dims.empty()) {
      TF_ASSIGN_OR_RETURN(
          auto dot, PartitionDotGroupOnNonContracting(
                        /*lhs_matching=*/false, rhs, lhs,
                        rhs_contracting_partitions, lhs_contracting_partitions,
                        matching_dims, lhs_non_contracting_partitions,
                        output_lhs_non_contracting_partitions,
                        output_base_shape, output_sharding, dims_mapping,
                        num_partitions, create_sharded_dot, conv_window, module,
                        original_hlo, require_matching_devices_to_group,
                        options, b, windowed_dot_general_loops));
      if (dot) {
        return dot;
      }
    }
  }

  // Case 3: Group partitions by contracting dimensions.
  if (lhs_contracting_partitions == rhs_contracting_partitions &&
      lhs_contracting_partitions > 1) {
    TF_ASSIGN_OR_RETURN(
        auto dot,
        PartitionDotGroupOnContracting(
            lhs, rhs, dims_mapping.contracting_dims, output_batch_partitions,
            output_lhs_non_contracting_partitions,
            output_rhs_non_contracting_partitions, output_base_shape,
            output_sharding, dims_mapping, num_partitions, create_sharded_dot,
            conv_window, module, original_hlo,
            require_matching_devices_to_group, options, b,
            windowed_dot_general_loops));
    if (dot) {
      return dot;
    }
  }
  if (lhs_contracting_partitions > 1 && rhs_contracting_partitions > 1) {
    // If part of contracting dims match, try them.
    std::vector<DotConvDimsMapping::DimsMapping> matching_dims;
    for (const auto& dim : dims_mapping.contracting_dims) {
      int64 lhs_partitions = lhs.sharding().tile_assignment().dim(dim.lhs);
      if (lhs_partitions > 1 &&
          lhs_partitions == rhs.sharding().tile_assignment().dim(dim.rhs)) {
        matching_dims.push_back(dim);
      }
    }
    if (!matching_dims.empty()) {
      TF_ASSIGN_OR_RETURN(
          auto dot, PartitionDotGroupOnContracting(
                        lhs, rhs, matching_dims, output_batch_partitions,
                        output_lhs_non_contracting_partitions,
                        output_rhs_non_contracting_partitions,
                        output_base_shape, output_sharding, dims_mapping,
                        num_partitions, create_sharded_dot, conv_window, module,
                        original_hlo, require_matching_devices_to_group,
                        options, b, windowed_dot_general_loops));
      if (dot) {
        return dot;
      }
    }
  }

  // Case 4: If operands are replicated but output is partially replicated,
  // recursive call with partial replication removed.
  if (lhs.sharding().IsReplicated() && rhs.sharding().IsReplicated() &&
      output_sharding.ReplicateOnLastTileDim()) {
    auto grouped_output =
        GroupShardingOnDims(output_sharding, {output_base_shape.rank()});
    auto inner_state = CreatePerGroupPartitioningState(
        lhs.state(), grouped_output.device_groups, b);
    TF_ASSIGN_OR_RETURN(
        auto dot,
        PartitionDot(PartitionedHlo(lhs.hlo(), lhs.base_shape(), inner_state),
                     PartitionedHlo(rhs.hlo(), rhs.base_shape(), inner_state),
                     output_base_shape, grouped_output.sharding, dims_mapping,
                     output_sharding.NumTiles(), create_sharded_dot,
                     conv_window, module, original_hlo, options, b,
                     windowed_dot_general_loops));
    if (dot) {
      return dot;
    }
  }

  // We failed to find partial matches, invoke base case again with
  // may_reshard_without_detecting_match.
  TF_ASSIGN_OR_RETURN(
      auto dot,
      PartitionBaseCase(
          lhs, rhs, output_base_shape, output_sharding, dims_mapping,
          num_partitions, create_sharded_dot, conv_window, module, original_hlo,
          lhs_batch_partitions, rhs_batch_partitions, output_batch_partitions,
          lhs_contracting_partitions, rhs_contracting_partitions,
          lhs_non_contracting_partitions, rhs_non_contracting_partitions,
          output_lhs_non_contracting_partitions,
          output_rhs_non_contracting_partitions, options, b,
          windowed_dot_general_loops,
          /*may_reshard_without_detecting_match=*/true));
  if (dot) {
    return dot;
  }
  return nullptr;
}

StatusOr<HloInstruction*> PartitionDot(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64 num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops) {
  // First try partitioning without resharding the groups, then try allow
  // resharding the groups.
  for (bool require_matching_devices_to_group : {true, false}) {
    TF_ASSIGN_OR_RETURN(
        auto try_partition,
        PartitionDot(lhs, rhs, output_base_shape, output_sharding, dims_mapping,
                     num_partitions, create_sharded_dot, conv_window, module,
                     original_hlo, require_matching_devices_to_group, options,
                     b, windowed_dot_general_loops));
    if (try_partition) {
      return try_partition;
    }
  }

  // Default action.
  TF_ASSIGN_OR_RETURN(
      auto dot, create_sharded_dot(lhs.Replicate().hlo(), rhs.Replicate().hlo(),
                                   b, conv_window));
  dot->set_sharding(HloSharding::Replicate());
  return PartitionedHlo(dot, output_base_shape, lhs.state())
      .Reshard(output_sharding)
      .hlo();
}

}  // namespace

Status SpmdPartitioningVisitor::HandleDotHelper(
    HloInstruction* hlo, const DotConvDimsMapping& dims_mapping,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot) {
  auto& lhs = GetPartitionedHlo(hlo->operand(0));
  auto& rhs = GetPartitionedHlo(hlo->operand(1));
  Window conv_window;
  if (hlo->opcode() == HloOpcode::kConvolution) {
    conv_window = hlo->window();
  }

  TF_ASSIGN_OR_RETURN(
      auto partitioned_dot,
      PartitionDot(lhs, rhs, hlo->shape(), hlo->sharding(), dims_mapping,
                   num_partitions_, create_sharded_dot, conv_window, module_,
                   hlo, options_, &b_, &windowed_dot_general_loops_));
  SetPartitionedHlo(hlo, [&] { return partitioned_dot; });
  return Status::OK();
}

namespace {

// Finds a cluster of nodes that produce the inputs for `hlo` which only depend
// on small operands, which means the cluster should start with broadcasts,
// constants and iotas. All other internal nodes must be non-side-effecting
// elemntwise ops. Returns the set of nodes, and the small operands. E.g., for
// the following graph,
//
//     a -> broadcast -> multiply
//     iota  ---> add--/
//     constant/
//
// FindInputNodesIfOnlyDependOnSmallOperands(multiply) will return
//    <{broadcast, iota, constant, add, multiply}, [a]>.
std::pair<absl::flat_hash_set<HloInstruction*>, std::vector<HloInstruction*>>
FindInputNodesIfOnlyDependOnSmallOperands(HloInstruction* hlo) {
  absl::flat_hash_set<HloInstruction*> nodes_found;
  std::vector<HloInstruction*> new_operands;
  absl::flat_hash_set<const HloInstruction*> new_operands_set;
  std::vector<HloInstruction*> worklist;
  worklist.push_back(hlo);
  while (!worklist.empty()) {
    auto inst = worklist.back();
    worklist.pop_back();
    if (nodes_found.count(inst) > 0) {
      continue;
    }
    if (inst->opcode() == HloOpcode::kBroadcast ||
        inst->opcode() == HloOpcode::kConstant ||
        inst->opcode() == HloOpcode::kIota) {
      nodes_found.insert(inst);
      for (auto o : inst->operands()) {
        auto res = new_operands_set.emplace(o);
        if (res.second) {
          new_operands.push_back(o);
        }
      }
    } else if (inst->IsElementwise() && !inst->HasSideEffectNoRecurse() &&
               inst->opcode() != HloOpcode::kAllReduce &&
               absl::c_all_of(inst->operands(),
                              [inst](const HloInstruction* o) {
                                return ShapeUtil::CompatibleIgnoringElementType(
                                    o->shape(), inst->shape());
                              })) {
      nodes_found.insert(inst);
      for (auto o : inst->operands()) {
        worklist.push_back(o);
      }
    } else {
      nodes_found.clear();
      new_operands.clear();
      break;
    }
  }
  return {std::move(nodes_found), std::move(new_operands)};
}

// Moves a cluster of memory-reducing nodes into the windowed dot-general loop
// on contracting dimensions. Such a loop has a dynamic slice on the
// non-windowed operand. If we move the input nodes into the loop, the
// dynamic-slice could be merged with them by later optimization passes, which
// reduces memory.
//
// small_operands             small_operands
//        |                          |
// input_nodes                loop { |
//        |          =>         input_nodes
// loop { |                          |
//    dynamic-slice             dynamic-slice
//    ...                       ...
// }                          }
//
// Later optimization passes (TpuPadSliceMover) will merge the dynamic slice
// with the input nodes.
Status SinkInputNodesIntoWindowedDotGeneralLoopOnContractingDimensions(
    HloInstruction* loop, int64 non_windowed_operand_index) {
  auto input_tuple = loop->mutable_operand(0);
  auto old_operand = input_tuple->mutable_operand(non_windowed_operand_index);
  auto input_nodes = FindInputNodesIfOnlyDependOnSmallOperands(old_operand);
  auto to_sink = std::move(input_nodes.first);
  auto new_operands = std::move(input_nodes.second);
  if (to_sink.empty()) {
    return Status::OK();
  }
  auto computation = loop->parent();
  // Replace the old operand with a tuple of the found small operands.
  auto new_input_subtuple =
      computation->AddInstruction(HloInstruction::CreateTuple(new_operands));
  TF_RETURN_IF_ERROR(input_tuple->ReplaceOperandWithDifferentShape(
      non_windowed_operand_index, new_input_subtuple));

  auto body = loop->while_body();
  auto body_param = body->parameter_instruction(0);
  auto old_body_param_users = body_param->users();
  // Update all tuple shapes.
  for (auto tuple : std::vector<HloInstruction*>{
           input_tuple, loop, loop->while_condition()->parameter_instruction(0),
           body_param, body->root_instruction()}) {
    *ShapeUtil::GetMutableSubshape(tuple->mutable_shape(),
                                   {non_windowed_operand_index}) =
        new_input_subtuple->shape();
  }
  // Now update the loop body.
  auto new_operand_tuple_inside =
      body->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_input_subtuple->shape(), body_param, non_windowed_operand_index));
  TF_RETURN_IF_ERROR(body->root_instruction()->ReplaceOperandWithDifferentShape(
      non_windowed_operand_index, new_operand_tuple_inside));

  // Create nodes inside the loop body.
  std::vector<HloInstruction*> worklist;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> outside_to_inside;
  auto add_users_if_available = [&](HloInstruction* inst) {
    for (auto u : inst->users()) {
      if (outside_to_inside.count(u) == 0 && to_sink.count(u) > 0 &&
          absl::c_all_of(u->operands(), [&](const HloInstruction* o) {
            return outside_to_inside.count(o) > 0;
          })) {
        worklist.push_back(u);
      }
    }
  };
  for (int64 i = 0; i < new_operands.size(); ++i) {
    outside_to_inside[new_operands[i]] =
        body->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_operands[i]->shape(), new_operand_tuple_inside, i));
    add_users_if_available(new_operands[i]);
  }
  // HLOs to sink without operands.
  std::vector<HloInstruction*> nullaries_to_sink;
  for (auto inst : to_sink) {
    if (inst->operand_count() == 0) {
      nullaries_to_sink.push_back(inst);
    }
  }
  // Sort nullaries_to_sink to make it deterministic.
  absl::c_sort(nullaries_to_sink,
               [](const HloInstruction* a, const HloInstruction* b) {
                 return a->unique_id() < b->unique_id();
               });
  worklist.reserve(nullaries_to_sink.size());
  for (auto inst : nullaries_to_sink) {
    worklist.push_back(inst);
  }
  while (!worklist.empty()) {
    auto inst = worklist.back();
    worklist.pop_back();
    std::vector<HloInstruction*> inst_new_operands(inst->operand_count());
    for (int64 i = 0; i < inst->operand_count(); ++i) {
      inst_new_operands[i] = outside_to_inside[inst->operand(i)];
    }
    outside_to_inside[inst] = body->AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), inst_new_operands));
    add_users_if_available(inst);
  }
  TF_RET_CHECK(outside_to_inside.count(old_operand) > 0);
  for (auto ou : old_body_param_users) {
    if (ou->opcode() == HloOpcode::kGetTupleElement &&
        ou->tuple_index() == non_windowed_operand_index) {
      TF_RETURN_IF_ERROR(
          ou->ReplaceAllUsesWith(outside_to_inside[old_operand]));
      TF_RETURN_IF_ERROR(body->RemoveInstruction(ou));
    }
  }
  return Status::OK();
}

// Moves a cluster of memory-reducing nodes (with reduce nodes at the end) into
// the windowed dot-general loop on non-contracting dimensions. Such a loop has
// a dynamic-update-slice at the output. If we move the user nodes into the loop
// and before the dynamic-update-slice, the user nodes can operate on smaller
// shapes, which reduces memory.
//
// small_operands                   small_operands
//  | |                 =>                  | |
//  | |  loop {                     loop {  | |
//  | |    conv                             | broadcast      conv
//  | |      |                              |     |           /
//  | | dynamic-update-slice                |  dynamic-slice /
//  | |         |                           |     |         /
//  | |  }      |                           |  multiply-----
//  |broadcast  /                           |    /
//  | |        /                            reduce
//  |multiply--                             |
//  \ |                                dynamic-update-slice
//   reduce                         }
//
// Later optimization passes (TpuPadSliceMover) will merge the dynamic slice
// with the input nodes (broadcast).
Status MoveUsersIntoWindowedDotGeneralLoopOnNonContractingDimensions(
    HloInstruction* loop) {
  CHECK_EQ(loop->user_count(), 1);
  // There should be a single direct user of the while loop, which is the
  // gte for element 2, i.e., the dot output.
  auto user_gte = loop->users().front();
  CHECK_EQ(user_gte->opcode(), HloOpcode::kGetTupleElement);
  CHECK_EQ(user_gte->tuple_index(), 2);
  auto computation = loop->parent();

  // Find the reduce outputs and the input nodes they depend on, if input nodes
  // only have small operands.
  absl::flat_hash_set<HloInstruction*> to_move;
  std::vector<HloInstruction*> new_operands;
  absl::flat_hash_set<const HloInstruction*> new_operands_set;
  std::vector<HloInstruction*> reduce_outputs;
  std::vector<HloInstruction*> worklist;
  Shape padded_shape = user_gte->shape();
  Shape unpadded_shape = user_gte->shape();
  auto original_output = user_gte;

  if (user_gte->user_count() == 1 &&
      user_gte->users().back()->opcode() == HloOpcode::kSlice) {
    original_output = user_gte->users().back();
    unpadded_shape = original_output->shape();
  }
  for (auto u : original_output->users()) {
    worklist.push_back(u);
  }
  to_move.insert(original_output);
  while (!worklist.empty()) {
    auto inst = worklist.back();
    worklist.pop_back();
    if (to_move.count(inst) > 0) {
      continue;
    }
    // We only support reduces with simple reduction function, since we may need
    // to accumulate across iterations manually.
    if (inst->opcode() == HloOpcode::kReduce &&
        inst->to_apply()->instruction_count() == 3 &&
        inst->to_apply()->num_parameters() == 2 &&
        inst->to_apply()->root_instruction()->IsElementwise()) {
      to_move.insert(inst);
      auto other_operand = inst->mutable_operand(1);
      auto res = new_operands_set.emplace(other_operand);
      if (res.second) {
        new_operands.push_back(other_operand);
      }
      reduce_outputs.push_back(inst);
    } else if (inst != computation->root_instruction() &&
               inst->user_count() > 0 && inst->IsElementwise() &&
               !inst->HasSideEffectNoRecurse() &&
               inst->opcode() != HloOpcode::kAllReduce &&
               absl::c_all_of(inst->operands(),
                              [inst](const HloInstruction* o) {
                                return ShapeUtil::CompatibleIgnoringElementType(
                                    o->shape(), inst->shape());
                              })) {
      // For an elementwise op, we need to make sure that they depend on only
      // nodes already in to_move and nodes with small operands.
      bool can_include = true;
      for (auto operand : inst->operands()) {
        if (to_move.count(operand) > 0) {
          continue;
        }
        auto find_result = FindInputNodesIfOnlyDependOnSmallOperands(operand);
        if (find_result.first.empty()) {
          can_include = false;
          break;
        }
        for (auto n : find_result.first) {
          to_move.insert(n);
        }
        for (auto new_operand : find_result.second) {
          auto res = new_operands_set.insert(new_operand);
          if (res.second) {
            new_operands.push_back(new_operand);
          }
        }
      }
      if (!can_include) {
        to_move.clear();
        break;
      }
      to_move.insert(inst);
      for (auto u : inst->users()) {
        worklist.push_back(u);
      }
    } else {
      to_move.clear();
      break;
    }
  }
  // If nothing is found, to_move could contain only original_output, or cleared
  // by the above code.
  if (to_move.size() <= 1) {
    return Status::OK();
  }

  // We will replace the original loop output with reduce-shape outputs. Create
  // the initial buffers before the loop.
  for (auto out : reduce_outputs) {
    auto padded_out_shape = out->shape();
    int64 operand_dim = 0;
    int64 output_dim = 0;
    while (output_dim < padded_out_shape.rank()) {
      if (absl::c_linear_search(out->dimensions(), operand_dim)) {
        // Dimension colapsed.
        ++operand_dim;
        continue;
      }
      // Kept dimensions have the same size of the padded shape.
      padded_out_shape.set_dimensions(output_dim,
                                      padded_shape.dimensions(operand_dim));
      ++operand_dim;
      ++output_dim;
    }
    auto broadcast =
        computation->AddInstruction(HloInstruction::CreateBroadcast(
            padded_out_shape,
            computation->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::Zero(out->shape().element_type()))),
            {}));
    new_operands.push_back(broadcast);
  }

  auto input_tuple = loop->mutable_operand(0);
  // Create the new input subtuple that contains the small operands and the
  // reduce-shape result buffers.
  auto new_input_subtuple =
      computation->AddInstruction(HloInstruction::CreateTuple(new_operands));
  TF_RETURN_IF_ERROR(
      input_tuple->ReplaceOperandWithDifferentShape(2, new_input_subtuple));
  auto body = loop->while_body();
  auto body_param = body->parameter_instruction(0);
  auto body_root = body->root_instruction();
  CHECK_EQ(body_root->opcode(), HloOpcode::kTuple);
  // Update tuple shapes.
  for (auto tuple : std::vector<HloInstruction*>{
           input_tuple, loop, loop->while_condition()->parameter_instruction(0),
           body_param, body_root}) {
    *ShapeUtil::GetMutableSubshape(tuple->mutable_shape(), {2}) =
        new_input_subtuple->shape();
  }
  auto new_loop_input =
      body->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_input_subtuple->shape(), body_param, 2));

  // Now create the moved nodes inside the loop body.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> outside_to_inside;
  worklist.clear();
  auto add_users_if_available = [&](HloInstruction* inst) {
    for (auto u : inst->users()) {
      if (outside_to_inside.count(u) == 0 && to_move.count(u) > 0 &&
          absl::c_all_of(u->operands(), [&](const HloInstruction* o) {
            return outside_to_inside.count(o) > 0;
          })) {
        worklist.push_back(u);
      }
    }
  };
  for (int64 i = 0; i < new_operands.size(); ++i) {
    outside_to_inside[new_operands[i]] =
        body->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_operands[i]->shape(), new_loop_input, i));
    add_users_if_available(new_operands[i]);
  }
  // The elementwise nodes will be created with sliced shape. The original loop
  // output corresponds to the dynamic-update-slice's update slice.
  auto dus = body_root->mutable_operand(2);
  CHECK_EQ(dus->opcode(), HloOpcode::kDynamicUpdateSlice);
  outside_to_inside[original_output] = dus->mutable_operand(1);
  add_users_if_available(original_output);
  std::vector<HloInstruction*> slice_offsets(padded_shape.rank());
  for (int64 i = 0; i < slice_offsets.size(); ++i) {
    slice_offsets[i] = dus->mutable_operand(i + 2);
  }
  auto get_slice = [&](HloInstruction* padded) {
    return body->AddInstruction(HloInstruction::CreateDynamicSlice(
        ShapeUtil::ChangeElementType(dus->operand(1)->shape(),
                                     padded->shape().element_type()),
        padded, slice_offsets, dus->operand(1)->shape().dimensions()));
  };
  // Helper functions to create nodes with small operands.
  auto add_broadcast = [&](const HloInstruction* broadcast) {
    auto padded_operand_shape = broadcast->operand(0)->shape();
    for (int64 i = 0; i < broadcast->dimensions().size(); ++i) {
      padded_operand_shape.set_dimensions(
          i, padded_shape.dimensions(broadcast->dimensions(i)));
    }
    auto padded_operand = PadToShape(outside_to_inside[broadcast->operand(0)],
                                     padded_operand_shape, nullptr, body);
    outside_to_inside[broadcast] =
        get_slice(body->AddInstruction(broadcast->CloneWithNewOperands(
            ShapeUtil::ChangeElementType(padded_shape,
                                         padded_operand_shape.element_type()),
            {padded_operand})));
  };
  auto add_iota = [&](const HloInstruction* iota) {
    outside_to_inside[iota] =
        get_slice(body->AddInstruction(iota->CloneWithNewOperands(
            ShapeUtil::ChangeElementType(padded_shape,
                                         iota->shape().element_type()),
            {})));
  };
  auto add_constant = [&](const HloInstruction* constant) {
    outside_to_inside[constant] = body->AddInstruction(constant->Clone());
    outside_to_inside[constant] = get_slice(
        PadToShape(outside_to_inside[constant],
                   ShapeUtil::ChangeElementType(
                       padded_shape, constant->shape().element_type()),
                   nullptr, body));
  };
  while (!worklist.empty()) {
    auto inst = worklist.back();
    worklist.pop_back();
    if (outside_to_inside.count(inst) > 0) {
      continue;
    }
    if (inst->opcode() == HloOpcode::kBroadcast) {
      add_broadcast(inst);
    } else if (inst->opcode() == HloOpcode::kIota) {
      add_iota(inst);
    } else if (inst->opcode() == HloOpcode::kConstant) {
      add_constant(inst);
    } else if (inst->opcode() == HloOpcode::kReduce) {
      // This is an output, for which we has special handling later.
    } else {
      std::vector<HloInstruction*> operands_inside(inst->operand_count());
      for (int64 i = 0; i < operands_inside.size(); ++i) {
        operands_inside[i] = outside_to_inside[inst->operand(i)];
      }
      outside_to_inside[inst] = body->AddInstruction(inst->CloneWithNewOperands(
          ShapeUtil::ChangeElementType(dus->operand(1)->shape(),
                                       inst->shape().element_type()),
          operands_inside));
    }
    add_users_if_available(inst);
  }
  std::vector<HloInstruction*> new_outputs_inside(new_operands.size());
  for (int64 i = 0; i < new_outputs_inside.size(); ++i) {
    new_outputs_inside[i] = outside_to_inside[new_operands[i]];
  }
  // Now create the reduce outpus inside of the loop.
  for (int64 i = 0; i < reduce_outputs.size(); ++i) {
    auto reduce_outside = reduce_outputs[i];
    CHECK_EQ(reduce_outside->opcode(), HloOpcode::kReduce);
    int64 index_in_operand = new_operands.size() - reduce_outputs.size() + i;
    auto last_iter_result = outside_to_inside[new_operands[index_in_operand]];
    auto operand0 = outside_to_inside[reduce_outside->operand(0)];
    auto operand1 = outside_to_inside[reduce_outside->operand(1)];
    TF_ASSIGN_OR_RETURN(auto reduce_shape,
                        ShapeInference::InferReduceShape(
                            {&operand0->shape(), &operand1->shape()},
                            reduce_outside->dimensions(),
                            reduce_outside->to_apply()->ComputeProgramShape()));
    *reduce_shape.mutable_layout() = reduce_outside->shape().layout();
    std::vector<HloInstruction*> reduce_dus_offsets;
    // If any collapsed dimension is windowed, we need to accumulate with last
    // iteration's result. If such a dimension has padding, we also need to mask
    // off invalid data.
    bool needs_accumulate = false;
    std::vector<int64> dims_to_mask;
    for (int64 i = 0; i < slice_offsets.size(); ++i) {
      if (absl::c_linear_search(reduce_outside->dimensions(), i)) {
        if (reduce_outside->operand(0)->shape().dimensions(i) !=
            operand0->shape().dimensions(i)) {
          needs_accumulate = true;
          if (unpadded_shape.dimensions(i) != padded_shape.dimensions(i)) {
            dims_to_mask.push_back(i);
          }
        }
        continue;
      }
      reduce_dus_offsets.push_back(slice_offsets[i]);
    }
    // Mask off invalid data in collapsed dimensions.
    for (int64 dim : dims_to_mask) {
      auto iota = body->AddInstruction(HloInstruction::CreateIota(
          ShapeUtil::ChangeElementType(operand0->shape(), S32), dim));
      auto add = body->AddInstruction(HloInstruction::CreateBinary(
          iota->shape(), HloOpcode::kAdd, iota,
          body->AddInstruction(HloInstruction::CreateBroadcast(
              iota->shape(), slice_offsets[dim], {}))));
      auto limit = body->AddInstruction(HloInstruction::CreateBroadcast(
          iota->shape(),
          body->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(
                  reduce_outside->operand(0)->shape().dimensions(dim)))),
          {}));
      auto compare = body->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::ChangeElementType(iota->shape(), PRED), add, limit,
          ComparisonDirection::kLt));
      operand0 = body->AddInstruction(HloInstruction::CreateTernary(
          operand0->shape(), HloOpcode::kSelect, compare, operand0,
          body->AddInstruction(HloInstruction::CreateBroadcast(
              operand0->shape(), operand1, {}))));
    }
    auto output_inside =
        body->AddInstruction(reduce_outside->CloneWithNewOperands(
            reduce_shape, {operand0, operand1}));
    // Accumulate with previous results if needed.
    if (needs_accumulate) {
      auto input_slice =
          body->AddInstruction(HloInstruction::CreateDynamicSlice(
              output_inside->shape(), last_iter_result, reduce_dus_offsets,
              output_inside->shape().dimensions()));
      output_inside = body->AddInstruction(HloInstruction::CreateBinary(
          output_inside->shape(),
          reduce_outside->to_apply()->root_instruction()->opcode(),
          output_inside, input_slice));
    }
    // Dynamic-update-slice if needed.
    if (!ShapeUtil::Compatible(output_inside->shape(),
                               last_iter_result->shape())) {
      output_inside =
          body->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
              last_iter_result->shape(), last_iter_result, output_inside,
              reduce_dus_offsets));
    }
    new_outputs_inside[index_in_operand] = output_inside;
  }
  // Body output.
  auto new_output_inside =
      body->AddInstruction(HloInstruction::CreateTuple(new_outputs_inside));
  TF_RETURN_IF_ERROR(
      body_root->ReplaceOperandWithDifferentShape(2, new_output_inside));
  TF_RETURN_IF_ERROR(body->RemoveInstructionAndUnusedOperands(dus));
  // Replace uses of the reduces outside the loop.
  auto new_output_gte =
      computation->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_output_inside->shape(), loop, 2));
  for (int64 i = 0; i < reduce_outputs.size(); ++i) {
    int64 index_in_operand = new_operands.size() - reduce_outputs.size() + i;
    auto new_output =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_outputs_inside[index_in_operand]->shape(), new_output_gte,
            index_in_operand));
    if (!ShapeUtil::Compatible(new_output->shape(),
                               reduce_outputs[i]->shape())) {
      new_output = computation->AddInstruction(HloInstruction::CreateSlice(
          reduce_outputs[i]->shape(), new_output,
          std::vector<int64>(new_output->shape().rank(), 0),
          reduce_outputs[i]->shape().dimensions(),
          std::vector<int64>(new_output->shape().rank(), 1)));
    }
    TF_RETURN_IF_ERROR(reduce_outputs[i]->ReplaceAllUsesWith(new_output));
    TF_RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(reduce_outputs[i]));
  }
  return Status::OK();
}

}  // namespace

Status SpmdPartitioningVisitor::DoCodeMotionForWindowedDotGeneralLoops(
    HloComputation* computation, const SpmdPartitionerOptions& options) {
  for (auto& loop : windowed_dot_general_loops_) {
    if (loop.windowed_in_contracting_dims || loop.windowed_in_batch_dims ||
        loop.operands_sharded_at_contracting_dims) {
      // We have a dynamic-slice for the non-windowed operand in
      // batch/contracting-dim/noncontracting-dim windowed dot-general. So
      // moving the broadcast/iota/elementwise ops into the loop could help
      // reduce memory via fusion.
      TF_RETURN_IF_ERROR(
          SinkInputNodesIntoWindowedDotGeneralLoopOnContractingDimensions(
              loop.while_loop, 1 - loop.windowed_operand));
    }
    // Currently unrolled loop does not support this optimization.
    if (!options.unroll_windowed_einsum && !loop.windowed_in_contracting_dims &&
        !loop.operands_sharded_at_contracting_dims) {
      // We have a dynamic-update-slice for the output in
      // batch/non-contracting-dim windowed dot-general. So moving reduce ops
      // into the loop could help reduce memory.
      TF_RETURN_IF_ERROR(
          MoveUsersIntoWindowedDotGeneralLoopOnNonContractingDimensions(
              loop.while_loop));
    }
  }
  return Status::OK();
}

}  // namespace spmd
}  // namespace xla
