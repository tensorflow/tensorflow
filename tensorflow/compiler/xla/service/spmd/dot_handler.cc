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
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/numbers.h"

namespace xla {
namespace spmd {

Status SpmdPartitioningVisitor::HandleDot(HloInstruction* hlo) {
  DotGeneralDimsMapping mapping;
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
  auto create_sharded_dot = [&](HloInstruction* l, HloInstruction* r,
                                SpmdBuilder* b) -> StatusOr<HloInstruction*> {
    TF_ASSIGN_OR_RETURN(
        auto sharded_dot_shape,
        ShapeInference::InferDotOpShape(l->shape(), r->shape(),
                                        hlo->dot_dimension_numbers()));
    return b->AddInstruction(HloInstruction::CreateDot(
        sharded_dot_shape, l, r, hlo->dot_dimension_numbers(),
        hlo->precision_config()));
  };
  return HandleDotHelper(hlo, mapping, create_sharded_dot);
}

Status SpmdPartitioningVisitor::HandleDotHelper(
    HloInstruction* hlo, const DotGeneralDimsMapping& dims_mapping,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*)>& create_sharded_dot) {
  const HloSharding& lhs_sharding = hlo->operand(0)->sharding();
  const HloSharding& rhs_sharding = hlo->operand(1)->sharding();

  // Similar to hlo_sharding_util::TransposeSharding(), but allows
  // removing/adding non-partitioned dimensions.
  auto transpose_sharding =
      [&](const HloSharding& source, absl::Span<int64 const> src_to_tgt,
          absl::Span<int64 const> tgt_to_src) -> absl::optional<HloSharding> {
    if (source.IsTileMaximal()) {
      return source;
    }
    std::vector<int64> tgt_dims_skipping_new(tgt_to_src.size(), -1);
    int64 skipped_tgt_dims = 0;
    for (int64 i = 0; i < tgt_to_src.size(); ++i) {
      if (tgt_to_src[i] < 0) {
        skipped_tgt_dims++;
      } else {
        tgt_dims_skipping_new[i] = i - skipped_tgt_dims;
      }
    }
    int64 skipped_src_dims = absl::c_count(src_to_tgt, -1);
    std::vector<int64> perm(src_to_tgt.size());
    for (int64 i = 0; i < src_to_tgt.size(); ++i) {
      if (src_to_tgt[i] < 0) {
        if (source.tile_assignment().dim(i) > 1) {
          return absl::nullopt;
        }
        perm[src_to_tgt.size() - skipped_src_dims] = i;
        skipped_src_dims--;
      } else {
        perm[tgt_dims_skipping_new[src_to_tgt[i]]] = i;
      }
    }
    auto tgt_sharding = hlo_sharding_util::TransposeSharding(source, perm);
    if (skipped_tgt_dims == 0) {
      return tgt_sharding;
    }
    auto reshape_tiles = tgt_sharding.tile_assignment();
    std::vector<int64> tgt_tiles(tgt_to_src.size(), 1);
    for (int64 i = 0; i < tgt_tiles.size(); ++i) {
      if (tgt_to_src[i] >= 0) {
        tgt_tiles[i] = reshape_tiles.dim(tgt_dims_skipping_new[i]);
      }
    }
    reshape_tiles.Reshape(tgt_tiles);
    return HloSharding::Tile(reshape_tiles);
  };

  std::vector<int64> lhs_to_rhs_indices(hlo->operand(0)->shape().rank(), -1);
  std::vector<int64> lhs_to_output_indices(hlo->operand(0)->shape().rank(), -1);
  std::vector<int64> rhs_to_lhs_indices(hlo->operand(1)->shape().rank(), -1);
  std::vector<int64> rhs_to_output_indices(hlo->operand(1)->shape().rank(), -1);
  std::vector<int64> output_to_lhs_indices(hlo->shape().rank(), -1);
  std::vector<int64> output_to_rhs_indices(hlo->shape().rank(), -1);
  auto populate_indices_mapping =
      [&](const DotGeneralDimsMapping::DimsMapping& mapping) {
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
  auto lhs_sharding_transposed_to_match_rhs =
      transpose_sharding(lhs_sharding, lhs_to_rhs_indices, rhs_to_lhs_indices);
  auto rhs_sharding_transposed_to_match_lhs =
      transpose_sharding(rhs_sharding, rhs_to_lhs_indices, lhs_to_rhs_indices);
  auto lhs_sharding_transposed_to_match_output = transpose_sharding(
      lhs_sharding, lhs_to_output_indices, output_to_lhs_indices);
  auto rhs_sharding_transposed_to_match_output = transpose_sharding(
      rhs_sharding, rhs_to_output_indices, output_to_rhs_indices);
  auto output_sharding_transposed_to_match_lhs = transpose_sharding(
      hlo->sharding(), output_to_lhs_indices, lhs_to_output_indices);
  auto output_sharding_transposed_to_match_rhs = transpose_sharding(
      hlo->sharding(), output_to_rhs_indices, rhs_to_output_indices);

  // lhs_rhs_or_output: 0 lhs, 1 rhs, 2 output.
  auto get_partitions_for_dims =
      [&](const HloSharding& sharding,
          absl::Span<const DotGeneralDimsMapping::DimsMapping> dims,
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
      get_partitions_for_dims(lhs_sharding, dims_mapping.batch_dims, 0);
  const int64 rhs_batch_partitions =
      get_partitions_for_dims(rhs_sharding, dims_mapping.batch_dims, 1);
  const int64 output_batch_partitions =
      get_partitions_for_dims(hlo->sharding(), dims_mapping.batch_dims, 2);
  const int64 lhs_contracting_partitions =
      get_partitions_for_dims(lhs_sharding, dims_mapping.contracting_dims, 0);
  const int64 rhs_contracting_partitions =
      get_partitions_for_dims(rhs_sharding, dims_mapping.contracting_dims, 1);
  const int64 lhs_non_contracting_partitions = get_partitions_for_dims(
      lhs_sharding, dims_mapping.lhs_non_contracting_dims, 0);
  const int64 rhs_non_contracting_partitions = get_partitions_for_dims(
      rhs_sharding, dims_mapping.rhs_non_contracting_dims, 1);
  const int64 output_lhs_non_contracting_partitions = get_partitions_for_dims(
      hlo->sharding(), dims_mapping.lhs_non_contracting_dims, 2);
  const int64 output_rhs_non_contracting_partitions = get_partitions_for_dims(
      hlo->sharding(), dims_mapping.rhs_non_contracting_dims, 2);

  auto& lhs = GetPartitionedHlo(hlo->operand(0));
  auto& rhs = GetPartitionedHlo(hlo->operand(1));
  // LHS and RHS are partitioned the same way and only partitioned in batch
  // dimensions.
  if (lhs_batch_partitions == rhs_batch_partitions &&
      rhs_batch_partitions == num_partitions_ &&
      lhs_sharding_transposed_to_match_rhs == rhs_sharding) {
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(lhs.hlo(), rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] {
      dot->set_sharding(*lhs_sharding_transposed_to_match_output);
      return PartitionedHlo(dot, hlo->shape(), MakePartitioningState())
          .Reshard(hlo->sharding())
          .hlo();
    });
    return Status::OK();
  }

  // Try emit batch-partitioned einsum with one operand resharded. Returns
  // whether the attempt succeeds. If may_reshard_with_allreduce is false,
  // reshard must be done using all-to-all; otherwise this attempt fails.
  auto try_emit_output_batch_partitioned_einsum_with_reshard =
      [&](bool may_reshard_with_allreduce) -> StatusOr<bool> {
    // LHS and output are batch partitioned in the same way.
    if (lhs_batch_partitions == num_partitions_ &&
        output_batch_partitions == num_partitions_ &&
        lhs_sharding_transposed_to_match_output == hlo->sharding()) {
      if (!may_reshard_with_allreduce &&
          !GetReshardAllToAllSourceTargetDims(
              rhs.sharding(), *lhs_sharding_transposed_to_match_rhs)) {
        return false;
      }
      auto resharded_rhs = rhs.Reshard(*lhs_sharding_transposed_to_match_rhs);
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(lhs.hlo(), resharded_rhs.hlo(), &b_));
      SetPartitionedHlo(hlo, [&] { return dot; });
      return true;
    }
    // RHS and output are batch partitioned in the same way.
    if (rhs_batch_partitions == num_partitions_ &&
        output_batch_partitions == num_partitions_ &&
        rhs_sharding_transposed_to_match_output == hlo->sharding()) {
      if (!may_reshard_with_allreduce &&
          !GetReshardAllToAllSourceTargetDims(
              lhs.sharding(), *rhs_sharding_transposed_to_match_lhs)) {
        return false;
      }
      auto resharded_lhs = lhs.Reshard(*rhs_sharding_transposed_to_match_lhs);
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(resharded_lhs.hlo(), rhs.hlo(), &b_));
      SetPartitionedHlo(hlo, [&] { return dot; });
      return true;
    }
    return false;
  };

  {
    // Try batch-parallel by resharding one operand, and not using all-reduce.
    TF_ASSIGN_OR_RETURN(
        bool emitted,
        try_emit_output_batch_partitioned_einsum_with_reshard(false));
    if (emitted) {
      return Status::OK();
    }
  }

  // Try to emit windowed DotGeneral when one operand is partitioned in the same
  // way as the output along non-contracting dimensions, but the other operand
  // is tiled in other dimensions.
  auto emit_windowed_dot_general = [&](int64 matching_operand,
                                       int64 windowing_operand,
                                       bool windowed_at_contracting_dims,
                                       bool windowed_at_batch_dims) {
    CHECK_EQ(matching_operand + windowing_operand, 1);
    CHECK(!windowed_at_batch_dims || !windowed_at_contracting_dims);
    auto unpadded_result_buffer_shape =
        MakePartitionedShape(hlo->shape(), hlo->sharding());
    auto padded_result_buffer_shape = unpadded_result_buffer_shape;
    // For windowing at batch/non-contracting dims, we produce the result one
    // partition at a time, so we need to pad the shape in case of uneven
    // partitioning in order to make dynamic-update-slice in-bound.
    if (!windowed_at_contracting_dims) {
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
          to_mask.PadWithValue(b_.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(hlo->shape().element_type()))));
    }
    auto result_buffer = CreateZero(padded_result_buffer_shape, &b_);
    auto iteration = b_.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32>(0)));

    // Create a while loop that computes one window per iteration. During each
    // iteration, each partition sends its input window to its neighbor using
    // collective-permute for the next iteration.
    SpmdBuilder body_b("windowed_dot_general_body", visiting_hlo_);
    auto param = body_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0,
        ShapeUtil::MakeTupleShape({lhs.hlo()->shape(), rhs.hlo()->shape(),
                                   result_buffer->shape(), iteration->shape()}),
        "param"));
    auto l = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(lhs.hlo()->shape(), param, 0));
    auto r = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(rhs.hlo()->shape(), param, 1));
    auto o = body_b.AddInstruction(HloInstruction::CreateGetTupleElement(
        result_buffer->shape(), param, 2));
    auto i = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(iteration->shape(), param, 3));

    auto partition_id = collective_ops_creator_.create_partition_id(&body_b);
    auto data_partition_id = body_b.AddInstruction(HloInstruction::CreateBinary(
        i->shape(), HloOpcode::kAdd, i, partition_id));
    auto partition_count = body_b.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<uint32>(num_partitions_)));
    data_partition_id = body_b.AddInstruction(HloInstruction::CreateBinary(
        i->shape(), HloOpcode::kRemainder, data_partition_id, partition_count));
    auto dot_lhs = l;
    auto dot_rhs = r;
    if (windowed_at_contracting_dims || windowed_at_batch_dims) {
      // Slice the matching operand according to the partitioned contracting
      // dimensions on the windowed operand. We do this by treating the matching
      // operand as replicated, and resharding it to match the windowed operand.
      auto slice_operand = matching_operand == 0 ? l : r;
      slice_operand->set_sharding(HloSharding::Replicate());
      auto state = MakePartitioningState();
      state.b = &body_b;
      state.partition_id = data_partition_id;
      auto slice = PartitionedHlo(slice_operand, slice_operand->shape(), state)
                       .Reshard(windowing_operand == 0
                                    ? *lhs_sharding_transposed_to_match_rhs
                                    : *rhs_sharding_transposed_to_match_lhs)
                       .hlo();
      slice_operand->clear_sharding();
      if (matching_operand == 0) {
        dot_lhs = slice;
      } else {
        dot_rhs = slice;
      }
    }
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(dot_lhs, dot_rhs, &body_b));
    if (windowed_at_contracting_dims) {
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

    // ++i
    i = body_b.AddInstruction(HloInstruction::CreateBinary(
        i->shape(), HloOpcode::kAdd, i,
        body_b.AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32>(1)))));
    auto has_more = body_b.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), i,
        body_b.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<uint32>(num_partitions_))),
        ComparisonDirection::kLt));
    // Collective-permute for the next window. We don't need it for the last
    // iteration, so we use a conditional around the collective-permute.
    HloInstruction* conditional;
    {
      SpmdBuilder cp_b("window_collective_permute", visiting_hlo_);
      {
        auto p = cp_b.AddInstruction(HloInstruction::CreateParameter(
            0, windowing_operand == 0 ? l->shape() : r->shape(), "window"));
        std::vector<std::pair<int64, int64>> sd_pairs(num_partitions_);
        for (int64 source = 0; source < num_partitions_; ++source) {
          // 0 -> n-1, 1 -> 0, 2 -> 1, ...
          sd_pairs[source] = {source,
                              (source - 1 + num_partitions_) % num_partitions_};
        }
        collective_ops_creator_.create_cross_partition_collective_permute(
            &cp_b, p, sd_pairs, (*next_channel_id_)++);
      }
      SpmdBuilder ncp_b("last_iteration_noop", visiting_hlo_);
      {
        ncp_b.AddInstruction(HloInstruction::CreateParameter(
            0, windowing_operand == 0 ? l->shape() : r->shape(), "window"));
      }
      conditional = body_b.AddInstruction(HloInstruction::CreateConditional(
          windowing_operand == 0 ? l->shape() : r->shape(), has_more,
          windowing_operand == 0 ? l : r,
          module_->AddEmbeddedComputation(cp_b.Build()),
          windowing_operand == 0 ? l : r,
          module_->AddEmbeddedComputation(ncp_b.Build())));
    }
    if (windowing_operand == 0) {
      l = conditional;
    } else {
      r = conditional;
    }
    body_b.AddInstruction(HloInstruction::CreateTuple({l, r, o, i}));

    SpmdBuilder cond_b("windowed_dot_general_cond", visiting_hlo_);
    auto cond_param = cond_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0,
        ShapeUtil::MakeTupleShape({lhs.hlo()->shape(), rhs.hlo()->shape(),
                                   result_buffer->shape(), iteration->shape()}),
        "param"));
    auto cond_i = cond_b.AddInstruction(HloInstruction::CreateGetTupleElement(
        iteration->shape(), cond_param, 3));
    cond_b.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), cond_i,
        cond_b.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<uint32>(num_partitions_))),
        ComparisonDirection::kLt));
    auto while_loop = b_.AddInstruction(HloInstruction::CreateWhile(
        cond_param->shape(), module_->AddEmbeddedComputation(cond_b.Build()),
        module_->AddEmbeddedComputation(body_b.Build()),
        b_.AddInstruction(HloInstruction::CreateTuple(
            {lhs.hlo(), rhs.hlo(), result_buffer, iteration}))));
    windowed_dot_general_loops_.push_back({while_loop, windowing_operand,
                                           windowed_at_contracting_dims,
                                           windowed_at_batch_dims});
    SetPartitionedHlo(hlo, [&] {
      auto result = b_.AddInstruction(HloInstruction::CreateGetTupleElement(
          result_buffer->shape(), while_loop, 2));
      if (!ShapeUtil::Compatible(padded_result_buffer_shape,
                                 unpadded_result_buffer_shape)) {
        result = b_.AddInstruction(HloInstruction::CreateSlice(
            unpadded_result_buffer_shape, result,
            std::vector<int64>(padded_result_buffer_shape.rank(), 0),
            unpadded_result_buffer_shape.dimensions(),
            std::vector<int64>(padded_result_buffer_shape.rank(), 1)));
      }
      return result;
    });
    return Status::OK();
  };
  if (output_lhs_non_contracting_partitions == num_partitions_ &&
      output_sharding_transposed_to_match_lhs == lhs_sharding &&
      ShapeSizeInBytes(hlo->operand(1)->shape()) >=
          options_.threshold_for_windowed_einsum_mib * 1024 * 1024) {
    if (rhs_contracting_partitions == num_partitions_) {
      return emit_windowed_dot_general(0, 1, true, false);
    }
    if (rhs_non_contracting_partitions == num_partitions_) {
      return emit_windowed_dot_general(0, 1, false, false);
    }
    if (rhs_batch_partitions == num_partitions_) {
      return emit_windowed_dot_general(0, 1, false, true);
    }
  }
  if (output_rhs_non_contracting_partitions == num_partitions_ &&
      output_sharding_transposed_to_match_rhs == rhs_sharding &&
      ShapeSizeInBytes(hlo->operand(0)->shape()) >=
          options_.threshold_for_windowed_einsum_mib * 1024 * 1024) {
    if (lhs_contracting_partitions == num_partitions_) {
      return emit_windowed_dot_general(1, 0, true, false);
    }
    if (lhs_non_contracting_partitions == num_partitions_) {
      return emit_windowed_dot_general(1, 0, false, false);
    }
    if (lhs_batch_partitions == num_partitions_) {
      return emit_windowed_dot_general(1, 0, false, true);
    }
  }

  {
    // Try batch-parallel by resharding one operand, and allowing all-reduce.
    TF_ASSIGN_OR_RETURN(
        bool emitted,
        try_emit_output_batch_partitioned_einsum_with_reshard(true));
    if (emitted) {
      return Status::OK();
    }
  }

  // LHS and RHS have the same partitioned contracting dimensions.
  if (lhs_contracting_partitions == rhs_contracting_partitions &&
      lhs_contracting_partitions == num_partitions_) {
    auto zero = b_.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(hlo->shape().element_type())));
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
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(lhs.hlo(), rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] {
      auto ar = collective_ops_creator_.create_cross_partition_all_reduce(
          &b_, dot, MakeBinaryAdd(hlo->shape().element_type(), module_),
          NewChannel());
      ar->set_sharding(HloSharding::Replicate());
      return PartitionedHlo(ar, hlo->shape(), MakePartitioningState())
          .Reshard(hlo->sharding())
          .hlo();
    });
    return Status::OK();
  }

  // LHS and output have the same partitioned non-contracting dimensions.
  if (lhs_non_contracting_partitions == num_partitions_ &&
      output_lhs_non_contracting_partitions == num_partitions_ &&
      lhs_sharding_transposed_to_match_output == hlo->sharding()) {
    auto rhs_replicated = rhs.Reshard(HloSharding::Replicate()).hlo();
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(lhs.hlo(), rhs_replicated, &b_));
    SetPartitionedHlo(hlo, [&] { return dot; });
    return Status::OK();
  }

  // RHS and output have the same partitioned non-contracting dimensions.
  if (rhs_non_contracting_partitions == num_partitions_ &&
      output_rhs_non_contracting_partitions == num_partitions_ &&
      rhs_sharding_transposed_to_match_output == hlo->sharding()) {
    auto lhs_replicated = lhs.Reshard(HloSharding::Replicate()).hlo();
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(lhs_replicated, rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] { return dot; });
    return Status::OK();
  }

  // Output is batch partitioned.
  if (output_batch_partitions == num_partitions_) {
    auto resharded_lhs = lhs.Reshard(*output_sharding_transposed_to_match_lhs);
    auto resharded_rhs = rhs.Reshard(*output_sharding_transposed_to_match_rhs);
    TF_ASSIGN_OR_RETURN(auto dot, create_sharded_dot(resharded_lhs.hlo(),
                                                     resharded_rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] { return dot; });
    return Status::OK();
  }
  // Output is partitioned along LHS non-contracting dimensions.
  if (output_lhs_non_contracting_partitions == num_partitions_) {
    auto resharded_lhs = lhs.Reshard(*output_sharding_transposed_to_match_lhs);
    auto replicated_rhs = rhs.Reshard(HloSharding::Replicate());
    TF_ASSIGN_OR_RETURN(
        auto dot,
        create_sharded_dot(resharded_lhs.hlo(), replicated_rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] { return dot; });
    return Status::OK();
  }
  // Output is partitioned along RHS non-contracting dimensions.
  if (output_rhs_non_contracting_partitions == num_partitions_) {
    auto replicated_lhs = lhs.Reshard(HloSharding::Replicate());
    auto resharded_rhs = rhs.Reshard(*output_sharding_transposed_to_match_rhs);
    TF_ASSIGN_OR_RETURN(auto dot, create_sharded_dot(replicated_lhs.hlo(),
                                                     resharded_rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] { return dot; });
    return Status::OK();
  }

  // Returns true if it is beneficial to reshard the operand at `operand_idx`
  // across the contracting dimension.
  const auto should_partition_contracting_dim = [&](int64 operand_idx) {
    if (!hlo->sharding().IsReplicated()) {
      return false;
    }

    if (operand_idx == 0) {
      // If LHS and output are replicated, we compare the cost of all-gather
      // on RHS vs all-reduce on the output.
      return (rhs_contracting_partitions == num_partitions_) &&
             lhs.sharding().IsReplicated() &&
             ShapeUtil::ElementsIn(hlo->operand(1)->shape()) >
                 ShapeUtil::ElementsIn(hlo->shape());
    } else {
      return (lhs_contracting_partitions == num_partitions_) &&
             rhs.sharding().IsReplicated() &&
             ShapeUtil::ElementsIn(hlo->operand(0)->shape()) >
                 ShapeUtil::ElementsIn(hlo->shape());
    }
  };

  // When the output is replicated and one of the operands is partitioned along
  // contracting dimension, align the other operand to be partitioned along
  // the contracting dimensions.
  if (hlo->sharding().IsReplicated() && (should_partition_contracting_dim(0) ||
                                         should_partition_contracting_dim(1))) {
    auto zero = b_.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(hlo->shape().element_type())));
    if (should_partition_contracting_dim(0)) {
      lhs =
          lhs.Reshard(*rhs_sharding_transposed_to_match_lhs).PadWithValue(zero);
      rhs = rhs.PadWithValue(zero);
    } else {
      lhs = lhs.PadWithValue(zero);
      rhs =
          rhs.Reshard(*lhs_sharding_transposed_to_match_rhs).PadWithValue(zero);
    }
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(lhs.hlo(), rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] {
      auto ar = collective_ops_creator_.create_cross_partition_all_reduce(
          &b_, dot, MakeBinaryAdd(hlo->shape().element_type(), module_),
          NewChannel());
      ar->set_sharding(HloSharding::Replicate());
      return PartitionedHlo(ar, hlo->shape(), MakePartitioningState()).hlo();
    });
    return Status::OK();
  }

  return DefaultAction(hlo);
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
std::pair<std::unordered_set<HloInstruction*>, std::vector<HloInstruction*>>
FindInputNodesIfOnlyDependOnSmallOperands(HloInstruction* hlo) {
  std::unordered_set<HloInstruction*> nodes_found;
  std::vector<HloInstruction*> new_operands;
  std::unordered_set<const HloInstruction*> new_operands_set;
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
  std::unordered_map<const HloInstruction*, HloInstruction*> outside_to_inside;
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
  std::unordered_set<HloInstruction*> to_move;
  std::vector<HloInstruction*> new_operands;
  std::unordered_set<const HloInstruction*> new_operands_set;
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
  std::unordered_map<const HloInstruction*, HloInstruction*> outside_to_inside;
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
    HloComputation* computation) {
  for (auto& loop : windowed_dot_general_loops_) {
    if (loop.windowed_in_contracting_dims || loop.windowed_in_batch_dims) {
      // We have a dynamic-slice for the non-windowed operand in
      // batch/contracting-dim windowed dot-general. So moving the
      // broadcast/iota/elementwise ops into the loop could help reduce memory
      // via fusion.
      TF_RETURN_IF_ERROR(
          SinkInputNodesIntoWindowedDotGeneralLoopOnContractingDimensions(
              loop.while_loop, 1 - loop.windowed_operand));
    }
    if (!loop.windowed_in_contracting_dims) {
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
