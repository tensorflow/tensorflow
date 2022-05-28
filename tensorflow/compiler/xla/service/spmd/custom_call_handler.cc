/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/spmd/custom_call_handler.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_lexer.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace spmd {

namespace {

StatusOr<absl::flat_hash_map<std::string, int64_t>> ParseOpaqueAsAttributes(
    const HloInstruction* hlo) {
  absl::string_view opaque = Cast<HloCustomCallInstruction>(hlo)->opaque();
  HloLexer lexer(opaque);
  absl::flat_hash_map<std::string, int64_t> result;
  while (lexer.Lex() != TokKind::kEof) {
    if (lexer.GetKind() != TokKind::kAttributeName) {
      return InvalidArgument("Expects attribute name, %s", opaque);
    }
    std::string attr_name = lexer.GetStrVal();
    if (lexer.Lex() != TokKind::kInt) {
      return InvalidArgument("expects integer attribute value");
    }
    result[attr_name] = lexer.GetInt64Val();
    if (lexer.Lex() != TokKind::kComma) {
      break;
    }
  }
  return result;
}

constexpr char kSPMDOpRotateRight[] = "_SPMDInternalOp_RotateRight";

}  // namespace

Status SpmdPartitioningVisitor::HandleCustomCallTopK(HloInstruction* hlo) {
  if (!hlo->operand(0)->has_sharding()) {
    return DefaultAction(hlo);
  }

  const HloSharding& sharding = hlo->operand(0)->sharding();
  // No support for partial replicate yet.
  if (sharding.IsTileMaximal() || sharding.IsReplicated() ||
      sharding.ReplicateOnLastTileDim()) {
    return DefaultAction(hlo);
  }

  const int64_t batch_dim = 0;
  const int64_t sort_dim = 1;
  const int64_t shard_count = sharding.tile_assignment().dim(sort_dim);

  if (shard_count <= 1) {
    return DefaultAction(hlo);
  }

  const int64_t batch_dim_partition = sharding.tile_assignment().dim(batch_dim);
  const int64_t input_size = hlo->operand(0)->shape().dimensions(sort_dim);
  const int64_t batch_size = hlo->shape().tuple_shapes(0).dimensions(batch_dim);
  const int64_t k = hlo->shape().tuple_shapes(0).dimensions(sort_dim);
  const int64_t per_partition_size = CeilOfRatio(input_size, shard_count);

  if (k >= per_partition_size) {
    return DefaultAction(hlo);
  }

  auto input = hlo->operand(0);
  const auto element_type = input->shape().element_type();

  auto partitioned_input = GetPartitionedHlo(input).PadWithValue(
      CreateFirstWithType(element_type, &b_));

  auto partition_state = partitioned_input.state();
  auto replicated_sharding = HloSharding::Replicate();
  // If batch dimension is partitioned, partial replicated on sort dimension.
  if (batch_dim_partition > 1) {
    auto sharding_grouped =
        hlo_sharding_util::GroupShardingOnDims(sharding, {batch_dim});
    partition_state = CreatePerGroupPartitioningState(
        partitioned_input.state(), sharding_grouped.device_groups,
        partitioned_input.state().b);
    auto reshape_tile_assignment = sharding.tile_assignment();
    auto reshape_dimensions = reshape_tile_assignment.dimensions();
    reshape_dimensions.push_back(reshape_dimensions.back());
    reshape_dimensions[sort_dim] = 1;
    reshape_tile_assignment.Reshape(reshape_dimensions);
    replicated_sharding = HloSharding::PartialTile(reshape_tile_assignment);
  }

  // Each partition needs to do TopK separately, thus the base shape
  // becomes [batch_size, k * shard_count].
  const Shape replicated_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(hlo->operand(0)->shape().element_type(),
                            {batch_size, k * shard_count}),
       ShapeUtil::MakeShape(S32, {batch_size, k * shard_count})});
  auto custom_call_sharding =
      sharding.GetTupleSharding(replicated_shape).ValueOrDie();
  auto shard_shape =
      MakePartitionedShape(replicated_shape, custom_call_sharding);
  auto topk = b_.AddInstruction(
      hlo->CloneWithNewOperands(shard_shape, {partitioned_input.hlo()}));
  topk->set_sharding(custom_call_sharding);
  // Partition customcall.
  PartitionedHlo partitioned_topk(topk, replicated_shape,
                                  MakePartitioningState());
  topk = partitioned_topk.hlo();

  // Get value from TopK.
  HloInstruction* value_gte =
      b_.AddInstruction(HloInstruction::CreateGetTupleElement(
          topk->shape().tuple_shapes(0), topk, 0));
  value_gte->set_sharding(sharding);
  // Partition GetTupleElement of value.
  PartitionedHlo value_partitioned_gte(
      value_gte, partitioned_topk.base_shape().tuple_shapes(0),
      MakePartitioningState());
  // Reshard value to be replicated.
  auto replicated_value_gte =
      value_partitioned_gte.Reshard(replicated_sharding).hlo();

  // Get index from TopK.
  HloInstruction* index_gte =
      b_.AddInstruction(HloInstruction::CreateGetTupleElement(
          topk->shape().tuple_shapes(1), topk, 1));
  auto partition_id_s32 = b_.AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::MakeShape(S32, partition_id_->shape().dimensions()),
      partition_state.partition_id));
  // Add per partition offset to index, index returned from CustomCall always
  // starts from 0.
  auto index_offset = b_.AddInstruction(HloInstruction::CreateBroadcast(
      index_gte->shape(),
      b_.AddInstruction(HloInstruction::CreateBinary(
          partition_id_s32->shape(), HloOpcode::kMultiply, partition_id_s32,
          b_.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32_t>(per_partition_size))))),
      {}));
  index_gte = b_.AddInstruction(HloInstruction::CreateBinary(
      index_offset->shape(), HloOpcode::kAdd, index_gte, index_offset));
  index_gte->set_sharding(sharding);
  // Parttion GetTupleElement of index.
  PartitionedHlo index_partitioned_gte(
      index_gte, partitioned_topk.base_shape().tuple_shapes(1),
      MakePartitioningState());
  // Reshard index to be replicated.
  auto replicated_index_gte =
      index_partitioned_gte.Reshard(replicated_sharding).hlo();

  // Creates replicated sort to do TopK, the input is value and index pairs
  // from all the partitions. The reason to use Sort instead of CustomCall TopK
  // is CustomCall only takes value as input. There will be an extra Gather
  // to get the correct index if CustomCall is used here.

  // Create comparator for the sort.
  XlaBuilder b("Sort.Compare");
  XlaComputation comparator = CreateScalarComparisonComputation(
      "compare-value-and-index", {input->shape().element_type(), S32}, {Gt, Lt},
      &b);
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, comparator.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module,
                      HloModule::CreateFromProto(comparator.proto(), config));
  HloCloneContext context(module_);
  auto compare_computation =
      module_->DeepCloneComputation(new_module->entry_computation(), &context);
  // Each partition needs to do TopK separately, thus the base shape for sort
  // becomes [ceil(batch_size / batch_dim_partition), k * shard_count].
  const Shape sort_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(
           hlo->operand(0)->shape().element_type(),
           {CeilOfRatio(batch_size, batch_dim_partition), k * shard_count}),
       ShapeUtil::MakeShape(S32, {CeilOfRatio(batch_size, batch_dim_partition),
                                  k * shard_count})});
  auto sort = b_.AddInstruction(HloInstruction::CreateSort(
      sort_shape, sort_dim, {replicated_value_gte, replicated_index_gte},
      compare_computation, true));
  sort->set_sharding(
      replicated_sharding.GetTupleSharding(sort->shape()).ValueOrDie());
  PartitionedHlo replicated_sort(sort, replicated_shape,
                                 MakePartitioningState());

  // Slice value and index from top-k for output.
  HloInstruction* sort_value_gte =
      b_.AddInstruction(HloInstruction::CreateGetTupleElement(
          replicated_sort.hlo()->shape().tuple_shapes(0), replicated_sort.hlo(),
          0));
  HloInstruction* sort_index_gte =
      b_.AddInstruction(HloInstruction::CreateGetTupleElement(
          replicated_sort.hlo()->shape().tuple_shapes(1), replicated_sort.hlo(),
          1));
  // Slice value from final sort.
  HloInstruction* slice_sort_value =
      SliceFirstK(sort_value_gte, &b_, sort_dim, k);
  // Slice index from final sort.
  HloInstruction* slice_index_value =
      SliceFirstK(sort_index_gte, &b_, sort_dim, k);
  auto create_tuple = b_.AddInstruction(
      HloInstruction::CreateTuple({slice_sort_value, slice_index_value}));
  create_tuple->set_sharding(
      replicated_sharding.GetTupleSharding(create_tuple->shape()).ValueOrDie());
  SetPartitionedHlo(
      hlo, PartitionedHlo(create_tuple, hlo->shape(), MakePartitioningState())
               .Reshard(hlo->sharding()));

  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleCustomCallSPMDInternal_RotateRight(
    HloInstruction* hlo) {
  TF_ASSIGN_OR_RETURN(auto attrs, ParseOpaqueAsAttributes(hlo));
  auto dim_it = attrs.find("dimension");
  TF_RET_CHECK(dim_it != attrs.end())
      << "No dimension attribute in SPMD rotate op";
  int64_t dim = dim_it->second;
  auto amount_it = attrs.find("amount");
  TF_RET_CHECK(amount_it != attrs.end())
      << "No amount attribute in SPMD rotate op";

  PartitionedHlo input =
      GetPartitionedHlo(hlo->operand(0)).Reshard(hlo->sharding());
  const int64_t full_size = hlo->shape().dimensions(dim);
  const int64_t shard_size = input.hlo()->shape().dimensions(dim);

  // We exclude shards that are entirely padding.
  const int64_t participating_shards = CeilOfRatio(full_size, shard_size);
  // The last included shard might still have padding on the right.
  const int64_t right_padding = participating_shards * shard_size - full_size;
  int64_t amount = amount_it->second;
  TF_RET_CHECK(amount >= 0)
      << "Rotate amount cannot be negative in SPMD rotate op";

  amount %= full_size;
  if (amount == 0) {
    SetPartitionedHlo(hlo, input);
    return Status::OK();
  }

  // First step: rotate `amount` on padded data. E.g., before
  //      012|345|678|9__     (_: padding)
  // after:
  //      678|9__|012|345     (amount: 6)
  auto rotate_with_padding = [&](int64_t rotate_amount) {
    int64_t current_size = 0;
    std::vector<HloInstruction*> concat_pieces;
    while (current_size < shard_size) {
      int64_t shard_distance =
          CeilOfRatio(rotate_amount - current_size, shard_size);
      int64_t offset_in_shard =
          shard_distance * shard_size - rotate_amount + current_size;

      int64_t halo_size =
          std::min(shard_size - offset_in_shard, shard_size - current_size);

      current_size += halo_size;
      Shape halo_shape = input.hlo()->shape();
      halo_shape.set_dimensions(dim, halo_size);
      HloInstruction* halo = input.hlo();
      if (halo_size != shard_size) {
        halo_shape.set_dimensions(dim, halo_size);
        std::vector<int64_t> slice_starts(hlo->shape().rank(), 0);
        slice_starts[dim] = offset_in_shard;
        std::vector<int64_t> slice_limits(
            input.hlo()->shape().dimensions().begin(),
            input.hlo()->shape().dimensions().end());
        slice_limits[dim] = offset_in_shard + halo_size;
        halo = b_.AddInstruction(HloInstruction::CreateSlice(
            halo_shape, halo, slice_starts, slice_limits,
            std::vector<int64_t>(halo_shape.rank(), 1)));
      }
      if (shard_distance != 0) {
        std::vector<std::pair<int64_t, int64_t>> pairs;
        hlo->sharding().tile_assignment().Each(
            [&](absl::Span<const int64_t> indices, int64_t device) {
              if (indices[dim] >= participating_shards) {
                return;
              }
              std::vector<int64_t> dst_idx(indices.begin(), indices.end());
              dst_idx[dim] += shard_distance;
              dst_idx[dim] %= participating_shards;
              pairs.emplace_back(device,
                                 hlo->sharding().tile_assignment()(dst_idx));
            });
        halo =
            collective_ops_creator_.create_cross_partition_collective_permute(
                &b_, halo, pairs, NewChannel());
      }
      concat_pieces.push_back(halo);
    }
    if (concat_pieces.size() > 1) {
      return b_.AddInstruction(HloInstruction::CreateConcatenate(
          input.hlo()->shape(), concat_pieces, dim));
    }
    return concat_pieces[0];
  };
  HloInstruction* rotated0 = rotate_with_padding(amount);
  if (right_padding == 0) {
    SetPartitionedHlo(hlo, [&] { return rotated0; });
    return Status::OK();
  }

  // Second step: perform another rotate from input, with `right_padding` added
  // to `amount`. E.g., before
  //      012|345|678|9__     (_: padding)
  // after:
  //      456|789|__0|123     (amount: 6 + 2)
  // combine (select) with first step:
  //      678|9__|012|345
  // now we get:
  //      456|789|012|3__

  HloInstruction* rotated1 = rotate_with_padding(
      (amount + right_padding) % (shard_size * participating_shards));
  HloInstruction* shard_offset = MakePartitionOffsets(
      hlo->shape(), hlo->sharding(), MakePartitioningState().partition_id, &b_,
      {dim})[dim];
  HloInstruction* iota = b_.AddInstruction(HloInstruction::CreateIota(
      ShapeUtil::ChangeElementType(rotated0->shape(), S32), dim));
  HloInstruction* selection_boundary =
      b_.AddInstruction(HloInstruction::CreateBroadcast(
          iota->shape(),
          b_.AddInstruction(HloInstruction::CreateBinary(
              shard_offset->shape(), HloOpcode::kSubtract,
              b_.AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(amount))),
              shard_offset)),
          {}));
  HloInstruction* pred = b_.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::ChangeElementType(iota->shape(), PRED), iota,
      selection_boundary, Comparison::Direction::kLt));
  SetPartitionedHlo(hlo, [&] {
    return b_.AddInstruction(HloInstruction::CreateTernary(
        rotated0->shape(), HloOpcode::kSelect, pred, rotated1, rotated0));
  });
  return Status::OK();
}

std::unique_ptr<HloInstruction> CreateCustomCallSPMDInternal_RotateRight(
    HloInstruction* input, int64_t dim, int64_t amount) {
  std::string opaque = absl::StrCat("dimension=", dim, ",amount=", amount);
  return HloInstruction::CreateCustomCall(input->shape(), {input},
                                          kSPMDOpRotateRight, opaque);
}

Status SpmdPartitioningVisitor::HandleCustomCall(HloInstruction* hlo) {
  if (hlo->custom_call_target() == "SPMDFullToShardShape") {
    // This op switches from auto partitioning to manual partitioning.
    auto input_partitioned = GetPartitionedHlo(hlo->operand(0));
    if (!EvenlyPartitions(hlo->shape(), input_partitioned.sharding())) {
      input_partitioned = input_partitioned.PadWithValue(
          CreateR0WithType(hlo->shape().element_type(), 0, &b_));
    }
    auto input = input_partitioned.hlo();
    CHECK(hlo->sharding().IsManual() || hlo->sharding().IsManualSubgroup());
    CHECK(ShapeUtil::Compatible(
        input->shape(), MakePartitionedShape(hlo->shape(), hlo->sharding())));
    auto copy = b_.AddInstruction(
        HloInstruction::CreateUnary(input->shape(), HloOpcode::kCopy, input));
    SetPartitionedHlo(hlo, [&] { return copy; });
    return Status::OK();
  }
  if (hlo->custom_call_target() == "SPMDShardToFullShape") {
    // This op switches from manual partitioning to auto partitioning.
    auto input = GetPartitionedHlo(hlo->operand(0)).hlo();
    CHECK(input->sharding().IsManual() || input->sharding().IsManualSubgroup());
    auto copy = b_.AddInstruction(
        HloInstruction::CreateUnary(input->shape(), HloOpcode::kCopy, input));
    CHECK(ShapeUtil::Compatible(
        copy->shape(), MakePartitionedShape(hlo->shape(), hlo->sharding())));
    SetPartitionedHlo(hlo, [&] { return copy; });
    return Status::OK();
  }

  if (hlo->custom_call_target() == "TopK") {
    return HandleCustomCallTopK(hlo);
  }

  if (hlo->custom_call_target() == kSPMDOpRotateRight) {
    return HandleCustomCallSPMDInternal_RotateRight(hlo);
  }

  return DefaultAction(hlo);
}

}  // namespace spmd
}  // namespace xla
