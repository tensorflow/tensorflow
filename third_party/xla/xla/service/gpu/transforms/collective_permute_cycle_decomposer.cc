/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/collective_permute_cycle_decomposer.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

namespace {

using SourceTargetPair = std::pair<int64_t, int64_t>;
using SourceTargetPairs = std::vector<SourceTargetPair>;

enum class CycleType { kUnknown, kForward, kBackward };

// Returns true if the CollectivePermute instruction has a cycle in its
// source-target pairs and should be decomposed.
CycleType ShouldDecomposeWithCycleType(
    const HloCollectivePermuteInstruction& collective_permute,
    int64_t threshold_in_bytes) {
  if (collective_permute.operand_count() != 1) {
    return CycleType::kUnknown;
  }

  // Skip the transformation if there is any context data.
  const Shape& result_shape = collective_permute.shape();
  if (result_shape.IsTuple()) {
    return CycleType::kUnknown;
  }

  CHECK(result_shape.IsArray());
  if (ShapeUtil::ByteSizeOf(result_shape) < threshold_in_bytes) {
    return CycleType::kUnknown;
  }

  const SourceTargetPairs& pairs = collective_permute.source_target_pairs();
  if (pairs.size() == 1) {
    return CycleType::kUnknown;
  }

  return IsForwardCycle(pairs)    ? CycleType::kForward
         : IsBackwardCycle(pairs) ? CycleType::kBackward
                                  : CycleType::kUnknown;
}

// Constructs the frontend attributes for the two decomposed CollectivePermute
// instructions.
absl::Status GetFrontendAttributes(HloCollectivePermuteInstruction* cp,
                                   CycleType cycle_type,
                                   xla::FrontendAttributes& cp1_attr,
                                   xla::FrontendAttributes& cp2_attr) {
  cp1_attr = cp->frontend_attributes();
  cp2_attr = cp->frontend_attributes();
  auto validation_it =
      cp->frontend_attributes().map().find(kSendRecvValidationAttr);
  if (validation_it == cp->frontend_attributes().map().end() ||
      validation_it->second == "invalid") {
    return absl::OkStatus();
  }

  auto statusor_bounds = ParseReplicaGroupsOnly(validation_it->second);
  if (!statusor_bounds.ok()) {
    return statusor_bounds.status();
  }
  const std::vector<ReplicaGroup>& bounds = statusor_bounds.value();
  if (bounds.size() < 2) {
    return Internal("Invalid number of replica groups");
  }

  int64_t num_pairs = bounds.size();
  // A forward cycle has its backedge at the end while a backward cycle has its
  // backedge at the beginning.
  auto backedge_start = cycle_type == CycleType::kBackward
                            ? bounds.begin()
                            : bounds.begin() + num_pairs - 1;
  auto other_edges_start =
      cycle_type == CycleType::kBackward ? bounds.begin() + 1 : bounds.begin();
  std::vector<ReplicaGroup> cp1_bounds(backedge_start, backedge_start + 1);
  std::vector<ReplicaGroup> cp2_bounds(other_edges_start,
                                       other_edges_start + num_pairs - 1);
  auto bounds_to_string = [](const std::vector<ReplicaGroup> groups) {
    return "{" +
           absl::StrJoin(groups, ",",
                         [](std::string* out, const ReplicaGroup& value) {
                           absl::StrAppend(out, "{", value.replica_ids(0), ",",
                                           value.replica_ids(1), "}");
                         }) +
           "}";
  };
  std::string cp1_validation_str = bounds_to_string(cp1_bounds);
  std::string cp2_validation_str = bounds_to_string(cp2_bounds);
  (*cp1_attr.mutable_map())[kSendRecvValidationAttr] = cp1_validation_str;
  (*cp2_attr.mutable_map())[kSendRecvValidationAttr] = cp2_validation_str;
  return absl::OkStatus();
}

// Decomposes a CollectivePermute instruction with a cycle in its source-target
// pairs into two CollectivePermute instructions.
absl::Status DecomposeCollectivePermuteCycle(
    HloCollectivePermuteInstruction* cp, HloComputation* computation,
    HloModule* module, int64_t next_channel_id, CycleType cycle_type) {
  const SourceTargetPairs& pairs = cp->source_target_pairs();
  int64_t num_pairs = pairs.size();
  // A forward cycle has its backedge at the end as in
  // {{0,1},{1,2},{2,3},{3,0}} while a backward cycle has its backedge at the
  // beginning as in {{0,3},{1,0},{2,1},{3,2}}.
  auto backedge_start = cycle_type == CycleType::kBackward
                            ? pairs.begin()
                            : pairs.begin() + num_pairs - 1;
  auto other_edges_start =
      cycle_type == CycleType::kBackward ? pairs.begin() + 1 : pairs.begin();
  SourceTargetPairs backedge(backedge_start, backedge_start + 1);
  SourceTargetPairs other_edges(other_edges_start,
                                other_edges_start + num_pairs - 1);
  const OpMetadata& metadata = cp->metadata();
  xla::FrontendAttributes cp1_attr, cp2_attr;
  TF_RETURN_IF_ERROR(GetFrontendAttributes(cp, cycle_type, cp1_attr, cp2_attr));

  TF_ASSIGN_OR_RETURN(
      CollectiveOpGroupMode mode,
      GetCollectiveOpGroupMode(cp->channel_id().has_value(), std::nullopt));

  // Create the CollectivePermute instruction for the communication represented
  // by the backedge.
  HloInstruction* cp1 = computation->AddInstruction(
      HloInstruction::CreateCollectivePermute(
          cp->shape(), cp->mutable_operand(0), backedge, cp->channel_id()),
      "cp.backward");
  cp1->set_metadata(metadata);
  cp1->set_frontend_attributes(cp1_attr);
  int64_t bwd_recv_id = backedge.back().second;

  // Create the CollectivePermute instruction for the communication represented
  // byt other edges.
  bool is_cross_partition = (mode == CollectiveOpGroupMode::kCrossPartition);
  HloInstruction* cp2 = computation->AddInstruction(
      HloInstruction::CreateCollectivePermute(
          cp->shape(), cp->mutable_operand(0), other_edges,
          is_cross_partition ? std::optional(next_channel_id) : std::nullopt),
      "cp.forward");

  cp2->set_metadata(metadata);
  cp2->set_frontend_attributes(cp2_attr);

  // Calculate the received data as follows:
  //   %partition = u32[] partition-id()
  //   %bwd_recv_id = u32[] constant(bwd-recv-partition-id)
  //   compare = pred[] compare(%partition, %bwd_recv_id), direction=EQ
  //   recv-data = type[?] select(compare, cp1_done, cp2_done)
  // If the collective is across replicas, then `partition` is replaced by
  // `replica = u32[] replica-id()`.
  HloInstruction* partition_or_replica = nullptr;
  switch (mode) {
    case CollectiveOpGroupMode::kCrossReplica:
      partition_or_replica =
          computation->AddInstruction(HloInstruction::CreateReplicaId());
      break;
    case CollectiveOpGroupMode::kCrossPartition:
      partition_or_replica =
          computation->AddInstruction(HloInstruction::CreatePartitionId());
      break;
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
    case CollectiveOpGroupMode::kFlattenedID:
      return absl::InternalError(absl::StrFormat(
          "Unexpected collective group mode for %s", cp->name()));
  };
  HloInstruction* constant = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(U32, bwd_recv_id)));
  HloInstruction* compare =
      computation->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), partition_or_replica, constant,
          Comparison::Direction::kEq));
  HloInstruction* recv_data =
      computation->AddInstruction(HloInstruction::CreateTernary(
          cp1->shape(), HloOpcode::kSelect, compare, cp1, cp2));

  TF_RETURN_IF_ERROR(cp->ReplaceAllUsesWith(recv_data));
  TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(cp));

  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<bool> CollectivePermuteCycleDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  int64_t next_channel_id;
  for (auto comp : module->computations(execution_threads)) {
    for (auto hlo : comp->MakeInstructionPostOrder()) {
      if (HloPredicateIsNotOp<HloOpcode::kCollectivePermute>(hlo)) {
        continue;
      }
      auto collective_permute = Cast<HloCollectivePermuteInstruction>(hlo);
      CycleType cycle_type = ShouldDecomposeWithCycleType(*collective_permute,
                                                          threshold_in_bytes_);
      if (cycle_type != CycleType::kUnknown) {
        if (changed == false) {
          next_channel_id = hlo_query::NextChannelId(*module);
          changed = true;
        }
        TF_RETURN_IF_ERROR(DecomposeCollectivePermuteCycle(
            collective_permute, comp, module, next_channel_id++, cycle_type));
      }
    }
  }
  return changed;
}

}  // namespace xla
