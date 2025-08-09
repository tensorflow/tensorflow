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

#include "xla/service/gpu/transforms/collectives/collective_permute_cycle_decomposer.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/source_target_pairs.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

using CycleType = collective_permute_cycle::CycleType;

using SourceTargetPairType = std::pair<int64_t, int64_t>;
using SourceTargetPairsType = std::vector<SourceTargetPairType>;

// Returns the cycle type and indices of the vertices that form cycles. For
// example, GetCycleTypeAndIndices({{0,3},{1,0},{2,1},{3,2}}) returns
// {kBackward, {0}}, since the communication pattern contains a backward cycle
// with the cycle-inducing vertex at index 0 in the input source-target pairs
// array. This function uses the assumption that, in practice, in forward
// cycles, most edges will have the target replica ID greater than the source
// replica ID except for the back edges that form cycles (similar logic applies
// to backward cycles).
std::pair<CycleType, std::set<int>> GetCycleTypeAndIndices(
    const SourceTargetPairsType& pairs) {
  std::set<int> seen_replica_ids;
  std::set<std::pair<int64_t, int64_t>> tentative_results;
  // first figure out if we're dealing with a potential forward or backward
  // cycle.
  int forward_edge_counter = 0;
  int backward_edge_counter = 0;
  for (auto pair : pairs) {
    pair.first < pair.second ? forward_edge_counter++ : backward_edge_counter++;
  }
  bool is_forward_cycle = forward_edge_counter > backward_edge_counter;
  for (int64_t i = 0; i < pairs.size(); ++i) {
    const SourceTargetPairType& pair = pairs[i];
    if (is_forward_cycle) {
      // check if the source of the current pair is smaller than the target
      if (pair.first < pair.second) {
        seen_replica_ids.insert(pair.first);
      } else {
        // the source of the current pair is larger than the target, so the
        // current pair may be part of a cycle. We keep track of the target ID
        // and the index of the pair in the original pairs array.
        tentative_results.insert(std::make_pair(pair.second, i));
      }
    } else {
      // The backward cycle check uses similar logic but in reverse.
      if (pair.first > pair.second) {
        seen_replica_ids.insert(pair.second);
      } else {
        tentative_results.insert(std::make_pair(pair.first, i));
      }
    }
  }
  std::set<int> final_results;
  // Iterate over the tentative results and only keep the indices that form an
  // actual cycle. This is done by checking if the target replica ID of the
  // pair is in the set of seen replica IDs. Note that the tentative results
  // array will be fairly small in practice, so this is not adding too much to
  // the runtime.
  for (auto& [replica_id, index] : tentative_results) {
    if (seen_replica_ids.find(replica_id) != seen_replica_ids.end()) {
      final_results.insert(index);
    }
  }
  CycleType cycle_type = final_results.empty() ? CycleType::kNone
                         : is_forward_cycle    ? CycleType::kForward
                                               : CycleType::kBackward;
  return std::make_pair(cycle_type, final_results);
}

// Returns the cycle type and indices of the vertices that form cycles. If the
// cycle type is kUnknown, the set of indices will be empty.
std::pair<CycleType, std::set<int>> GetCycleTypeAndIndicesArray(
    const HloCollectivePermuteInstruction& collective_permute,
    int64_t threshold_in_bytes) {
  if (collective_permute.operand_count() != 1) {
    return std::make_pair(CycleType::kNone, std::set<int>{});
  }

  // Skip the transformation if there is any context data.
  const Shape& result_shape = collective_permute.shape();
  if (result_shape.IsTuple()) {
    return std::make_pair(CycleType::kNone, std::set<int>{});
  }

  CHECK(result_shape.IsArray());
  if (ShapeUtil::ByteSizeOf(result_shape) < threshold_in_bytes) {
    return std::make_pair(CycleType::kNone, std::set<int>{});
  }

  const std::vector<std::pair<int64_t, int64_t>>& pairs =
      collective_permute.source_target_pairs();
  if (pairs.size() == 1) {
    return std::make_pair(CycleType::kNone, std::set<int>{});
  }

  return GetCycleTypeAndIndices(pairs);
}

// Copies the frontend attributes from the original CP and splits the
// _xla_send_recv_validation attribute;
absl::StatusOr<std::pair<FrontendAttributes, FrontendAttributes>>
DecomposeFrontendAttributes(const FrontendAttributes& orig,
                            const CycleType cycle_type) {
  FrontendAttributes attr1 = orig, attr2 = orig;
  auto it = orig.map().find(kSendRecvValidationAttr);
  if (it == orig.map().end() || it->second == "invalid") {
    return std::make_pair(attr1, attr2);
  }

  TF_ASSIGN_OR_RETURN(SourceTargetPairs bounds,
                      SourceTargetPairs::FromString(it->second));
  int64_t num_pairs = bounds.size();
  if (num_pairs < 2) {
    return Internal("Invalid number of replica groups");
  }

  // TODO: b/391377472 - this also need to be able to work with multiple cycles.
  auto [cp1_bounds, cp2_bounds] =
      collective_permute_cycle::SplitEdges(bounds, cycle_type);
  (*attr1.mutable_map())[kSendRecvValidationAttr] = cp1_bounds.ToString();
  (*attr2.mutable_map())[kSendRecvValidationAttr] = cp2_bounds.ToString();
  return std::make_pair(attr1, attr2);
}

// Adds a CollectivePermute instruction based on the original CP.
HloInstruction* AddCP(HloCollectivePermuteInstruction* orig,
                      HloComputation* computation,
                      const std::vector<std::pair<int64_t, int64_t>>& pairs,
                      absl::string_view name_suffix,
                      const FrontendAttributes& attrs,
                      const std::optional<int64_t> channel_id) {
  HloInstruction* cp1 = computation->AddInstruction(
      HloInstruction::CreateCollectivePermute(
          orig->shape(), orig->mutable_operand(0), pairs, channel_id),
      absl::StrCat(orig->name(), name_suffix));
  cp1->set_metadata(orig->metadata());
  cp1->set_frontend_attributes(attrs);
  return cp1;
}

// Creates a partition-id or replica-id instruction based on the collective
// group mode.
absl::StatusOr<HloInstruction*> CreatePartitionOrReplicaId(
    HloComputation* computation, CollectiveOpGroupMode mode,
    absl::string_view cp_name) {
  switch (mode) {
    case CollectiveOpGroupMode::kCrossReplica:
      return computation->AddInstruction(HloInstruction::CreateReplicaId(),
                                         absl::StrCat(cp_name, "-rep-id"));
    case CollectiveOpGroupMode::kCrossPartition:
      return computation->AddInstruction(HloInstruction::CreatePartitionId(),
                                         absl::StrCat(cp_name, "-part-id"));
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
    case CollectiveOpGroupMode::kFlattenedID:
      return absl::InternalError(
          absl::StrFormat("Unexpected collective group mode for %s", cp_name));
  }
}

// Decomposes a CollectivePermute instruction with a cycle in its source-target
// pairs into two CollectivePermute instructions.
absl::Status DecomposeCollectivePermuteCycle(
    HloCollectivePermuteInstruction* cp, HloComputation* computation,
    HloModule* module, int64_t next_channel_id, CycleType cycle_type,
    std::set<int> indices_to_break_out) {
  const std::vector<std::pair<int64_t, int64_t>>& pairs =
      cp->source_target_pairs();
  absl::string_view cp_name = cp->name();
  std::vector<std::pair<int64_t, int64_t>> back_pairs, fwd_pairs;
  for (int i = 0; i < pairs.size(); ++i) {
    if (indices_to_break_out.find(i) != indices_to_break_out.end()) {
      back_pairs.push_back(pairs[i]);
    } else {
      fwd_pairs.push_back(pairs[i]);
    }
  }
  TF_ASSIGN_OR_RETURN(
      CollectiveOpGroupMode mode,
      GetCollectiveOpGroupMode(cp->channel_id().has_value(), std::nullopt));

  TF_ASSIGN_OR_RETURN(auto attrs, DecomposeFrontendAttributes(
                                      cp->frontend_attributes(), cycle_type));

  // Backward edge.
  HloInstruction* back_cp =
      AddCP(cp, computation, back_pairs, "-bwd", attrs.first, cp->channel_id());

  // Forward edge.
  bool is_cross_partition = (mode == CollectiveOpGroupMode::kCrossPartition);
  std::optional<int64_t> fwd_channel_id =
      is_cross_partition ? std::optional(next_channel_id) : std::nullopt;
  HloInstruction* fwd_cp =
      AddCP(cp, computation, fwd_pairs, "-fwd", attrs.second, fwd_channel_id);

  // Calculate the received data as follows:
  //   %partition = u32[] partition-id()
  //   %bwd_recv_id = u32[] constant(bwd-recv-partition-id)
  //   compare = pred[] compare(%partition, %bwd_recv_id), direction=EQ
  //   recv-data = type[?] select(compare, cp1_done, cp2_done)
  // If the collective is across replicas, then `partition` is replaced by
  // `replica = u32[] replica-id()`.
  TF_ASSIGN_OR_RETURN(HloInstruction * partition_or_replica,
                      CreatePartitionOrReplicaId(computation, mode, cp_name));
  int64_t bwd_recv_id = back_pairs.back().second;
  HloInstruction* constant = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(U32, bwd_recv_id)),
      absl::StrCat(cp_name, "-bwd-recv-id"));
  HloInstruction* compare = computation->AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}),
                                    partition_or_replica, constant,
                                    Comparison::Direction::kEq),
      absl::StrCat(cp_name, "-cmp"));

  // Later in the pipeline, CollectivePermuteDecomposer uses post order
  //  to chain the send/recv instructions. It's important that the back
  //  edge is placed before forward edge in select operands for more optimal
  //  chaining.
  HloInstruction* recv_data = computation->AddInstruction(
      HloInstruction::CreateTernary(cp->shape(), HloOpcode::kSelect, compare,
                                    back_cp, fwd_cp),
      absl::StrCat(cp_name, "-sel"));

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
      std::pair<CycleType, std::set<int>> cycle_type_and_indices =
          GetCycleTypeAndIndicesArray(*collective_permute, threshold_in_bytes_);
      CycleType cycle_type = cycle_type_and_indices.first;
      std::set<int> indices_to_break_out = cycle_type_and_indices.second;
      if (cycle_type != CycleType::kNone) {
        if (changed == false) {
          next_channel_id = hlo_query::NextChannelId(*module);
          changed = true;
        }
        TF_RETURN_IF_ERROR(DecomposeCollectivePermuteCycle(
            collective_permute, comp, module, next_channel_id++, cycle_type,
            indices_to_break_out));
      }
    }
  }
  return changed;
}

}  // namespace xla
