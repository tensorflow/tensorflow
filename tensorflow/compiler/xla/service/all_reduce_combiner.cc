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

#include "tensorflow/compiler/xla/service/all_reduce_combiner.h"

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

// Combines the elements of to_combine into a single AllReduce op. All
// entries in to_combine must be AllReduce ops with exactly one operand
// and the same reduction operation.
Status CombineAllReduces(absl::Span<HloInstruction* const> to_combine) {
  if (to_combine.size() < 2) {
    return Status::OK();
  }
  VLOG(1) << "Combined " << to_combine.size() << " CRS ops";

  HloComputation& computation = *to_combine.back()->parent();
  HloComputation* reduction = to_combine[0]->to_apply();
  const HloOpcode type = reduction->root_instruction()->opcode();

  // Create a single bigger AllReduce of the operands of the smaller
  // AllReduces.
  std::vector<HloInstruction*> operands;
  std::vector<Shape> operand_shapes;
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kAllReduce);
    TF_RET_CHECK(hlo->operands().size() == 1);
    TF_RET_CHECK(hlo->to_apply() == reduction ||
                 (hlo->to_apply()->instruction_count() == 3 &&
                  hlo->to_apply()->num_parameters() == 2 &&
                  hlo->to_apply()->root_instruction()->opcode() == type));
    TF_RET_CHECK(hlo->shape().IsArray());
    for (HloInstruction* operand : hlo->operands()) {
      operands.push_back(operand);
      operand_shapes.push_back(operand->shape());
    }
  }

  HloInstruction* combined;
  // AllReduce ops with more than one operand produce a tuple.
  TF_RET_CHECK(operands.size() >= 2);
  combined = computation.AddInstruction(HloInstruction::CreateAllReduce(
      ShapeUtil::MakeTupleShape(operand_shapes), operands, reduction,
      to_combine.front()->replica_groups(),
      /*constrain_layout=*/false, to_combine.front()->channel_id(),
      Cast<HloAllReduceInstruction>(to_combine.front())
          ->use_global_device_ids()));

  // We have to propagate the sharding manually because Domain instructions are
  // not guaranteed to preserve it for side effecting instructions.
  if (to_combine.front()->has_sharding()) {
    combined->set_sharding(to_combine.front()->sharding());
  }
  VLOG(1) << "Replacing with : " << combined->ToString();

  // Replace all the smaller AllReduces with elements of the tuple output
  // of the single bigger AllReduce.
  for (int64 i = 0; i < to_combine.size(); ++i) {
    auto replace_with = HloInstruction::CreateGetTupleElement(
        to_combine[i]->shape(), combined, i);
    TF_RETURN_IF_ERROR(computation.ReplaceWithNewInstruction(
        to_combine[i], std::move(replace_with)));
  }
  return Status::OK();
}

struct GroupKey {
  GroupKey(const HloInstruction* hlo, const HloDomainMap& domain_map)
      : opcode(hlo->to_apply()->root_instruction()->opcode()),
        accum_type(hlo->to_apply()->root_instruction()->shape().element_type()),
        domain_id(domain_map.GetDomainMetadataId(hlo)),
        is_cross_shard(hlo->channel_id().has_value()),
        use_global_device_ids(
            Cast<HloAllReduceInstruction>(hlo)->use_global_device_ids()),
        replica_groups(hlo->replica_groups()) {}

  bool operator<(const GroupKey& other) const {
    if (opcode != other.opcode) {
      return opcode < other.opcode;
    }
    if (accum_type != other.accum_type) {
      return accum_type < other.accum_type;
    }
    if (domain_id != other.domain_id) {
      return domain_id < other.domain_id;
    }
    if (is_cross_shard != other.is_cross_shard) {
      return is_cross_shard < other.is_cross_shard;
    }
    if (use_global_device_ids != other.use_global_device_ids) {
      return use_global_device_ids < other.use_global_device_ids;
    }
    if (replica_groups.size() != other.replica_groups.size()) {
      return replica_groups.size() < other.replica_groups.size();
    }
    for (int64 i = 0; i < replica_groups.size(); ++i) {
      const auto& rg = replica_groups[i];
      const auto& org = other.replica_groups[i];
      if (rg.replica_ids_size() != org.replica_ids_size()) {
        return rg.replica_ids_size() < org.replica_ids_size();
      }
      for (int64 j = 0; j < rg.replica_ids_size(); ++j) {
        if (rg.replica_ids(j) != org.replica_ids(j)) {
          return rg.replica_ids(j) < org.replica_ids(j);
        }
      }
    }
    return false;
  }

  HloOpcode opcode;
  PrimitiveType accum_type;
  int64 domain_id;
  bool is_cross_shard;
  bool use_global_device_ids;
  std::vector<ReplicaGroup> replica_groups;
};

// Group AllReduce instructions by the reduction types, e.g., add, min,
// max, replica groups and domain. For cross-module all reduce instructions
// we group them by the set of domains they are reducing across.
//
// Note that the shape of the reduction computation is not included in the
// reduction types, e.g.: "f32[] add" and "bf16[] add" will be the same type. We
// need to disallow combining CRS instructions with different domain metadata as
// well as that could end up short-cutting two or more different domains.
//
// In each group, the instructions should be in post order. We will then iterate
// each group and try to combine them, so to prevent non-determinism, we use
// std::map here.
//
// The return value is a list of groups where every group contains a list of
// all-reduce instruction sets in topological order and with a deterministic
// order within the set. Additionally due to the above constraints every all
// reduce set within a group will contain the same number of elements
// and every instruction within an all reduce set will have the same
// all-reduce-id (if specified) and thus shape (all reduce sets without an
// all-reduce-id will have a single instruction).
using InstructionGroups =
    std::vector<std::vector<std::vector<HloInstruction*>>>;
StatusOr<InstructionGroups> CreateComputationGroups(
    HloComputation* computation) {
  TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));

  // Group instructions by opcode, domain id and replica group.
  std::map<GroupKey, std::vector<HloInstruction*>> opcode_groups;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kAllReduce) {
      continue;
    }
    if (instruction->to_apply()->instruction_count() != 3 ||
        instruction->to_apply()->num_parameters() != 2) {
      VLOG(1) << "Skipping due to non-trivial reduction function.";
      continue;
    }
    opcode_groups[GroupKey(instruction, *domain_map)].push_back(instruction);
  }

  // Generate a unique all-reduce-id for instructions without one by negating
  // the unique id of the hlo. This way we can treat cross module and normal CRS
  // instructions uniformly.
  auto channel_id = [](const HloInstruction* all_reduce) {
    return all_reduce->IsCrossModuleAllReduce()
               ? all_reduce->channel_id().value()
               : -1 * all_reduce->unique_id();
  };

  // Group instructions by all-reduce id with instructions for an all-reduce id
  // is listed along their group id and the (group id, instruction) pairs are
  // sorted by group id in the vector.
  std::map<int64, std::vector<std::pair<int64, HloInstruction*>>>
      all_reduce_sets;
  int64 group_id = 0;
  for (auto& domain_groups : opcode_groups) {
    for (HloInstruction* hlo : domain_groups.second) {
      all_reduce_sets[channel_id(hlo)].emplace_back(group_id, hlo);
    }
    ++group_id;
  }

  // Group instructions by participating group ids. Instructions within a group
  // are sorted by topological order and instructions within an all reduce group
  // is still sorted by group id.
  std::map<std::vector<int64>, std::vector<std::vector<HloInstruction*>>>
      all_reduce_group_map;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kAllReduce) {
      continue;
    }
    if (instruction->to_apply()->instruction_count() != 3 ||
        instruction->to_apply()->num_parameters() != 2) {
      VLOG(1) << "Skipping due to non-trivial reduction function.";
      continue;
    }

    int64 arid = channel_id(instruction);
    if (all_reduce_sets.count(arid) == 0) {
      // Already processed.
      continue;
    }

    std::vector<int64> group_ids;
    std::vector<HloInstruction*> instructions;
    for (const auto& hlo : all_reduce_sets[arid]) {
      group_ids.push_back(hlo.first);
      instructions.push_back(hlo.second);
    }
    all_reduce_group_map[group_ids].push_back(std::move(instructions));
    all_reduce_sets.erase(arid);
  }
  CHECK(all_reduce_sets.empty());

  InstructionGroups groups;
  for (const auto& all_reduce_group : all_reduce_group_map) {
    groups.push_back(all_reduce_group.second);
  }
  return std::move(groups);
}

}  // namespace

AllReduceCombiner::AllReduceCombiner(int64 combine_threshold_in_bytes,
                                     int64 combine_threshold_count)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count) {}

StatusOr<bool> AllReduceCombiner::Run(HloModule* module) {
  VLOG(1) << "Running AllReduceCombiner with threshold of "
          << combine_threshold_in_bytes_ << " bytes";

  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1) << "Skip AllReduceCombiner because the module contains all-reduce "
               "with constrained layouts";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(auto groups, CreateComputationGroups(computation));
    for (auto group : groups) {
      // Recompute reachability after every combine group because we can't
      // maintain a cross group topolgical order to be able to rely on the
      // transitive dependencies to detect cycles.
      auto reachability = HloReachabilityMap::Build(computation);

      // Create a map to be able to find an instruction group based on the first
      // instruction in the group. It will be used during the post order
      // iteration to be able to process full groups at a time. Doing it only
      // for one instruction in every group will be sufficient because all
      // instruction have to schedule at the same time due to cross core
      // dependencies.
      absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>*>
          group_map;
      for (auto& instruction : group) {
        group_map[instruction.front()] = &instruction;
      }

      // Collect sets of AllReduce instructions to combine.
      std::vector<std::vector<std::vector<HloInstruction*>>> combine_sets(1);
      int64 current_size_in_bytes = 0;
      int64 current_operand_count = 0;

      // Iterate all instructions in post order and skip the ones not in the
      // current group. We have to create a new post order iteration for every
      // group because merging instructions in the previous group can made the
      // original post order no longer hold.
      // This will make it likely that we won't increase memory pressure much
      // above combine_threshold_in_bytes, since two AllReduces that are
      // near in post order are most likely, but not for sure, also near in
      // scheduled order.
      //
      // TODO(b/70235266): This should usually be fine, but it's probably
      // possible to construct some case where the memory usage increases beyond
      // the threshold due to reordering of the instructions in scheduling. If
      // this ever comes up as a real problem, it would be nice to implement
      // safeguards so that that cannot possibly happen.
      for (const HloInstruction* inst :
           computation->MakeInstructionPostOrder()) {
        auto it = group_map.find(inst);
        if (it == group_map.end()) {
          // Instruction belongs to a different group.
          continue;
        }
        const auto& instructions = *it->second;

        VLOG(1) << "Considering HLO " << instructions.front()->ToString()
                << " with current set size of " << current_size_in_bytes
                << " and current operand count of " << current_operand_count;

        // We do not handle AllReduce ops that do not have exactly 1
        // operand since that is simpler and this pass is the only way to
        // generate such ops and it should rarely be important to consider the
        // same ops again.
        if (instructions.front()->operands().size() != 1) {
          VLOG(1) << "Skipping due to "
                  << instructions.front()->operands().size() << " operands";
          continue;
        }

        int64 size_in_bytes;
        TF_RET_CHECK(instructions.front()->shape().IsArray());
        size_in_bytes = ShapeUtil::ByteSizeOf(instructions.front()->shape());

        if (size_in_bytes > combine_threshold_in_bytes_) {
          VLOG(1) << "Skipping due to size " << size_in_bytes
                  << " above threshold";
          // If the instruction is greather than the threshold, then we can
          // never combine it with anything.
          continue;
        }

        // If the current set is dependent on the instruction, then create a new
        // one to avoid the dependency. We move on from the current set instead
        // of ignoring the instruction since otherwise a single AllReduce
        // instruction that all the other ones depend on (such as one on the
        // forward pass of a model) could disable this optimization entirely.
        TF_RET_CHECK(!combine_sets.empty());
        for (const auto& previous : combine_sets.back()) {
          // The reachability information does not reflect the planned
          // combination from combine_sets. We cannot just bring it up to date
          // cheaply since HloReachabilityMap does not track reachability
          // updates transitively and doing it directly is expensive. However,
          // leaving it stale has no effect on the reachability queries that we
          // are doing here because we are considering the ops in a topological
          // order, so we can just leave it stale.
          //
          // Proof: Suppose A is the instruction we are looking to combine and B
          // is an element of the current combine set that we are looking to
          // combine A into.
          //
          // First of all, we check that all elements in each set do not depend
          // on each other, so combining the *current* combine set cannot create
          // new dependencies between A and B. It remains to prove that
          // combining the prior combine sets also cannot create a dependency
          // between A and B.
          //
          // Assume to get a contradiction that there are two AllReduce
          // ops C and D in combine_sets that will be combined and that A and B
          // are not connected now but that they will be after combining C and
          // D. Then there exist paths in the dependency graph such that one of
          // these cases is true:
          //
          //   A -> ... -> C and D -> ... -> B
          //   A -> ... -> D and C -> ... -> B
          //   B -> ... -> C and D -> ... -> A
          //   B -> ... -> D and C -> ... -> A
          //
          // None of these cases are possible because we are visiting the nodes
          // in a topological order, so C and D cannot be in-between A and B.
          // That is a contradiction, so combining the prior combine sets also
          // cannot create a dependency between A and B.
          bool new_set = false;
          for (int64 i = 0; i < instructions.size(); ++i) {
            if (reachability->IsReachable(previous[i], instructions[i])) {
              VLOG(1) << "Starting new set due to dependency between "
                      << previous[i]->ToString() << " AND "
                      << instructions[i]->ToString();
              new_set = true;
              break;
            }
          }
          if (new_set) {
            combine_sets.emplace_back();
            current_size_in_bytes = 0;
            current_operand_count = 0;
            break;
          }
        }

        if (current_size_in_bytes + size_in_bytes >
                combine_threshold_in_bytes_ ||
            current_operand_count + 1 > combine_threshold_count_) {
          VLOG(1) << "The instruction cannot be entered into the set due "
                     "to the combined size being too large.";
          // In this case we cannot include the instruction into the current set
          // since then it would grow beyond the threshold. The set of
          // instructions to carry forward will either be the current set or the
          // instruction by itself, whichever is smaller, since that maximizes
          // the chance of being able to combine with the next instruction.
          if (size_in_bytes > current_size_in_bytes) {
            VLOG(1) << "Skipping as the instruction is larger than the set.";
            continue;  // keep the current set
          }
          VLOG(1)
              << "Resetting the set as the set is larger than the instruction.";
          combine_sets.emplace_back();
          current_size_in_bytes = 0;
          current_operand_count = 0;
        }

        VLOG(1) << "Adding instruction to set.";
        combine_sets.back().push_back(instructions);
        current_size_in_bytes += size_in_bytes;
        current_operand_count += 1;
        TF_RET_CHECK(current_size_in_bytes <= combine_threshold_in_bytes_);
        TF_RET_CHECK(current_operand_count <= combine_threshold_count_);
      }
      VLOG(1) << "Done constructing sets. Final set size is "
              << current_size_in_bytes << " bytes and " << current_operand_count
              << " operands";

      // Combine the collected sets of AllReduce instructions.
      for (const auto& combine_set : combine_sets) {
        if (combine_set.size() >= 2) {
          changed = true;
          for (int64 i = 0; i < combine_set.front().size(); ++i) {
            std::vector<HloInstruction*> to_combine;
            to_combine.reserve(combine_set.size());
            for (const auto& c : combine_set) {
              to_combine.push_back(c[i]);
            }
            TF_RETURN_IF_ERROR(CombineAllReduces(to_combine));
          }
        }
      }
    }
  }

  return changed;
}

}  // namespace xla
