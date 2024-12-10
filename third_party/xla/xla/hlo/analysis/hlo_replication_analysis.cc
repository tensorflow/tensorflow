/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/analysis/hlo_replication_analysis.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {
// When cross_partition_spmd is true, returns the partition IDs of all
// replica groups in which a given replica participates. Specfically, the k-th
// element of the outermost vector in the returned data structure holds the
// partition IDs converted from the global IDs in a collective's
// replica_groups field for replica k.
//
// When cross_partition_spmd is false, returns the replica IDs of all
// replica groups in which a given partition participates. Specfically, the k-th
// element of the outermost vector in the returned data structure holds the
// replica IDs converted from the global IDs in a collective's replica_groups
// field for partition k.
std::vector<absl::flat_hash_set<std::vector<int64_t>>> GroupsForReplicas(
    absl::Span<const ReplicaGroup> groups, int64_t num_partitions,
    int64_t replica_count, bool cross_partition_spmd) {
  int64_t num_replicas = cross_partition_spmd ? replica_count : num_partitions;
  std::vector<absl::flat_hash_set<std::vector<int64_t>>> groups_for_replicas(
      num_replicas);
  for (const ReplicaGroup& group : groups) {
    absl::flat_hash_map<int64_t, std::vector<int64_t>> id_to_ids;
    for (int64_t id : group.replica_ids()) {
      int64_t rid = id / num_partitions;
      int64_t pid = id % num_partitions;
      if (cross_partition_spmd) {
        id_to_ids[rid].push_back(pid);
      } else {
        id_to_ids[pid].push_back(rid);
      }
    }
    for (const auto& [id, ids] : id_to_ids) {
      groups_for_replicas[id].emplace(std::move(ids));
    }
  }
  return groups_for_replicas;
}

// Returns a set of vectors such that two ints appear in the same vector iff
// they appear in the same vector of both groups0 and groups1.
//
// Example:
//
//  groups0 = {{0,1,2,3},{4,5,6,7}}
//  groups1 = {{0,3,4,5},{1,2,6,7}}
//
//  MergeGroups(groups0, groups1) returns {{0,3},{1,2},{4,5},{6,7}}.
absl::flat_hash_set<std::vector<int64_t>> MergeGroups(
    const absl::flat_hash_set<std::vector<int64_t>>& groups0,
    const absl::flat_hash_set<std::vector<int64_t>>& groups1) {
  absl::flat_hash_map<int64_t, int64_t> groups1_idx_for_target;
  int64_t groups1_idx = 0;
  for (std::vector<int64_t> group1 : groups1) {
    for (int64_t element : group1) {
      groups1_idx_for_target[element] = groups1_idx;
    }
    ++groups1_idx;
  }

  absl::flat_hash_map<std::pair<int64_t, int64_t>, std::vector<int64_t>>
      groups_idxs_to_elements;
  int64_t groups0_idx = 0;
  for (std::vector<int64_t> group0 : groups0) {
    for (int64_t element : group0) {
      int64_t groups1_idx = groups1_idx_for_target[element];
      groups_idxs_to_elements[{groups0_idx, groups1_idx}].push_back(element);
    }
    ++groups0_idx;
  }
  absl::flat_hash_set<std::vector<int64_t>> merged_groups;
  for (const auto& [pair, group] : groups_idxs_to_elements) {
    merged_groups.emplace(std::move(group));
  }
  return merged_groups;
}
}  // namespace

// Determines whether an HLO instruction is replicated at index based on current
// knowledge in hlo_replication. When cross_partition_spmd is true, the
// instruction must be replicated across all partitions on each replica.
// Similarly, when cross_partition_spmd is false, the instruction must be
// replicated across all replicas on each partition.
HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::DetermineHloInstructionIsReplicated(
    const HloInstruction* hlo, const ShapeIndex& index,
    bool cross_partition_spmd,
    const absl::flat_hash_map<const HloInstruction*, ShapeTree<HloReplication>>&
        hlo_replication,
    bool support_partial_replication) {
  const auto merge_operand_replication = [&hlo_replication](
                                             const HloInstruction* inst) {
    HloReplication replication = HloReplication::ReplicatedOnAllDevices();
    for (auto operand : inst->operands()) {
      auto operand_it = hlo_replication.find(operand);
      if (operand_it == hlo_replication.end()) {
        replication = replication.Merge(HloReplication::UniqueOnAllDevices());
      } else {
        replication = replication.Merge(operand_it->second.element({}));
      }
    }
    return replication;
  };

  if (hlo->opcode() == HloOpcode::kAllReduce ||
      hlo->opcode() == HloOpcode::kAllGather) {
    // All-reduce/all-gather returns same values across partitions/replicas as
    // long as its operands are replicated.
    HloReplication replication = merge_operand_replication(hlo);
    if (replication.IsReplicatedOnAllDevices()) {
      return replication;
    }
    if (!hlo->channel_id().has_value()) {
      // This is cross-replica-only.
      if (cross_partition_spmd) {
        return replication;
      }
      if (hlo->replica_groups().empty() || hlo->replica_groups().size() == 1) {
        return HloReplication::ReplicatedOnAllDevices();
      }
      if (support_partial_replication) {
        const int64_t num_partitions =
            hlo->GetModule()->config().num_partitions();
        std::vector<absl::flat_hash_set<std::vector<int64_t>>>
            groups_for_replicas(num_partitions);
        for (const ReplicaGroup& replica_group : hlo->replica_groups()) {
          std::vector<int64_t> group_for_replica;
          for (auto id : replica_group.replica_ids()) {
            group_for_replica.push_back(id);
          }
          groups_for_replicas[0].insert(group_for_replica);
        }
        std::fill(groups_for_replicas.begin() + 1, groups_for_replicas.end(),
                  groups_for_replicas.front());
        return HloReplication::PartiallyReplicated(groups_for_replicas);
      } else {
        return HloReplication::UniqueOnAllDevices();
      }
    } else {
      bool global_id;
      if (hlo->opcode() == HloOpcode::kAllReduce) {
        global_id = Cast<HloAllReduceInstruction>(hlo)->use_global_device_ids();
      } else {
        global_id = Cast<HloAllGatherInstruction>(hlo)->use_global_device_ids();
      }
      if (global_id) {
        const int64_t num_partitions =
            hlo->GetModule()->config().num_partitions();
        const int64_t replica_count =
            hlo->GetModule()->config().replica_count();
        std::vector<absl::flat_hash_set<std::vector<int64_t>>>
            groups_for_replicas =
                GroupsForReplicas(hlo->replica_groups(), num_partitions,
                                  replica_count, cross_partition_spmd);

        // In the fully replicated case, there is one set of partition or
        // replica IDs on each replica or partition. Since the flattened ID
        // replica groups must contain every device, the size of the set is the
        // number of partitions or replicas.
        bool fully_replicated = true;
        for (auto groups_for_replica : groups_for_replicas) {
          fully_replicated &=
              groups_for_replica.size() == 1 &&
              (*groups_for_replica.begin()).size() ==
                  (cross_partition_spmd ? num_partitions : replica_count);
        }
        if (fully_replicated) {
          return HloReplication::ReplicatedOnAllDevices();
        } else if (support_partial_replication) {
          return HloReplication::PartiallyReplicated(groups_for_replicas);
        } else {
          return HloReplication::UniqueOnAllDevices();
        }
      }
      if (cross_partition_spmd) {
        return HloReplication::ReplicatedOnAllDevices();
      }
      if (hlo->replica_groups().empty() || hlo->replica_groups().size() == 1) {
        return HloReplication::ReplicatedOnAllDevices();
      } else {
        return HloReplication::UniqueOnAllDevices();
      }
    }
  }
  if (hlo->HasSideEffectNoRecurse()) {
    return HloReplication::UniqueOnAllDevices();
  }
  if (hlo->opcode() == HloOpcode::kReplicaId) {
    // ReplicaId returns the same value for all partitions in each replica.
    return cross_partition_spmd ? HloReplication::ReplicatedOnAllDevices()
                                : HloReplication::UniqueOnAllDevices();
  }
  if (hlo->opcode() == HloOpcode::kPartitionId) {
    // PartitionId returns the same value for all replicas in each partition.
    return cross_partition_spmd ? HloReplication::UniqueOnAllDevices()
                                : HloReplication::ReplicatedOnAllDevices();
  }
  auto it = hlo_replication.find(hlo);
  if (hlo->opcode() == HloOpcode::kParameter) {
    // Parameters should have been processed.
    CHECK(it != hlo_replication.end());
    return it->second.element(index);
  }
  if (it != hlo_replication.end() &&
      it->second.element(index).IsUniqueOnAllDevices()) {
    // The HLO is already marked as non-replicated.
    return it->second.element(index);
  }

  if (hlo->opcode() == HloOpcode::kConstant) {
    return HloReplication::ReplicatedOnAllDevices();
  }

  if (hlo->opcode() == HloOpcode::kCustomCall &&
      (hlo->custom_call_target() == "X64SplitLow" ||
       hlo->custom_call_target() == "X64SplitHigh" ||
       hlo->custom_call_target() == "X64Combine")) {
    return merge_operand_replication(hlo);
  }

  // Pattern-match and process cases where the HLO is partially replicated.
  if (support_partial_replication) {
    // Below is a very specific pattern to match the SPMD pipeline case.
    if (hlo->opcode() == HloOpcode::kDynamicSlice) {
      const HloInstruction* ds_buffer = hlo->operand(0);
      if (hlo->dynamic_slice_sizes().size() == 1 &&
          hlo->dynamic_slice_sizes()[0] == 1 &&
          ds_buffer->opcode() == HloOpcode::kConstant &&
          ds_buffer->shape().rank() == 1 &&
          ds_buffer->shape().element_type() == PrimitiveType::S32 &&
          ((cross_partition_spmd &&
            hlo->operand(1)->opcode() == HloOpcode::kPartitionId) ||
           (!cross_partition_spmd &&
            hlo->operand(1)->opcode() == HloOpcode::kReplicaId))) {
        const HloModule* hlo_module = hlo->GetModule();
        int64_t num_devices = cross_partition_spmd
                                  ? hlo_module->config().num_partitions()
                                  : hlo_module->config().replica_count();
        absl::flat_hash_map<int64_t, std::vector<int64_t>> value_to_device_set;
        for (int64_t device_id = 0; device_id < num_devices; ++device_id) {
          std::optional<int64_t> value =
              ds_buffer->literal().GetIntegralAsS64({device_id});
          value_to_device_set[*value].push_back(device_id);
        }
        int64_t num_replicas = cross_partition_spmd
                                   ? hlo_module->config().replica_count()
                                   : hlo_module->config().num_partitions();
        std::vector<absl::flat_hash_set<std::vector<int64_t>>>
            groups_for_replicas(num_replicas);
        for (const auto& value_and_device_set : value_to_device_set) {
          groups_for_replicas[0].insert(value_and_device_set.second);
        }
        std::fill(groups_for_replicas.begin() + 1, groups_for_replicas.end(),
                  groups_for_replicas.front());
        return HloReplication::PartiallyReplicated(groups_for_replicas);
      }
    }
  }

  if (hlo->IsElementwise() ||                             //
      hlo->opcode() == HloOpcode::kConcatenate ||         //
      hlo->opcode() == HloOpcode::kConvolution ||         //
      hlo->opcode() == HloOpcode::kDot ||                 //
      hlo->opcode() == HloOpcode::kReduce ||              //
      hlo->opcode() == HloOpcode::kBroadcast ||           //
      hlo->opcode() == HloOpcode::kTranspose ||           //
      hlo->opcode() == HloOpcode::kReshape ||             //
      hlo->opcode() == HloOpcode::kBitcast ||             //
      hlo->opcode() == HloOpcode::kReverse ||             //
      hlo->opcode() == HloOpcode::kGather ||              //
      hlo->opcode() == HloOpcode::kScatter ||             //
      hlo->opcode() == HloOpcode::kIota ||                //
      hlo->opcode() == HloOpcode::kPad ||                 //
      hlo->opcode() == HloOpcode::kSlice ||               //
      hlo->opcode() == HloOpcode::kDynamicSlice ||        //
      hlo->opcode() == HloOpcode::kDynamicUpdateSlice ||  //
      hlo->opcode() == HloOpcode::kReduceWindow ||        //
      hlo->opcode() == HloOpcode::kCopy) {
    return merge_operand_replication(hlo);
  }
  return HloReplication::UniqueOnAllDevices();
}

bool HloReplicationAnalysis::ComputeHloReplicationOnComputation(
    const HloComputation* computation, bool mark_everything_not_replicated) {
  bool changed = false;
  for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
    // Assigns the shape tree to dest if dest doesn't have one yet, or combines
    // it with the existing one by and'ing them. Returns if anything is updated.
    auto assign_or_combine_shapetree =
        [&](ShapeTree<HloReplication>&& to_combine,
            const HloInstruction* dest) {
          auto it = hlo_replication_.find(dest);
          if (it == hlo_replication_.end()) {
            hlo_replication_[dest] = std::move(to_combine);
            return true;
          }
          bool updated = false;
          it->second.ForEachMutableElement(
              [&](const ShapeIndex& index, HloReplication* element) {
                HloReplication new_replication =
                    element->Merge(to_combine.element(index));
                if (!element->Equal(new_replication)) {
                  *element = std::move(new_replication);
                  updated = true;
                }
              });
          return updated;
        };
    // Assigns or combines source's shape tree to dest. Returns if anything is
    // updated.
    auto propagate_shapetree = [&](const HloInstruction* source,
                                   const HloInstruction* dest) {
      auto source_it = hlo_replication_.find(source);
      if (source_it == hlo_replication_.end()) {
        return false;
      }
      return assign_or_combine_shapetree(
          ShapeTree<HloReplication>(source_it->second), dest);
    };
    // For the opcodes below that we do special handling, we don't need to
    // explicitly check mark_everything_not_replicated because if it is set, the
    // operands should already be marked as not replicated.
    if (inst->opcode() == HloOpcode::kWhile) {
      // Since while body's input and output alias each other, we need to run it
      // multiple times until a fixed point is reached.
      while (true) {
        // First, propagate the input's and body root's shape trees to the
        // parameters of the body and condition.
        bool updated = propagate_shapetree(
            inst->operand(0),
            inst->while_condition()->parameter_instruction(0));
        updated |= propagate_shapetree(
            inst->while_body()->root_instruction(),
            inst->while_condition()->parameter_instruction(0));
        updated |= propagate_shapetree(
            inst->operand(0), inst->while_body()->parameter_instruction(0));
        updated |=
            propagate_shapetree(inst->while_body()->root_instruction(),
                                inst->while_body()->parameter_instruction(0));
        // Compute the condition.
        updated |= ComputeHloReplicationOnComputation(
            inst->while_condition(), mark_everything_not_replicated);
        // Compute the body. If the condition is not replicated, the while body
        // should be different across replicas.
        if (!ContainsKey(loops_known_with_same_iterations_, inst) &&
            !hlo_replication_[inst->while_condition()->root_instruction()]
                 .element({})
                 .IsReplicatedOnAllDevices()) {
          updated |= ComputeHloReplicationOnComputation(
              inst->while_body(), /*mark_everything_not_replicated=*/true);
        } else {
          updated |= ComputeHloReplicationOnComputation(
              inst->while_body(), mark_everything_not_replicated);
        }
        if (!updated) {
          break;
        }
        changed = true;
      }
      // Propagate the input's and body root's shape trees to the while HLO.
      changed |= propagate_shapetree(inst->operand(0), inst);
      changed |=
          propagate_shapetree(inst->while_body()->root_instruction(), inst);
    } else if (inst->opcode() == HloOpcode::kCall ||
               inst->opcode() == HloOpcode::kFusion) {
      auto called = inst->called_computations().front();
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        changed |= propagate_shapetree(inst->operand(i),
                                       called->parameter_instruction(i));
      }
      changed |= ComputeHloReplicationOnComputation(
          called, mark_everything_not_replicated);
      changed |= propagate_shapetree(called->root_instruction(), inst);
    } else if (inst->opcode() == HloOpcode::kConditional) {
      // Propagate inputs' shape trees to the called computations' parameters.
      for (int64_t i = 0; i < inst->called_computations().size(); ++i) {
        changed |= propagate_shapetree(
            inst->operand(i + 1),
            inst->called_computations()[i]->parameter_instruction(0));
      }
      // If the condition is not replicated, the conditional result should be
      // different across replicas.
      if (!hlo_replication_[inst->operand(0)]
               .element({})
               .IsReplicatedOnAllDevices()) {
        for (auto called : inst->called_computations()) {
          changed |= ComputeHloReplicationOnComputation(
              called,
              /*mark_everything_not_replicated=*/true);
        }
        changed |= assign_or_combine_shapetree(
            ShapeTree<HloReplication>(inst->shape(),
                                      HloReplication::UniqueOnAllDevices()),
            inst);
      } else {
        for (auto called : inst->called_computations()) {
          changed |= ComputeHloReplicationOnComputation(
              called, mark_everything_not_replicated);
          changed |= propagate_shapetree(called->root_instruction(), inst);
        }
      }
    } else if (inst->opcode() == HloOpcode::kTuple) {
      ShapeTree<HloReplication> shape_tree(
          inst->shape(), HloReplication::ReplicatedOnAllDevices());
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        shape_tree.CopySubtreeFrom(hlo_replication_[inst->operand(i)], {}, {i});
      }
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kOptimizationBarrier) {
      ShapeTree<HloReplication> shape_tree = hlo_replication_[inst->operand(0)];
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
      ShapeTree<HloReplication> shape_tree(
          inst->shape(), HloReplication::ReplicatedOnAllDevices());
      shape_tree.CopySubtreeFrom(hlo_replication_[inst->operand(0)],
                                 {inst->tuple_index()}, {});
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kInfeed && cross_partition_spmd_) {
      ShapeTree<HloReplication> shape_tree(
          inst->shape(), HloReplication::UniqueOnAllDevices());
      if (inst->has_sharding()) {
        auto sharding = inst->sharding().GetAsShapeTree(inst->shape());
        shape_tree.ForEachMutableElement(
            [&sharding](const ShapeIndex& index, HloReplication* data) {
              *data = sharding.element(index).IsReplicated()
                          ? HloReplication::ReplicatedOnAllDevices()
                          : HloReplication::UniqueOnAllDevices();
            });
      }
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else {
      if (mark_everything_not_replicated) {
        changed |= assign_or_combine_shapetree(
            ShapeTree<HloReplication>(inst->shape(),
                                      HloReplication::UniqueOnAllDevices()),
            inst);
      } else {
        ShapeTree<HloReplication> shape_tree(
            inst->shape(), HloReplication::ReplicatedOnAllDevices());
        ShapeUtil::ForEachSubshape(
            inst->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
              *shape_tree.mutable_element(index) =
                  DetermineHloInstructionIsReplicated(
                      inst, index, cross_partition_spmd_, hlo_replication_,
                      support_partial_replication_);
            });
        changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
      }
    }
  }
  return changed;
}

absl::Status HloReplicationAnalysis::ComputeHloReplication() {
  // Add entry parameters to the above sets according to user annotation.
  // Replicated modules read from `parameter_replicated_at_leaf_buffers` whereas
  // SPMD partitioned modules read from HloSharding attributes.
  auto entry = module_->entry_computation();
  for (int i = 0; i < entry->num_parameters(); ++i) {
    auto param = entry->parameter_instruction(i);
    ShapeTree<HloReplication> shape_tree(param->shape(),
                                         HloReplication::UniqueOnAllDevices());
    const auto& replication = param->parameter_replicated_at_leaf_buffers();
    int leaf_index = 0;
    absl::Status status = ShapeUtil::ForEachSubshapeWithStatus(
        param->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (!ShapeUtil::IsLeafIndex(param->shape(), index)) {
            return absl::OkStatus();
          }
          if (cross_partition_spmd_ && param->has_sharding()) {
            // In cross-partition spmd mode, set parameter replication status
            // based on the parameter's sharding.
            TF_ASSIGN_OR_RETURN(auto sharding_tree,
                                param->sharding().AsShapeTree(param->shape()));
            *shape_tree.mutable_element(index) =
                sharding_tree.element(index).IsReplicated()
                    ? HloReplication::ReplicatedOnAllDevices()
                    : HloReplication::UniqueOnAllDevices();
          }
          if (replication) {
            // If parameter replication status has been set explicitly, use that
            // instead.
            if (!cross_partition_spmd_ && (*replication)[leaf_index]) {
              // Setting parameter replication status for replicas in
              // non cross-partition spmd mode.
              *shape_tree.mutable_element(index) =
                  HloReplication::ReplicatedOnAllDevices();
            }
            if (cross_partition_spmd_ && !(*replication)[leaf_index]) {
              // Setting paramemter replication status for partitions in
              // cross-partition spmd mode.
              *shape_tree.mutable_element(index) =
                  HloReplication::UniqueOnAllDevices();
            }
            ++leaf_index;
          }
          return absl::OkStatus();
        });
    TF_RETURN_IF_ERROR(status);
    hlo_replication_[param] = std::move(shape_tree);
  }
  ComputeHloReplicationOnComputation(entry,
                                     /*mark_everything_not_replicated=*/false);
  return absl::OkStatus();
}

bool HloReplicationAnalysis::HloInstructionIsReplicatedAt(
    const HloInstruction* inst, const ShapeIndex& index) const {
  auto it = hlo_replication_.find(inst);
  if (it == hlo_replication_.end()) {
    return false;
  }
  return it->second.element(index).IsReplicatedOnAllDevices();
}

bool HloReplicationAnalysis::HloInstructionIsReplicatedAt(
    const HloInstruction* inst, const ShapeIndex& index,
    absl::Span<const ReplicaGroup> replica_groups) const {
  auto it = hlo_replication_.find(inst);
  if (it == hlo_replication_.end()) {
    return false;
  }
  VLOG(5) << "HloInstructionIsReplicatedAt is called on " << inst->name()
          << ", index: " << index.ToString()
          << ", replication: " << it->second.element(index).ToString();
  if (replica_groups.empty()) {
    return it->second.element(index).IsReplicatedOnAllDevices();
  }
  if (it->second.element(index).IsReplicatedOnAllDevices()) {
    return true;
  }
  if (it->second.element(index).IsUniqueOnAllDevices()) {
    return false;
  }
  for (const ReplicaGroup& replica_group : replica_groups) {
    if (!it->second.element(index).IsReplicatedWithinSubgroup(
            replica_group.replica_ids())) {
      return false;
    }
  }
  return true;
}

/* static */ absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::Run(const HloModule* module,
                            bool cross_partition_spmd) {
  const absl::flat_hash_set<const HloInstruction*> empty;
  return Run(module, cross_partition_spmd, &empty);
}

/* static */ absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::Run(const HloModule* module, bool cross_partition_spmd,
                            const absl::flat_hash_set<const HloInstruction*>*
                                loops_known_with_same_iterations) {
  auto analysis = absl::WrapUnique(new HloReplicationAnalysis(
      module, cross_partition_spmd, loops_known_with_same_iterations,
      /*support_partial_replication=*/false));
  TF_RETURN_IF_ERROR(analysis->ComputeHloReplication());
  return analysis;
}

/* static */ absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::RunWithPartialReplication(const HloModule* module,
                                                  bool cross_partition_spmd) {
  const absl::flat_hash_set<const HloInstruction*> empty;
  auto analysis = absl::WrapUnique(
      new HloReplicationAnalysis(module, cross_partition_spmd, &empty,
                                 /*support_partial_replication=*/true));
  TF_RETURN_IF_ERROR(analysis->ComputeHloReplication());
  return analysis;
}

HloReplicationAnalysis::HloReplication::HloReplication()
    : state_(State::kReplicatedOnAllDevices) {}

HloReplicationAnalysis::HloReplication::HloReplication(
    HloReplicationAnalysis::HloReplication::State state,
    absl::Span<const absl::flat_hash_set<std::vector<int64_t>>>
        groups_for_replicas)
    : state_(state),
      groups_for_replicas_(groups_for_replicas.begin(),
                           groups_for_replicas.end()) {
  CHECK(state == State::kPartiallyReplicated || groups_for_replicas_.empty());
}

HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::ReplicatedOnAllDevices() {
  return HloReplication(State::kReplicatedOnAllDevices, {});
}

HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::UniqueOnAllDevices() {
  return HloReplication(State::kUniqueOnAllDevices, {});
}

HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::PartiallyReplicated(
    absl::Span<const absl::flat_hash_set<std::vector<int64_t>>>
        groups_for_replicas) {
  return HloReplication(State::kPartiallyReplicated, groups_for_replicas);
}

HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::Merge(
    const HloReplication& other) const {
  switch (state_) {
    case State::kReplicatedOnAllDevices:
      return other;
    case State::kUniqueOnAllDevices:
      return *this;
    case State::kPartiallyReplicated: {
      switch (other.state_) {
        case State::kReplicatedOnAllDevices:
          return *this;
        case State::kUniqueOnAllDevices:
          return other;
        case State::kPartiallyReplicated: {
          CHECK_EQ(groups_for_replicas_.size(),
                   other.groups_for_replicas_.size());
          // Merge the groups for each replica or partition.
          std::vector<absl::flat_hash_set<std::vector<int64_t>>>
              merged_groups_for_replicas(groups_for_replicas_.size());
          std::transform(groups_for_replicas_.begin(),
                         groups_for_replicas_.end(),
                         other.groups_for_replicas_.begin(),
                         merged_groups_for_replicas.begin(), MergeGroups);
          return PartiallyReplicated(merged_groups_for_replicas);
        }
      }
    }
  }
}

bool HloReplicationAnalysis::HloReplication::Equal(
    const HloReplication& other) const {
  if (state_ != other.state_) {
    return false;
  }
  CHECK_EQ(groups_for_replicas_.size(), other.groups_for_replicas_.size());
  for (int k = 0; k < groups_for_replicas_.size(); ++k) {
    if (groups_for_replicas_[k] != other.groups_for_replicas_[k]) {
      return false;
    }
  }
  return true;
}

bool HloReplicationAnalysis::HloReplication::IsReplicatedOnAllDevices() const {
  return state_ == State::kReplicatedOnAllDevices;
}

bool HloReplicationAnalysis::HloReplication::IsUniqueOnAllDevices() const {
  return state_ == State::kUniqueOnAllDevices;
}

bool HloReplicationAnalysis::HloReplication::IsReplicatedWithinSubgroup(
    absl::Span<const int64_t> device_ids) const {
  for (absl::flat_hash_set<std::vector<int64_t>> groups_for_replica :
       groups_for_replicas_) {
    // All groups for all replicas must contain all or none of the elements of
    // device_ids.
    for (std::vector<int64_t> group_for_replica : groups_for_replica) {
      int match_count = 0;
      for (int64_t device_id : device_ids) {
        match_count +=
            std::find(group_for_replica.begin(), group_for_replica.end(),
                      device_id) != group_for_replica.end();
      }
      if (match_count && match_count != device_ids.size()) {
        return false;
      }
    }
  }

  return true;
}

std::string HloReplicationAnalysis::HloReplication::ToString() const {
  switch (state_) {
    case State::kReplicatedOnAllDevices:
      return "ReplicatedOnAllDevices";
    case State::kUniqueOnAllDevices:
      return "UniqueOnAllDevices";
    case State::kPartiallyReplicated:
      std::ostringstream oss;
      oss << "PartiallyReplicated";
      for (int k = 0; k < groups_for_replicas_.size(); ++k) {
        oss << " device" << k << "{";
        int l = 0;
        for (std::vector<int64_t> group_for_replica : groups_for_replicas_[k]) {
          oss << "{";
          oss << absl::StrJoin(group_for_replica, ",");
          oss << (++l == groups_for_replicas_[k].size() ? "}" : "},");
        }
        oss << (k == groups_for_replicas_.size() - 1 ? "}" : "},");
      }
      oss << std::endl;
      return oss.str();
  }
}

}  // namespace xla
