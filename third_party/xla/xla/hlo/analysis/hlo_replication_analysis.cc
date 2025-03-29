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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/map_util.h"
#include "xla/shape_tree.h"
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
std::vector<std::vector<std::vector<int64_t>>> GroupsForReplicas(
    absl::Span<const ReplicaGroup> groups, int64_t num_partitions,
    int64_t replica_count, bool cross_partition_spmd) {
  int64_t num_replicas = cross_partition_spmd ? replica_count : num_partitions;
  std::vector<std::vector<std::vector<int64_t>>> groups_for_replicas(
      num_replicas);
  for (const ReplicaGroup& group : groups) {
    absl::flat_hash_map<int64_t, std::vector<int64_t>> id_to_ids;
    for (int64_t id : group.replica_ids()) {
      int64_t rid = id / num_partitions;
      int64_t pid = id % num_partitions;
      if (cross_partition_spmd) {
        CHECK_LT(rid, num_replicas)
            << "Got replica ID " << rid
            << " which is greater or equal to the number of replicas: "
            << num_replicas;
        id_to_ids[rid].push_back(pid);
      } else {
        CHECK_LT(pid, num_partitions)
            << "Got partition ID " << rid
            << " which is greater or equal to the number of partitions: "
            << num_partitions;
        id_to_ids[pid].push_back(rid);
      }
    }
    for (const auto& [id, ids] : id_to_ids) {
      groups_for_replicas[id].push_back(std::move(ids));
    }
  }

  return groups_for_replicas;
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
        std::vector<std::vector<std::vector<int64_t>>> device_sets_per_replica(
            1);
        for (const ReplicaGroup& replica_group : hlo->replica_groups()) {
          std::vector<int64_t> device_set;
          for (auto id : replica_group.replica_ids()) {
            device_set.push_back(id);
          }
          device_sets_per_replica[0].push_back(device_set);
        }
        return HloReplication::PartiallyReplicated(device_sets_per_replica);
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
        std::vector<std::vector<std::vector<int64_t>>> device_sets_per_replica =
            GroupsForReplicas(hlo->replica_groups(), num_partitions,
                              replica_count, cross_partition_spmd);

        // In the fully replicated case, there is one set of partition or
        // replica IDs on each replica or partition. Since the flattened ID
        // replica groups must contain every device, the size of the set is the
        // number of partitions or replicas.
        bool fully_replicated = true;
        for (auto device_sets : device_sets_per_replica) {
          fully_replicated &=
              device_sets.size() == 1 &&
              (*device_sets.begin()).size() ==
                  (cross_partition_spmd ? num_partitions : replica_count);
        }
        if (fully_replicated) {
          return HloReplication::ReplicatedOnAllDevices();
        } else if (support_partial_replication) {
          return HloReplication::PartiallyReplicated(device_sets_per_replica);
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
          ds_buffer->shape().dimensions_size() == 1 &&
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
        std::vector<std::vector<std::vector<int64_t>>> device_sets_per_replica(
            1);
        for (const auto& value_and_device_set : value_to_device_set) {
          device_sets_per_replica[0].push_back(value_and_device_set.second);
        }
        return HloReplication::PartiallyReplicated(device_sets_per_replica);
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

    std::unique_ptr<ShapeTree<HloSharding>> sharding_tree = nullptr;
    if (cross_partition_spmd_ && param->has_sharding()) {
      TF_ASSIGN_OR_RETURN(auto result,
                          param->sharding().AsShapeTree(param->shape()));
      sharding_tree =
          std::make_unique<ShapeTree<HloSharding>>(std::move(result));
    }

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
            *shape_tree.mutable_element(index) =
                sharding_tree->element(index).IsReplicated()
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
    absl::Span<const std::vector<int64_t>> device_set_root_per_replica)
    : state_(state),
      device_set_root_per_replica_(device_set_root_per_replica.begin(),
                                   device_set_root_per_replica.end()) {
  CHECK(state == State::kPartiallyReplicated ||
        device_set_root_per_replica_.empty());
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
    absl::Span<const std::vector<std::vector<int64_t>>>
        device_sets_per_replica) {
  std::vector<std::vector<int64_t>> device_set_root_per_replica;
  for (int i = 0; i < device_sets_per_replica.size(); ++i) {
    const std::vector<std::vector<int64_t>>& device_sets =
        device_sets_per_replica[i];
    int64_t max_device_id = 0;
    for (const std::vector<int64_t>& device_set : device_sets) {
      for (int64_t device_id : device_set) {
        max_device_id = std::max(max_device_id, device_id);
      }
    }
    std::vector<int64_t> device_set_root;
    device_set_root.resize(max_device_id + 1);
    for (const std::vector<int64_t>& device_set : device_sets) {
      int64_t min_device_id = *absl::c_min_element(device_set);
      for (int64_t device_id : device_set) {
        device_set_root[device_id] = min_device_id;
      }
    }
    device_set_root_per_replica.push_back(std::move(device_set_root));
  }
  return HloReplication(State::kPartiallyReplicated,
                        device_set_root_per_replica);
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
          bool unique_on_all_devices = true;
          std::vector<std::vector<std::vector<int64_t>>>
              device_sets_per_replica;
          CHECK_EQ(device_set_root_per_replica_.size(),
                   other.device_set_root_per_replica_.size());
          for (int i = 0; i < device_set_root_per_replica_.size(); ++i) {
            const std::vector<int64_t>& my_device_set_root =
                device_set_root_per_replica_[i];
            const std::vector<int64_t>& other_device_set_root =
                other.device_set_root_per_replica_[i];
            absl::flat_hash_map<int64_t, std::vector<int64_t>>
                value_to_device_set;
            size_t num_devices = my_device_set_root.size();
            for (int64_t device_id = 0; device_id < num_devices; ++device_id) {
              int64_t new_value = my_device_set_root[device_id] * num_devices +
                                  other_device_set_root[device_id];
              value_to_device_set[new_value].push_back(device_id);
            }
            CHECK_LE(value_to_device_set.size(), num_devices);
            std::vector<std::vector<int64_t>> device_sets;
            for (const auto& value_and_device_set : value_to_device_set) {
              device_sets.push_back(value_and_device_set.second);
            }
            device_sets_per_replica.push_back(std::move(device_sets));
            unique_on_all_devices &= value_to_device_set.size() == num_devices;
          }
          if (unique_on_all_devices) {
            return HloReplication::UniqueOnAllDevices();
          }
          return HloReplication::PartiallyReplicated(device_sets_per_replica);
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
  for (int i = 0; i < device_set_root_per_replica_.size(); ++i) {
    if (device_set_root_per_replica_[i] !=
        other.device_set_root_per_replica_[i]) {
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
  if (device_ids.empty()) return true;
  for (std::vector<int64_t> device_set_roots : device_set_root_per_replica_) {
    if (!absl::c_all_of(device_ids,
                        [&device_ids, &device_set_roots](int device_id) {
                          return device_set_roots[device_id] ==
                                 device_set_roots[device_ids.front()];
                        })) {
      return false;
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
      oss << "PartiallyReplicated{";
      for (int k = 0; k < device_set_root_per_replica_.size(); ++k) {
        if (k > 0) {
          oss << ", ";
        }
        oss << absl::StrCat(
            "{", absl::StrJoin(device_set_root_per_replica_[k], ","), "}");
      }
      oss << "}";
      return oss.str();
  }
}

}  // namespace xla
