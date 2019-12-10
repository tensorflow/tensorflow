/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_replication_analysis.h"

#include <memory>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

namespace {

// Determines whether an HLO instruction is replicated at index based on current
// knowledge in hlo_replication.
bool DetermineHloInstructionIsReplicated(
    const HloInstruction* hlo, const ShapeIndex& index,
    bool cross_partition_spmd,
    const absl::flat_hash_map<const HloInstruction*, ShapeTree<bool>>&
        hlo_replication) {
  // Returns true if all operands are known to be replicated.
  const auto all_operands_replicated =
      [&hlo_replication](const HloInstruction* inst) {
        for (auto operand : inst->operands()) {
          auto operand_it = hlo_replication.find(operand);
          if (operand_it == hlo_replication.end() ||
              !operand_it->second.element({})) {
            return false;
          }
        }
        return true;
      };

  if (hlo->IsCrossReplicaAllReduce()) {
    if (cross_partition_spmd) {
      // Cross-replica all-reduce returns same values across partitions as long
      // as its operands are replicated.
      return all_operands_replicated(hlo);
    }
    // Only all-reduce across all cores are replicated, which means there
    // is only one subgroup.
    return hlo->replica_groups().empty() || hlo->replica_groups().size() == 1;
  }
  if (hlo->IsCrossModuleAllReduce()) {
    return cross_partition_spmd;
  }
  if (hlo->HasSideEffectNoRecurse()) {
    return false;
  }
  if (hlo->opcode() == HloOpcode::kReplicaId) {
    // ReplicaId returns the same value for all partitions in each replica.
    return cross_partition_spmd;
  }
  if (hlo->opcode() == HloOpcode::kPartitionId) {
    // PartitionId returns the same value for all replicas in each partition.
    return !cross_partition_spmd;
  }
  auto it = hlo_replication.find(hlo);
  if (hlo->opcode() == HloOpcode::kParameter) {
    // Parameters should have been processed.
    return it != hlo_replication.end() && it->second.element(index);
  }
  if (it != hlo_replication.end() && !it->second.element(index)) {
    // The HLO is already marked as non-replicated.
    return false;
  }
  if (hlo->opcode() == HloOpcode::kConstant) {
    return true;
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
    return all_operands_replicated(hlo);
  }
  return false;
}

}  // namespace

bool HloReplicationAnalysis::ComputeHloReplicationOnComputation(
    const HloComputation* computation, bool mark_everything_not_replicated) {
  bool changed = false;
  for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
    // Assigns the shape tree to dest if dest doesn't have one yet, or combines
    // it with the existing one by and'ing them. Returns if anything is updated.
    auto assign_or_combine_shapetree = [&](ShapeTree<bool>&& to_combine,
                                           const HloInstruction* dest) {
      auto it = hlo_replication_.find(dest);
      if (it == hlo_replication_.end()) {
        hlo_replication_[dest] = std::move(to_combine);
        return true;
      }
      bool updated = false;
      it->second.ForEachMutableElement(
          [&](const ShapeIndex& index, bool* element) {
            if (*element && !to_combine.element(index)) {
              *element = false;
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
      return assign_or_combine_shapetree(ShapeTree<bool>(source_it->second),
                                         dest);
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
                 .element({})) {
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
      for (int64 i = 0; i < inst->operand_count(); ++i) {
        changed |= propagate_shapetree(inst->operand(i),
                                       called->parameter_instruction(i));
      }
      changed |= ComputeHloReplicationOnComputation(
          called, mark_everything_not_replicated);
      changed |= propagate_shapetree(called->root_instruction(), inst);
    } else if (inst->opcode() == HloOpcode::kConditional) {
      // Propagate inputs' shape trees to the called computations' parameters.
      for (int64 i = 0; i < inst->called_computations().size(); ++i) {
        changed |= propagate_shapetree(
            inst->operand(i + 1),
            inst->called_computations()[i]->parameter_instruction(0));
      }
      // If the condition is not replicated, the conditional result should be
      // different across replicas.
      if (!hlo_replication_[inst->operand(0)].element({})) {
        for (auto called : inst->called_computations()) {
          changed |= ComputeHloReplicationOnComputation(
              called,
              /*mark_everything_not_replicated=*/true);
        }
        changed |= assign_or_combine_shapetree(
            ShapeTree<bool>(inst->shape(), false), inst);
      } else {
        for (auto called : inst->called_computations()) {
          changed |= ComputeHloReplicationOnComputation(
              called, mark_everything_not_replicated);
          changed |= propagate_shapetree(called->root_instruction(), inst);
        }
      }
    } else if (inst->opcode() == HloOpcode::kTupleSelect) {
      if (!hlo_replication_[inst->operand(0)].element({})) {
        // The predicate is not replicated, so the result is different across
        // replicas.
        changed |= assign_or_combine_shapetree(
            ShapeTree<bool>(inst->shape(), false), inst);
      } else {
        changed |= propagate_shapetree(inst->operand(1), inst);
        changed |= propagate_shapetree(inst->operand(2), inst);
      }
    } else if (inst->opcode() == HloOpcode::kTuple) {
      ShapeTree<bool> shape_tree(inst->shape(), true);
      for (int64 i = 0; i < inst->operand_count(); ++i) {
        shape_tree.CopySubtreeFrom(hlo_replication_[inst->operand(i)], {}, {i});
      }
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
      ShapeTree<bool> shape_tree(inst->shape(), true);
      shape_tree.CopySubtreeFrom(hlo_replication_[inst->operand(0)],
                                 {inst->tuple_index()}, {});
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else {
      if (mark_everything_not_replicated) {
        changed |= assign_or_combine_shapetree(
            ShapeTree<bool>(inst->shape(), false), inst);
      } else {
        ShapeTree<bool> shape_tree(inst->shape(), true);
        ShapeUtil::ForEachSubshape(
            inst->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
              *shape_tree.mutable_element(index) =
                  DetermineHloInstructionIsReplicated(
                      inst, index, cross_partition_spmd_, hlo_replication_);
              return Status::OK();
            });
        changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
      }
    }
  }
  return changed;
}

void HloReplicationAnalysis::ComputeHloReplication() {
  // Add entry parameters to the above sets according to user annotation.
  // Replicated modules read from `parameter_replicated_at_leaf_buffers` whereas
  // SPMD partitioned modules read from HloSharding attributes.
  auto entry = module_->entry_computation();
  for (int i = 0; i < entry->num_parameters(); ++i) {
    auto param = entry->parameter_instruction(i);
    ShapeTree<bool> shape_tree(param->shape(), false);
    if (cross_partition_spmd_ && param->has_sharding()) {
      auto sharding_tree =
          param->sharding().AsShapeTree(param->shape()).ValueOrDie();
      ShapeUtil::ForEachSubshape(
          param->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
            if (!ShapeUtil::IsLeafIndex(param->shape(), index)) {
              return Status::OK();
            }
            *shape_tree.mutable_element(index) =
                sharding_tree.element(index).IsReplicated();
            return Status::OK();
          });
    } else if (!cross_partition_spmd_) {
      const auto& replication = param->parameter_replicated_at_leaf_buffers();
      int leaf_index = 0;
      ShapeUtil::ForEachSubshape(
          param->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
            if (!ShapeUtil::IsLeafIndex(param->shape(), index)) {
              return Status::OK();
            }
            if (replication && replication->at(leaf_index)) {
              *shape_tree.mutable_element(index) = true;
            }
            ++leaf_index;
            return Status::OK();
          });
    }
    hlo_replication_[param] = std::move(shape_tree);
  }
  ComputeHloReplicationOnComputation(entry,
                                     /*mark_everything_not_replicated=*/false);
}

bool HloReplicationAnalysis::HloInstructionIsReplicatedAt(
    const HloInstruction* inst, const ShapeIndex& index) const {
  auto it = hlo_replication_.find(inst);
  if (it == hlo_replication_.end()) {
    return false;
  }
  return it->second.element(index);
}

/* static */ StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::Run(const HloModule* module,
                            bool cross_partition_spmd) {
  const absl::flat_hash_set<const HloInstruction*> empty;
  return Run(module, cross_partition_spmd, &empty);
}

/* static */ StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::Run(const HloModule* module, bool cross_partition_spmd,
                            const absl::flat_hash_set<const HloInstruction*>*
                                loops_known_with_same_iterations) {
  auto analysis = absl::WrapUnique(new HloReplicationAnalysis(
      module, cross_partition_spmd, loops_known_with_same_iterations));
  analysis->ComputeHloReplication();
  return analysis;
}

}  // namespace xla
