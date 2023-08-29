/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/priority_fusion.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/meta/type_traits.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"
#include "tensorflow/compiler/xla/service/fusion_queue.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_performance_model.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {
namespace gpu {

namespace {
bool ElementIsF32OrF16(const Shape& shape) {
  PrimitiveType type = shape.element_type();
  return type == F32 || type == F16;
}

// An implementation of FusionQueue that determines whether to fuse instructions
// according to a cost model, and chooses the next fusion candidate according to
// dynamically updated priorities. The elements in the queue are producer nodes
// that could be fused, and the priority of a producer is the benefit in
// performance when fusing it to all of its fusible users. We greedily pick the
// max-benefit producer to fuse, and update the estimated benefits of the fused
// nodes and their operands.
class GpuPriorityFusionQueue : public FusionQueue {
  using Priority = int64_t;
  using CanFuseCallback = std::function<bool(
      HloInstruction* /*producer*/, int64_t /*consumer operand_index*/)>;

 public:
  GpuPriorityFusionQueue(
      HloComputation* computation, GpuHloCostAnalysis* cost_analysis,
      const std::function<bool(HloInstruction*, int64_t)>& can_fuse)
      : computation_(computation),
        cost_analysis_(cost_analysis),
        can_fuse_(can_fuse) {
    VLOG(2) << "Running full HLO cost analysis for " << computation_->name();
    TF_CHECK_OK(computation_->Accept(cost_analysis_));

    // Initializes the priority queue.
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->user_count() == 0 || !instruction->IsFusible() ||
          instruction->opcode() == HloOpcode::kTuple ||
          instruction->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      Priority priority = CalculateProducerPriority(instruction);
      auto emplace_result = producer_priority_queue_.emplace(
          std::make_pair(priority, instruction->unique_id()), instruction);
      CHECK(emplace_result.second);
      reverse_map_.emplace(instruction, emplace_result.first);
      producer_user_count_[instruction] = instruction->user_count();
    }
  }

  std::pair<HloInstruction*, std::vector<int64_t>>
  DequeueNextInstructionAndOperandsToFuseInOrder() override {
    while (current_consumers_.empty()) {
      if (producer_priority_queue_.empty()) {
        return {};
      }
      auto next_it = std::prev(producer_priority_queue_.end());
      auto priority = next_it->first.first;

      current_producer_ = next_it->second;
      producer_priority_queue_.erase(next_it);
      reverse_map_.erase(current_producer_);

      // If the priority is negative, it's not helpful to perform fusion on this
      // instruction.
      if (priority < 0) {
        continue;
      }
      current_consumers_ = GetFusibleUsers(current_producer_);
    }

    auto next_consumer = current_consumers_.back();
    int64_t producer_operand_index =
        next_consumer->operand_index(current_producer_);
    current_consumers_.pop_back();
    VLOG(5) << "next: " << next_consumer->name() << "(" << next_consumer
            << ") + " << current_producer_->name() << "(" << current_producer_
            << ")";
    return {next_consumer, {producer_operand_index}};
  }

  // Calculates the compute cost and free computation of the new fusion in the
  // PreFusion callback.
  void PreFusion(HloInstruction* producer, HloInstruction* consumer) override {}

  // Updates data for the new fusion instruction and its users and operands.
  void OnFusingInstruction(HloInstruction* fusion,
                           HloInstruction* original_producer,
                           HloInstruction* original_consumer) override {
    // The original consumer was replaced with the fusion, but it's pointer can
    // still be referenced somewhere, for example, in to_update_priority_.
    // Priority recomputation is called before DCE. Remove all references to
    // the original consumer here.
    if (fusion != original_consumer) {
      RemoveInstruction(original_consumer);
    }

    // Detach 'original_producer' from its operands if it has no users.
    // This avoids having it appear as a "phantom" user in subsequent priority
    // calculations on 'fusion.operands' below, before it is finally removed
    // in 'RemoveInstruction'.
    if (original_producer->user_count() == 0) {
      original_producer->DetachFromOperandsAndUsers();
    }

    // Collect the instructions whose priorities need to be updated.
    for (HloInstruction* operand : fusion->operands()) {
      if (operand == original_producer ||
          original_producer->opcode() == HloOpcode::kBroadcast ||
          operand->opcode() == HloOpcode::kBroadcast ||
          operand->opcode() == HloOpcode::kConstant ||
          operand->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      // Need to consider only instructions that are fusible, e.g., rng with
      // greater than one user is not fusible.
      if (!operand->IsFusible()) {
        continue;
      }
      // We only update the priority for the operand when its user count has
      // changed, which is the main cause of priority change. The priority could
      // change in other cases, but we skip them to improve compile time.
      auto user_count_it = producer_user_count_.find(operand);
      if (user_count_it != producer_user_count_.end() &&
          user_count_it->second == operand->user_count()) {
        continue;
      }
      producer_user_count_[operand] = operand->user_count();
      to_update_priority_.insert(operand);
    }
    to_update_priority_.insert(fusion);

    // When current_consumers_ is empty, we will need to dequeue a new producer
    // next time, so we update the priorities now.
    if (current_consumers_.empty()) {
      // Revisit costs of all updated ops. It's important to update cost
      // analysis before recalculating priorities.
      for (auto instruction : to_update_priority_) {
        TF_CHECK_OK(cost_analysis_->RevisitInstruction(instruction));
      }

      for (auto instruction : to_update_priority_) {
        auto reverse_it = reverse_map_.find(instruction);
        const auto new_priority = CalculateProducerPriority(instruction);
        const auto new_key =
            std::make_pair(new_priority, instruction->unique_id());
        if (reverse_it != reverse_map_.end()) {
          if (new_key == reverse_it->second->first) {
            continue;
          }
          producer_priority_queue_.erase(reverse_it->second);
        }
        auto emplace_result =
            producer_priority_queue_.emplace(new_key, instruction);
        CHECK(emplace_result.second);
        if (reverse_it != reverse_map_.end()) {
          reverse_it->second = emplace_result.first;
        } else {
          reverse_map_.emplace(instruction, emplace_result.first);
        }
      }
      to_update_priority_.clear();
    }
  }

  // Removes data for the instruction.
  void RemoveInstruction(HloInstruction* instruction) override {
    to_update_priority_.erase(instruction);
    producer_user_count_.erase(instruction);

    auto reverse_it = reverse_map_.find(instruction);
    if (reverse_it == reverse_map_.end()) {
      return;
    }
    producer_priority_queue_.erase(reverse_it->second);
    reverse_map_.erase(reverse_it);
  }

  const std::vector<bool>* FusionConfiguration() override { return nullptr; }

 private:
  // Returns the priority of the producer based on its current operands and
  // users.
  Priority CalculateProducerPriority(HloInstruction* producer) {
    std::vector<HloInstruction*> fusible_users = GetFusibleUsers(producer);

    // Don't bother computing cost for non-fusible ops.
    if (fusible_users.empty()) {
      return std::numeric_limits<Priority>::min();
    }

    GpuPerformanceModel::RunTimes run_times =
        GpuPerformanceModel::EstimateRunTimes(producer, cost_analysis_,
                                              std::nullopt, fusible_users);
    return absl::ToInt64Nanoseconds(run_times.time_unfused -
                                    run_times.time_fused);
  }

  std::vector<HloInstruction*> GetFusibleUsers(HloInstruction* producer) const {
    std::vector<HloInstruction*> fusible_users;
    for (auto user : producer->users()) {
      int64_t operand_index = user->operand_index(producer);

      if (can_fuse_(user, operand_index)) {
        fusible_users.push_back(user);
      }
    }

    return fusible_users;
  }

  // Store computation for cost analysis.
  HloComputation* computation_;

  // Reference to cost model that defines priorities in the queue.
  GpuHloCostAnalysis* cost_analysis_;

  // The priority queue of producers, implemented as an ordered map, where a
  // key is a pair: the first element is the priority and the second element is
  // the unique ID of the instruction to break ties.
  using PriorityQueue = std::map<std::pair<Priority, int>, HloInstruction*>;
  PriorityQueue producer_priority_queue_;

  // A reverse map that helps find an instruction in the priority queue.
  absl::flat_hash_map<HloInstruction*, PriorityQueue::iterator> reverse_map_;

  // The current producer being visited.
  HloInstruction* current_producer_;

  // The current consumers being visited.
  std::vector<HloInstruction*> current_consumers_;

  // Callbacks passed from the caller to check if we can fuse a pair of
  // producer and consumer, where the consumer is given as a HloInstruction*
  // and the producer is given as the consumer's operand index.
  CanFuseCallback can_fuse_;

  // The user counts of producers, used to determine whether we update their
  // priorities when fusion happens.
  absl::flat_hash_map<HloInstruction*, int64_t> producer_user_count_;

  // The set of producers whose priorities need to be updated. Their
  // priorities are changed because their neighbors got fused, but we delay
  // the priority updates until current_consumers_ becomes empty. This is to
  // avoid recomputing priorities multiple times before we dequeue a new
  // producer.
  absl::flat_hash_set<HloInstruction*> to_update_priority_;
};

}  // namespace

/*static*/ bool GpuPriorityFusion::IsExpensive(
    const HloInstruction& instruction) {
  // Some floating-point math ops are cheap on the GPU.
  switch (instruction.opcode()) {
    case HloOpcode::kDivide:
    case HloOpcode::kSqrt:
    case HloOpcode::kRsqrt:
    case HloOpcode::kExp:
      if (ElementIsF32OrF16(instruction.shape())) {
        return false;
      }
      break;
    // Loop fusions are cheap.
    case HloOpcode::kFusion:
      return false;
    default:
      break;
  }
  return InstructionFusion::IsExpensive(instruction);
}

FusionDecision GpuPriorityFusion::ShouldFuseInexpensiveChecks(
    HloInstruction* consumer, int64_t operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // Cost condition: not fuse (simple, expensive producers) and (consumers who
  // reuse operand elements).
  if (producer->opcode() != HloOpcode::kFusion && is_expensive(*producer) &&
      ReusesOperandElements(consumer, operand_index)) {
    return "the producer is expensive, and the consumer reuses inputs";
  }

  // Do not fuse into fusions if the resulting kernel would suffer from
  // uncoalesced reads due to a transposed memory access pattern.
  if (IsInputFusibleReduction(*consumer) &&
      IsPhysicallyTransposing(*producer)) {
    return "fusing the producer would break read coalescing";
  }

  if (auto fusible = IsProducerConsumerFusible(*producer, *consumer);
      !fusible) {
    return fusible;
  }

  if (CreatesHeavyComputation(*producer, *consumer)) {
    return "the fusion would create a heavy computation";
  }

  return InstructionFusion::ShouldFuse(consumer, operand_index);
}

FusionDecision GpuPriorityFusion::ShouldFuse(HloInstruction* consumer,
                                             int64_t operand_index) {
  if (auto fusible = ShouldFuseInexpensiveChecks(consumer, operand_index);
      !fusible) {
    return fusible;
  }

  auto producer = consumer->operand(operand_index);

  // The following checks are potentially expensive.
  if (auto fusible = FusionFitsInBudget(*consumer, *producer,
                                        *cost_analysis_->device_info_,
                                        /*is_consumer_producer_fusion=*/true);
      !fusible) {
    return fusible;
  }

  // Also check that our emitter can handle the fusion node. We currently can
  // have exponential time/memory requirements for emitting certain fusion
  // kernels, in which case we don't want to fuse.
  // TODO(b/119692968): Remove this once we have fixed our fusion emitter.
  // TODO(kramerb): Re-enable caching of FusionNodeIndexingEvaluation. It
  // doesn't get invalidated when fusions are merged.
  if (consumer->opcode() == HloOpcode::kFusion &&
      FusionNodeIndexingEvaluation(consumer).CodeDuplicationTooHigh(producer)) {
    return "the fusion would result in an overly large code duplication";
  }

  return {};
}

HloInstruction::FusionKind GpuPriorityFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  return ChooseFusionKind(*producer, *consumer);
}

HloInstruction* GpuPriorityFusion::FuseInstruction(
    HloInstruction* fusion_instruction, HloInstruction* producer) {
  HloInstruction* result = fusion_instruction;
  if (producer->opcode() == HloOpcode::kFusion) {
    fusion_instruction->MergeFusionInstruction(producer);
  } else {
    result = InstructionFusion::FuseInstruction(fusion_instruction, producer);
  }
  return result;
}

std::unique_ptr<FusionQueue> GpuPriorityFusion::GetFusionQueue(
    HloComputation* computation) {
  return std::unique_ptr<FusionQueue>(new GpuPriorityFusionQueue(
      computation, &*cost_analysis_,
      [this](HloInstruction* consumer, int64_t operand_index) {
        return ShouldFuse(consumer, operand_index).CanFuse();
      }));
}

}  // namespace gpu
}  // namespace xla
