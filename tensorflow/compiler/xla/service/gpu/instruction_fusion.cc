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

#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"
#include "tensorflow/compiler/xla/service/fusion_queue.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"

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
 public:
  GpuPriorityFusionQueue(
      HloComputation* computation,
      const std::function<bool(HloInstruction*, int64_t)>& can_fuse)
      : can_fuse_(can_fuse) {
    // Initializes the priority queue.
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->user_count() == 0 || !instruction->IsFusible() ||
          instruction->opcode() == HloOpcode::kTuple ||
          instruction->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      const std::pair<double, double> priority =
          CalculateProducerPriority(instruction);
      auto emplace_result = producer_priority_queue_.emplace(
          std::make_tuple(priority.first, priority.second,
                          instruction->unique_id()),
          instruction);
      CHECK(emplace_result.second);
      reverse_map_.emplace(instruction, emplace_result.first);
    }
  }

  std::pair<HloInstruction*, std::vector<int64_t>>
  DequeueNextInstructionAndOperandsToFuseInOrder() override {
    while (current_consumers_.empty()) {
      if (producer_priority_queue_.empty()) {
        return std::pair<HloInstruction*, std::vector<int64_t>>(nullptr, {});
      }
      auto next_it = producer_priority_queue_.end();
      --next_it;

      current_producer_ = next_it->second;
      producer_priority_queue_.erase(next_it);
      reverse_map_.erase(current_producer_);

      current_consumers_ = GetFusibleUsers(current_producer_);
    }

    auto next_consumer = current_consumers_.back();
    current_consumers_.pop_back();
    VLOG(5) << "next: " << next_consumer->name() << "(" << next_consumer
            << ") + " << current_producer_->name() << "(" << current_producer_
            << ")";
    auto indices = next_consumer->OperandIndices(current_producer_);
    return std::make_pair(next_consumer,
                          std::vector<int64_t>(indices.begin(), indices.end()));
  }

  // Calculates the compute cost and free computation of the new fusion in the
  // PreFusion callback.
  void PreFusion(HloInstruction* producer, HloInstruction* consumer) override {}

  // Updates data for the new fusion instruction and its users and operands.
  void OnFusingInstruction(HloInstruction* fusion,
                           HloInstruction* original_producer,
                           HloInstruction* original_consumer) override {}

  // Removes data for the instruction.
  void RemoveInstruction(HloInstruction* instruction) override {
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
  std::pair<double, int64_t> CalculateProducerPriority(
      HloInstruction* producer) {
    // TODO(shyshkov): Use cost model.
    return {1., 1};
  }

  std::vector<HloInstruction*> GetFusibleUsers(HloInstruction* producer) const {
    auto fusible = [&](HloInstruction* user) {
      for (int64_t i = 0; i < user->operand_count(); ++i) {
        if (user->operand(i) == producer && can_fuse_(user, i)) {
          return true;
        }
      }
      return false;
    };
    std::vector<HloInstruction*> fusible_users;
    std::vector<HloInstruction*> prod_users(producer->users());
    for (auto user : prod_users) {
      if (fusible(user)) {
        fusible_users.push_back(user);
      }
    }

    return fusible_users;
  }

  // The priority queue of producers, implemented as an ordered map, where a
  // key is a tuple: the first two elements are the primary and secondary
  // priorities, and the last one is the unique ID of the instruction to break
  // ties.
  using PriorityQueue =
      std::map<std::tuple<double, double, int>, HloInstruction*>;
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
  std::function<bool(HloInstruction*, int64_t)> can_fuse_;
};

}  // namespace

/*static*/ bool GpuInstructionFusion::IsExpensive(
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
    default:
      break;
  }
  return InstructionFusion::IsExpensive(instruction);
}

FusionDecision GpuInstructionFusion::ShouldFuseInexpensiveChecks(
    HloInstruction* consumer, int64_t operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // Output fusions are not currently supported on GPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    return "the producer is a fusion";
  }
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

  if (NoFusionPossible fusible =
          !IsProducerConsumerFusible(*producer, *consumer)) {
    return !fusible;
  }

  if (CreatesHeavyComputation(*producer, *consumer)) {
    return "the fusion would create a heavy computation";
  }

  if (NoFusionPossible fusible =
          !InstructionFusion::ShouldFuse(consumer, operand_index)) {
    return !fusible;
  }
  return {};
}

FusionDecision GpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                                int64_t operand_index) {
  if (NoFusionPossible fusible =
          !ShouldFuseInexpensiveChecks(consumer, operand_index)) {
    return !fusible;
  }

  auto producer = consumer->operand(operand_index);

  // The following checks are potentially expensive.
  if (NoFusionPossible too_large =
          !FusionFitsInBudget(*consumer, *producer, device_info_,
                              /*is_consumer_producer_fusion=*/true)) {
    return !too_large;
  }

  if (consumer->opcode() != HloOpcode::kFusion) {
    return {};
  }

  // Also check that our emitter can handle the fusion node. We currently can
  // have exponential time/memory requirements for emitting certain fusion
  // kernels, in which case we don't want to fuse.
  // TODO(b/119692968): Remove this once we have fixed our fusion emitter.
  if (fusion_node_evaluations_.find(consumer) ==
      fusion_node_evaluations_.end()) {
    // We have no cached results for this fusion node yet. This can happen when
    // we run the InstructionFusion pass more than once. We can only cache the
    // results within one run.
    fusion_node_evaluations_.emplace(consumer,
                                     FusionNodeIndexingEvaluation(consumer));
  }
  if (fusion_node_evaluations_.at(consumer).CodeDuplicationTooHigh(producer)) {
    return "the fusion would result in an overly large code duplication";
  }
  return {};
}

HloInstruction::FusionKind GpuInstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  return ChooseFusionKind(*producer, *consumer);
}

HloInstruction* GpuInstructionFusion::FuseInstruction(
    HloInstruction* fusion_instruction, HloInstruction* producer) {
  auto evaluation = fusion_node_evaluations_.find(fusion_instruction);
  if (evaluation == fusion_node_evaluations_.end()) {
    evaluation = fusion_node_evaluations_
                     .emplace(fusion_instruction,
                              FusionNodeIndexingEvaluation(fusion_instruction))
                     .first;
  }
  auto indexing_users = evaluation->second.RemoveFusionOperand(producer);
  HloInstruction* new_producer =
      InstructionFusion::FuseInstruction(fusion_instruction, producer);
  evaluation->second.UpdateEvaluationCache(new_producer, indexing_users);
  return new_producer;
}

std::unique_ptr<FusionQueue> GpuInstructionFusion::GetFusionQueue(
    HloComputation* computation) {
  if (priority_fusion_) {
    return std::unique_ptr<FusionQueue>(new GpuPriorityFusionQueue(
        computation, [this](HloInstruction* consumer, int64_t operand_index) {
          return ShouldFuse(consumer, operand_index).CanFuse();
        }));
  }

  return InstructionFusion::GetFusionQueue(computation);
}

}  // namespace gpu
}  // namespace xla
