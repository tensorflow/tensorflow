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

#include "xla/service/gpu/priority_fusion.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/meta/type_traits.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/dump.h"
#include "xla/service/fusion_node_indexing_evaluation.h"
#include "xla/service/fusion_queue.h"
#include "xla/service/gpu/fusion_process_dump.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

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
  using CanFuseCallback = std::function<FusionDecision(
      HloInstruction* /*producer*/, int64_t /*consumer operand_index*/)>;

 public:
  GpuPriorityFusionQueue(
      HloComputation* computation,
      const GpuHloCostAnalysis::Options& cost_analysis_options,
      const se::DeviceDescription* device_info, const CanFuseCallback& can_fuse,
      FusionProcessDumpProto* fusion_process_dump)
      : computation_(computation),
        cost_analysis_(cost_analysis_options, device_info),
        can_fuse_(can_fuse),
        fusion_process_dump_(fusion_process_dump) {
    VLOG(2) << "Running full HLO cost analysis for " << computation_->name();
    TF_CHECK_OK(computation_->Accept(&cost_analysis_));

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
      current_consumers_ = current_producer_->users();
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
    if (fusion_process_dump_) {
      auto* fusion_step = fusion_process_dump_->add_fusion_steps();

      // Explicit std::string is needed for OSS proto implementation.
      fusion_step->set_fusion_name(std::string(fusion->name()));
      fusion_step->set_producer_name(std::string(original_producer->name()));
      fusion_step->set_consumer_name(std::string(original_consumer->name()));
    }

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
        TF_CHECK_OK(cost_analysis_.RevisitInstruction(instruction));
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
    // Don't fuse if we can't fuse in all users.
    if (auto fusion_decision = CanFuseWithAllUsers(producer);
        !fusion_decision) {
      return std::numeric_limits<Priority>::min();
    }

    GpuPerformanceModel::RunTimes run_times =
        GpuPerformanceModel::EstimateRunTimes(producer, &cost_analysis_,
                                              producer->users());
    return absl::ToInt64Nanoseconds(run_times.time_unfused -
                                    run_times.time_fused);
  }

  FusionDecision CanFuseWithAllUsers(HloInstruction* producer) const {
    FusionDecision result;
    for (const auto& user : producer->users()) {
      if (auto fusion_decision = can_fuse_(user, user->operand_index(producer));
          !fusion_decision) {
        VLOG(10) << "Cannot fuse " << producer->name() << " with "
                 << user->name() << ", because: " << fusion_decision.Explain();
        return fusion_decision;
      }
    }
    return {};
  }

  // Store computation for cost analysis.
  HloComputation* computation_;

  // Reference to cost model that defines priorities in the queue.
  GpuHloCostAnalysis cost_analysis_;

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

  // Proto with structured logs of fusion decisions. Used only for debugging. If
  // null, logging is disabled.
  FusionProcessDumpProto* fusion_process_dump_;
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

StatusOr<bool> GpuPriorityFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool dump_enabled = DumpingEnabledForHloModule(*module);
  if (dump_enabled) {
    fusion_process_dump_ = std::make_unique<FusionProcessDumpProto>();
  }

  auto result = InstructionFusion::Run(module, execution_threads);

  if (dump_enabled) {
    DumpPerModuleProtobufToFile(*module, *fusion_process_dump_,
                                module->config().debug_options(),
                                "priority_fusion_dump");
  }

  return result;
}

FusionDecision GpuPriorityFusion::ShouldFuse(HloInstruction* consumer,
                                             int64_t operand_index) {
  auto isFusible = [](const HloInstruction& instr) {
    // Side-effecting operations are not fusible.
    if (!instr.IsFusible()) {
      return false;
    }

    // Element-wise operations are always fusible.
    if (instr.IsElementwise()) {
      return true;
    }

    // Other non-elementwise ops also supported by elemental fusion.
    switch (instr.opcode()) {
      case HloOpcode::kFusion:
        return instr.fusion_kind() != HloInstruction::FusionKind::kCustom;

      case HloOpcode::kCopy:
      case HloOpcode::kIota:
      case HloOpcode::kConstant:
      case HloOpcode::kReduce:
      case HloOpcode::kBitcast:
      case HloOpcode::kBroadcast:
      case HloOpcode::kConcatenate:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kGather:
      case HloOpcode::kPad:
      case HloOpcode::kReduceWindow:
      case HloOpcode::kReshape:
      case HloOpcode::kReverse:
      case HloOpcode::kScatter:
      case HloOpcode::kSlice:
      case HloOpcode::kTranspose:
        return true;
      default:
        return false;
    }
  };

  HloInstruction* producer = consumer->mutable_operand(operand_index);
  if (!isFusible(*producer)) {
    return "the producer is not fusible";
  }

  if (!isFusible(*consumer)) {
    return "the consumer is not fusible";
  }

  // Scatter is special as it has no elemental version but is still input
  // fusible. Block attempts to create scatter fusions we can't codegen.
  if (auto can_fuse = CanEmitInputFusedScatter(*producer, *consumer);
      !can_fuse) {
    return can_fuse;
  }

  // Avoid cases where we'd create a fusion that hit limitations in ptxas. Would
  // be nice to model this with cost instead.
  if (auto fits_budget =
          FusionFitsInBudget(*consumer, *producer, device_info_,
                             /*is_consumer_producer_fusion=*/true);
      !fits_budget) {
    return fits_budget;
  }

  return InstructionFusion::ShouldFuse(consumer, operand_index);
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
      computation, cost_analysis_options_, &device_info_,
      [this](HloInstruction* consumer, int64_t operand_index) {
        return ShouldFuse(consumer, operand_index);
      },
      fusion_process_dump_.get()));
}

}  // namespace gpu
}  // namespace xla
