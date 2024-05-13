/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/instruction_fusion.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/meta/type_traits.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/fusion_node_indexing_evaluation.h"
#include "xla/service/fusion_queue.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {
bool ElementIsF32OrF16(const Shape& shape) {
  PrimitiveType type = shape.element_type();
  return type == F32 || type == F16;
}
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

  if (consumer->IsCustomFusion()) {
    return "the consumer is a custom fusion";
  }

  // Cost condition: not fuse (simple, expensive producers) and (consumers who
  // reuse operand elements).
  if (is_expensive(*producer) &&
      ReusesOperandElements(consumer, operand_index)) {
    return "the producer is expensive, and the consumer reuses inputs";
  }

  // Do not fuse into fusions if the resulting kernel would suffer from
  // uncoalesced reads due to a transposed memory access pattern.
  if (IsInputFusibleReduction(*consumer) &&
      IsPhysicallyTransposing(*producer)) {
    return "fusing the producer would break read coalescing";
  }

  RETURN_IF_NOT_FUSIBLE(IsProducerConsumerFusible(*producer, *consumer));

  if (CreatesHeavyComputation(*producer, *consumer)) {
    return "the fusion would create a heavy computation";
  }

  return InstructionFusion::ShouldFuse(consumer, operand_index);
}

FusionDecision GpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                                int64_t operand_index) {
  RETURN_IF_NOT_FUSIBLE(ShouldFuseInexpensiveChecks(consumer, operand_index));

  auto producer = consumer->operand(operand_index);

  // The following checks are potentially expensive.
  RETURN_IF_NOT_FUSIBLE(
      FusionFitsInBudget(*consumer, *producer, device_info_,
                         /*is_consumer_producer_fusion=*/true));

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
  return InstructionFusion::GetFusionQueue(computation);
}

}  // namespace gpu
}  // namespace xla
