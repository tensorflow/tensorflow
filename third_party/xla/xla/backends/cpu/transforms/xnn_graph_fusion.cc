/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/transforms/xnn_graph_fusion.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
#include "xla/backends/cpu/xnn_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/instruction_fusion.h"
#include "xla/xla.pb.h"

namespace xla::cpu {

namespace {

bool IsWideningConvert(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kConvert &&
         primitive_util::BitWidth(instr->operand(0)->shape().element_type()) <
             primitive_util::BitWidth(instr->shape().element_type());
}

}  // namespace

FusionDecision XnnGraphFusion::ShouldFuse(HloInstruction* consumer,
                                          int64_t operand_index) {
  if (!IsXnnGraphFusion(consumer) && !IsOpSupported(consumer)) {
    return FusionDecision::Forbid("Unsupported consumer");
  }

  if (consumer->opcode() == HloOpcode::kBroadcast) {
    return FusionDecision::Forbid(
        "Do not start growing fusions from broadcasts");
  }

  if (IsWideningConvert(consumer)) {
    // We don't want to start a fusion with a widening convert, because that
    // makes the buffer the fusion writes to bigger, and it would be better to
    // fuse the convert into the consumer of the convert.
    return FusionDecision::Forbid(
        "Do not start growing fusions from widening converts");
  }

  HloInstruction* producer = consumer->mutable_operand(operand_index);
  if (!(producer->opcode() == HloOpcode::kParameter ||
        IsOpSupported(producer))) {
    return FusionDecision::Forbid("Unsupported producer");
  }
  return FusionDecision::Allow();
}

HloInstruction::FusionKind XnnGraphFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  return HloInstruction::FusionKind::kCustom;
}

HloInstruction* XnnGraphFusion::Fuse(HloInstruction* producer,
                                     HloInstruction* consumer,
                                     HloComputation* computation) {
  HloInstruction* fusion =
      InstructionFusion::Fuse(producer, consumer, computation);

  BackendConfig backend_config;
  FusionBackendConfig* fusion_config = backend_config.mutable_fusion_config();
  fusion_config->set_kind(kXnnFusionKind);
  CHECK(backend_config.has_fusion_config());
  CHECK_OK(fusion->set_backend_config(backend_config));
  return fusion;
}

std::vector<HloComputation*> XnnGraphFusion::GetNonFusionComputations(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloComputation*> non_fusion_computations =
      InstructionFusion::GetNonFusionComputations(module, execution_threads);
  std::unique_ptr<CallGraph> call_graph =
      CallGraph::Build(module, execution_threads);
  auto SkipComputation = [&](HloComputation* c) {
    auto callers = call_graph->GetComputationCallers(c);
    return std::any_of(
        callers.begin(), callers.end(),
        [&](HloInstruction* caller) { return caller->has_to_apply(); });
  };
  auto it = std::remove_if(non_fusion_computations.begin(),
                           non_fusion_computations.end(), SkipComputation);
  non_fusion_computations.erase(it, non_fusion_computations.end());
  return non_fusion_computations;
}

bool XnnGraphFusion::IsOpSupported(const HloInstruction* instr) {
  if (!IsLayoutSupportedByXnn(instr->shape())) {
    return false;
  }
  if (!XnnDatatype(instr->shape().element_type()).ok()) {
    return false;
  }
  if (instr->IsConstant()) {
    return IsConstantSupportedByXnn(instr);
  }
  if (instr->IsElementwise()) {
    return IsElementwiseOpSupportedByXnn(instr);
  }

  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
      return IsBitcastOpSupportedByXnn(instr);
    case HloOpcode::kBroadcast:
      return IsBroadcastOpSupportedByXnn(instr);
    case HloOpcode::kReduce:
      return IsReduceOpSupportedByXnn(instr);
    default:
      return false;
  }
}

bool XnnGraphFusion::IsXnnGraphFusion(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kFusion) {
    return false;
  }
  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instr);
  if (fusion->fusion_kind() != HloInstruction::FusionKind::kCustom) {
    return false;
  }
  auto backend_config = fusion->backend_config<BackendConfig>();
  if (!backend_config.ok() || !backend_config->has_fusion_config()) {
    return false;
  }
  return backend_config->fusion_config().kind() == kXnnFusionKind;
}

}  // namespace xla::cpu
