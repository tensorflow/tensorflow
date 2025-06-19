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

#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "xla/backends/cpu/xnn_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/instruction_fusion.h"
#include "xla/tsl/platform/status.h"

namespace xla {
namespace cpu {

FusionDecision XnnGraphFusion::ShouldFuse(HloInstruction* consumer,
                                          int64_t operand_index) {
  if (!IsXnnGraphFusion(consumer) && !IsOpSupported(consumer)) {
    return FusionDecision::Forbid("Unsupported consumer");
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
  fusion_config->set_kind(std::string{kXnnFusionKind});
  CHECK(backend_config.has_fusion_config());
  TF_CHECK_OK(fusion->set_backend_config(backend_config));
  return fusion;
}

bool XnnGraphFusion::IsOpSupported(const HloInstruction* instr) const {
  if (instr->IsConstant()) {
    return IsConstantSupportedByXnn(instr);
  }
  if (instr->IsElementwise()) {
    return IsElementwiseOpSupportedByXnn(instr);
  }
  return false;
}

bool XnnGraphFusion::IsXnnGraphFusion(const HloInstruction* instr) const {
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
}  // namespace cpu
}  // namespace xla
