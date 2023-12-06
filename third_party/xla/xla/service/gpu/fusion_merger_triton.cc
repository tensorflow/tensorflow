/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

#include "xla/service/gpu/fusion_merger_triton.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

// Taking in a producer HloFusionInstruction, tries to merge into consumer
// triton softmax fusion.
// The following is assumed:
//  * The producer is an HloFusionInstruction
//  * The (sole) consumer of the producer is a triton softmax fusion
//
// Returns StatusOr<HloInstruction*>, i.e. returns a pointer to the new (fused)
// triton softmax instruction if the producer was successfully merged into the
// consumer. If the merge was unsuccessful, the original computation remains
// unchanged and a non-ok status is returned.
std::optional<HloFusionInstruction*>
TryMergeFusionProducerIntoTritonSoftmaxConsumer(
    HloFusionInstruction* producer) {
  // TODO(b/313026024): Add support for multiple users
  CHECK_EQ(producer->user_count(), 1);

  HloComputation* computation = producer->parent();
  HloInstruction* original_softmax_instruction = producer->users().front();
  CHECK_EQ(original_softmax_instruction->opcode(), HloOpcode::kFusion);

  std::unique_ptr<HloInstruction> candidate =
      original_softmax_instruction->Clone();
  HloInstruction* candidate_fusion =
      static_cast<HloInstruction*>(candidate.get());

  // Try to merge the producer into candidate fusion
  candidate_fusion->MergeFusionInstruction(producer);

  HloComputation* fused_computation =
      candidate_fusion->called_computations().front();

  const auto analysis = TritonFusionAnalysis::Execute(*fused_computation);

  if (!analysis.ok()) {
    return std::nullopt;
  }

  computation->AddInstruction(std::move(candidate));

  if (original_softmax_instruction->IsRoot()) {
    computation->set_root_instruction(candidate_fusion);
  }

  TF_CHECK_OK(
      original_softmax_instruction->ReplaceAllUsesWith(candidate_fusion));
  TF_CHECK_OK(computation->RemoveInstruction(original_softmax_instruction));

  CHECK_EQ(0, producer->user_count()) << producer->ToString();
  TF_CHECK_OK(computation->RemoveInstruction(producer));

  return Cast<HloFusionInstruction>(candidate_fusion);
}

bool TryMergeProducerAndConsumerFusionsIntoTritonSoftmax(
    HloFusionInstruction* softmax_fusion) {
  // The softmax_fusion should come directly from the matcher, and have a single
  // operand.
  CHECK_EQ(softmax_fusion->operand_count(), 1);

  bool producer_is_fusion =
      softmax_fusion->operand(0)->opcode() == HloOpcode::kFusion;

  if (producer_is_fusion) {
    HloFusionInstruction* producer =
        Cast<HloFusionInstruction>(softmax_fusion->mutable_operand(0));

    VLOG(6) << "Fusing producer " << producer->ToShortString() << " into "
            << softmax_fusion->ToShortString();

    std::optional<HloFusionInstruction*> result =
        TryMergeFusionProducerIntoTritonSoftmaxConsumer(producer);

    if (!result.has_value()) {
      VLOG(6) << "Did not fuse producer into "
              << softmax_fusion->ToShortString();
    } else {
      return true;
    }
  }

  return false;
}

}  // anonymous namespace

StatusOr<bool> FusionMergerTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  int fused_comps = 0;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    if (comp->IsCustomCallComputation()) {
      continue;
    }

    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kFusion &&
          instr->fusion_kind() == HloInstruction::FusionKind::kCustom &&
          instr->backend_config<FusionBackendConfig>().ok() &&
          instr->backend_config<FusionBackendConfig>()->kind() ==
              kTritonSoftmaxFusionKind) {
        VLOG(6) << "Matched triton_softmax fusion: " << instr->ToShortString();

        HloFusionInstruction* softmax = Cast<HloFusionInstruction>(instr);

        bool result =
            TryMergeProducerAndConsumerFusionsIntoTritonSoftmax(softmax);

        if (!result) {
          VLOG(6) << "Did not fuse producer into " << instr->ToShortString();
        } else {
          ++fused_comps;
        }
      }
    }
  }
  return fused_comps > 0;
}
}  // namespace xla::gpu
