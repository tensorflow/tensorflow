/* Copyright 2023 The OpenXLA Authors.
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
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace xla::gpu {
namespace {

// Taking in a producer HloFusionInstruction, tries to merge into consumer
// triton softmax fusion.
// The following is assumed:
//  * The producer is an HloFusionInstruction
//  * The (sole) consumer of the producer is a triton softmax fusion
//
// Returns std::optional<HloFusionInstruction*>, pointing to the new (fused)
// triton softmax instruction if the producer was successfully merged into the
// consumer. If the merge was unsuccessful, the original computation remains
// unchanged and a nullopt is returned.
std::optional<HloFusionInstruction*>
TryMergeFusionProducerIntoTritonSoftmaxConsumer(
    HloFusionInstruction* producer) {
  // TODO(b/313026024): Add support for multiple users
  CHECK_EQ(producer->user_count(), 1);

  HloComputation* computation = producer->parent();
  HloModule* parent_module = computation->parent();
  HloInstruction* original_softmax_instruction = producer->users().front();
  CHECK_EQ(original_softmax_instruction->opcode(), HloOpcode::kFusion);

  std::unique_ptr<HloInstruction> candidate =
      original_softmax_instruction->Clone();
  HloInstruction* candidate_fusion = candidate.get();

  // Try to merge the producer into candidate fusion.
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

  HloComputation* original_softmax_computation =
      original_softmax_instruction->fused_instructions_computation();
  TF_CHECK_OK(computation->RemoveInstruction(original_softmax_instruction));
  TF_CHECK_OK(
      parent_module->RemoveEmbeddedComputation(original_softmax_computation));

  CHECK_EQ(0, producer->user_count()) << producer->ToString();
  HloComputation* original_producer_computation =
      producer->fused_instructions_computation();
  TF_CHECK_OK(computation->RemoveInstruction(producer));
  TF_CHECK_OK(
      parent_module->RemoveEmbeddedComputation(original_producer_computation));

  return Cast<HloFusionInstruction>(candidate_fusion);
}

// Taking in a consumer HloFusionInstruction and a HloInstruction for a triton
// softmax fusion, tries to merge the consumer fusion into the softmax fusion.
// The following is assumed:
//  * The consumer is an HloFusionInstruction
//  * consumer->shape().IsArray(), i.e. not a multi-output consumer
//  * The original_softmax_instr is a triton softmax fusion
//  * The consumer is the sole user of original_softmax_instr
//
// Returns std::optional<HloFusionInstruction*>, pointing to the new (fused)
// triton softmax instruction if the consumer was successfully merged into the
// producer. If the merge was unsuccessful, the original computation remains
// unchanged and a nullopt is returned.
std::optional<HloFusionInstruction*>
TryMergeFusionConsumerIntoTritonSoftmaxProducer(
    HloFusionInstruction* consumer,
    HloFusionInstruction* original_softmax_instr) {
  CHECK_EQ(original_softmax_instr->opcode(), HloOpcode::kFusion);
  CHECK_EQ(original_softmax_instr->user_count(), 1);
  CHECK_EQ(original_softmax_instr->users().front(), consumer);
  CHECK(consumer->shape().IsArray());
  CHECK_OK(original_softmax_instr->backend_config<GpuBackendConfig>());
  CHECK_EQ(original_softmax_instr->backend_config<GpuBackendConfig>()
               ->fusion_backend_config()
               .kind(),
           kTritonSoftmaxFusionKind);
  HloComputation* parent_computation = consumer->parent();
  HloModule* parent_module = parent_computation->parent();

  // We clone the consumer to generate a candidate that we fuse into.
  std::unique_ptr<HloInstruction> candidate_instr_ptr = consumer->Clone();
  HloInstruction* consumer_candidate_instr = candidate_instr_ptr.get();

  // Try to merge the producer into candidate fusion.
  consumer_candidate_instr->MergeFusionInstruction(original_softmax_instr);
  HloComputation* fused_computation =
      consumer_candidate_instr->fused_instructions_computation();

  const auto analysis = TritonFusionAnalysis::Execute(*fused_computation);

  if (!analysis.ok()) {
    return std::nullopt;
  }

  // We want our joined fusion to have the correct fusion_kind, backend_config,
  // etc for a triton fusion. So we assemble a new instruction rather than
  // using consumer_candidate_instr, which would not get triton codegen'd.
  std::unique_ptr<HloInstruction> new_softmax_instr_ptr =
      HloInstruction::CreateFusion(
          /*shape=*/consumer_candidate_instr->shape(),
          /*fusion_kind=*/original_softmax_instr->fusion_kind(),
          /*operands=*/consumer_candidate_instr->operands(),
          /*fusion_computation=*/fused_computation,
          /*prefix=*/"triton_softmax_");

  HloInstruction* new_softmax_instr = new_softmax_instr_ptr.get();

  new_softmax_instr->CopyBackendConfigFrom(original_softmax_instr);

  // Now, we incorporate new_softmax_instr into our module.
  parent_computation->AddInstruction(std::move(new_softmax_instr_ptr));

  if (consumer->IsRoot()) {
    parent_computation->set_root_instruction(new_softmax_instr);
  }

  TF_CHECK_OK(consumer->ReplaceAllUsesWith(new_softmax_instr));

  // Remove the replaced instructions and computations from the module.
  HloComputation* original_consumer_computation =
      consumer->fused_instructions_computation();
  TF_CHECK_OK(parent_computation->RemoveInstruction(consumer));
  TF_CHECK_OK(
      parent_module->RemoveEmbeddedComputation(original_consumer_computation));

  CHECK_EQ(0, original_softmax_instr->user_count());

  // Keep a ptr to the original computation so we can remove it from the module.
  HloComputation* original_softmax_computation =
      original_softmax_instr->fused_instructions_computation();

  TF_CHECK_OK(parent_computation->RemoveInstruction(original_softmax_instr));
  TF_CHECK_OK(
      parent_module->RemoveEmbeddedComputation(original_softmax_computation));

  return Cast<HloFusionInstruction>(new_softmax_instr);
}

bool TryMergeProducerAndConsumerFusionsIntoTritonSoftmax(
    HloFusionInstruction* softmax_fusion) {
  // The softmax_fusion should come directly from the matcher. They might have
  // more than a single operand, in this case attempt to fuse into the first
  // operand only.
  if (softmax_fusion->operand_count() > 1) {
    LOG(INFO) << "More than one parameter detected. Will attempt to merge "
                 "fusions only for operand 0 (diamond producer).";
  }

  // TODO(b/313026024): Add support for multiple users
  bool should_try_merging_producer =
      softmax_fusion->operand(0)->user_count() == 1 &&
      softmax_fusion->operand(0)->opcode() == HloOpcode::kFusion;
  // TODO(b/315040476): generalize for multiple users and multi-output
  bool should_try_merging_consumer =
      softmax_fusion->user_count() == 1 &&
      softmax_fusion->users().front()->opcode() == HloOpcode::kFusion &&
      softmax_fusion->users().front()->shape().IsArray();

  bool changed = false;
  if (should_try_merging_producer) {
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
      softmax_fusion = result.value();
      changed = true;
    }
  }

  if (should_try_merging_consumer) {
    HloFusionInstruction* consumer =
        Cast<HloFusionInstruction>(softmax_fusion->users().front());

    VLOG(6) << "Fusing consumer " << consumer->ToShortString() << " into "
            << softmax_fusion->ToShortString();

    std::optional<HloFusionInstruction*> result =
        TryMergeFusionConsumerIntoTritonSoftmaxProducer(consumer,
                                                        softmax_fusion);

    if (!result.has_value()) {
      VLOG(6) << "Did not fuse consumer into "
              << softmax_fusion->ToShortString();
    } else {
      softmax_fusion = result.value();
      changed = true;
    }
  }
  return changed;
}

}  // anonymous namespace

absl::StatusOr<bool> FusionMergerTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  int fused_comps = 0;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    if (comp->IsCustomCallComputation()) {
      continue;
    }

    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (!IsTritonSoftmaxFusion(*instr)) continue;

      VLOG(6) << "Matched triton_softmax fusion: " << instr->ToShortString();

      HloFusionInstruction* softmax = Cast<HloFusionInstruction>(instr);

      bool result =
          TryMergeProducerAndConsumerFusionsIntoTritonSoftmax(softmax);

      if (!result) {
        VLOG(6) << "Did not fuse producer or consumer into "
                << instr->ToShortString();
      } else {
        ++fused_comps;
      }
    }
  }
  return fused_comps > 0;
}
}  // namespace xla::gpu
