/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/gpu/transforms/explicit_stream_annotation_async_wrapper.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// Returns true if `computation` is, or is transitively called from, a fusion
// computation. Sort comparators, reduce functions, etc. embedded inside a
// fusion fall into this category and must not be asynchronized.
static bool IsNestedInFusionComputation(const HloComputation* computation) {
  for (const auto& [caller, count] : computation->caller_computations()) {
    if (caller->IsFusionComputation() || IsNestedInFusionComputation(caller)) {
      return true;
    }
  }
  return false;
}

void ClearSchedulingAnnotations(HloInstruction* instr) {
  // These attributes are only valid on the async pairs.
  instr->erase_frontend_attribute(kXlaSchedulingGroupIdAttr);
  instr->erase_frontend_attribute(kXlaStreamAnnotationAttr);
}

static absl::StatusOr<bool> AsynchronizeInstruction(HloInstruction* instr) {
  if (!instr->frontend_attributes().map().contains(kXlaStreamAnnotationAttr)) {
    return false;
  }
  HloComputation* computation = instr->parent();

  // Instructions inside a fusion or async computation body — or in any
  // computation transitively nested inside a fusion (e.g. sort comparators,
  // reduce functions) — cannot be asynchronized directly.
  if (instr->parent()->IsFusionComputation() ||
      instr->parent()->IsAsyncComputation() ||
      IsNestedInFusionComputation(instr->parent())) {
    return false;
  }

  // Already async or a collective start/done — nothing to do.
  if (instr->IsAsynchronous() || IsNonFusionCollective(instr) ||
      instr->opcode() == HloOpcode::kCopyStart ||
      instr->opcode() == HloOpcode::kCopyDone) {
    return false;
  }

  // If not already a kCall, wrap the instruction in one and move all
  // frontend attributes to the wrapper so the inner instruction is clean.
  if (instr->opcode() != HloOpcode::kCall) {
    auto original_attributes = instr->frontend_attributes();
    instr->set_frontend_attributes(FrontendAttributes{});
    instr = computation->CreateCallInstruction({instr});
    instr->set_frontend_attributes(original_attributes);
  }

  auto original_attributes = instr->frontend_attributes();

  // These annotations are only legal on the async instructions and
  // can cause issues if the annotations remain on the inner operations,
  // so we clear them before creating the async pair.
  for (auto* inner_instr : instr->called_computations()[0]->instructions()) {
    ClearSchedulingAnnotations(inner_instr);
  }
  ClearSchedulingAnnotations(instr);

  ASSIGN_OR_RETURN(
      HloInstruction * done,
      computation->CreateAsyncInstructions(
          instr, {}, ExplicitStreamAnnotationAsyncWrapper::kMainExecutionThread,
          /*replace=*/true));
  // Replace the original attributes after creating the async pair.
  done->set_frontend_attributes(original_attributes);
  done->mutable_operand(0)->set_frontend_attributes(original_attributes);
  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   done->backend_config<GpuBackendConfig>());
  // Set earliest schedule of done op to be false so it can be scheduled
  // far apart from start.
  gpu_config.set_force_earliest_schedule(false);
  RETURN_IF_ERROR(done->set_backend_config(gpu_config));
  VLOG(5) << "Created async instruction: " << done->ToString();
  return true;
}
}  // namespace

absl::StatusOr<bool> ExplicitStreamAnnotationAsyncWrapper::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  // Iterate in reverse post-order (callers before callees) so that when a kCall
  // is asynchronized and ClearSchedulingAnnotations is applied to its called
  // computation's instructions, those annotations are already gone by the time
  // the inner computation is visited.
  auto computations = module->MakeNonfusionComputations(execution_threads);
  for (auto it = computations.rbegin(); it != computations.rend(); ++it) {
    if ((*it)->IsAsyncComputation() || IsNestedInFusionComputation(*it)) {
      continue;
    }
    for (HloInstruction* instr : (*it)->instructions()) {
      ASSIGN_OR_RETURN(bool result, AsynchronizeInstruction(instr));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace xla::gpu
