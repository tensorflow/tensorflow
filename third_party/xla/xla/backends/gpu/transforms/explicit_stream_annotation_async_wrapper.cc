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
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
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

bool IsLocalityAnnotation(absl::string_view annotation) {
  return absl::StartsWith(annotation, "locality_domain");
}

bool HasStreamAnnotation(const HloInstruction* instr,
                         absl::string_view annotation) {
  auto it = instr->frontend_attributes().map().find(kXlaStreamAnnotationAttr);
  return it != instr->frontend_attributes().map().end() &&
         it->second == annotation;
}

bool IsMatchingLocalityAsyncPair(const HloInstruction* instr,
                                 absl::string_view annotation) {
  if (instr->opcode() == HloOpcode::kAsyncStart) {
    for (const HloInstruction* user : instr->users()) {
      if (user->opcode() == HloOpcode::kAsyncDone &&
          user->operand(0) == instr && HasStreamAnnotation(user, annotation)) {
        return true;
      }
    }
    return false;
  }
  return instr->opcode() == HloOpcode::kAsyncDone &&
         instr->operand(0)->opcode() == HloOpcode::kAsyncStart &&
         HasStreamAnnotation(instr->operand(0), annotation);
}

bool IsCollectiveOrSendRecv(const HloInstruction* instr) {
  return IsNonFusionCollective(instr) ||
         instr->has_frontend_attribute(kCollectiveGroupMarkerAttr) ||
         HloPredicateIsOp<HloOpcode::kSend, HloOpcode::kSendDone,
                          HloOpcode::kRecv, HloOpcode::kRecvDone>(instr);
}

absl::Status ValidateLocalityComputation(
    const HloComputation* computation,
    absl::flat_hash_set<const HloComputation*>& visited) {
  if (!visited.insert(computation).second) {
    return absl::OkStatus();
  }
  for (const HloInstruction* instr : computation->instructions()) {
    if (IsCollectiveOrSendRecv(instr)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Locality-domain stream call contains unsupported communication "
          "instruction: ",
          instr->name()));
    }
    if (auto it =
            instr->frontend_attributes().map().find(kXlaStreamAnnotationAttr);
        it != instr->frontend_attributes().map().end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Locality-domain stream call contains nested stream annotation \"",
          it->second, "\" on instruction ", instr->name()));
    }
    for (const HloComputation* called : instr->called_computations()) {
      ABSL_RETURN_IF_ERROR(ValidateLocalityComputation(called, visited));
    }
  }
  return absl::OkStatus();
}

absl::Status RejectNestedLocalityAnnotations(
    const HloComputation* computation, absl::string_view enclosing_annotation,
    absl::flat_hash_set<const HloComputation*>& visited) {
  if (!visited.insert(computation).second) {
    return absl::OkStatus();
  }
  for (const HloInstruction* instr : computation->instructions()) {
    if (auto it =
            instr->frontend_attributes().map().find(kXlaStreamAnnotationAttr);
        it != instr->frontend_attributes().map().end() &&
        IsLocalityAnnotation(it->second)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Locality-domain stream annotation \"", it->second,
                       "\" on instruction ", instr->name(),
                       " is nested inside a call with stream annotation \"",
                       enclosing_annotation, "\""));
    }
    for (const HloComputation* called : instr->called_computations()) {
      ABSL_RETURN_IF_ERROR(RejectNestedLocalityAnnotations(
          called, enclosing_annotation, visited));
    }
  }
  return absl::OkStatus();
}

void ClearSchedulingAnnotations(HloInstruction* instr) {
  // These attributes are only valid on the async pairs.
  instr->erase_frontend_attribute(kXlaSchedulingGroupIdAttr);
  instr->erase_frontend_attribute(kXlaStreamAnnotationAttr);
}

static absl::StatusOr<bool> AsynchronizeInstruction(HloInstruction* instr) {
  auto annotation_it =
      instr->frontend_attributes().map().find(kXlaStreamAnnotationAttr);
  if (annotation_it == instr->frontend_attributes().map().end()) {
    return false;
  }
  absl::string_view annotation = annotation_it->second;
  bool is_locality_annotation = IsLocalityAnnotation(annotation);
  if (is_locality_annotation) {
    ABSL_RETURN_IF_ERROR(ParseLocalityDomainStreamAnnotation(annotation).status());
    if (instr->has_frontend_attribute(kCollectiveGroupMarkerAttr)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Locality-domain stream annotation cannot be combined with ",
          kCollectiveGroupMarkerAttr, ": ", instr->name()));
    }
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

  // A previous invocation may already have wrapped a valid locality call.
  // Accept the annotated call-start/call-done pair without trying to wrap it
  // again. Other annotated async operations are not valid locality regions.
  if (instr->IsAsynchronous()) {
    if (is_locality_annotation) {
      const auto* async = DynCast<HloAsyncInstruction>(instr);
      if (async == nullptr ||
          !async->async_wrapped_computation()
               ->CanExpandIntoSingleInstruction() ||
          async->async_wrapped_opcode() != HloOpcode::kCall ||
          !IsMatchingLocalityAsyncPair(instr, annotation)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Locality-domain stream annotation requires a call-start/"
            "call-done pair; got ",
            HloOpcodeString(instr->opcode()), " on ", instr->name()));
      }
      if (instr->opcode() == HloOpcode::kAsyncStart) {
        absl::flat_hash_set<const HloComputation*> visited;
        ABSL_RETURN_IF_ERROR(ValidateLocalityComputation(
            async->async_wrapped_instruction()->called_computations()[0],
            visited));
      }
    }
    return false;
  }

  if (is_locality_annotation) {
    if (instr->opcode() != HloOpcode::kCall) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Locality-domain stream annotation requires an existing kCall; got ",
          HloOpcodeString(instr->opcode()), " on ", instr->name()));
    }
    auto inlineable_it = instr->frontend_attributes().map().find("inlineable");
    if (inlineable_it == instr->frontend_attributes().map().end() ||
        inlineable_it->second != "false") {
      return absl::InvalidArgumentError(absl::StrCat(
          "Locality-domain stream annotation requires a non-inlineable kCall "
          "with inlineable=\"false\": ",
          instr->name()));
    }
    absl::flat_hash_set<const HloComputation*> visited;
    ABSL_RETURN_IF_ERROR(
        ValidateLocalityComputation(instr->called_computations()[0], visited));
  }

  // Collective start/done and copy start/done — nothing to do.
  if (IsNonFusionCollective(instr) ||
      instr->opcode() == HloOpcode::kCopyStart ||
      instr->opcode() == HloOpcode::kCopyDone) {
    return false;
  }

  // Instructions for which no thunks are emitted do no work on any stream, so
  // there is nothing to run asynchronously, and wrapping them would only add a
  // synchronization point on the annotated stream. Drop the annotations
  // instead, as they are only legal on async pairs.
  if (EmitsNoThunks(*instr)) {
    ClearSchedulingAnnotations(instr);
    return true;
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

  // The cleanup below would silently drop a nested locality placement.
  if (!is_locality_annotation) {
    absl::flat_hash_set<const HloComputation*> visited;
    ABSL_RETURN_IF_ERROR(RejectNestedLocalityAnnotations(
        instr->called_computations()[0],
        original_attributes.map().at(kXlaStreamAnnotationAttr), visited));
  }

  // These annotations are only legal on the async instructions and
  // can cause issues if the annotations remain on the inner operations,
  // so we clear them before creating the async pair.
  for (auto* inner_instr : instr->called_computations()[0]->instructions()) {
    ClearSchedulingAnnotations(inner_instr);
  }
  ClearSchedulingAnnotations(instr);

  ABSL_ASSIGN_OR_RETURN(HloInstruction * done,
                   computation->CreateAsyncInstructions(
                       instr, {}, HloInstruction::kParallelExecutionThread,
                       /*replace=*/true));
  // Replace the original attributes after creating the async pair.
  done->set_frontend_attributes(original_attributes);
  done->mutable_operand(0)->set_frontend_attributes(original_attributes);
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   done->backend_config<GpuBackendConfig>());
  // Set earliest schedule of done op to be false so it can be scheduled
  // far apart from start.
  gpu_config.set_force_earliest_schedule(false);
  ABSL_RETURN_IF_ERROR(done->set_backend_config(gpu_config));
  VLOG(5) << "Created async instruction: " << done->ToString();
  return true;
}
}  // namespace

absl::StatusOr<bool> ExplicitStreamAnnotationAsyncWrapper::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Locality annotations are errors, rather than ignored annotations, when
  // embedded in computations that cannot be independently asynchronized.
  for (HloComputation* computation : module->computations()) {
    if (!computation->IsFusionComputation() &&
        !computation->IsAsyncComputation() &&
        !IsNestedInFusionComputation(computation)) {
      continue;
    }
    for (HloInstruction* instr : computation->instructions()) {
      auto it =
          instr->frontend_attributes().map().find(kXlaStreamAnnotationAttr);
      if (it == instr->frontend_attributes().map().end() ||
          !IsLocalityAnnotation(it->second)) {
        continue;
      }
      return absl::InvalidArgumentError(absl::StrCat(
          "Locality-domain stream annotation is not allowed inside an async "
          "or fusion computation: ",
          instr->name()));
    }
  }

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
      ABSL_ASSIGN_OR_RETURN(bool result, AsynchronizeInstruction(instr));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace xla::gpu
