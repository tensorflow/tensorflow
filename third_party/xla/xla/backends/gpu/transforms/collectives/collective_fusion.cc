/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/collectives/collective_fusion.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/codegen/triton/collective_emitter.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"

namespace xla::gpu {

namespace {

static constexpr bool kMultimemDisabled = false;

// Manually creates a fusion instruction for a collective op.
// We cannot use HloComputation::CreateFusionInstruction because it checks
// HloInstruction::IsFusible(), which returns false for collectives because of
// side effects being true.
HloFusionInstruction* CreateCollectiveFusionInstruction(
    HloComputation* computation, const HloInstruction* instr,
    HloInstruction::FusionKind fusion_kind) {
  HloModule* module = computation->parent();

  HloComputation::Builder builder(absl::StrCat("fused_", instr->name()));

  std::vector<HloInstruction*> parameters;
  parameters.reserve(instr->operand_count());
  for (auto const& [i, operand] : llvm::enumerate(instr->operands())) {
    parameters.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i, operand->shape(), absl::StrCat("param_", i))));
  }
  HloInstruction* fused_root = builder.AddInstruction(
      instr->CloneWithNewOperands(instr->shape(), parameters));
  HloComputation* fused_computation =
      module->AddEmbeddedComputation(builder.Build(fused_root));
  HloInstruction* fusion_instr =
      computation->AddInstruction(HloInstruction::CreateFusion(
          instr->shape(), fusion_kind,
          absl::MakeSpan(instr->operands().begin(), instr->operands().end()),
          fused_computation, absl::StrCat(instr->name(), "-")));
  // Propagate metadata.
  fusion_instr->set_metadata(instr->metadata());
  fusion_instr->set_frontend_attributes(fused_root->frontend_attributes());
  fusion_instr->CopyBackendConfigFrom(instr);
  return Cast<HloFusionInstruction>(fusion_instr);
}

// Collect candidates for fusion.
std::vector<HloInstruction*> GetFusionCandidates(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::flat_hash_set<HloOpcode>& instructions_to_fuse) {
  std::vector<HloInstruction*> candidates;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      if (!instructions_to_fuse.contains(instr->opcode())) {
        continue;
      }
      auto gpu_config_status = instr->backend_config<GpuBackendConfig>();
      if (!gpu_config_status.ok()) {
        continue;
      }

      if (IsTritonCollectiveKernel(
              gpu_config_status->collective_backend_config()
                  .kernel_strategy())) {
        candidates.push_back(instr);
      }
    }
  }
  return candidates;
}

bool ShouldFlatten(const HloInstruction* instr) {
  const int64_t size_bytes =
      ShapeUtil::ElementsIn(instr->shape()) *
      primitive_util::ByteWidth(instr->shape().element_type());
  const bool has_rank_higher_than_1 =
      instr->shape().IsArray() && instr->shape().dimensions().size() > 1;
  return has_rank_higher_than_1 &&
         GetAllReduceStrategy(size_bytes, kMultimemDisabled) ==
             se::gpu::AllReduceStrategy::kTwoShot;
}

// Normalizes legacy all-reduce-start/done into modern generic async boundaries.
// Normally this does not exist but it can be created by explicitly calling
// psum_start/end in jax.
absl::StatusOr<HloInstruction*> NormalizeAsyncCandidate(
    HloAllReduceInstruction* start_instr) {
  HloComputation* computation = start_instr->parent();

  HloInstruction* done_instr = nullptr;
  for (HloInstruction* user : start_instr->users()) {
    if (user->opcode() == HloOpcode::kAllReduceDone) {
      done_instr = user;
      break;
    }
  }
  TF_RET_CHECK(done_instr != nullptr)
      << "Missing all-reduce-done for " << start_instr->ToString();

  // Create a temporary synchronous instruction for working with create async.
  HloInstruction* sync_ar =
      computation->AddInstruction(HloInstruction::CreateAllReduce(
          done_instr->shape(), start_instr->operands(), start_instr->to_apply(),
          start_instr->device_list(), start_instr->constrain_layout(),
          start_instr->channel_id(), start_instr->use_global_device_ids()));
  sync_ar->CopyBackendConfigFrom(start_instr);
  sync_ar->set_metadata(start_instr->metadata());
  sync_ar->set_frontend_attributes(start_instr->frontend_attributes());
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * new_done,
      computation->CreateAsyncInstructions(sync_ar, /*context_shapes=*/{},
                                           HloInstruction::kMainExecutionThread,
                                           /*replace=*/false));

  HloInstruction* new_start = new_done->mutable_operand(0);

  // new_start has a tuple.
  ABSL_RETURN_IF_ERROR(start_instr->ReplaceAllUsesWithDifferentShape(new_start));
  ABSL_RETURN_IF_ERROR(done_instr->ReplaceAllUsesWith(new_done));
  ABSL_RETURN_IF_ERROR(start_instr->CopyAllControlDepsTo(new_start, new_start));
  ABSL_RETURN_IF_ERROR(done_instr->CopyAllControlDepsTo(new_done, new_done));
  new_start->set_frontend_attributes(start_instr->frontend_attributes());
  new_done->set_frontend_attributes(start_instr->frontend_attributes());
  // Clean up.
  ABSL_RETURN_IF_ERROR(start_instr->DropAllControlDeps());
  ABSL_RETURN_IF_ERROR(done_instr->DropAllControlDeps());
  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(done_instr));
  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(start_instr));
  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(sync_ar));

  return new_start->called_computations().front()->root_instruction();
}

absl::Status FuseCandidate(HloInstruction* candidate,
                           const GpuTopology& gpu_topology,
                           const DeviceAssignment* device_assignment) {
  if (candidate->opcode() == HloOpcode::kAllReduceStart) {
    ABSL_ASSIGN_OR_RETURN(candidate, NormalizeAsyncCandidate(
                                    Cast<HloAllReduceInstruction>(candidate)));
  }
  const bool should_flatten = ShouldFlatten(candidate);
  HloComputation* computation = candidate->parent();
  HloFusionInstruction* fusion_instr = CreateCollectiveFusionInstruction(
      computation, candidate, HloInstruction::FusionKind::kCustom);
  ABSL_RETURN_IF_ERROR(candidate->ReplaceAllUsesWith(fusion_instr));
  if (computation->root_instruction() == candidate) {
    computation->set_root_instruction(fusion_instr);
  }
  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(candidate));
  if (should_flatten) {
    ABSL_RETURN_IF_ERROR(FlattenCollectiveFusion(fusion_instr));
  }
  // NB: Must be done after flattening.
  ABSL_RETURN_IF_ERROR(TrySetGpuBackendConfigForCollective(
      gpu_topology, fusion_instr, device_assignment));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> CollectiveFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  ABSL_ASSIGN_OR_RETURN(
      absl::flat_hash_set<HloOpcode> instructions_to_fuse,
      OpcodesForTritonCollectives(module->config().debug_options()));
  if (instructions_to_fuse.empty()) {
    VLOG(3) << "Collective fusion is not enabled for module " << module->name();
    return false;
  }

  const DeviceAssignment* device_assignment = nullptr;
  if (module->config().has_static_device_assignment()) {
    device_assignment = &module->config().static_device_assignment();
  }
  std::vector<HloInstruction*> candidates =
      GetFusionCandidates(module, execution_threads, instructions_to_fuse);
  VLOG(3) << "Found " << candidates.size()
          << " candidates for collective fusion.";
  for (HloInstruction* candidate : candidates) {
    ABSL_RETURN_IF_ERROR(FuseCandidate(candidate, gpu_topology_, device_assignment));
  }
  return !candidates.empty();
}

}  // namespace xla::gpu
