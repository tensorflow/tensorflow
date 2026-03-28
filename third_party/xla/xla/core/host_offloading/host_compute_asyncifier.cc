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

#include "xla/core/host_offloading/host_compute_asyncifier.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/core/host_offloading/hlo_host_device_type_call_wrapper.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/layout.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {

namespace {

// Removes tiles and memory spaces from all instructions inside `computation`.
void RemoveTilesAndMemorySpaces(HloComputation* computation) {
  for (HloInstruction* instruction : computation->instructions()) {
    VLOG(1) << absl::StreamFormat(
        "Removing tiles and memory spaces from \"%s\".",
        instruction->ToString());
    ShapeUtil::ForEachMutableSubshape(
        instruction->mutable_shape(),
        [](Shape* subshape, const ShapeIndex& subshape_index) {
          if (!subshape->has_layout()) {
            return;
          }
          Layout* layout = subshape->mutable_layout();
          layout->clear_tiles();
          layout->set_memory_space(Layout::kDefaultMemorySpace);
        });
  }
}
}  // namespace

absl::StatusOr<bool> HostComputeAsyncifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool modified = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* call : computation->instructions()) {
      if (call->opcode() != HloOpcode::kCall) {
        continue;
      }
      if (!backend_config_device_type_is_host_fn_(call)) {
        continue;
      }
      VLOG(1) << "Host Call: " << call->name();

      HloComputation* parent_computation = call->parent();
      HloComputation* host_computation = call->called_computations().front();
      HloCallInstruction* call_instr =
          absl::down_cast<HloCallInstruction*>(call);
      CHECK_NE(call_instr, nullptr);

      TF_ASSIGN_OR_RETURN(
          HloCallInstruction * call_instr_no_tuple_operands,
          HloHostDeviceTypeCallWrapper::RemoveTupleParameters(call_instr));
      TF_RETURN_IF_ERROR(TupleSimplifier().Run(module).status());
      TF_ASSIGN_OR_RETURN(
          HloInstruction * call_instr_no_constants,
          HloHostDeviceTypeCallWrapper::MaterializeConstantsOnHostComputation(
              call_instr_no_tuple_operands));

      VLOG(1) << "Call instruction without constants: "
              << call_instr_no_constants->name();

      TF_RET_CHECK(call_instr_no_constants->operands().size() ==
                   host_computation->num_parameters())
          << "Expected the number of operands to match the number of "
             "parameters "
             "of the host called computation.";
      TF_RET_CHECK(
          ShapeUtil::Equal(host_computation->root_instruction()->shape(),
                           call_instr_no_constants->shape()))
          << "Shape mismatch between the host computation and the "
             "corresponding "
             "host call.";

      TF_ASSIGN_OR_RETURN(
          HloInstruction * async_done,
          parent_computation->CreateAsyncInstructions(
              call_instr_no_constants, {ShapeUtil::MakeScalarShape(U32)},
              HloInstruction::kHostThread,
              /*replace=*/true, /*override_names=*/true));
      if (call_instr_no_constants->has_frontend_attributes()) {
        HloInstruction* async_start = async_done->async_chain_start();
        async_start->set_frontend_attributes(
            call_instr_no_constants->frontend_attributes());
        async_done->set_frontend_attributes(
            call_instr_no_constants->frontend_attributes());
      }
      VLOG(1) << "Turning " << call_instr_no_constants->name()
              << " into an async instruction " << async_done->name();

      VLOG(1) << "Replacing" << call_instr_no_constants->name() << " with "
              << async_done->name();
      TF_RETURN_IF_ERROR(
          call_instr_no_constants->ReplaceAllUsesWith(async_done));

      RemoveTilesAndMemorySpaces(host_computation);

      modified = true;
    }
  }

  if (modified) {
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());

    if (module->has_schedule()) {
      TF_RETURN_IF_ERROR(module->schedule().Update());
    }
  }
  return modified;
}

}  // namespace xla
