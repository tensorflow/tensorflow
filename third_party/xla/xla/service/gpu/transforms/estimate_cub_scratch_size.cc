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

#include "xla/service/gpu/transforms/estimate_cub_scratch_size.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/cub_sort_thunk.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

// Rewrites a single sort instruction with a custom call.
absl::StatusOr<bool> EstimateCubScratchSize::RunOnInstruction(
    HloCustomCallInstruction* custom_call) {
  CHECK_EQ(custom_call->custom_call_target(),
           kCubDeviceRadixSortUnassignedScratchSizeTarget);
  const Shape& key_shape = custom_call->operand(0)->shape();
  PrimitiveType key_type = key_shape.element_type();
  std::optional<PrimitiveType> value_type;
  if (custom_call->operand_count() == 2) {
    value_type = custom_call->operand(1)->shape().element_type();
  }

  ASSIGN_OR_RETURN(
      std::unique_ptr<CubSortRunnerInterface> runner,
      CubSortRunnerInterface::Create(key_type, value_type, platform_name_));

  int64_t num_elements = Product(key_shape.dimensions());
  // It is assumed that the sorting happens on the innermost dimension.
  int64_t batch_size = num_elements / key_shape.dimensions().back();

  ASSIGN_OR_RETURN(int64_t scratch_size,
                   runner->GetScratchSize(num_elements, batch_size));

  // Align and increase scratch size to fit the offsets.
  if (batch_size > 1) {
    scratch_size += sizeof(int) - scratch_size % sizeof(int);
    scratch_size += (batch_size + 1) * sizeof(int);
  }

  // Update the custom call.
  Shape new_shape = custom_call->shape();
  new_shape.mutable_tuple_shapes()->back() =
      ShapeUtil::MakeShape(U8, {scratch_size});
  HloInstruction* new_custom_call =
      custom_call->AddInstruction(HloInstruction::CreateCustomCall(
          new_shape, absl::MakeSpan(custom_call->operands()),
          kCubDeviceRadixSortTarget));
  new_custom_call->SetupDerivedInstruction(custom_call);
  RETURN_IF_ERROR(custom_call->parent()->ReplaceInstructionWithDifferentShape(
      custom_call, new_custom_call));
  return true;
}

// Rewrites the sorts in the given computation into calls to CUB.
absl::StatusOr<bool> EstimateCubScratchSize::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloCustomCallInstruction*> custom_calls;
  for (auto* inst : computation->instructions()) {
    if (auto custom_call = DynCast<HloCustomCallInstruction>(inst)) {
      if (custom_call->custom_call_target() ==
          kCubDeviceRadixSortUnassignedScratchSizeTarget) {
        custom_calls.push_back(custom_call);
      }
    }
  }
  bool changed = false;
  for (auto* call : custom_calls) {
    ASSIGN_OR_RETURN(bool result, RunOnInstruction(call));
    changed |= result;
  }
  return changed;
}

// Replace compatible sort operations with custom calls.
absl::StatusOr<bool> EstimateCubScratchSize::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      3, "EstimateCubScratchSize::RunImpl(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(
      3, "EstimateCubScratchSize::RunImpl(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
