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

#include "xla/backends/gpu/transforms/scan_rewriter.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

absl::StatusOr<bool> ScanRewriter::RunOnComputation(
    HloComputation* computation) {
  bool changed = false;
  std::vector<HloScanInstruction*> scans;
  for (HloInstruction* inst : computation->instructions()) {
    if (inst->opcode() == HloOpcode::kScan) {
      scans.push_back(xla::Cast<HloScanInstruction>(inst));
    }
  }

  for (HloScanInstruction* scan : scans) {
    // Skip if not a plain inclusive sum.
    if (scan->is_reverse() || !scan->is_associative()) {
      continue;
    }
    if (scan->inputs().size() != 1 || scan->inits().size() != 1) {
      continue;
    }
    if (absl::c_any_of(scan->users(), [](HloInstruction* user) {
          return user->opcode() != HloOpcode::kGetTupleElement ||
                 user->tuple_index() > 0;
        })) {
      continue;
    }
    HloInstruction* init = scan->inits().front();
    if (!init->IsConstant() || !init->literal().IsZero({})) {
      continue;
    }
    HloInstruction* root = scan->to_apply()->root_instruction();
    if (root->opcode() != HloOpcode::kTuple || root->operand_count() != 2 ||
        root->operand(0) != root->operand(1)) {
      continue;
    }
    auto binary_op = root->operand(0)->opcode();
    if (binary_op != HloOpcode::kAdd) {
      continue;
    }
    const Shape& shape = scan->shape().tuple_shapes(0);
    if (!shape.IsArray()) {
      continue;
    }

    int64_t scan_dim = scan->scan_dimension();
    int64_t row_length = shape.dimensions(scan_dim);
    int64_t vector_length = 1;
    int64_t column_length = 1;
    bool found_scan_dim = false;
    for (int64_t dim : shape.layout().minor_to_major()) {
      if (dim == scan_dim) {
        found_scan_dim = true;
      } else if (!found_scan_dim) {
        vector_length *= shape.dimensions(dim);
      } else {
        column_length *= shape.dimensions(dim);
      }
    }

    // Skip if scan dimension is not the major dimension.
    if (vector_length > 1) {
      continue;
    }

    // Create the custom call.
    Shape scratch_shape =
        ShapeUtil::MakeShape(U8, {0});  // Empty shape, assigned later.
    Shape new_result_shape = ShapeUtil::MakeTupleShape({shape, scratch_shape});

    HloInstruction* custom_call =
        computation->AddInstruction(HloInstruction::CreateCustomCall(
            new_result_shape, absl::MakeSpan(scan->operands()).first(1),
            kCubDeviceScanUnassignedScratchSizeTarget));

    CubScanOptions::Kind kind = [&]() {
      switch (binary_op) {
        case HloOpcode::kAdd:
          return CubScanOptions::SUM;
        default:
          return CubScanOptions::KIND_INVALID;
      }
    }();

    // Pass attributes via backend_config
    xla::CubScanOptions options;
    options.set_vector_length(vector_length);
    options.set_row_length(row_length);
    options.set_column_length(column_length);
    options.set_kind(kind);
    options.set_is_reverse(scan->is_reverse());
    RETURN_IF_ERROR(custom_call->set_backend_config(options));

    RETURN_IF_ERROR(scan->ReplaceAllUsesWithDifferentShape(custom_call));
    RETURN_IF_ERROR(computation->RemoveInstruction(scan));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> ScanRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace xla::gpu
