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
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"

namespace xla::gpu {

absl::StatusOr<bool> ScanRewriter::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloScanInstruction*> scans;
  for (HloInstruction* inst : computation->instructions()) {
    if (hlo_query::IsStandardAssociativeScan(inst)) {
      scans.push_back(xla::Cast<HloScanInstruction>(inst));
    }
  }

  bool changed = false;
  for (HloScanInstruction* scan : scans) {
    // IsStandardAssociativeScan ensured the single input, single carry form.
    TF_RET_CHECK(scan->num_carries() == 1 && scan->operand_count() == 2);
    // A zero init needs no seeding. A non zero init is a seed: by
    // associativity, the seeded inclusive scan is the unseeded scan combined
    // elementwise with the broadcast seed, applied after the custom call.
    HloInstruction* seed = scan->inits().front();
    const HloInstruction* init = seed;
    while (init->opcode() == HloOpcode::kBroadcast) {
      init = init->operand(0);
    }
    const bool init_is_zero = init->IsConstant() && init->literal().IsAll(0);
    // Check that the applied op is an inclusive add.
    const HloInstruction* root = scan->to_apply()->root_instruction();
    if (root->opcode() != HloOpcode::kTuple || root->operand_count() != 2 ||
        root->operand(0) != root->operand(1)) {
      continue;
    }
    auto binary_op = root->operand(0)->opcode();
    if (binary_op != HloOpcode::kAdd) {
      continue;
    }

    // Check same shape/layout for input and output.
    HloInstruction* input = scan->inputs().front();
    const Shape& shape = input->shape();
    if (shape != scan->shape().tuple_shapes(0)) {
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
      } else if (found_scan_dim) {
        column_length *= shape.dimensions(dim);
      } else {
        vector_length *= shape.dimensions(dim);
      }
    }

    // Skip if the scan dimension is not the minormost dimension (rows must
    // be contiguous), or is empty (nothing to scan, and the final carry
    // would have no boundary element to slice).
    if (vector_length > 1 || row_length == 0) {
      continue;
    }

    // The final carry of an inclusive scan is its boundary output element,
    // so live carry users can be fed a slice of the output. Live users are
    // get-tuple-elements (IsStandardAssociativeScan); dead users may have
    // any opcode.
    bool carry_used = false;
    bool has_live_user = false;
    for (const HloInstruction* user : scan->users()) {
      if (user->user_count() == 0 && !user->IsRoot()) {
        continue;
      }
      has_live_user = true;
      if (user->tuple_index() == 1) {
        carry_used = true;
      }
    }
    // A scan with no live users is dead code; leave it to DCE.
    if (!has_live_user) {
      continue;
    }

    // Create the custom call.
    Shape scratch_shape =
        ShapeUtil::MakeShape(U8, {0});  // Empty shape, assigned later.
    Shape result_shape = ShapeUtil::MakeTupleShape({shape, scratch_shape});

    // Create a layout-constrained custom call.
    HloInstruction* custom_call =
        computation->AddInstruction(HloInstruction::CreateCustomCall(
            result_shape, {input}, kCubDeviceScanUnassignedScratchSizeTarget,
            {shape}));

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

    HloInstruction* out = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(shape, custom_call, 0));
    if (!init_is_zero) {
      // out[i] = seed + unseeded_out[i], by associativity of the combiner.
      std::vector<int64_t> broadcast_dims;
      broadcast_dims.reserve(shape.dimensions().size() - 1);
      for (int64_t dim = 0; dim < shape.dimensions().size(); ++dim) {
        if (dim != scan_dim) {
          broadcast_dims.push_back(dim);
        }
      }
      HloInstruction* broadcast_seed = computation->AddInstruction(
          HloInstruction::CreateBroadcast(shape, seed, broadcast_dims));
      out = computation->AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, out, broadcast_seed));
    }

    HloInstruction* carry = nullptr;
    if (carry_used) {
      // The final carry is the output element at the last scanned position:
      // the end of the row, or its start for reverse scans.
      std::vector<int64_t> starts(shape.dimensions().size(), 0);
      std::vector<int64_t> limits(shape.dimensions().begin(),
                                  shape.dimensions().end());
      std::vector<int64_t> strides(shape.dimensions().size(), 1);
      if (scan->is_reverse()) {
        limits[scan_dim] = 1;
      } else {
        starts[scan_dim] = row_length - 1;
      }
      Shape slice_shape = shape;
      slice_shape.set_dimensions(scan_dim, 1);
      HloInstruction* carry_slice =
          computation->AddInstruction(HloInstruction::CreateSlice(
              slice_shape, out, starts, limits, strides));
      // With one output and one carry, the carry shape is the tuple's last
      // element (checked by the TF_RET_CHECK above).
      carry = computation->AddInstruction(HloInstruction::CreateReshape(
          scan->shape().tuple_shapes().back(), carry_slice));
    }

    // Rewire the live get-tuple-element users. The scan and its dead users
    // stay behind with their shapes intact until DCE collects them.
    std::vector<HloInstruction*> users(scan->users().begin(),
                                       scan->users().end());
    for (HloInstruction* user : users) {
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        continue;
      }
      if (user->tuple_index() == 0) {
        RETURN_IF_ERROR(computation->ReplaceInstruction(user, out));
      } else if (carry != nullptr) {
        RETURN_IF_ERROR(computation->ReplaceInstruction(user, carry));
      }
    }
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
