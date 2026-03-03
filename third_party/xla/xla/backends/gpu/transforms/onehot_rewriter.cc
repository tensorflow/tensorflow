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

#include "xla/backends/gpu/transforms/onehot_rewriter.h"

#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

namespace {

namespace m = xla::match;

HloInstruction* TraceIota(HloInstruction* inst, int64_t expected_dim) {
  HloInstruction* current = inst;
  int64_t current_dim = expected_dim;
  constexpr int64_t kMaxTraceDepth = 20;

  for (int64_t depth = 0; depth < kMaxTraceDepth; ++depth) {
    if (current->opcode() == HloOpcode::kIota) {
      if (current->shape().element_type() != S32) {
        // TODO: b/477516620 - Support other index types.
        VLOG(4) << "TraceIota: Iota element type mismatch on "
                << current->name() << ". Expected S32, got "
                << current->shape().element_type();
        return nullptr;
      }
      if (Cast<HloIotaInstruction>(current)->iota_dimension() == current_dim) {
        return current;
      }
      VLOG(4) << "TraceIota: Iota dimension mismatch on " << current->name()
              << ". Expected " << current_dim << ", got "
              << Cast<HloIotaInstruction>(current)->iota_dimension();
      return nullptr;
    }

    HloInstruction* next = nullptr;
    if (Match(current, m::AnyOf<HloInstruction>(m::Broadcast(m::Op(&next)),
                                                m::Reshape(m::Op(&next))))) {
      auto next_dim = current->MapUnaryOutputDimToOperandDim(current_dim);
      if (!next_dim.has_value()) {
        VLOG(4) << "TraceIota: MapUnaryOutputDimToOperandDim failed for "
                << current->name() << " dim " << current_dim;
        return nullptr;
      }
      current_dim = *next_dim;
      current = next;
      continue;
    }

    return nullptr;
  }
  VLOG(4) << "TraceIota: Max trace depth reached.";
  return nullptr;
}

// Traces Indices back through Broadcast.
HloInstruction* TraceIndices(HloInstruction* inst, int64_t contract_dim) {
  HloInstruction* current = inst;
  HloInstruction* next = nullptr;
  if (Match(current, m::Broadcast(m::Op(&next)))) {
    // Broadcast must not permute dimensions.
    if (!absl::c_is_sorted(current->dimensions())) {
      return nullptr;
    }
    // Check if the broadcast only adds the contracting dimension.
    bool adds_only_contract_dim = true;
    for (int64_t i = 0; i < current->shape().dimensions().size(); ++i) {
      if (i != contract_dim) {
        if (!current->MapUnaryOutputDimToOperandDim(i).has_value()) {
          adds_only_contract_dim = false;
          break;
        }
      }
    }
    if (adds_only_contract_dim) {
      return next;
    }
  }
  return current;
}

struct OneHotMatch {
  HloInstruction* indices;
  HloInstruction* weights;
  bool lhs_is_one_hot;
};

bool ShouldRewrite(const HloDotInstruction* dot, const OneHotMatch& match) {
  int64_t weights_contract_dim =
      (match.lhs_is_one_hot)
          ? dot->dot_dimension_numbers().rhs_contracting_dimensions(0)
          : dot->dot_dimension_numbers().lhs_contracting_dimensions(0);
  int64_t depth = match.weights->shape().dimensions(weights_contract_dim);

  if (depth == 0 || depth > std::numeric_limits<int32_t>::max()) {
    return false;
  }

  // No rewrite at low depth/high batch, where dot is likely more efficient.
  // Please see go/onehot-microbenchmark to re-evaluate this threshold.
  // Note that the current threshold is defensive (too strict on some hardware),
  // to avoid a complex heuristic.
  int64_t batch = ShapeUtil::ElementsIn(match.indices->shape());
  if (depth < 256 && batch > 1024) {
    VLOG(3) << "Skipping OneHot rewrite for " << dot->name()
            << " due to low depth/high batch ratio. depth: " << depth
            << ", batch: " << batch;
    return false;
  }
  return true;
}

// Returns a match if the dot instruction is a One-Hot encoded matmul.
// 1. Identifies a Dot instruction.
// 2. Checks if one operand is a comparison (EQ) involving an Iota and some
//    Indices.
// 3. Verifies the Iota dimension matches the Dot's contracting dimension.
// 4. Traces through transparent ops (Broadcast, Reshape, Convert, Bitcast) to
//    find the underlying Iota and Indices.
std::optional<OneHotMatch> TryMatchOneHotDot(HloInstruction* instr) {
  HloInstruction *lhs, *rhs;
  if (!Match(instr, m::Dot(m::Op(&lhs), m::Op(&rhs)))) {
    return std::nullopt;
  }
  auto* dot = Cast<HloDotInstruction>(instr);
  const auto& dnums = dot->dot_dimension_numbers();

  if (dnums.lhs_batch_dimensions_size() > 0 ||
      dnums.rhs_batch_dimensions_size() > 0 ||
      dnums.lhs_contracting_dimensions_size() != 1) {
    VLOG(3) << "Dot instruction " << dot->name()
            << " failed dimension check for OneHot match.";
    return std::nullopt;
  }

  // Check both operands for the One-Hot pattern
  for (int i = 0; i < 2; ++i) {
    int64_t contract_dim = (i == 0) ? dnums.lhs_contracting_dimensions(0)
                                    : dnums.rhs_contracting_dimensions(0);

    HloInstruction* root = dot->mutable_operand(i);
    HloInstruction* operand;
    // Strip Convert
    while (Match(root, m::Convert(m::Op(&operand)))) {
      root = operand;
    }

    HloInstruction* compare = nullptr;

    // One-hot encodings often appear as a comparison (Indices == Iota).
    if (Match(root, m::Compare())) {
      compare = root;
    }

    if (!compare ||
        compare->comparison_direction() != ComparisonDirection::kEq) {
      VLOG(3) << "Operand " << i << " of " << dot->name()
              << " is not a valid comparison for OneHot.";
      continue;
    }

    // Check operands of the compare: one must be Iota, one Indices
    HloInstruction* cmp_lhs = compare->mutable_operand(0);
    HloInstruction* cmp_rhs = compare->mutable_operand(1);

    bool lhs_iota = TraceIota(cmp_lhs, contract_dim) != nullptr;
    bool rhs_iota = TraceIota(cmp_rhs, contract_dim) != nullptr;

    HloInstruction* found_indices = nullptr;
    if (lhs_iota && !rhs_iota) {
      found_indices = TraceIndices(cmp_rhs, contract_dim);
    } else if (rhs_iota && !lhs_iota) {
      found_indices = TraceIndices(cmp_lhs, contract_dim);
    } else {
      VLOG(3) << "Tracing failed for operand " << i << " of " << dot->name();
      continue;
    }

    if (found_indices) {
      HloInstruction* weights = dot->mutable_operand(1 - i);
      // Verify shapes: The resulting Gather must have the same rank as the
      // original Dot. Gather rank = indices_rank + weights_rank - 1.
      if (found_indices->shape().dimensions().size() +
              weights->shape().dimensions().size() - 1 !=
          dot->shape().dimensions().size()) {
        VLOG(3) << "Shape mismatch for OneHot match on " << dot->name();
        continue;
      }
      if (dot->shape().element_type() != weights->shape().element_type()) {
        VLOG(3) << "Type mismatch for OneHot match on " << dot->name();
        continue;
      }
      VLOG(2) << "Matched OneHot pattern on " << dot->name();
      OneHotMatch match{found_indices, weights, i == 0};
      if (ShouldRewrite(dot, match)) {
        return match;
      }
    }
  }
  return std::nullopt;
}

// Performs the actual transformation from Dot to Gather.
//
// The transformation ensures robustness against out-of-bounds indices to match
// the semantics of the matched pattern (e.g. Iota == Indices).
// In the original graph, out-of-bounds indices result in a zero one-hot vector
// (as the equality check fails everywhere) and thus a zero result.
//
// To preserve this behavior:
// 1. Clamp the indices to the valid range [0, depth-1] to ensure the Gather
//    instruction does not access invalid memory.
// 2. Compute an "in-bounds" mask (0 <= indices < depth).
// 3. Select between the Gather result and zeros based on the mask.
absl::Status RewriteOneHotDotToGather(HloComputation* computation,
                                      HloDotInstruction* dot,
                                      const OneHotMatch& match) {
  VLOG(2) << "Rewriting OneHot Dot " << dot->name() << " to Gather.";
  HloInstruction* indices = match.indices;
  HloInstruction* weights = match.weights;
  const Shape& dot_shape = dot->shape();

  const auto& dnums = dot->dot_dimension_numbers();
  int64_t weights_contract_dim = (match.lhs_is_one_hot)
                                     ? dnums.rhs_contracting_dimensions(0)
                                     : dnums.lhs_contracting_dimensions(0);

  // Identify bounds and create constants for clamping.
  int64_t depth = weights->shape().dimensions(weights_contract_dim);
  HloInstruction* zero_s32 = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  HloInstruction* depth_s32 = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(depth)));
  HloInstruction* depth_minus_one =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32_t>(depth - 1)));

  HloInstruction* zero_s32_broadcast = computation->AddInstruction(
      HloInstruction::CreateBroadcast(indices->shape(), zero_s32, {}));
  HloInstruction* depth_minus_one_broadcast = computation->AddInstruction(
      HloInstruction::CreateBroadcast(indices->shape(), depth_minus_one, {}));

  // Clamp indices to valid memory range [0, depth-1].
  HloInstruction* clamped_indices =
      computation->AddInstruction(HloInstruction::CreateTernary(
          indices->shape(), HloOpcode::kClamp, zero_s32_broadcast, indices,
          depth_minus_one_broadcast));

  // Create Gather
  Shape indices_shape = indices->shape();
  std::vector<int64_t> new_dims(indices_shape.dimensions().begin(),
                                indices_shape.dimensions().end());
  new_dims.push_back(1);
  Shape reshaped_shape =
      ShapeUtil::MakeShape(indices_shape.element_type(), new_dims);

  HloInstruction* reshaped_indices = computation->AddInstruction(
      HloInstruction::CreateReshape(reshaped_shape, clamped_indices));

  int64_t indices_rank = indices->shape().dimensions().size();
  int64_t weights_rank = weights->shape().dimensions().size();
  int64_t weights_nc_rank = weights_rank - 1;

  GatherDimensionNumbers gnums;
  gnums.mutable_offset_dims()->Reserve(weights_nc_rank);
  int offset = match.lhs_is_one_hot ? indices_rank : 0;
  for (int i = 0; i < weights_nc_rank; ++i) {
    gnums.add_offset_dims(i + offset);
  }

  gnums.add_collapsed_slice_dims(weights_contract_dim);
  gnums.add_start_index_map(weights_contract_dim);
  gnums.set_index_vector_dim(indices_rank);

  std::vector<int64_t> slice_sizes;
  slice_sizes.reserve(weights_rank);
  for (int i = 0; i < weights_rank; ++i) {
    slice_sizes.push_back(
        i == weights_contract_dim ? 1 : weights->shape().dimensions(i));
  }

  HloInstruction* gather =
      computation->AddInstruction(HloInstruction::CreateGather(
          dot->shape(), weights, reshaped_indices, gnums, slice_sizes, false));
  gather->set_metadata(dot->metadata());

  // Create in-bounds mask: (indices >= 0) && (indices < depth).
  Shape mask_shape = ShapeUtil::ChangeElementType(indices->shape(), PRED);
  HloInstruction* depth_s32_broadcast = computation->AddInstruction(
      HloInstruction::CreateBroadcast(indices->shape(), depth_s32, {}));

  HloInstruction* ge_zero =
      computation->AddInstruction(HloInstruction::CreateCompare(
          mask_shape, indices, zero_s32_broadcast, ComparisonDirection::kGe));
  HloInstruction* lt_depth =
      computation->AddInstruction(HloInstruction::CreateCompare(
          mask_shape, indices, depth_s32_broadcast, ComparisonDirection::kLt));
  HloInstruction* in_bounds =
      computation->AddInstruction(HloInstruction::CreateBinary(
          mask_shape, HloOpcode::kAnd, ge_zero, lt_depth));

  // Broadcast mask to match the Dot output shape.
  // The Indices dimensions are either at the start or end of the output.
  std::vector<int64_t> broadcast_dims(indices->shape().dimensions().size());
  int start_dim = match.lhs_is_one_hot ? 0 : weights_nc_rank;
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), start_dim);

  Shape broadcast_mask_shape = ShapeUtil::ChangeElementType(dot_shape, PRED);
  HloInstruction* broadcast_mask =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_mask_shape, in_bounds, broadcast_dims));

  // Select between Gather result and Zeros.
  HloInstruction* zero_scalar =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(dot_shape.element_type())));
  HloInstruction* zeros = computation->AddInstruction(
      HloInstruction::CreateBroadcast(dot_shape, zero_scalar, {}));

  HloInstruction* robust_result =
      computation->AddInstruction(HloInstruction::CreateTernary(
          dot_shape, HloOpcode::kSelect, broadcast_mask, gather, zeros));
  robust_result->set_metadata(dot->metadata());

  RETURN_IF_ERROR(computation->ReplaceInstruction(dot, robust_result));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> OneHotGatherRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // We gate this rewrite behind --xla_enable_fast_math because it propagates 0
  // instead of NaNs in some cases. See
  // b/477516620#comment12 for details.
  if (!module->config().debug_options().xla_enable_fast_math()) {
    VLOG(2) << "Skipping OneHot rewrite due to --xla_enable_fast_math=false.";
    return false;
  }

  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    // We identify candidates in a first pass and perform rewrites in a second
    // pass, after re-running the pattern matching. This avoids issues where
    // a rewrite of one Dot instruction invalidates the operands of another
    // candidate.
    std::vector<HloDotInstruction*> rewrite_candidates;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (TryMatchOneHotDot(instr).has_value()) {
        rewrite_candidates.push_back(Cast<HloDotInstruction>(instr));
      }
    }
    for (HloDotInstruction* dot : rewrite_candidates) {
      if (auto match = TryMatchOneHotDot(dot)) {
        RETURN_IF_ERROR(RewriteOneHotDotToGather(computation, dot, *match));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
