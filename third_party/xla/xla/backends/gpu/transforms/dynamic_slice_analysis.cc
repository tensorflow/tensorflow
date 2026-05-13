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

#include "xla/backends/gpu/transforms/dynamic_slice_analysis.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

//===-----------------------------------------------------------------------===/
// DynamicSliceDescriptor
//===-----------------------------------------------------------------------===/

static std::optional<absl::InlinedVector<int64_t, 4>>
ComputeByteStridesForSubByte(const Shape& shape) {
  absl::InlinedVector<int64_t, 4> strides(shape.dimensions().size());
  int64_t stride_in_bits = ShapeUtil::ElementSizeInBits(shape);
  for (int32_t i : shape.layout().minor_to_major()) {
    if (stride_in_bits % CHAR_BIT != 0) {
      strides[i] = -1;
    } else {
      strides[i] = stride_in_bits / CHAR_BIT;
    }
    stride_in_bits *= shape.dimensions(i);
  }
  return strides;
}

static std::optional<absl::InlinedVector<int64_t, 4>> ComputeByteStrides(
    const Shape& shape) {
  auto strides = ShapeUtil::ByteStrides(shape);
  if (strides) {
    return strides;
  }
  return ComputeByteStridesForSubByte(shape);
}

static bool IsZeroOffset(const HloInstruction* slice, int32_t dim) {
  return GetSliceSize(slice, dim) == slice->operand(0)->shape().dimensions(dim);
}

int32_t GetFirstOffsetOperandIndex(const HloInstruction* slice) {
  CHECK(slice->opcode() == HloOpcode::kDynamicSlice ||
        slice->opcode() == HloOpcode::kDynamicUpdateSlice);
  return slice->opcode() == HloOpcode::kDynamicSlice ? 1 : 2;
}

int64_t GetSliceSize(const HloInstruction* slice, int32_t dim) {
  if (slice->opcode() == HloOpcode::kDynamicSlice) {
    return slice->dynamic_slice_sizes()[dim];
  }
  CHECK_EQ(slice->opcode(), HloOpcode::kDynamicUpdateSlice);
  return slice->operand(1)->shape().dimensions(dim);
}

// Evaluates the total byte offset for a DS/DUS at a given induction variable
// value by substituting into HloEvaluator.
static absl::StatusOr<int64_t> EvaluateByteOffsetAtIteration(
    const HloInstruction* instr, absl::Span<const int64_t> byte_strides,
    const HloInstruction* induction_var, int64_t ivar_value) {
  int32_t first_offset_index = GetFirstOffsetOperandIndex(instr);
  int32_t rank = instr->operand(0)->shape().dimensions().size();

  Literal ivar_literal(induction_var->shape());
  RETURN_IF_ERROR(ivar_literal.SetIntegralAsS64({}, ivar_value));

  absl::flat_hash_map<const HloInstruction*, const LiteralBase*> substitutions;
  substitutions[induction_var] = &ivar_literal;

  HloEvaluator evaluator(/*max_loop_iterations=*/0);
  int64_t total_byte_offset = 0;

  // Iterate over each dimension's offset operand of the DS/DUS.
  for (int32_t i = 0; i < rank; ++i) {
    const HloInstruction* operand = instr->operand(i + first_offset_index);

    if (IsZeroOffset(instr, i)) {
      continue;
    }

    if (byte_strides[i] < 0) {
      return Internal("Non-byte-aligned dimension in slice.");
    }

    if (operand->opcode() == HloOpcode::kConstant) {
      auto value = LiteralUtil::LiteralAsScalarInt64(operand->literal());
      if (!value) {
        return Internal("Failed to read constant offset.");
      }
      total_byte_offset += *value * byte_strides[i];
      continue;
    }

    ASSIGN_OR_RETURN(Literal offset_literal,
                     evaluator.Evaluate(operand, {}, true, substitutions));

    auto offset_value = LiteralUtil::LiteralAsScalarInt64(offset_literal);
    if (!offset_value) {
      return Internal("Failed to evaluate dynamic offset.");
    }

    total_byte_offset += *offset_value * byte_strides[i];
  }

  return total_byte_offset;
}

// Attempts to identify a staggered induction variable created by loop
// pipelining. The pipeliner peels the first iteration and creates a new tuple
// slot that carries the induction variable value from the previous iteration.
// Pattern: body parameter GTE(staggered_index), where the ROOT tuple assigns
// GTE(param, ivar_index) to that slot.
struct StaggeredVariable {
  const HloInstruction* loop;
  const HloInstruction* gte;
  int64_t init_value;
};

static const HloInstruction* StripCopies(const HloInstruction* instr) {
  while (instr->opcode() == HloOpcode::kCopy) {
    instr = instr->operand(0);
  }
  return instr;
}

static std::optional<StaggeredVariable> TryResolveStaggeredVariable(
    const HloInstruction* operand) {
  namespace m = match;

  // Step 1: The operand must be GTE(parameter), possibly through copies
  // inserted by copy-insertion for scheduling.
  const HloInstruction* gte = StripCopies(operand);
  if (!Match(gte, m::GetTupleElement(m::Parameter()))) {
    return std::nullopt;
  }
  const HloInstruction* param = gte->operand(0);
  int64_t staggered_index = gte->tuple_index();

  // Step 2: The parameter's parent computation must be the body of exactly
  // one while loop.
  const HloComputation* body = param->parent();
  auto maybe_while_loop = body->GetUniqueCaller(HloOpcode::kWhile);
  if (!maybe_while_loop.has_value()) {
    return std::nullopt;
  }
  const HloInstruction* while_loop = *maybe_while_loop;
  if (while_loop->while_body() != body) {
    return std::nullopt;
  }

  // Step 3: The while loop must have annotated induction variable metadata
  // (set by WhileLoopTripCountAnnotator). The operand's tuple index must
  // differ from the primary induction variable — otherwise the standard
  // ResolveFunctionalDependencyOnInductionVariable would have succeeded.
  auto config = while_loop->backend_config<WhileLoopBackendConfig>();
  if (!config.ok() || !config->has_known_induction_variable() ||
      !config->has_known_init_step()) {
    return std::nullopt;
  }
  int64_t ivar_index = config->known_induction_variable().tuple_index();
  if (staggered_index == ivar_index) {
    return std::nullopt;
  }

  // Step 4: Verify the pipelining pattern — the body ROOT tuple's element at
  // staggered_index must be GTE(param, ivar_index), possibly through copies.
  // This means the staggered slot receives the current induction variable
  // value, making it carry the "previous iteration's" value on the next
  // iteration.
  const HloInstruction* root = body->root_instruction();
  if (root->opcode() != HloOpcode::kTuple ||
      staggered_index >= root->operand_count()) {
    return std::nullopt;
  }
  const HloInstruction* staggered_update =
      StripCopies(root->operand(staggered_index));
  if (!Match(staggered_update,
             m::GetTupleElement(m::Op().Is(param), ivar_index))) {
    return std::nullopt;
  }

  // Step 5: Read the staggered slot's initial value from the while loop's
  // init tuple. It must be a constant (possibly through copies) so we can
  // compute the offset formula. Typically this is (init - step), e.g. 0 when
  // init=1, step=1.
  if (!Match(while_loop->operand(0), m::Tuple()) ||
      staggered_index >= while_loop->operand(0)->operand_count()) {
    return std::nullopt;
  }
  const HloInstruction* init_constant =
      StripCopies(while_loop->operand(0)->operand(staggered_index));
  if (init_constant->opcode() != HloOpcode::kConstant) {
    return std::nullopt;
  }
  auto init_value = LiteralUtil::LiteralAsScalarInt64(init_constant->literal());
  if (!init_value) {
    return std::nullopt;
  }

  VLOG(3) << "Found staggered induction variable: " << operand->name()
          << " (index=" << staggered_index << ", init=" << *init_value
          << ") in " << while_loop->name();

  return StaggeredVariable{while_loop, operand, *init_value};
}

absl::StatusOr<std::optional<DynamicSliceDescriptor>> AnalyzeDynamicSlice(
    const HloInstruction* instr) {
  // Step 1: Only analyze dynamic-slice and dynamic-update-slice instructions.
  if (instr->opcode() != HloOpcode::kDynamicSlice &&
      instr->opcode() != HloOpcode::kDynamicUpdateSlice) {
    return std::nullopt;
  }

  // Step 2: The slice must be contiguous in memory (i.e. it slices along the
  // most-major dimension only, with full extent in all minor dimensions).
  if (!IsContiguousSlice(*instr)) {
    VLOG(5) << "Slice is not contiguous: " << instr->name();
    return std::nullopt;
  }

  // Step 3: Compute byte strides per dimension of the sliced buffer. These
  // convert index offsets to byte offsets (e.g. f32[4,8] has byte strides
  // [32, 4]: moving one position along dim0 skips 8*4=32 bytes).
  const Shape& slice_input_shape = instr->operand(0)->shape();
  std::optional<absl::InlinedVector<int64_t, 4>> strides =
      ComputeByteStrides(slice_input_shape);
  if (!strides) {
    VLOG(5) << "Failed to get byte strides: " << instr->name();
    return std::nullopt;
  }

  // Step 4: For each dynamic offset operand, resolve it to either a primary
  // or staggered induction variable. Collect results into a vector; we
  // validate consistency (same loop, same kind) after the loop.
  int32_t first_offset_index = GetFirstOffsetOperandIndex(instr);
  int32_t rank = slice_input_shape.dimensions().size();

  struct ResolvedOffset {
    const HloInstruction* loop;
    const HloInstruction* ivar;
    bool is_staggered;
    int64_t staggered_init;
    int64_t loop_index;
  };

  std::vector<ResolvedOffset> resolved_offsets;

  // Iterate over each dimension's offset operand of the DS/DUS.
  for (int32_t i = 0; i < rank; ++i) {
    const HloInstruction* operand = instr->operand(i + first_offset_index);

    // Skip dimensions where the slice covers the full extent or the offset is
    // a compile-time constant — these don't depend on the induction variable.
    if (IsZeroOffset(instr, i) || operand->opcode() == HloOpcode::kConstant) {
      continue;
    }

    // Step 4a: Try the standard analysis — traces the operand through the
    // computation graph back to a while loop's primary induction variable.
    auto functional_dependency =
        ResolveFunctionalDependencyOnInductionVariable(operand);
    if (functional_dependency) {
      // Use the induction variable from the while body by default. If the
      // DUS is inside a called computation (async/fusion/call), find the
      // local parameter that corresponds to the induction variable so that
      // HloEvaluator can substitute it without crossing computation
      // boundaries.
      const HloInstruction* local_ivar = functional_dependency->induction_var;
      const HloComputation* instr_comp = instr->parent();
      auto it = functional_dependency->required_parameters.find(instr_comp);
      if (it != functional_dependency->required_parameters.end()) {
        auto param_it = absl::c_find(it->second, true);
        if (param_it != it->second.end()) {
          local_ivar =
              instr_comp->parameter_instruction(param_it - it->second.begin());
        }
      }

      resolved_offsets.push_back({functional_dependency->loop, local_ivar,
                                  /*is_staggered=*/false, /*staggered_init=*/0,
                                  /*loop_index=*/0});
      continue;
    }

    // Step 4b: Standard analysis failed. It only recognizes the primary
    // induction variable (or entries in dynamic_variable_tuple_indices), not
    // staggered copies. Try to detect a staggered induction variable — a
    // copy of the primary ivar at a different tuple index, created by
    // CollectivePipeliner when it peels the first iteration.
    auto staggered_var = TryResolveStaggeredVariable(operand);
    if (staggered_var) {
      resolved_offsets.push_back({staggered_var->loop, staggered_var->gte,
                                  /*is_staggered=*/true,
                                  staggered_var->init_value,
                                  /*loop_index=*/0});
      continue;
    }

    VLOG(5) << "Offset for dimension " << i << " is not statically known.";
    return std::nullopt;
  }

  // Step 5: All offsets are constant — compute the static byte offset directly.
  if (resolved_offsets.empty()) {
    int64_t byte_offset = 0;
    for (int32_t i = 0; i < rank; ++i) {
      if (IsZeroOffset(instr, i)) {
        continue;
      }
      const HloInstruction* operand = instr->operand(i + first_offset_index);
      auto value = LiteralUtil::LiteralAsScalarInt64(operand->literal());
      if (!value.has_value()) {
        return std::nullopt;
      }
      byte_offset += *value * (*strides)[i];
    }
    return DynamicSliceDescriptor{std::nullopt, std::nullopt, byte_offset, 0};
  }

  // Step 5: All resolved offsets must be consistent — same while loop, same
  // kind (all primary or all staggered), and same induction variable GTE.
  const ResolvedOffset& front = resolved_offsets.front();
  bool consistent = absl::c_all_of(resolved_offsets, [&](const auto& r) {
    return r.loop == front.loop && r.is_staggered == front.is_staggered &&
           r.ivar == front.ivar && r.loop_index == front.loop_index;
  });

  if (!consistent) {
    VLOG(3) << "Skipping " << instr->name()
            << ": inconsistent dynamic offsets (different loops, mixed "
               "primary/staggered, or different induction variables).";
    return std::nullopt;
  }

  const HloInstruction* while_loop = front.loop;
  const HloInstruction* induction_var = front.ivar;
  bool all_staggered = front.is_staggered;

  // Step 6: Read the loop's init/step/trip_count from WhileLoopBackendConfig
  // (set by WhileLoopTripCountAnnotator).
  ASSIGN_OR_RETURN(auto loop_config,
                   while_loop->backend_config<WhileLoopBackendConfig>());
  if (!loop_config.has_known_init_step() ||
      !loop_config.has_known_trip_count()) {
    VLOG(5) << "Loop does not have known init/step/trip_count.";
    return std::nullopt;
  }

  int64_t init = loop_config.known_init_step().init();
  int64_t step = loop_config.known_init_step().step();
  int64_t trip_count = loop_config.known_trip_count().n();

  if (trip_count < 1) {
    return std::nullopt;
  }

  // Step 7: For staggered variables, the iteration sequence uses the staggered
  // init value rather than the primary induction variable's init. E.g. if the
  // primary ivar starts at 1 with step 1, a staggered copy starts at 0.
  int64_t effective_init =
      all_staggered ? resolved_offsets.front().staggered_init : init;

  // Step 8: Evaluate the byte offset for every iteration by substituting the
  // induction variable value into HloEvaluator.
  std::vector<int64_t> offsets(trip_count);
  for (int64_t iter = 0; iter < trip_count; ++iter) {
    int64_t ivar = effective_init + iter * step;
    ASSIGN_OR_RETURN(offsets[iter], EvaluateByteOffsetAtIteration(
                                        instr, *strides, induction_var, ivar));
    VLOG(3) << instr->name() << ": iteration " << iter << " (ivar=" << ivar
            << ") -> byte_offset=" << offsets[iter];
  }

  // Step 9: Verify linearity — all consecutive differences must be equal.
  // This confirms the offset is an affine function of the iteration count:
  //   byte_address = base + byte_offset + byte_stride * iteration
  int64_t byte_offset = offsets[0];
  int64_t byte_stride = (trip_count > 1) ? (offsets[1] - offsets[0]) : 0;

  for (int64_t iter = 2; iter < trip_count; ++iter) {
    int64_t actual_stride = offsets[iter] - offsets[iter - 1];
    if (actual_stride != byte_stride) {
      VLOG(3) << instr->name() << ": non-linear offset pattern at iteration "
              << iter << ": stride " << actual_stride << " != " << byte_stride;
      return std::nullopt;
    }
  }

  VLOG(2) << instr->name() << ": linear pattern confirmed over " << trip_count
          << " iterations: offset=" << byte_offset
          << ", stride=" << byte_stride;

  return DynamicSliceDescriptor{while_loop, front.loop_index, byte_offset,
                                byte_stride};
}

//===-----------------------------------------------------------------------===/
// DynamicSliceChain
//===-----------------------------------------------------------------------===/

// Walks a DS/DUS buffer operand (operand 0) back through bitcasts and DUS
// chains to find the buffer root: either a GTE of a tuple parameter or a
// non-tuple parameter directly. DUS chains arise when multiple updates write
// to different slices of the same buffer sequentially:
//
//   buf = GTE(param, 1)
//   updated1 = DUS(buf, v1, off1)
//   updated2 = DUS(updated1, v2, off2)
//
// All three (buf, updated1, updated2) share the same buffer root.
static const HloInstruction* TraceBufferSource(const HloInstruction* instr) {
  const HloInstruction* buffer = instr->operand(0);
  while (buffer->opcode() == HloOpcode::kBitcast ||
         buffer->opcode() == HloOpcode::kDynamicUpdateSlice) {
    buffer = buffer->operand(0);
  }
  if (buffer->opcode() == HloOpcode::kGetTupleElement &&
      buffer->operand(0)->opcode() == HloOpcode::kParameter) {
    return buffer;
  }
  if (buffer->opcode() == HloOpcode::kParameter) {
    return buffer;
  }
  return nullptr;
}

absl::StatusOr<DynamicSliceChain> FindDynamicSliceChain(
    const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kDynamicSlice &&
      instr->opcode() != HloOpcode::kDynamicUpdateSlice) {
    return Internal("FindDynamicSliceChain: expected DS or DUS, got %s",
                    HloOpcodeString(instr->opcode()));
  }

  const HloComputation* computation = instr->parent();
  auto maybe_caller = computation->GetUniqueCaller(HloOpcode::kWhile);
  if (!maybe_caller.has_value() ||
      (*maybe_caller)->while_body() != computation) {
    return Internal("FindDynamicSliceChain: %s is not inside a while loop body",
                    instr->name());
  }

  const HloInstruction* buffer = TraceBufferSource(instr);
  if (buffer == nullptr) {
    return Internal(
        "FindDynamicSliceChain: buffer operand of %s does not trace back to a "
        "parameter",
        instr->name());
  }

  DynamicSliceChain result;
  result.buffer = buffer;

  for (const HloInstruction* other : computation->instructions()) {
    if (other->opcode() != HloOpcode::kDynamicSlice &&
        other->opcode() != HloOpcode::kDynamicUpdateSlice) {
      continue;
    }
    const HloInstruction* other_buffer = TraceBufferSource(other);
    if (other_buffer != buffer) {
      continue;
    }
    if (auto* ds = DynCast<HloDynamicSliceInstruction>(other)) {
      result.slices.push_back(ds);
    } else if (auto* dus = DynCast<HloDynamicUpdateSliceInstruction>(other)) {
      result.updates.push_back(dus);
    }
  }

  // Find the last DUS in the chain: the one that is not consumed by another
  // DUS in the set (i.e. its result feeds into the ROOT tuple or other non-DUS
  // consumers).
  for (const auto* dus : result.updates) {
    bool is_consumed_by_another_dus = false;
    for (const HloInstruction* user : dus->users()) {
      auto* dus_user = DynCast<HloDynamicUpdateSliceInstruction>(user);
      if (dus_user &&
          absl::c_find(result.updates, dus_user) != result.updates.end()) {
        is_consumed_by_another_dus = true;
        break;
      }
    }
    if (!is_consumed_by_another_dus) {
      result.result = dus;
      break;
    }
  }

  return result;
}

std::optional<bool> IsNonOverlapping(const DynamicSliceChain& chain) {
  struct SliceRange {
    int64_t byte_offset;
    int64_t byte_size;
  };

  auto analyze_instr = [](const HloInstruction* instr)
      -> std::optional<std::pair<DynamicSliceDescriptor, int64_t>> {
    auto result = AnalyzeDynamicSlice(instr);
    if (!result.ok() || !result->has_value()) {
      return std::nullopt;
    }
    int64_t slice_byte_size = ShapeUtil::ByteSizeOf(instr->operand(1)->shape());
    return std::make_pair(**result, slice_byte_size);
  };

  // Only check DUS-vs-DUS overlap. DS reads and DUS writes to the same slice
  // are allowed (the read happens before the write within an iteration).
  std::vector<std::pair<DynamicSliceDescriptor, int64_t>> dus_descriptors;

  for (const auto* dus : chain.updates) {
    auto desc = analyze_instr(dus);
    if (!desc.has_value()) {
      return std::nullopt;
    }
    dus_descriptors.push_back(*std::move(desc));
  }

  if (dus_descriptors.size() <= 1) {
    return true;
  }

  auto front_loop = dus_descriptors.front().first.while_loop;
  if (!front_loop.has_value()) {
    return std::nullopt;
  }
  const HloInstruction* while_loop = *front_loop;
  for (const auto& [desc, _] : dus_descriptors) {
    if (desc.while_loop != front_loop) {
      return std::nullopt;
    }
  }

  auto loop_config = while_loop->backend_config<WhileLoopBackendConfig>();
  if (!loop_config.ok() || !loop_config->has_known_trip_count()) {
    return std::nullopt;
  }
  int64_t trip_count = loop_config->known_trip_count().n();

  for (int64_t iter = 0; iter < trip_count; ++iter) {
    std::vector<SliceRange> ranges;
    ranges.reserve(dus_descriptors.size());
    for (const auto& [desc, byte_size] : dus_descriptors) {
      int64_t offset = desc.byte_offset + desc.byte_stride * iter;
      ranges.push_back({offset, byte_size});
    }

    for (size_t i = 0; i < ranges.size(); ++i) {
      for (size_t j = i + 1; j < ranges.size(); ++j) {
        int64_t start_i = ranges[i].byte_offset;
        int64_t end_i = start_i + ranges[i].byte_size;
        int64_t start_j = ranges[j].byte_offset;
        int64_t end_j = start_j + ranges[j].byte_size;
        if (start_i < end_j && start_j < end_i) {
          return false;
        }
      }
    }
  }

  return true;
}

}  // namespace xla::gpu
