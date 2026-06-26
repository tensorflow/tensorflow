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

#include "xla/backends/gpu/transforms/dus_accumulator_zero_init_elimination.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/status/status_macros.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/constant_value.h"
#include "xla/service/value_range.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

// Matches broadcast(scalar_constant) through layout-only wrappers
// (bitcast/copy/reshape) above the broadcast and convert-like casts below it.
// Strictly wider than hlo_query::IsBroadcastOfScalarConstant.
bool IsBroadcastOfScalarConstantThroughLayoutWrappers(const HloInstruction* h) {
  while (h->opcode() == HloOpcode::kBitcast ||
         h->opcode() == HloOpcode::kCopy ||
         h->opcode() == HloOpcode::kReshape) {
    h = h->operand(0);
  }
  return h->opcode() == HloOpcode::kBroadcast &&
         hlo_query::IsScalarConstant(hlo_query::StripCastLike(h->operand(0)));
}

// Init value is irrelevant; soundness comes from the dead-input check.
bool IsInitReplaceable(const HloInstruction* init) {
  if (IsBroadcastOfScalarConstantThroughLayoutWrappers(init)) {
    return true;
  }
  if (init->opcode() == HloOpcode::kFusion && init->operand_count() == 0) {
    return IsBroadcastOfScalarConstantThroughLayoutWrappers(
        init->fused_expression_root());
  }
  if (init->opcode() == HloOpcode::kGetTupleElement) {
    const HloInstruction* fusion = init->operand(0);
    if (fusion->opcode() == HloOpcode::kFusion &&
        fusion->operand_count() == 0) {
      const HloInstruction* root = fusion->fused_expression_root();
      if (root->opcode() == HloOpcode::kTuple) {
        return IsBroadcastOfScalarConstantThroughLayoutWrappers(
            root->operand(init->tuple_index()));
      }
    }
  }
  return false;
}

// Holds the subset of while-loop facts this pass needs. We don't reuse
// while_loop_unroller's WhileLoopConfig because we want a mutable while_instr
// and don't need the `init` field.
struct CandidateLoop {
  HloInstruction* while_instr;
  int64_t trip_count;
  int64_t induction_var_idx;
};

absl::StatusOr<bool> AllUsersKillBuffer(const HloInstruction* buffer,
                                        const CandidateLoop& cfg);
absl::StatusOr<bool> UserKillsBuffer(const HloInstruction* user,
                                     const HloInstruction* operand,
                                     const CandidateLoop& cfg);

// Read from the trip-count annotation: upstream rewrites defeat structural
// IV detection.
// TODO: the annotation can outlive the IR shape it was derived from —
// upstream IV-rewriting passes should refresh or drop it.
Range LoopIterationRange(const CandidateLoop& cfg) {
  return Range{ConstantValue::GetSigned(0, /*bitwidth=*/64),
               ConstantValue::GetSigned(cfg.trip_count - 1, /*bitwidth=*/64),
               ConstantValue::GetSigned(1, /*bitwidth=*/64),
               /*is_linear=*/true};
}

// RecursivelyIdentifyRange handles the recursion into fused parameters, so
// this works inside or outside fusions. Caller is responsible for checking
// IsBounded()/IsStepKnown() before using the returned range.
Range ResolveIndexRangeAsFunctionOfIv(const HloInstruction* idx,
                                      const CandidateLoop& cfg) {
  Range loop_range = LoopIterationRange(cfg);
  absl::flat_hash_map<const HloInstruction*, Range> predefined;
  HloInstruction* body_param =
      cfg.while_instr->while_body()->parameter_instruction(0);
  for (HloInstruction* u : body_param->users()) {
    if (u->opcode() == HloOpcode::kGetTupleElement &&
        u->tuple_index() == cfg.induction_var_idx) {
      predefined[u] = loop_range;
    }
  }
  return RecursivelyIdentifyRange(idx, predefined, nullptr);
}

// Generalizes AdvancedMatchShapeCoveringDynamicIndexInstruction to indices
// that are arithmetic expressions over fused parameters (e.g. descending DUS).
bool IsKillingDus(const HloInstruction* dus, const HloInstruction* operand,
                  const CandidateLoop& cfg) {
  if (dus->opcode() != HloOpcode::kDynamicUpdateSlice ||
      dus->operand(0) != operand) {
    return false;
  }

  const Shape& slice_shape = dus->operand(1)->shape();
  const Shape& input_shape = dus->operand(0)->shape();
  if (slice_shape.dimensions().size() != input_shape.dimensions().size()) {
    return false;
  }

  // Find the single dynamic dim; static dims must have start=0 and full size.
  std::optional<int64_t> dyn_dim_opt;
  Range dyn_dim_range;
  const int64_t num_dims = input_shape.dimensions().size();
  for (int64_t d = 0; d < num_dims; ++d) {
    const HloInstruction* idx = dus->operand(2 + d);
    Range r = ResolveIndexRangeAsFunctionOfIv(idx, cfg);
    if (!r.IsBounded() || !r.IsStepKnown() || r.step()->GetSignedValue() == 0) {
      return false;
    }
    if (r.IsSingleValue() && r.min().GetSignedValue() == 0) {
      if (slice_shape.dimensions(d) != input_shape.dimensions(d)) {
        return false;
      }
      continue;
    }
    // Reject if we already saw a dynamic dim.
    if (dyn_dim_opt.has_value()) {
      return false;
    }
    dyn_dim_opt = d;
    dyn_dim_range = r;
  }
  if (!dyn_dim_opt.has_value()) {
    return false;
  }
  int64_t dyn_dim = *dyn_dim_opt;

  const int64_t dim_size = input_shape.dimensions(dyn_dim);
  const int64_t slice_size = slice_shape.dimensions(dyn_dim);
  if (slice_size <= 0 || slice_size > dim_size) {
    return false;
  }
  // Left edge, right edge, and no interior gaps.
  return dyn_dim_range.min().GetSignedValue() <= 0 &&
         dyn_dim_range.max()->GetSignedValue() >= dim_size - slice_size &&
         slice_size >= dyn_dim_range.step()->GetSignedValue();
}

// TODO: scatter killing-write support; needs reasoning over scatter index
// expressions and update_window_dims.

absl::StatusOr<bool> FusionParamIsKilled(const HloInstruction* fusion,
                                         int64_t param_idx,
                                         const CandidateLoop& cfg) {
  HloComputation* fc = fusion->fused_instructions_computation();
  HloInstruction* fused_param = fc->parameter_instruction(param_idx);
  bool ok = false;
  ASSIGN_OR_RETURN(ok, AllUsersKillBuffer(fused_param, cfg));
  return ok;
}

absl::StatusOr<bool> AllUsersKillBuffer(const HloInstruction* buffer,
                                        const CandidateLoop& cfg) {
  if (buffer->user_count() == 0) {
    return true;  // trivially dead
  }
  for (const HloInstruction* user : buffer->users()) {
    ASSIGN_OR_RETURN(bool ok, UserKillsBuffer(user, buffer, cfg));
    if (!ok) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<bool> UserKillsBuffer(const HloInstruction* user,
                                     const HloInstruction* operand,
                                     const CandidateLoop& cfg) {
  bool result = false;
  switch (user->opcode()) {
    case HloOpcode::kDynamicUpdateSlice:
      result = IsKillingDus(user, operand, cfg);
      break;

    case HloOpcode::kBitcast:
      // Bitcast is layout-only, so recurse to its users. kCopy intentionally
      // falls to default: it observes the init bytes we are eliding.
      result = AllUsersKillBuffer(user, cfg).value_or(false);
      break;

    case HloOpcode::kFusion: {
      int64_t op_idx = user->operand_index(operand);
      if (op_idx < 0) {
        result = false;
        break;
      }
      result = FusionParamIsKilled(user, op_idx, cfg).value_or(false);
      break;
    }

    case HloOpcode::kConditional: {
      // Scope: operand passed straight to a branch (not inside a tuple);
      // each receiving branch's parameter_0 must be killed.
      bool every_branch_kills = true;
      bool flows_into_some_branch = false;
      for (int b = 0; b < user->branch_count(); ++b) {
        const HloInstruction* branch_input = user->operand(b + 1);
        if (branch_input != operand) {
          continue;
        }
        flows_into_some_branch = true;
        HloComputation* branch = user->branch_computation(b);
        HloInstruction* branch_param = branch->parameter_instruction(0);
        ASSIGN_OR_RETURN(bool branch_ok, AllUsersKillBuffer(branch_param, cfg));
        if (!branch_ok) {
          every_branch_kills = false;
          break;
        }
      }
      result = flows_into_some_branch && every_branch_kills;
      break;
    }

    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
      // Conservatively bail on tuple/GTE rather than tracing aliasing.
      result = false;
      break;

    default:
      result = false;  // anything else reads the buffer, so it's not killed
      break;
  }
  return result;
}

// Two checks: (1) body_param[slot] is overwritten before any read, and
// (2) body_root[slot] carries the writer's output, not a passthrough.
absl::StatusOr<bool> SlotIsDeadInput(int64_t slot, const CandidateLoop& cfg) {
  HloComputation* body = cfg.while_instr->while_body();
  HloInstruction* body_param = body->parameter_instruction(0);
  HloInstruction* body_root = body->root_instruction();
  if (body_root->opcode() != HloOpcode::kTuple) {
    return false;
  }

  HloInstruction* slot_gte = nullptr;
  for (HloInstruction* u : body_param->users()) {
    if (u->opcode() == HloOpcode::kGetTupleElement &&
        u->tuple_index() == slot) {
      if (slot_gte != nullptr) {
        return false;  // multiple GTEs at slot
      }
      slot_gte = u;
    }
  }
  if (slot_gte == nullptr) {
    return true;  // never read; trivially dead
  }

  ASSIGN_OR_RETURN(bool all_kill, AllUsersKillBuffer(slot_gte, cfg));
  if (!all_kill) {
    return false;
  }

  // If the body passes slot through unchanged (root traces to slot_gte via
  // bitcast/copy), post-loop uses would see garbage if we elided the init.
  const HloInstruction* root_v = body_root->operand(slot);
  while (root_v->opcode() == HloOpcode::kBitcast ||
         root_v->opcode() == HloOpcode::kCopy) {
    root_v = root_v->operand(0);
  }
  if (root_v == slot_gte) {
    return false;  // unmodified passthrough
  }
  return true;
}

// Filters a single while op against the pass's loop-shape preconditions and
// returns a CandidateLoop if it's a candidate, nullopt otherwise.
std::optional<CandidateLoop> ClassifyCandidateWhile(HloInstruction* while_op) {
  if (while_op->has_sharding()) {
    return std::nullopt;
  }
  auto loop_cfg = while_op->backend_config<WhileLoopBackendConfig>();
  if (!loop_cfg.ok() || !loop_cfg->has_known_trip_count()) {
    return std::nullopt;
  }
  int64_t trip_count = loop_cfg->known_trip_count().n();
  if (trip_count <= 0) {
    return std::nullopt;
  }
  if (!loop_cfg->has_known_init_step() ||
      loop_cfg->known_init_step().init() != 0 ||
      loop_cfg->known_init_step().step() != 1) {
    return std::nullopt;
  }
  if (while_op->operand(0)->opcode() != HloOpcode::kTuple) {
    return std::nullopt;
  }
  int64_t iv_tuple_idx;
  if (loop_cfg->has_known_induction_variable()) {
    iv_tuple_idx = loop_cfg->known_induction_variable().tuple_index();
  } else {
    std::optional<int64_t> iv_idx_opt = GetLoopInductionVarTupleIdx(while_op);
    if (!iv_idx_opt.has_value()) {
      return std::nullopt;
    }
    iv_tuple_idx = *iv_idx_opt;
  }
  return CandidateLoop{/*while_instr=*/while_op,
                       /*trip_count=*/trip_count,
                       /*induction_var_idx=*/iv_tuple_idx};
}

}  // namespace

absl::StatusOr<bool> DusAccumulatorZeroInitElimination::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->config()
           .debug_options()
           .xla_gpu_enable_dus_accumulator_zero_init_elimination()) {
    return false;
  }

  std::vector<CandidateLoop> candidates;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* ins : comp->instructions()) {
      if (ins->opcode() != HloOpcode::kWhile) {
        continue;
      }
      if (std::optional<CandidateLoop> cfg = ClassifyCandidateWhile(ins)) {
        candidates.push_back(*cfg);
      }
    }
  }

  bool changed = false;
  for (const CandidateLoop& cfg : candidates) {
    HloInstruction* while_op = cfg.while_instr;
    HloInstruction* init_tuple = while_op->mutable_operand(0);
    int64_t n_slots = init_tuple->operand_count();
    for (int64_t slot = 0; slot < n_slots; ++slot) {
      if (slot == cfg.induction_var_idx) {
        continue;
      }

      HloInstruction* init = init_tuple->mutable_operand(slot);
      if (!IsInitReplaceable(init)) {
        continue;
      }
      // Init must feed only this while and not carry its own sharding.
      if (init->has_sharding() || init->user_count() != 1) {
        continue;
      }
      const Shape& alloc_shape = init->shape();
      if (alloc_shape.dimensions().empty() ||
          alloc_shape.dimensions(0) != cfg.trip_count) {
        continue;
      }

      ASSIGN_OR_RETURN(bool dead, SlotIsDeadInput(slot, cfg));
      if (!dead) {
        continue;
      }

      HloInstruction* alloc =
          while_op->parent()->AddInstruction(HloInstruction::CreateCustomCall(
              alloc_shape, /*operands=*/{}, "AllocateBuffer"));
      alloc->set_metadata(init->metadata());
      alloc->set_frontend_attributes(init->frontend_attributes());
      alloc->set_statistics_viz(init->statistics_viz());
      RETURN_IF_ERROR(init_tuple->ReplaceOperandWith(slot, alloc));
      changed = true;
    }
  }

  // TODO: extend to multi-iter coverage analysis (a buffer may be killed
  // across iters 0..k-1 jointly even if iter 0 alone reads stale slots).
  return changed;
}

}  // namespace xla::gpu
