/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/status/status_macros.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/host_offload_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

std::optional<int64_t> ScalarConstantInitOperand(const HloInstruction* while_op,
                                                 int64_t tuple_idx) {
  const HloInstruction* init = while_op->operand(0);
  if (init->opcode() != HloOpcode::kTuple ||
      tuple_idx >= init->operand_count()) {
    return std::nullopt;
  }
  const HloInstruction* init_val = init->operand(tuple_idx);
  if (init_val->opcode() != HloOpcode::kConstant) {
    return std::nullopt;
  }
  return LiteralUtil::LiteralAsScalarInt64(init_val->literal());
}

struct AffineMatch {
  int64_t source_tuple_idx;
  int64_t step_offset;
};

struct LinearTerm {
  std::optional<int64_t> source_tuple_idx;
  int64_t constant_offset;
};

std::optional<LinearTerm> DecomposeLinear(const HloInstruction* expr,
                                          const HloInstruction* param) {
  if (expr->opcode() == HloOpcode::kGetTupleElement &&
      expr->operand(0) == param) {
    return LinearTerm{expr->tuple_index(), 0};
  }
  if (expr->opcode() == HloOpcode::kConstant) {
    std::optional<int64_t> v =
        LiteralUtil::LiteralAsScalarInt64(expr->literal());
    if (!v.has_value()) {
      return std::nullopt;
    }
    return LinearTerm{std::nullopt, *v};
  }
  if (expr->opcode() == HloOpcode::kAdd ||
      expr->opcode() == HloOpcode::kSubtract) {
    std::optional<LinearTerm> lhs = DecomposeLinear(expr->operand(0), param);
    std::optional<LinearTerm> rhs = DecomposeLinear(expr->operand(1), param);
    if (!lhs.has_value() || !rhs.has_value()) {
      return std::nullopt;
    }
    bool is_subtract = expr->opcode() == HloOpcode::kSubtract;
    if (lhs->source_tuple_idx.has_value() &&
        rhs->source_tuple_idx.has_value()) {
      return std::nullopt;
    }
    if (rhs->source_tuple_idx.has_value() && is_subtract) {
      return std::nullopt;
    }
    int64_t offset = is_subtract ? lhs->constant_offset - rhs->constant_offset
                                 : lhs->constant_offset + rhs->constant_offset;
    return LinearTerm{lhs->source_tuple_idx.has_value() ? lhs->source_tuple_idx
                                                        : rhs->source_tuple_idx,
                      offset};
  }
  return std::nullopt;
}

std::optional<AffineMatch> MatchAffineBodyUpdate(const HloInstruction* root,
                                                 const HloInstruction* param,
                                                 int64_t tuple_idx) {
  if (root->opcode() != HloOpcode::kTuple ||
      tuple_idx >= root->operand_count()) {
    return std::nullopt;
  }
  std::optional<LinearTerm> term =
      DecomposeLinear(root->operand(tuple_idx), param);
  if (!term.has_value() || !term->source_tuple_idx.has_value()) {
    return std::nullopt;
  }
  return AffineMatch{*term->source_tuple_idx, term->constant_offset};
}

// Tries to recover the affine (init, step) parameters describing how the
// tuple-element at `tuple_idx` of `while_op` evolves across iterations, i.e.
// value_at_iteration(k) = init + k * step.
//
// Two body-update patterns are supported:
//
//   1. Self-affine:  body_root[tuple_idx] = body_param[tuple_idx] + step
//      → init comes from the scalar constant operand of the while at
//        `tuple_idx`; step comes from the body update.
//
//   2. Lagged copy of the primary induction variable:
//        body_root[tuple_idx] = body_param[primary_idx] + delta
//      The tuple slot trails the primary induction variable by one iteration,
//      so it shares the primary's step. `init` is derived as
//      `primary_init - primary_step + delta` and is cross-checked against the
//      actual scalar init operand of the while.
//
// Returns nullopt if neither pattern matches, or if the consistency check
// fails. `primary_*` arguments are only consulted by pattern (2) and must all
// be set together for that pattern to apply.
std::optional<std::pair<int64_t, int64_t>> ResolveAffine(
    const HloInstruction* while_op, int64_t tuple_idx,
    std::optional<int64_t> primary_idx, std::optional<int64_t> primary_init,
    std::optional<int64_t> primary_step) {
  const HloComputation* body = while_op->while_body();
  const HloInstruction* root = body->root_instruction();
  const HloInstruction* param = body->parameter_instruction(0);

  std::optional<AffineMatch> match =
      MatchAffineBodyUpdate(root, param, tuple_idx);
  if (!match.has_value()) {
    return std::nullopt;
  }

  std::optional<int64_t> init = ScalarConstantInitOperand(while_op, tuple_idx);
  if (!init.has_value()) {
    return std::nullopt;
  }

  if (match->source_tuple_idx == tuple_idx) {
    return std::make_pair(*init, match->step_offset);
  }

  if (primary_idx.has_value() && match->source_tuple_idx == *primary_idx &&
      primary_init.has_value() && primary_step.has_value()) {
    int64_t derived_init = *primary_init - *primary_step + match->step_offset;
    if (*init != derived_init) {
      return std::nullopt;
    }
    return std::make_pair(derived_init, *primary_step);
  }

  return std::nullopt;
}

}  // namespace

// For a while loop with known init, step, and trip count, replace all
// get-tuple-element instructions that extract the induction variable from the
// while result with a constant equal to (init + step * trip_count).
static bool ForwardInductionVarToConstants(HloInstruction* while_instr,
                                           int64_t indvar_index,
                                           int64_t final_value) {
  bool changed = false;

  // Collect GTEs first to avoid modifying the user list while iterating.
  std::vector<HloInstruction*> indvar_gtes;
  for (HloInstruction* user : while_instr->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == indvar_index) {
      indvar_gtes.push_back(user);
    }
  }

  for (HloInstruction* gte : indvar_gtes) {
    HloComputation* comp = gte->parent();
    Literal literal =
        LiteralUtil::CreateR0(gte->shape().element_type(), final_value);
    HloInstruction* constant = comp->AddInstruction(
        HloInstruction::CreateConstant(std::move(literal)));
    CHECK_OK(gte->ReplaceAllUsesWith(constant));
    changed = true;
  }

  return changed;
}

absl::StatusOr<bool> WhileLoopTripCountAnnotator::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() != HloOpcode::kWhile) {
        continue;
      }

      auto induction_variable_index = GetLoopInductionVarTupleIdx(instr);
      if (!induction_variable_index) {
        continue;
      }

      ASSIGN_OR_RETURN(WhileLoopBackendConfig existing_config,
                       instr->backend_config<WhileLoopBackendConfig>());
      if (existing_config.ByteSizeLong() != 0) {
        LOG(WARNING) << absl::StrFormat(
            "WhileLoopTripCountAnnotator is overwriting an existing non-empty "
            "WhileLoopBackendConfig on %s: %s",
            instr->name(), existing_config.ShortDebugString());
      }

      WhileLoopBackendConfig config;
      config.mutable_known_induction_variable()->set_tuple_index(
          *induction_variable_index);

      absl::flat_hash_set<int64_t> dynamic_variable_tuple_indices =
          host_offload_utils::CollectDynamicVariableTupleIndices(instr);
      std::vector<int64_t> sorted_dynamic_indices(
          dynamic_variable_tuple_indices.begin(),
          dynamic_variable_tuple_indices.end());
      std::sort(sorted_dynamic_indices.begin(), sorted_dynamic_indices.end());
      for (int64_t tuple_idx : sorted_dynamic_indices) {
        config.add_dynamic_variables()->set_tuple_index(tuple_idx);
      }

      std::optional<int64_t> primary_init;
      std::optional<int64_t> primary_step;
      if (auto range = MatchTrivialLoopRange(instr);
          range.has_value() && range->IsBounded() && range->IsStepKnown() &&
          // We store the values in signed integers, so we need to verify
          // they fit.
          range->max()->GetSignedValue() >= 0 &&
          range->min().GetSignedValue() >= 0 &&
          range->step()->GetSignedValue() > 0) {
        int64_t max = range->max()->GetUnsignedValue();
        int64_t min = range->min().GetUnsignedValue();
        int64_t step = range->step()->GetSignedValue();
        int64_t trip_count = (max - min) / step + 1;

        config.mutable_known_trip_count()->set_n(trip_count);
        config.mutable_known_init_step()->set_init(min);
        config.mutable_known_init_step()->set_step(step);
        primary_init = min;
        primary_step = step;

        int64_t final_value = min + step * trip_count;
        changed |= ForwardInductionVarToConstants(
            instr, *induction_variable_index, final_value);
      } else if (auto trip_count = ComputeWhileLoopTripCount(instr)) {
        config.mutable_known_trip_count()->set_n(*trip_count);
      }

      for (auto& dv : *config.mutable_dynamic_variables()) {
        auto resolved =
            ResolveAffine(instr, dv.tuple_index(), induction_variable_index,
                          primary_init, primary_step);
        if (!resolved.has_value()) {
          continue;
        }
        dv.set_init(resolved->first);
        dv.set_step(resolved->second);
      }

      RETURN_IF_ERROR(instr->set_backend_config(config));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
