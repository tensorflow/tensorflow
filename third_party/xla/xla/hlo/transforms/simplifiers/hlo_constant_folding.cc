/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

// Checks whether instr is or transitively contains an instruction that we
// shouldn't fold.
//
// Specifically, we don't fold kRng or kAfterAll instructions:
//
//  - kRng is already marked as side-effecting and so is skipped elsewhere, but
//    we check for it here.  Even kRng weren't side-effecting and took an
//    explicit seed, we *still* wouldn't want to constant-fold it, because the
//    evaluator's handling of rng is not guaranteed to be identical to any
//    particular backend's rng.
//
//  - kAfterAll needs to be skipped because a kAfterAll op with no args can
//    currently materialize a token "out of thin air".  TODO(b/110532604):
//    Remove this check once AfterAll requires at least one operand, in which
//    case constant folding will be impossible.
static bool IsOrContainsIllegalInstr(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kAfterAll ||
      instr->opcode() == HloOpcode::kRng) {
    return true;
  }
  for (const HloComputation* c : instr->called_computations()) {
    if (absl::c_any_of(c->instructions(), IsOrContainsIllegalInstr)) {
      return true;
    }
  }
  return false;
}

// Checks if  any of the operands of the instruction are constants.
// Any tuples are recursively checked.
bool AnyOperandsConstant(const HloInstruction* instr) {
  for (const HloInstruction* operand : instr->operands()) {
    HloOpcode opcode = operand->opcode();

    if (opcode == HloOpcode::kTuple) {
      if (AnyOperandsConstant(operand)) {
        return true;
      }
    } else if (opcode == HloOpcode::kConstant) {
      return true;
    }
  }

  return false;
}

// Checks if all of the operands of the instruction are constants or broadcasts
// of constants (iota is a broadcast of a constant from the standpoint of
// constant folding).
// Any tuples are recursively checked.
bool AllOperandsConstantOrBroadcastConstant(const HloInstruction* instr) {
  for (const HloInstruction* operand : instr->operands()) {
    HloOpcode opcode = operand->opcode();

    if (opcode == HloOpcode::kTuple) {
      if (!AllOperandsConstantOrBroadcastConstant(operand)) {
        return false;
      }
    } else if (opcode != HloOpcode::kConstant &&
               !(opcode == HloOpcode::kBroadcast &&
                 operand->operand(0)->opcode() == HloOpcode::kConstant) &&
               opcode != HloOpcode::kIota) {
      return false;
    }
  }

  return true;
}

/*static*/ std::atomic<int64_t> HloConstantFolding::slow_op_counter_{0};

// Removes a dead instruction and any operands that die as a result,
// transitively. Every removed instruction is recorded in
// `removed_instructions` so that callers iterating over an instruction list
// taken before the removal can recognize removed entries by pointer identity
// instead of dereferencing them.
absl::Status RecursivelyRemoveDeadInstructionAndDeadOperands(
    HloComputation& computation, HloInstruction* instruction,
    absl::flat_hash_set<const HloInstruction*>& removed_instructions) {
  std::vector<HloInstruction*> dead_instructions = {instruction};
  while (!dead_instructions.empty()) {
    HloInstruction* dead_instruction = dead_instructions.back();
    dead_instructions.pop_back();
    if (!removed_instructions.insert(dead_instruction).second) {
      continue;
    }

    // Save the operands before calling RemoveInstruction which clears them.
    auto operands = dead_instruction->operands();

    // First remove the instruction itself.
    ABSL_RETURN_IF_ERROR(computation.RemoveInstruction(dead_instruction));

    // Now check if some of its operands are dead as a result of the removal.
    for (auto operand : operands) {
      if (operand->IsDead()) {
        dead_instructions.push_back(operand);
      }
    }
  }
  return absl::OkStatus();
}

namespace {

// Ops that move, select, or reinterpret bytes without arithmetic; folding
// them cannot change numerics for any element type. (Pad copies the padding
// value verbatim, dynamic-slice start indices are clamped per spec, and
// select passes one operand's bits through.)
bool IsPureDataMovement(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kConcatenate:
    case HloOpcode::kCopy:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kPad:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSelect:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}

// True if every array leaf of `shape` is integer or pred typed.
bool ShapeIsIntegralOrPred(const Shape& shape) {
  bool ok = true;
  ShapeUtil::ForEachSubshape(
      shape, [&ok](const Shape& subshape, const ShapeIndex& /*index*/) {
        if (subshape.IsArray() &&
            !primitive_util::IsIntegralType(subshape.element_type()) &&
            subshape.element_type() != PRED) {
          ok = false;
        }
      });
  return ok;
}

// Structural ops that are exempt from per-instruction options policies: they
// read or regroup existing values and are only reachable through the fold of
// an enclosing instruction.
bool IsStructural(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return true;
    default:
      return false;
  }
}

bool IsFoldable(
    const HloInstruction* instruction,
    const HloConstantFolding::Options& options,
    absl::flat_hash_map<HloComputation*, bool>& is_foldable_computation) {
  const HloConstantFolding::Level level = options.level;
  // Broadcasts dramatically increase the size of constants, which is often
  // detrimental to performance and memory capacity, so do not fold
  // broadcasts.
  if (instruction->opcode() == HloOpcode::kBroadcast ||
      instruction->opcode() == HloOpcode::kIota) {
    return false;
  }

  // Do not fold FFT. Evaluating it may significantly increase compile time.
  if (instruction->opcode() == HloOpcode::kFft) {
    return false;
  }

  switch (instruction->opcode()) {
    // Opaque, runtime-dependent, or cross-device: never fold, not even inside
    // called computations. Most of these are unimplemented in the evaluator
    // today; the explicit list keeps the policy sound on its own.
    case HloOpcode::kAllGather:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kPartitionId:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kReplicaId:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kRngGetAndUpdateState:
      return false;
    // Sub-byte types are stored unpacked in literals, so evaluating a
    // width-changing bitcast of a packed type reinterprets unpacked host
    // bytes and produces a constant that differs from the backend result.
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert: {
      const PrimitiveType from =
          instruction->operand(0)->shape().element_type();
      const PrimitiveType to = instruction->shape().element_type();
      if (primitive_util::BitWidth(from) != primitive_util::BitWidth(to) &&
          (primitive_util::IsSubByteNonPredType(from) ||
           primitive_util::IsSubByteNonPredType(to))) {
        return false;
      }
      break;
    }
    default:
      break;
  }

  // Skip while loops as they can significantly increase compile times.
  if (level == HloConstantFolding::Level::kDefault &&
      instruction->opcode() == HloOpcode::kWhile) {
    return false;
  }

  if (options.can_fold_shape != nullptr &&
      !IsStructural(instruction->opcode()) &&
      !options.can_fold_shape(instruction->shape())) {
    return false;
  }

  if (!options.fold_float_arithmetic && !IsStructural(instruction->opcode()) &&
      // Ops with called computations are judged by the recursion below.
      instruction->called_computations().empty() &&
      !IsPureDataMovement(instruction->opcode())) {
    if (!ShapeIsIntegralOrPred(instruction->shape())) {
      return false;
    }
    for (const HloInstruction* operand : instruction->operands()) {
      if (!ShapeIsIntegralOrPred(operand->shape())) {
        return false;
      }
    }
  }

  // Don't fold across async execution thread if it's not supposed to be
  // changed by this pass.
  if (instruction->IsAsynchronous() &&
      instruction->async_execution_thread() !=
          instruction->parent()->execution_thread()) {
    return false;
  }

  // Don't fold if any of the subcomputations are not foldable. Note that this
  // will recurse into deeper called computations.
  for (HloComputation* subcomputation : instruction->called_computations()) {
    auto iter = is_foldable_computation.find(subcomputation);
    if (iter == is_foldable_computation.end()) {
      for (auto* sub_instruction : subcomputation->MakeInstructionPostOrder()) {
        if (!IsFoldable(sub_instruction, options, is_foldable_computation)) {
          is_foldable_computation[subcomputation] = false;
          return false;
        }
      }
      is_foldable_computation[subcomputation] = true;
    } else if (!iter->second) {
      return false;
    }
  }

  // Check for instructions that we can't fold even if they appear inside of
  // a subcomputation (e.g. a kCall).
  if (IsOrContainsIllegalInstr(instruction)) {
    return false;
  }

  // Don't constant-fold side-effecting instructions or instructions which
  // contain side-effecting instructions.
  if (instruction->HasSideEffect()) {
    return false;
  }

  // Skip constant folding for instructions that have control dependencies.
  if (instruction->HasControlDependencies()) {
    return false;
  }

  // Reduce the compile time by skipping the constant folding of pad
  // instruction with broadcast operand. With 45m shape limit the compile
  // time could be more than 30 seconds. According to the current
  // benchmarks it does not affect the performance.
  if (instruction->opcode() == HloOpcode::kPad &&
      instruction->operand(0)->opcode() == HloOpcode::kBroadcast &&
      instruction->operand(1)->opcode() == HloOpcode::kConstant) {
    return false;
  }

  // Don't constant fold unless output and operand sizes are small.
  if (level == HloConstantFolding::Level::kDefault &&
      instruction->shape().IsArray()) {
    int64_t elements_in_operands = 0;
    for (HloInstruction* operand : instruction->operands()) {
      if (operand->shape().IsArray()) {
        elements_in_operands += ShapeUtil::ElementsIn(operand->shape());
      }
    }
    int64_t elements_in_constant = ShapeUtil::ElementsIn(instruction->shape());

    static const int64_t kMaximumConstantSizeElements = 45 * 1000 * 1000;
    if (std::max(elements_in_constant, elements_in_operands) >
        kMaximumConstantSizeElements) {
      VLOG(2) << "Ignore constant folding: result shape size is "
              << elements_in_constant << " total size of arguments is "
              << elements_in_operands;
      return false;
    }
  }
  return true;
}

// Makes a constant declared with the exact shape of the instruction it
// replaces: literals normalize away layout decorations (tiles,
// element_size_in_bits, tail padding). The evaluator only guarantees the
// minor_to_major of array results, so relayout the literal first when it
// disagrees with `shape` (e.g. tuple leaves come back in default layouts).
std::unique_ptr<HloInstruction> MakeFoldedConstant(Literal literal,
                                                   const Shape& shape) {
  if (!LayoutUtil::HasLayout(shape)) {
    return HloInstruction::CreateConstant(std::move(literal));
  }
  if (!LayoutUtil::LayoutsInShapesEqual(literal.shape(), shape,
                                        Layout::Equal().MinorToMajorOnly())) {
    literal = literal.Relayout(shape);
  }
  return std::make_unique<HloConstantInstruction>(std::move(literal), shape);
}

// In layout sensitive mode, a producer with a tuple shape can only be folded
// by rewriting each of its get-tuple-element users to a leaf constant.
bool CanRewriteTupleUsers(const HloInstruction* instruction) {
  if (instruction == instruction->parent()->root_instruction()) {
    return false;
  }
  for (const HloInstruction* user : instruction->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement ||
        !user->shape().IsArray() || user->HasControlDependencies()) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<bool> PropagateIdenticalConstantArguments(
    HloComputation* computation) {
  // For each parameter, figure out if all the arguments passed to that
  // parameter in the various call sites are identical constants.
  std::vector<bool> identical_constant_parameters(computation->num_parameters(),
                                                  true);
  for (int i = 0; i < computation->num_parameters(); ++i) {
    for (HloInstruction* call_site : computation->caller_instructions()) {
      if (call_site->operand(i)->opcode() != HloOpcode::kConstant ||
          (call_site->operand(i)->literal() !=
           computation->caller_instructions()[0]->operand(i)->literal())) {
        identical_constant_parameters[i] = false;
        break;
      }
    }
  }
  // If *all* parameters are identical constants, we can let the regular
  // constant-folding path handle it.
  if (absl::c_all_of(identical_constant_parameters, [](bool b) { return b; })) {
    return false;
  }

  bool changed = false;
  for (int i = 0; i < computation->num_parameters(); ++i) {
    if (identical_constant_parameters[i]) {
      HloInstruction* parameter = computation->parameter_instruction(i);
      if (parameter->IsDead()) {
        continue;
      }
      auto caller_instructions = computation->caller_instructions();
      if (caller_instructions.size() > 1) {
        // Sort the caller instructions by their unique id to make the
        // compilation deterministic.
        absl::c_sort(caller_instructions,
                     [](const HloInstruction* a, const HloInstruction* b) {
                       return a->unique_id() < b->unique_id();
                     });
      }
      const HloInstruction* constant = caller_instructions[0]->operand(i);
      ABSL_RETURN_IF_ERROR(parameter->ReplaceAllUsesWith(
          computation->AddInstruction(constant->Clone())));
      changed = true;
    }
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> HloConstantFolding::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Limit the constant folding to 0 iterations to skip folding loops in the
  // default case. This retains the behavior from before while loop support in
  // HloEvaluator and may be revised.
  auto evaluator = std::make_unique<HloEvaluator>(
      /*max_loop_iterations=*/options_.level == Level::kAggressive ? -1 : 0);
  // fast-path lets us e.g. use Eigen for matmuls.
  evaluator->set_use_fast_path(true);

  bool changed = false;

  // For each computation, cache whether we can fold all the instructions in it.
  absl::flat_hash_map<HloComputation*, bool> is_foldable_computation;

  std::vector<HloComputation*> computations =
      module->MakeNonfusionComputations(execution_threads);

  // Visit computations in reverse post-order, so that we can propagate constant
  // arguments from callers to callees.
  for (auto it = computations.rbegin(); it != computations.rend(); ++it) {
    HloComputation* computation = *it;
    // If the computation is only used by call instructions, check whether for
    // any of the parameters of the computation, the argument passed by the
    // call-sites is always the same constant. In that case, we can sink the
    // parameter into the computation before we perform constant folding on its
    // body.
    if (absl::c_all_of(computation->caller_instructions(),
                       [](HloInstruction* instruction) {
                         return instruction->opcode() == HloOpcode::kCall;
                       })) {
      ABSL_ASSIGN_OR_RETURN(bool did_change,
                       PropagateIdenticalConstantArguments(computation));
      changed |= did_change;
    }
    // Instructions removed while folding earlier entries of the snapshot
    // iterated below. Removed entries (e.g. the get-tuple-element users of a
    // folded tuple shaped producer, which appear after the producer in post
    // order) are recognized by pointer identity and skipped without being
    // dereferenced.
    absl::flat_hash_set<const HloInstruction*> removed_instructions;
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (removed_instructions.contains(instruction)) {
        continue;
      }
      // Skip dead code.
      if (instruction->IsDead()) {
        continue;
      }

      // We only handle instructions where
      //
      //  - at least one operand is a constant, and
      //  - all other operands are either constants or broadcast(constant).
      //
      // Why this particular set of rules around broadcasts?
      //
      //  - We don't want to fold broadcast(constant) on its own, because in
      //    general it's "simpler" to remember that it's a broadcast.  Also,
      //    algsimp will fold an all-one-value constant into a broadcast, so
      //    we'd just end up fighting with it.
      //
      //  - We don't want to fold an op where all operands are broadcasts of
      //    constants, because algsimp will transform op(broadcast(constant) =>
      //    broadcast(op(constant)).  Then we can constant-fold the smaller op.
      //
      //  - So the only remaining case is where some but not all operands are
      //    broadcasts of constants, e.g. op(constant, broadcast(constant)).
      //
      if (options_.level == HloConstantFolding::Level::kDefault &&
          !AnyOperandsConstant(instruction)) {
        continue;
      }
      if (!AllOperandsConstantOrBroadcastConstant(instruction)) {
        continue;
      }

      // Don't fold Constant, Parameter, and Tuple instructions.  Tuple
      // constants are not directly supported by any backends, hence folding
      // Tuple is not useful and would in fact be expanded back into kTuple by
      // Algebraic Simplifier.
      //
      // (We do allow folding subcomputations that contain these instructions.)
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->opcode() == HloOpcode::kConstant ||
          instruction->opcode() == HloOpcode::kTuple) {
        continue;
      }

      if (!IsFoldable(instruction, options_, is_foldable_computation)) {
        continue;
      }

      // In layout sensitive mode a tuple shaped constant may never be
      // materialized: fold a tuple shaped producer by rewriting each of its
      // get-tuple-element users to a leaf constant instead.
      const bool rewrite_tuple_users =
          options_.is_layout_sensitive && instruction->shape().IsTuple();
      if (rewrite_tuple_users && !CanRewriteTupleUsers(instruction)) {
        continue;
      }
      VLOG(5) << "Constant folding: " << instruction->ToString();

      absl::Duration slow_timeout =
          absl::Seconds(uint64_t{1} << slow_op_counter_.load());
      SlowOperationAlarm slow_alarm(slow_timeout, [instruction, slow_timeout] {
#if NDEBUG
        absl::string_view explanation_msg =
            "This isn't necessarily a bug; constant-folding is "
            "inherently a trade-off between compilation time and speed "
            "at runtime. XLA has some guards that attempt to keep "
            "constant folding from taking too long, but fundamentally "
            "you'll always be able to come up with an input program that "
            "takes a long time.\n\n"
            "If you'd like to file a bug, run with envvar "
            "XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.";
#else
        absl::string_view explanation_msg =
            "XLA was built without compiler optimizations, which can be "
            "slow. Try rebuilding with -c opt.";
#endif
        return absl::StrFormat(
            "Constant folding an instruction is taking > %s:\n\n"
            "  %s\n\n"  // instruction->name() or instruction->ToString()
            "%s",       // explanation_msg
            absl::FormatDuration(slow_timeout), instruction->ToString(),
            explanation_msg);
      });

      // Currently we skip unimplemented operations.
      Literal result;
      if (!evaluator->TryEvaluate(
              instruction, &result,
              /*recursively_evaluate_nonconstant_operands=*/true)) {
        VLOG(2) << "Constant folding failed for instruction: "
                << instruction->ToString();
        continue;
      }

      slow_alarm.cancel();
      if (slow_alarm.fired()) {
        slow_op_counter_++;
      }

      VLOG(4) << "Constant folded: " << instruction->ToString();
      changed = true;
      if (rewrite_tuple_users) {
        std::vector<Literal> leaves = result.DecomposeTuple();
        std::vector<HloInstruction*> users(instruction->users().begin(),
                                           instruction->users().end());
        for (HloInstruction* user : users) {
          if (!user->IsDead()) {
            HloInstruction* leaf_constant =
                user->AddInstruction(MakeFoldedConstant(
                    leaves[user->tuple_index()].Clone(), user->shape()));
            ABSL_RETURN_IF_ERROR(user->ReplaceAllUsesWith(leaf_constant));
          }
          ABSL_RETURN_IF_ERROR(RecursivelyRemoveDeadInstructionAndDeadOperands(
              *computation, user, removed_instructions));
        }
      } else {
        HloInstruction* new_constant = instruction->AddInstruction(
            MakeFoldedConstant(std::move(result), instruction->shape()));
        ABSL_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(new_constant));
        ABSL_RETURN_IF_ERROR(RecursivelyRemoveDeadInstructionAndDeadOperands(
            *computation, instruction, removed_instructions));
      }
    }
  }
  return changed;
}

}  // namespace xla
