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

#include "xla/tools/hlo_isolation/hlo_inf_nan_intent_analyzer.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"

namespace xla {
namespace hlo_isolation {
namespace {

// Returns true if the instruction is a constant containing Inf or NaN, or a
// broadcast of one.
bool IsConstantInfOrNan(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kBroadcast) {
    instr = instr->operand(0);
  }
  return instr->opcode() == HloOpcode::kConstant &&
         LiteralContainsInfOrNan(instr->literal());
}

// Returns true if `operand` flows into `user` via a structural (data movement
// or indexing) operation that passes element values through unchanged.
bool IsStructuralOperand(const HloInstruction* user,
                         const HloInstruction* operand) {
  switch (user->opcode()) {
    case HloOpcode::kBroadcast:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kCopy:
    case HloOpcode::kBitcast:
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTranspose:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReverse:
    case HloOpcode::kPad:
      return true;
    case HloOpcode::kSelect:
      // Only the value branches (operand 1: on_true, operand 2: on_false) pass
      // data.
      return user->operand(1) == operand || user->operand(2) == operand;
    case HloOpcode::kDynamicSlice:
      // Operand 0 is data; operands 1..N are slice start indices.
      return user->operand(0) == operand;
    case HloOpcode::kDynamicUpdateSlice:
      // Operands 0 (base) and 1 (update) are data; operands 2..N are indices.
      return user->operand(0) == operand || user->operand(1) == operand;
    default:
      return false;
  }
}

// Returns true if `shape` is a floating-point or complex type, or if it is a
// tuple that contains at least one floating-point or complex leaf element.
bool ShapeContainsFloatingOrComplex(const Shape& shape) {
  bool has_floating = false;
  ShapeUtil::ForEachLeafShape(
      shape, [&](const Shape& leaf_shape, const ShapeIndex& /*index*/) {
        if (ShapeUtil::ElementIsFloating(leaf_shape) ||
            ShapeUtil::ElementIsComplex(leaf_shape)) {
          has_floating = true;
        }
      });
  return has_floating;
}

// Returns true if an Inf/NaN at `operand` is guaranteed to propagate through
// `user`'s output. Uses a denylist of operations and operand roles that can
// extinguish Inf/NaN (e.g. clamp, saturating activations like tanh/sigmoid,
// exp(-inf)=0) or consume operands as non-data (e.g. comparison, select
// condition, indices).
bool IsInfNanPropagatingOperand(const HloInstruction* user,
                                const HloInstruction* operand) {
  // 1. Output must be capable of containing floating-point values.
  if (!ShapeContainsFloatingOrComplex(user->shape())) {
    return false;
  }

  // 2. Denylist of non-propagating operand positions.
  switch (user->opcode()) {
    case HloOpcode::kSelect:
      // Operand 0 is the boolean predicate, not data.
      if (user->operand(0) == operand) return false;
      break;
    case HloOpcode::kDynamicSlice:
      // Operand 0 is data; operands 1..N are indices.
      if (user->operand(0) != operand) return false;
      break;
    case HloOpcode::kDynamicUpdateSlice:
      // Operands 0 and 1 are data; operands 2..N are indices.
      if (user->operand(0) != operand && user->operand(1) != operand) {
        return false;
      }
      break;
    case HloOpcode::kGather:
      // Operand 0 is data; operand 1 is start indices.
      if (user->operand(0) != operand) return false;
      break;
    case HloOpcode::kScatter:
      // Operand 1 is scatter indices.
      if (user->operand(1) == operand) return false;
      break;
    case HloOpcode::kReduce:
      // Operand 0 is data to reduce; operand 1 is init value.
      if (user->operand(0) != operand) return false;
      break;
    default:
      break;
  }

  // 3. Denylist of operations that extinguish Inf/NaN to finite values or
  // do not propagate data.
  switch (user->opcode()) {
    case HloOpcode::kClamp:
    case HloOpcode::kLogistic:
    case HloOpcode::kTanh:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kRng:
      return false;
    case HloOpcode::kDivide:
      // Denominator inf can produce 0 (x / inf = 0).
      if (user->operand(1) == operand) return false;
      break;
    case HloOpcode::kPower:
      // Power can extinguish inf (e.g. inf^0 = 1).
      return false;
    default:
      break;
  }

  return true;
}

// Returns true if the computation contains operations (e.g., sqrt, divide, log)
// that could produce NaN or Inf from finite inputs.
bool CanProduceInfNan(const HloComputation* comp) {
  for (const HloInstruction* instr : comp->instructions()) {
    switch (instr->opcode()) {
      // Roots and powers:
      case HloOpcode::kSqrt:
      case HloOpcode::kRsqrt:
      case HloOpcode::kPower:
      // Division and remainder:
      case HloOpcode::kDivide:
      case HloOpcode::kRemainder:
      // Logarithms:
      case HloOpcode::kLog:
      case HloOpcode::kLog1p:
      // Inverse trigonometric and hyperbolic functions:
      case HloOpcode::kAsin:
      case HloOpcode::kAcos:
      case HloOpcode::kAcosh:
      case HloOpcode::kAtanh:
      // Matrix factorizations and triangular solvers:
      case HloOpcode::kCholesky:
      case HloOpcode::kTriangularSolve:
        return true;
      default:
        break;
    }
  }
  return false;
}

// Detects masked reductions (e.g., masked attention) where masked elements are
// replaced with an infinity fallback that matches the reduction's neutral
// identity:
//   select(mask, data, -inf) -> reduce_max(..., init=-inf)
bool IsIntentionalMaskedReduction(const HloComputation* comp,
                                  const InfNanIntentOptions& options) {
  if (options.reject_unconstrained_ops && CanProduceInfNan(comp)) {
    return false;
  }
  for (const HloInstruction* instr : comp->instructions()) {
    if (instr->opcode() != HloOpcode::kSelect) continue;

    // Check if either the on_true (operand 1) or on_false (operand 2) branch
    // provides an Inf/NaN fallback.
    bool has_inf_fallback = IsConstantInfOrNan(instr->operand(1)) ||
                            IsConstantInfOrNan(instr->operand(2));
    if (!has_inf_fallback) continue;

    for (const HloInstruction* user : instr->users()) {
      if (user->opcode() == HloOpcode::kReduce && user->operand(0) == instr &&
          IsConstantInfOrNan(user->operand(1))) {
        return true;
      }
    }
  }
  return false;
}

// Traverses forward from an Inf/NaN constant instruction to determine whether
// it propagates to the computation root.
//
// If `allow_arithmetic_propagation` is false, it only checks if there is a path
// from the constant Inf/NaN to root through structural data movement ops
// (broadcast, slice, select, pad, etc.).
// If `allow_arithmetic_propagation` is true, it checks if the constant Inf/NaN
// can propagate to root through arithmetic ops without being extinguished
// (e.g. 1/inf -> 0 extinguishes the inf).
bool PropagatesInfNanToRoot(const HloInstruction* inf_nan_constant,
                            const HloComputation* comp,
                            bool allow_arithmetic_propagation) {
  const HloInstruction* root = comp->root_instruction();
  if (root == nullptr) return false;

  std::queue<const HloInstruction*> queue;
  absl::flat_hash_set<const HloInstruction*> visited;

  queue.push(inf_nan_constant);
  visited.insert(inf_nan_constant);

  while (!queue.empty()) {
    const HloInstruction* curr = queue.front();
    queue.pop();

    if (curr == root) {
      return true;
    }

    for (const HloInstruction* user : curr->users()) {
      if (user->parent() != comp) continue;
      bool is_valid_step = allow_arithmetic_propagation
                               ? IsInfNanPropagatingOperand(user, curr)
                               : IsStructuralOperand(user, curr);
      if (is_valid_step && visited.insert(user).second) {
        queue.push(user);
      }
    }
  }
  return false;
}

}  // namespace

bool LiteralContainsInfOrNan(const LiteralSlice& literal) {
  if (literal.shape().IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(literal.shape()); ++i) {
      if (LiteralContainsInfOrNan(LiteralSlice(literal, {i}))) {
        return true;
      }
    }
    return false;
  }
  return primitive_util::PrimitiveTypeSwitch<bool>(
      [&](auto type) -> bool {
        if constexpr (primitive_util::IsFloatingPointType(type)) {
          using NativeT = primitive_util::NativeTypeOf<type>;
          if (!std::numeric_limits<NativeT>::has_infinity &&
              !std::numeric_limits<NativeT>::has_quiet_NaN) {
            return false;
          }
          bool found = false;
          literal.EachCellUntilFailure<NativeT>(
              [&](absl::Span<const int64_t> /*indices*/,
                  NativeT value) -> bool {
                if (std::isinf(value) || std::isnan(value)) {
                  found = true;
                  return false;
                }
                return true;
              });
          return found;
        } else if constexpr (primitive_util::IsComplexType(type)) {
          using NativeT = primitive_util::NativeTypeOf<type>;
          bool found = false;
          literal.EachCellUntilFailure<NativeT>(
              [&](absl::Span<const int64_t> /*indices*/,
                  NativeT value) -> bool {
                if (std::isinf(value.real()) || std::isinf(value.imag()) ||
                    std::isnan(value.real()) || std::isnan(value.imag())) {
                  found = true;
                  return false;
                }
                return true;
              });
          return found;
        }
        return false;
      },
      literal.shape().element_type());
}

bool ModuleContainsConstantInfOrNan(const HloModule& module) {
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kConstant &&
          LiteralContainsInfOrNan(instr->literal())) {
        return true;
      }
    }
  }
  return false;
}

bool IsIntentionalInfNan(const HloModule& module,
                         const InfNanIntentOptions& options) {
  // 1. Fast check: return false if the module contains no constant Inf/NaN.
  if (!ModuleContainsConstantInfOrNan(module)) {
    VLOG(2) << "Module " << module.name()
            << " has no constant Inf/NaN literals.";
    return false;
  }

  // 2. Check each computation for intentional paths.
  for (const HloComputation* comp : module.computations()) {
    // Check for masked reductions (e.g. attention mask to reduce_max).
    if (IsIntentionalMaskedReduction(comp, options)) {
      VLOG(2) << "Module " << module.name()
              << " detected intentional masked reduction in " << comp->name();
      return true;
    }

    // Check forward dataflow reachability from Inf/NaN constants to ROOT:
    bool comp_can_produce_inf_nan = CanProduceInfNan(comp);

    for (const HloInstruction* inf_nan_constant : comp->instructions()) {
      if (inf_nan_constant->opcode() != HloOpcode::kConstant ||
          !LiteralContainsInfOrNan(inf_nan_constant->literal())) {
        continue;
      }

      // Arithmetic propagation is only permitted if the computation cannot
      // produce Inf/NaN from finite inputs (e.g. no sqrt, divide, log). If
      // hazard ops are present, only pure structural propagation is allowed.
      bool allow_arithmetic = !comp_can_produce_inf_nan;
      if (PropagatesInfNanToRoot(inf_nan_constant, comp, allow_arithmetic)) {
        VLOG(2) << "Module " << module.name()
                << " detected intentional path in " << comp->name()
                << " from constant " << inf_nan_constant->name()
                << " to ROOT (allow_arithmetic=" << allow_arithmetic << ").";
        return true;
      }
    }
  }

  VLOG(2) << "Module " << module.name()
          << " contains constant Inf/NaN but no intentional path to ROOT was "
             "found.";
  return false;
}

}  // namespace hlo_isolation
}  // namespace xla
