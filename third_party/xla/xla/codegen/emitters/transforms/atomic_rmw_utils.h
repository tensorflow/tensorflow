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
#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_ATOMIC_RMW_UTILS_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_ATOMIC_RMW_UTILS_H_

#include <optional>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla {
namespace emitters {

enum class AtomicRMWKind {
  kAdd,
  kSub,
  kMaxS,
  kMinS,
  kMaxU,
  kMinU,
  kAnd,
  kOr,
  kXor,
  kFAdd,
  kFSub,
  // `kFMax` and `kFMin` have `arith.maximumf` and `arith.minimumf` semantics
  // (IEEE 754-2019 maximum and minimum, `llvm.atomicrmw fmaximum` and
  // `fminimum`): a NaN operand yields NaN, and -0.0 orders below +0.0. They are
  // not `llvm.atomicrmw fmax` and `fmin`, which are IEEE 754-2008 maxNum and
  // minNum: those return the other operand when one is NaN, and either zero for
  // -0.0 and +0.0. Lowering to them is only correct under `nnan` and `nsz`.
  kFMax,
  kFMin,
};

// Describes an `xla.atomic_rmw` computation as `current <kind> modifier`.
struct AtomicModifierParameters {
  // Either an SSA value defined outside the `xla.atomic_rmw` region or a
  // TypedAttr constant narrowed losslessly to the atomic value type.
  mlir::OpFoldResult modifier;
  AtomicRMWKind kind;
};

// Matches an `xla.atomic_rmw` whose yielded value computes
// `current <kind> modifier`, looking through widening `arith.extf` and
// narrowing `arith.truncf` casts for low precision float reductions.
std::optional<AtomicModifierParameters> GetAtomicModifierParameters(
    AtomicRMWOp op);

// Returns `modifier` as an SSA Value, materializing an `arith.constant` when
// `modifier` holds a narrowed constant attribute.
mlir::Value MaterializeAtomicModifier(mlir::OpBuilder& b, mlir::Location loc,
                                      mlir::OpFoldResult modifier);

}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_ATOMIC_RMW_UTILS_H_
