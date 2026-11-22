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

#include "xla/codegen/emitters/transforms/atomic_rmw_utils.h"

#include <cassert>
#include <optional>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla {
namespace emitters {
namespace {

using mlir::Operation;
using mlir::Type;
using mlir::Value;

namespace arith = ::mlir::arith;

struct BinaryOpMatch {
  AtomicRMWKind kind;
  bool is_commutative;
};

bool IsFloatAtomicRMWKind(AtomicRMWKind kind) {
  switch (kind) {
    case AtomicRMWKind::kFAdd:
    case AtomicRMWKind::kFSub:
    case AtomicRMWKind::kFMax:
    case AtomicRMWKind::kFMin:
      return true;
    default:
      return false;
  }
}

std::optional<BinaryOpMatch> MatchBinaryOpKind(Operation* modifier_op) {
  if (!modifier_op) {
    return std::nullopt;
  }
  return llvm::TypeSwitch<Operation*, std::optional<BinaryOpMatch>>(modifier_op)
      .Case([](arith::AddFOp op) -> std::optional<BinaryOpMatch> {
        if (op.getRoundingmode().has_value() &&
            *op.getRoundingmode() != arith::RoundingMode::to_nearest_even) {
          return std::nullopt;
        }
        return BinaryOpMatch{AtomicRMWKind::kFAdd, true};
      })
      .Case([](arith::SubFOp op) -> std::optional<BinaryOpMatch> {
        if (op.getRoundingmode().has_value() &&
            *op.getRoundingmode() != arith::RoundingMode::to_nearest_even) {
          return std::nullopt;
        }
        return BinaryOpMatch{AtomicRMWKind::kFSub, false};
      })
      .Case([](arith::MaximumFOp) {
        return BinaryOpMatch{AtomicRMWKind::kFMax, true};
      })
      .Case([](arith::MinimumFOp) {
        return BinaryOpMatch{AtomicRMWKind::kFMin, true};
      })
      .Case([](arith::AddIOp) {
        return BinaryOpMatch{AtomicRMWKind::kAdd, true};
      })
      .Case([](arith::SubIOp) {
        return BinaryOpMatch{AtomicRMWKind::kSub, false};
      })
      .Case([](arith::MaxSIOp) {
        return BinaryOpMatch{AtomicRMWKind::kMaxS, true};
      })
      .Case([](arith::MinSIOp) {
        return BinaryOpMatch{AtomicRMWKind::kMinS, true};
      })
      .Case([](arith::MaxUIOp) {
        return BinaryOpMatch{AtomicRMWKind::kMaxU, true};
      })
      .Case([](arith::MinUIOp) {
        return BinaryOpMatch{AtomicRMWKind::kMinU, true};
      })
      .Case([](arith::AndIOp) {
        return BinaryOpMatch{AtomicRMWKind::kAnd, true};
      })
      .Case(
          [](arith::OrIOp) { return BinaryOpMatch{AtomicRMWKind::kOr, true}; })
      .Case([](arith::XOrIOp) {
        return BinaryOpMatch{AtomicRMWKind::kXor, true};
      })
      .Default([](Operation*) { return std::nullopt; });
}

Value UnwrapExtF(Value value, Type target_type) {
  if (auto ext = value.getDefiningOp<arith::ExtFOp>()) {
    if (ext.getIn().getType() == target_type) {
      return ext.getIn();
    }
  }
  return Value{};
}

std::optional<llvm::APFloat> ConvertLosslessly(
    llvm::APFloat value, mlir::FloatType target_float_type) {
  const llvm::fltSemantics& target_sem = target_float_type.getFloatSemantics();
  if (value.isNaN() ||
      (value.isInfinity() && !llvm::APFloat::semanticsHasInf(target_sem))) {
    return std::nullopt;
  }
  const llvm::fltSemantics& src_sem = value.getSemantics();
  llvm::APFloat narrowed = value;
  bool loses_info = false;
  narrowed.convert(target_sem, llvm::APFloat::rmNearestTiesToEven, &loses_info);
  if (loses_info) {
    return std::nullopt;
  }
  llvm::APFloat roundtrip = narrowed;
  bool roundtrip_loses_info = false;
  roundtrip.convert(src_sem, llvm::APFloat::rmNearestTiesToEven,
                    &roundtrip_loses_info);
  if (roundtrip_loses_info || !roundtrip.bitwiseIsEqual(value)) {
    return std::nullopt;
  }
  return narrowed;
}

// Recovers a `target_type` modifier from a widened float operand, either by
// looking through `arith.extf` or by narrowing a constant whose value is exact
// in `target_type` (for example after `arith.extf(cst)` constant folded to a
// wider float constant).
std::optional<mlir::OpFoldResult> RecoverNarrowFloatModifier(
    AtomicRMWOp atomic_op, Value widened_value, Type target_type) {
  if (Value narrow = UnwrapExtF(widened_value, target_type)) {
    if (atomic_op.isDefinedOutsideOfLoop(narrow)) {
      return narrow;
    }
    return std::nullopt;
  }

  if (auto target_float_type = mlir::dyn_cast<mlir::FloatType>(target_type)) {
    mlir::FloatAttr float_attr;
    if (mlir::matchPattern(widened_value, mlir::m_Constant(&float_attr))) {
      if (std::optional<llvm::APFloat> narrowed =
              ConvertLosslessly(float_attr.getValue(), target_float_type)) {
        return mlir::FloatAttr::get(target_float_type, *narrowed);
      }
    }
    return std::nullopt;
  }

  if (auto target_vec_type = mlir::dyn_cast<mlir::VectorType>(target_type)) {
    auto target_elem_type =
        mlir::dyn_cast<mlir::FloatType>(target_vec_type.getElementType());
    mlir::DenseFPElementsAttr dense_attr;
    if (!target_elem_type ||
        !mlir::matchPattern(widened_value, mlir::m_Constant(&dense_attr)) ||
        dense_attr.getNumElements() != target_vec_type.getNumElements()) {
      return std::nullopt;
    }
    llvm::SmallVector<llvm::APFloat> narrowed_values;
    narrowed_values.reserve(dense_attr.getNumElements());
    for (const llvm::APFloat& elem : dense_attr.getValues<llvm::APFloat>()) {
      std::optional<llvm::APFloat> narrowed =
          ConvertLosslessly(elem, target_elem_type);
      if (!narrowed.has_value()) {
        return std::nullopt;
      }
      narrowed_values.push_back(*narrowed);
    }
    return mlir::DenseElementsAttr::get(target_vec_type, narrowed_values);
  }

  return std::nullopt;
}

std::optional<AtomicModifierParameters> MatchDirectBinaryModifier(
    AtomicRMWOp atomic_op, Operation* modifier_op, Value current,
    Type expected_type) {
  std::optional<BinaryOpMatch> match = MatchBinaryOpKind(modifier_op);
  if (!match.has_value()) {
    return std::nullopt;
  }
  Value lhs = modifier_op->getOperand(0);
  Value rhs = modifier_op->getOperand(1);
  Value modifier_arg;
  if (lhs == current && rhs != current) {
    modifier_arg = rhs;
  } else if (match->is_commutative && rhs == current && lhs != current) {
    modifier_arg = lhs;
  } else {
    return std::nullopt;
  }
  if (modifier_arg.getType() != expected_type ||
      !atomic_op.isDefinedOutsideOfLoop(modifier_arg)) {
    return std::nullopt;
  }
  return AtomicModifierParameters{modifier_arg, match->kind};
}

}  // namespace

std::optional<AtomicModifierParameters> GetAtomicModifierParameters(
    AtomicRMWOp op) {
  mlir::Block* body = op.getBody();
  if (!llvm::all_of(body->without_terminator(), [](Operation& body_op) {
        return mlir::wouldOpBeTriviallyDead(&body_op);
      })) {
    return std::nullopt;
  }
  Operation* terminator = body->getTerminator();
  if (terminator->getNumOperands() != 1) {
    return std::nullopt;
  }
  Value current = op.getCurrentValue();
  Type expected_type = current.getType();
  if (auto vec_type = mlir::dyn_cast<mlir::VectorType>(expected_type);
      vec_type && (vec_type.getRank() != 1 || vec_type.isScalable())) {
    return std::nullopt;
  }
  Value yielded = terminator->getOperand(0);

  if (std::optional<AtomicModifierParameters> direct =
          MatchDirectBinaryModifier(op, yielded.getDefiningOp(), current,
                                    expected_type)) {
    return direct;
  }

  // Widened low precision float case (e.g. bf16 computed in f32):
  //   %0 = arith.extf %current  : bf16 to f32
  //   %1 = arith.extf %modifier : bf16 to f32  (or lossless f32 constant)
  //   %2 = arith.<binop> %0, %1 : f32
  //   %3 = arith.truncf %2      : f32 to bf16
  //   xla.yield %3              : bf16
  // Restrict to sub 32 bit float target types promoted by FloatNormalization so
  // explicit f32 to f64 precision elevations stay in f64.
  auto expected_float_type = mlir::dyn_cast<mlir::FloatType>(
      mlir::getElementTypeOrSelf(expected_type));
  if (!expected_float_type || expected_float_type.getWidth() >= 32) {
    return std::nullopt;
  }
  auto trunc_op = yielded.getDefiningOp<arith::TruncFOp>();
  if (!trunc_op || trunc_op.getType() != expected_type ||
      (trunc_op.getRoundingmode().has_value() &&
       *trunc_op.getRoundingmode() != arith::RoundingMode::to_nearest_even)) {
    return std::nullopt;
  }
  Operation* modifier_op = trunc_op.getIn().getDefiningOp();
  std::optional<BinaryOpMatch> match = MatchBinaryOpKind(modifier_op);
  if (!match.has_value() || !IsFloatAtomicRMWKind(match->kind)) {
    return std::nullopt;
  }
  auto wide_float_type = mlir::dyn_cast<mlir::FloatType>(
      mlir::getElementTypeOrSelf(trunc_op.getIn().getType()));
  if (!wide_float_type) {
    return std::nullopt;
  }
  const llvm::fltSemantics& narrow_sem =
      expected_float_type.getFloatSemantics();
  const llvm::fltSemantics& wide_sem = wide_float_type.getFloatSemantics();
  unsigned int narrow_prec = llvm::APFloat::semanticsPrecision(narrow_sem);
  unsigned int wide_prec = llvm::APFloat::semanticsPrecision(wide_sem);
  // Require lossless widening into the intermediate format and, for addition
  // and subtraction, satisfy wide_prec >= 2 * narrow_prec + 2 so widened
  // rounding matches single rounding in the narrow type.
  if (wide_prec <= narrow_prec ||
      !llvm::APFloat::isLosslesslyConvertibleTo(narrow_sem, wide_sem,
                                                /*IgnoreNaNs=*/true)) {
    return std::nullopt;
  }
  if ((match->kind == AtomicRMWKind::kFAdd ||
       match->kind == AtomicRMWKind::kFSub) &&
      wide_prec < 2 * narrow_prec + 2) {
    return std::nullopt;
  }

  Value raw_lhs = modifier_op->getOperand(0);
  Value raw_rhs = modifier_op->getOperand(1);
  Value unwrapped_lhs = UnwrapExtF(raw_lhs, expected_type);
  Value unwrapped_rhs = UnwrapExtF(raw_rhs, expected_type);

  Value raw_modifier;
  if (unwrapped_lhs == current && unwrapped_rhs != current) {
    raw_modifier = raw_rhs;
  } else if (match->is_commutative && unwrapped_rhs == current &&
             unwrapped_lhs != current) {
    raw_modifier = raw_lhs;
  } else {
    return std::nullopt;
  }

  std::optional<mlir::OpFoldResult> narrow_modifier =
      RecoverNarrowFloatModifier(op, raw_modifier, expected_type);
  if (!narrow_modifier.has_value()) {
    return std::nullopt;
  }
  return AtomicModifierParameters{*narrow_modifier, match->kind};
}

Value MaterializeAtomicModifier(mlir::OpBuilder& b, mlir::Location loc,
                                mlir::OpFoldResult modifier) {
  assert(modifier && "expected nonnull atomic modifier");
  if (auto value = mlir::dyn_cast<Value>(modifier)) {
    return value;
  }
  return arith::ConstantOp::create(
      b, loc,
      mlir::cast<mlir::TypedAttr>(mlir::cast<mlir::Attribute>(modifier)));
}

}  // namespace emitters
}  // namespace xla
