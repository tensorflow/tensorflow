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

#include <optional>
#include <utility>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ilist.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla {
namespace emitters {
namespace {

using mlir::Operation;
using mlir::Type;
using mlir::Value;

namespace ml = ::mlir::LLVM;
namespace arith = ::mlir::arith;

bool IsAtomicIntegral(Type element_type) {
  if (!element_type.isInteger()) {
    return false;
  }
  unsigned element_bitwidth = element_type.getIntOrFloatBitWidth();
  return element_bitwidth == 32 || element_bitwidth == 64;
}

std::optional<ml::AtomicBinOp> GetAtomicBinOp(Operation* modifier_op,
                                              Type element_type) {
  return llvm::TypeSwitch<Operation*, std::optional<ml::AtomicBinOp>>(
             modifier_op)
      // Floating-point operations.
      .Case([](arith::AddFOp op) { return ml::AtomicBinOp::fadd; })
      .Case([](arith::MaximumFOp op) { return ml::AtomicBinOp::fmax; })
      .Case([](arith::MinimumFOp op) { return ml::AtomicBinOp::fmin; })
      // Integer operations.
      .Case([&](arith::AddIOp op) {
        return IsAtomicIntegral(element_type)
                   ? std::make_optional(ml::AtomicBinOp::add)
                   : std::nullopt;
      })
      .Case([&](arith::MaxUIOp op) {
        return IsAtomicIntegral(element_type)
                   ? std::make_optional(ml::AtomicBinOp::umax)
                   : std::nullopt;
      })
      .Case([&](arith::MinUIOp op) {
        return IsAtomicIntegral(element_type)
                   ? std::make_optional(ml::AtomicBinOp::umin)
                   : std::nullopt;
      })
      .Case([&](arith::MaxSIOp op) {
        return IsAtomicIntegral(element_type)
                   ? std::make_optional(ml::AtomicBinOp::max)
                   : std::nullopt;
      })
      .Case([&](arith::MinSIOp op) {
        return IsAtomicIntegral(element_type)
                   ? std::make_optional(ml::AtomicBinOp::min)
                   : std::nullopt;
      })
      .Default([](Operation* op) { return std::nullopt; });
}

// Looks through an arith.extf widening cast and returns the narrower source
// value. Low-precision floating-point reductions (e.g. bf16) are computed in a
// wider type, so the atomic modifier appears in the body as extf(modifier).
Value LookThroughExtF(Value value) {
  if (auto ext = value.getDefiningOp<arith::ExtFOp>()) {
    return ext.getIn();
  }
  return value;
}

}  // namespace

// Returns atomic op modifier and the atomic bin op kind.
std::optional<std::pair<Value, ml::AtomicBinOp>> GetAtomicModifierParameters(
    AtomicRMWOp op) {
  Type element_type = op.getInput().getType().getElementType();
  auto& operations = op.getBody()->getOperations();
  auto terminator = op.getBody()->getTerminator();
  Value block_arg = op.getBody()->getArgument(0);

  // If the body contains only the terminator, then it is an atomic store.
  if (operations.size() == 1) {
    // TODO(b/336367145): Support complex<f32> atomic store.
    if (element_type.isF32() || IsAtomicIntegral(element_type)) {
      return std::make_pair(terminator->getOperand(0), ml::AtomicBinOp::xchg);
    }
    return std::nullopt;
  }

  // Simple case: a single binary modifier op followed by the terminator,
  // operating directly on the atomic element type.
  if (operations.size() == 2) {
    mlir::Operation* modifier_op = &operations.front();
    auto kind = GetAtomicBinOp(modifier_op, element_type);
    if (!kind.has_value()) {
      return std::nullopt;
    }
    // Find the modifier arg that does not match the argument of `atomic_rmw`
    // body.
    Value modifier_arg = modifier_op->getOperand(0) == block_arg
                             ? modifier_op->getOperand(1)
                             : modifier_op->getOperand(0);
    return std::make_pair(modifier_arg, *kind);
  }

  // Widened low-precision case (e.g. bf16): the reduction is computed in a
  // wider type, so the body looks like:
  //   %0 = arith.extf %current  : bf16 to f32
  //   %1 = arith.extf %modifier : bf16 to f32
  //   %2 = arith.<binop> %0, %1 : f32
  //   %3 = arith.truncf %2      : f32 to bf16
  //   xla.yield %3              : bf16
  //
  // bf16 has no native arithmetic on the relevant targets, so
  // FloatNormalization (driven by GpuFloatSupport, which reports bf16
  // add/mul/etc. as unsupported) legitimately wraps the combiner in f32
  // conversions. We do NOT undo that normalization globally; we only look
  // through it *here*, where we are about to emit a hardware atomic (e.g.
  // global_atomic_pk_add_bf16) that performs the very same widen-add-round
  // internally. Recovering the narrow (bf16) modifier lets the lowering use
  // that packed atomic instead of a slow compare-and-swap loop, without
  // claiming a native bf16 arithmetic instruction.
  auto trunc_op = terminator->getOperand(0).getDefiningOp<arith::TruncFOp>();
  if (!trunc_op || trunc_op.getType() != element_type) {
    return std::nullopt;
  }
  mlir::Operation* modifier_op = trunc_op.getIn().getDefiningOp();
  if (!modifier_op || modifier_op->getNumOperands() != 2) {
    return std::nullopt;
  }
  auto kind = GetAtomicBinOp(modifier_op, element_type);
  if (!kind.has_value()) {
    return std::nullopt;
  }
  Value lhs = LookThroughExtF(modifier_op->getOperand(0));
  Value rhs = LookThroughExtF(modifier_op->getOperand(1));
  Value modifier_arg;
  if (lhs == block_arg) {
    modifier_arg = rhs;
  } else if (rhs == block_arg) {
    modifier_arg = lhs;
  } else {
    return std::nullopt;
  }
  // The recovered modifier must have the atomic element type for the downstream
  // direct-atomic emission to be valid.
  if (modifier_arg.getType() != element_type) {
    return std::nullopt;
  }
  return std::make_pair(modifier_arg, *kind);
}

}  // namespace emitters
}  // namespace xla
