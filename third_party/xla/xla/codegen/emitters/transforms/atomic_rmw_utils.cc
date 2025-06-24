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

}  // namespace

// Returns atomic op modifier and the atomic bin op kind.
std::optional<std::pair<Value, ml::AtomicBinOp>> GetAtomicModifierParameters(
    AtomicRMWOp op) {
  Type element_type = op.getInput().getType().getElementType();
  auto& operations = op.getBody()->getOperations();
  auto terminator = op.getBody()->getTerminator();
  if (operations.size() > 2) {
    return std::nullopt;
  }
  // If the body contains only the terminator, then it is an atomic store.
  if (operations.size() == 1) {
    // TODO(b/336367145): Support complex<f32> atomic store.
    if (element_type.isF32() || IsAtomicIntegral(element_type)) {
      return std::make_pair(terminator->getOperand(0), ml::AtomicBinOp::xchg);
    }
    return std::nullopt;
  }
  // Match the kind of the atomic op.
  // TODO(rocm): Match bf16 ops
  mlir::Operation* modifier_op = &operations.front();
  auto kind = GetAtomicBinOp(modifier_op, element_type);
  if (!kind.has_value()) {
    return std::nullopt;
  }
  // Find the modifier arg that does not match the argument of `atomic_rmw`
  // body.
  Value block_arg = op.getBody()->getArgument(0);
  Value modifier_arg = modifier_op->getOperand(0) == block_arg
                           ? modifier_op->getOperand(1)
                           : modifier_op->getOperand(0);
  return std::make_pair(modifier_arg, *kind);
}

}  // namespace emitters
}  // namespace xla
