/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_H_
#define XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_H_

#include <cassert>
#include <cstdint>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/sharding_param.h"

namespace xla {
namespace ifrt {

class IfrtControlType;
class IfrtPairwiseArrayOpInterface;

namespace detail {

// Forwards `control_inputs` to all users of `control_output`, removing
// `control_output` from each user's control dependencies.
void ForwardControlInputsToUsers(mlir::Value control_output,
                                 mlir::ValueRange control_inputs,
                                 mlir::RewriterBase& rewriter);

template <typename OpT>
IfrtPairwiseArrayOpInterface ClonePairwiseArrayOpWithKeptOutputs(
    OpT op, mlir::RewriterBase& rewriter, const llvm::BitVector& keep);

template <typename OpT>
IfrtPairwiseArrayOpInterface ReplacePairwiseArrayOpWithKeptOutputs(
    OpT op, mlir::RewriterBase& rewriter, const llvm::BitVector& keep);

}  // namespace detail

}  // namespace ifrt
}  // namespace xla

namespace mlir {
namespace OpTrait {
namespace xla {
namespace ifrt {

namespace impl {

// Verifies `op` used in a FuncOp with `ifrt.function` attr.
LogicalResult verifyNestedInIfrtFunc(Operation* op);

}  // namespace impl

template <typename ConcreteType>
class NestedInIfrtFuncTrait
    : public TraitBase<ConcreteType, NestedInIfrtFuncTrait> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return impl::verifyNestedInIfrtFunc(op);
  }
};

template <typename CalleeOpType>
class IfrtCallLikeTrait {
 public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, Impl> {
   public:
    // Verifies getCallee() is a valid SymbolRefAttr to CalleeOpType.
    static LogicalResult verifyTrait(Operation* op) {
      mlir::SymbolTableCollection symbol_table;
      ConcreteType concrete = llvm::cast<ConcreteType>(op);
      CalleeOpType callee = concrete.getCalleeOp(symbol_table);
      if (callee == nullptr) {
        return op->emitOpError() << "requires '" << concrete.getCallee()
                                 << "' to reference a valid `"
                                 << CalleeOpType::getOperationName() << "`";
      }
      if (callee->hasAttr(::xla::ifrt::kIfrtFunctionAttrName)) {
        return op->emitOpError() << "requires callee not with attr `"
                                 << ::xla::ifrt::kIfrtFunctionAttrName << "`";
      }
      return success();
    }

    CalleeOpType getCalleeOp(mlir::SymbolTableCollection& symbol_table) {
      SymbolRefAttr callee_attr = static_cast<ConcreteType*>(this)->getCallee();
      return symbol_table.lookupNearestSymbolFrom<CalleeOpType>(
          this->getOperation(), callee_attr);
    }
  };
};

}  // namespace ifrt
}  // namespace xla
}  // namespace OpTrait
}  // namespace mlir

// IWYU pragma: begin_exports

// Generated definitions.
#define GET_ATTR_INTERFACE_CLASSES
#include "xla/python/ifrt/ir/ifrt_attr_interfaces.h.inc"

#define GET_OP_INTERFACE_CLASSES
#include "xla/python/ifrt/ir/ifrt_op_interfaces.h.inc"

// IWYU pragma: end_exports

namespace xla {
namespace ifrt {
namespace detail {

template <typename OpT>
IfrtPairwiseArrayOpInterface ClonePairwiseArrayOpWithKeptOutputs(
    OpT op, mlir::RewriterBase& rewriter, const llvm::BitVector& keep) {
  const mlir::Operation::operand_range inputs = op.getInputs();
  const mlir::Operation::result_range outputs = op.getOutputs();
  const mlir::Operation::operand_range control_inputs = op.getControlInputs();
  assert(inputs.size() == outputs.size());
  assert(keep.size() == outputs.size());
  if (keep.none()) {
    return nullptr;
  }

  const unsigned num_kept = keep.count();
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<mlir::Type> result_types;
  operands.reserve(num_kept + control_inputs.size());
  result_types.reserve(num_kept + 1);
  for (int i : keep.set_bits()) {
    operands.push_back(inputs[i]);
    result_types.push_back(outputs[i].getType());
  }
  operands.append(control_inputs.begin(), control_inputs.end());
  result_types.push_back(op.getControlOutput().getType());

  typename OpT::Properties props = op.getProperties();
  props.operandSegmentSizes = {static_cast<int32_t>(num_kept),
                               static_cast<int32_t>(control_inputs.size())};

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  OpT new_op = OpT::create(rewriter, op.getLoc(), result_types, operands, props,
                           op->getDiscardableAttrDictionary().getValue());
  return llvm::cast<IfrtPairwiseArrayOpInterface>(new_op.getOperation());
}

template <typename OpT>
IfrtPairwiseArrayOpInterface ReplacePairwiseArrayOpWithKeptOutputs(
    OpT op, mlir::RewriterBase& rewriter, const llvm::BitVector& keep) {
  const mlir::Operation::result_range outputs = op.getOutputs();
  assert(keep.size() == outputs.size());
  for (unsigned i = 0, e = outputs.size(); i < e; ++i) {
    assert(keep.test(i) || outputs[i].use_empty());
  }
  if (keep.all()) {
    return llvm::cast<IfrtPairwiseArrayOpInterface>(op.getOperation());
  }
  if (keep.none()) {
    ForwardControlInputsToUsers(op.getControlOutput(), op.getControlInputs(),
                                rewriter);
    rewriter.eraseOp(op);
    return nullptr;
  }

  IfrtPairwiseArrayOpInterface new_op =
      ClonePairwiseArrayOpWithKeptOutputs(op, rewriter, keep);
  unsigned new_idx = 0;
  for (int i : keep.set_bits()) {
    rewriter.replaceAllUsesWith(outputs[i], new_op.getOutputs()[new_idx++]);
  }
  rewriter.replaceAllUsesWith(op.getControlOutput(), new_op.getControlOutput());
  rewriter.eraseOp(op);
  return new_op;
}

}  // namespace detail
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_H_
