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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
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

// Pair of `(input_index, output_index)` representing an `io_aliases` entry on
// an IFRT call op where `inputs[input_index]` may alias
// `outputs[output_index]`.
struct IoAlias {
  int32_t input_index;
  int32_t output_index;

  bool operator==(const IoAlias& other) const {
    return input_index == other.input_index &&
           output_index == other.output_index;
  }
  bool operator!=(const IoAlias& other) const { return !(*this == other); }
};

// Parses an `io_aliases` `mlir::ArrayAttr` (containing 2-element
// `mlir::DenseI32ArrayAttr` entries) into typed `IoAlias` pairs. Returns an
// empty vector if `io_aliases` is null. Index bounds (including
// non-negativity) are validated during op verification.
llvm::SmallVector<IoAlias, 4> GetIoAliases(mlir::ArrayAttr io_aliases);

namespace detail {

template <typename OpT>
IfrtPairwiseArrayOpInterface ClonePairwiseArrayOpWithNewInputsAndOutputs(
    OpT op, mlir::RewriterBase& rewriter, mlir::Location loc,
    mlir::ValueRange inputs, mlir::TypeRange output_types,
    mlir::ValueRange control_inputs);

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

// Verifies that `op` is directly nested in a `FuncOp` with `ifrt.function` or
// `ifrt.reshard_function` attr.
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

// Returns true if `control_inputs` can be forwarded to all users of
// `control_output` via `ForwardControlInputsToUsers` (either all users
// implement `IfrtControlOpInterface`, or there is a single unique control
// input).
bool CanForwardControlInputs(mlir::Value control_output,
                             mlir::ValueRange control_inputs);

// Forwards `control_inputs` to all users of `control_output`, removing
// `control_output` from each user's control dependencies.
void ForwardControlInputsToUsers(mlir::Value control_output,
                                 mlir::ValueRange control_inputs,
                                 mlir::RewriterBase& rewriter);

// Filters an array of per-argument or per-result dictionary attributes (e.g.,
// `arg_attrs` or `res_attrs`) by dropping entries whose index is set in
// `erase`. Returns `nullptr` if `dict_array` is null, all entries are erased,
// or no kept entry is a non-empty `mlir::DictionaryAttr`.
mlir::ArrayAttr FilterDictArrayAttr(mlir::ArrayAttr dict_array,
                                    const llvm::BitVector& erase);

// Partition of an `IfrtArrayOpInterface` op's array outputs into used (kept)
// and unused (erased) results, along with index remapping and kept types.
struct UsedOutputsSlice {
  llvm::BitVector keep;
  llvm::BitVector erase;
  llvm::SmallVector<int32_t> old_to_new;
  llvm::SmallVector<mlir::Type> kept_types;
};

// Inspects the array outputs of `op` and returns a `UsedOutputsSlice` recording
// which outputs are used vs. unused, their compacted output indices, and the
// types of the kept outputs.
UsedOutputsSlice CollectUsedOutputs(IfrtArrayOpInterface op);

// Replaces uses of kept array outputs of `old_op` with the corresponding
// compacted outputs of `new_op` according to `old_to_new` (-1 for erased
// outputs), forwards `old_op`'s control output to `new_op`'s control output,
// and erases `old_op`. `CHECK`-fails if `old_to_new.size()` does not match
// `old_op.getOutputs().size()`, any erased output still has uses, or any kept
// output index is outside `[0, new_op.getOutputs().size())`.
void ReplaceKeptOutputs(mlir::RewriterBase& rewriter,
                        IfrtArrayOpInterface old_op,
                        IfrtArrayOpInterface new_op,
                        llvm::ArrayRef<int32_t> old_to_new);

// Collects input indices of `call_op` that are donated
// (`donated_input_indices`) or aliased to an output (`io_aliases`).
llvm::SmallDenseSet<int32_t, 8> GetDonatedOrAliasedInputIndices(
    IfrtCallOpInterface call_op);

// Collects output indices of `call_op` that are aliased to an input in
// `io_aliases`.
llvm::SmallDenseSet<int32_t, 8> GetAliasedOutputIndices(
    IfrtCallOpInterface call_op);

// Returns true if `operand` is an array input marked as donated via
// `donated_input_indices` (on `IfrtCallOpInterface`) or `donated = true`.
// Note: `io_aliases` on `IfrtCallOpInterface` also consumes input buffer
// ownership; use `IsDonatedOrAliasedOperand` when checking whether an input
// buffer is consumed or aliased.
bool IsDonatedOperand(mlir::OpOperand& operand);

// Returns true if `operand` is an array input paired with an output via
// `io_aliases` (on `IfrtCallOpInterface`, which consumes the input buffer) or
// aliased without donation via `reuse = true` (`CopyArraysOp`) or
// `donated = false` (`BitcastArraysOp`, `RemapArraysOp`, `AssembleOp`,
// `DisassembleOp`).
bool IsAliasedOperand(mlir::OpOperand& operand);

// Returns true if `operand` is donated or aliased to its owner IFRT operation.
bool IsDonatedOrAliasedOperand(mlir::OpOperand& operand);

// Returns true if `value` is donated or aliased in any user `mlir::OpOperand`.
bool IsDonatedOrAliasedByUser(mlir::Value value);

namespace detail {

template <typename OpT>
IfrtPairwiseArrayOpInterface ClonePairwiseArrayOpWithNewInputsAndOutputs(
    OpT op, mlir::RewriterBase& rewriter, mlir::Location loc,
    mlir::ValueRange inputs, mlir::TypeRange output_types,
    mlir::ValueRange control_inputs) {
  assert(!inputs.empty() && inputs.size() == output_types.size());
  llvm::SmallVector<mlir::Value> operands;
  operands.reserve(inputs.size() + control_inputs.size());
  operands.append(inputs.begin(), inputs.end());
  operands.append(control_inputs.begin(), control_inputs.end());

  llvm::SmallVector<mlir::Type> result_types;
  result_types.reserve(output_types.size() + 1);
  result_types.append(output_types.begin(), output_types.end());
  result_types.push_back(op.getControlOutput().getType());

  typename OpT::Properties props = op.getProperties();
  props.operandSegmentSizes = {static_cast<int32_t>(inputs.size()),
                               static_cast<int32_t>(control_inputs.size())};

  OpT new_op = OpT::create(rewriter, loc, result_types, operands, props,
                           op->getDiscardableAttrDictionary().getValue());
  return llvm::cast<IfrtPairwiseArrayOpInterface>(new_op.getOperation());
}

template <typename OpT>
IfrtPairwiseArrayOpInterface ClonePairwiseArrayOpWithKeptOutputs(
    OpT op, mlir::RewriterBase& rewriter, const llvm::BitVector& keep) {
  const mlir::Operation::operand_range inputs = op.getInputs();
  const mlir::Operation::result_range outputs = op.getOutputs();
  assert(inputs.size() == outputs.size());
  assert(keep.size() == outputs.size());
  if (keep.none()) {
    return nullptr;
  }

  const unsigned num_kept = keep.count();
  llvm::SmallVector<mlir::Value> kept_inputs;
  llvm::SmallVector<mlir::Type> kept_output_types;
  kept_inputs.reserve(num_kept);
  kept_output_types.reserve(num_kept);
  for (int i : keep.set_bits()) {
    kept_inputs.push_back(inputs[i]);
    kept_output_types.push_back(outputs[i].getType());
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  return ClonePairwiseArrayOpWithNewInputsAndOutputs(
      op, rewriter, op.getLoc(), kept_inputs, kept_output_types,
      op.getControlInputs());
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
