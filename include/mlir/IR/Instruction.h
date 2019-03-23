//===- Instruction.h - MLIR Instruction Class -------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines the Instruction class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INSTRUCTION_H
#define MLIR_IR_INSTRUCTION_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/InstructionSupport.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

namespace mlir {
class BlockAndValueMapping;
class Location;
class MLIRContext;
template <typename OpType> class OpPointer;
class OperandIterator;
class ResultIterator;
class ResultTypeIterator;

/// Terminator operations can have Block operands to represent successors.
using BlockOperand = IROperandImpl<Block>;

/// Instruction is a basic unit of execution within a function. Instructions can
/// be nested within other instructions effectively forming a tree. Child
/// instructions are organized into instruction blocks represented by a 'Block'
/// class.
class Instruction final
    : public llvm::ilist_node_with_parent<Instruction, Block>,
      private llvm::TrailingObjects<Instruction, InstResult, BlockOperand,
                                    unsigned, Region, detail::OperandStorage> {
public:
  /// Create a new Instruction with the specific fields.
  static Instruction *create(Location location, OperationName name,
                             ArrayRef<Value *> operands,
                             ArrayRef<Type> resultTypes,
                             ArrayRef<NamedAttribute> attributes,
                             ArrayRef<Block *> successors, unsigned numRegions,
                             bool resizableOperandList, MLIRContext *context);

  /// Overload of create that takes an existing NamedAttributeList to avoid
  /// unnecessarily uniquing a list of attributes.
  static Instruction *create(Location location, OperationName name,
                             ArrayRef<Value *> operands,
                             ArrayRef<Type> resultTypes,
                             const NamedAttributeList &attributes,
                             ArrayRef<Block *> successors, unsigned numRegions,
                             bool resizableOperandList, MLIRContext *context);

  /// The name of an operation is the key identifier for it.
  OperationName getName() { return name; }

  /// If this operation has a registered operation description, return it.
  /// Otherwise return null.
  const AbstractOperation *getAbstractOperation() {
    return getName().getAbstractOperation();
  }

  /// Remove this instruction from its parent block and delete it.
  void erase();

  /// Create a deep copy of this instruction, remapping any operands that use
  /// values outside of the instruction using the map that is provided (leaving
  /// them alone if no entry is present).  Replaces references to cloned
  /// sub-instructions to the corresponding instruction that is copied, and adds
  /// those mappings to the map.
  Instruction *clone(BlockAndValueMapping &mapper, MLIRContext *context);
  Instruction *clone(MLIRContext *context);

  /// Returns the instruction block that contains this instruction.
  Block *getBlock() { return block; }

  /// Return the context this operation is associated with.
  MLIRContext *getContext();

  /// The source location the operation was defined or derived from.
  Location getLoc() { return location; }

  /// Set the source location the operation was defined or derived from.
  void setLoc(Location loc) { location = loc; }

  /// Returns the closest surrounding instruction that contains this instruction
  /// or nullptr if this is a top-level instruction.
  Instruction *getParentInst();

  /// Returns the function that this instruction is part of.
  /// The function is determined by traversing the chain of parent instructions.
  /// Returns nullptr if the instruction is unlinked.
  Function *getFunction();

  /// Destroys this instruction and its subclass data.
  void destroy();

  /// This drops all operand uses from this instruction, which is an essential
  /// step in breaking cyclic dependences between references when they are to
  /// be deleted.
  void dropAllReferences();

  /// Unlink this instruction from its current block and insert it right before
  /// `existingInst` which may be in the same or another block in the same
  /// function.
  void moveBefore(Instruction *existingInst);

  /// Unlink this operation instruction from its current block and insert it
  /// right before `iterator` in the specified block.
  void moveBefore(Block *block, llvm::iplist<Instruction>::iterator iterator);

  /// Given an instruction 'other' that is within the same parent block, return
  /// whether the current instruction is before 'other' in the instruction list
  /// of the parent block.
  /// Note: This function has an average complexity of O(1), but worst case may
  /// take O(N) where N is the number of instructions within the parent block.
  bool isBeforeInBlock(Instruction *other);

  void print(raw_ostream &os);
  void dump();

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  /// Returns if the operation has a resizable operation list, i.e. operands can
  /// be added.
  bool hasResizableOperandsList() { return getOperandStorage().isResizable(); }

  /// Replace the current operands of this operation with the ones provided in
  /// 'operands'. If the operands list is not resizable, the size of 'operands'
  /// must be less than or equal to the current number of operands.
  void setOperands(ArrayRef<Value *> operands) {
    getOperandStorage().setOperands(this, operands);
  }

  unsigned getNumOperands() { return getOperandStorage().size(); }

  Value *getOperand(unsigned idx) { return getInstOperand(idx).get(); }
  void setOperand(unsigned idx, Value *value) {
    return getInstOperand(idx).set(value);
  }

  // Support operand iteration.
  using operand_iterator = OperandIterator;
  using operand_range = llvm::iterator_range<operand_iterator>;

  operand_iterator operand_begin();
  operand_iterator operand_end();

  /// Returns an iterator on the underlying Value's (Value *).
  operand_range getOperands();

  MutableArrayRef<InstOperand> getInstOperands() {
    return getOperandStorage().getInstOperands();
  }

  InstOperand &getInstOperand(unsigned idx) { return getInstOperands()[idx]; }

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  /// Return true if there are no users of any results of this operation.
  bool use_empty();

  unsigned getNumResults() { return numResults; }

  Value *getResult(unsigned idx) { return &getInstResult(idx); }

  // Support result iteration.
  using result_iterator = ResultIterator;
  result_iterator result_begin();
  result_iterator result_end();
  llvm::iterator_range<result_iterator> getResults();

  MutableArrayRef<InstResult> getInstResults() {
    return {getTrailingObjects<InstResult>(), numResults};
  }

  InstResult &getInstResult(unsigned idx) { return getInstResults()[idx]; }

  // Support result type iteration.
  using result_type_iterator = ResultTypeIterator;
  result_type_iterator result_type_begin();
  result_type_iterator result_type_end();
  llvm::iterator_range<result_type_iterator> getResultTypes();

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  // Instructions may optionally carry a list of attributes that associate
  // constants to names.  Attributes may be dynamically added and removed over
  // the lifetime of an instruction.

  /// Return all of the attributes on this instruction.
  ArrayRef<NamedAttribute> getAttrs() { return attrs.getAttrs(); }

  /// Return the specified attribute if present, null otherwise.
  Attribute getAttr(Identifier name) { return attrs.get(name); }
  Attribute getAttr(StringRef name) { return attrs.get(name); }

  template <typename AttrClass> AttrClass getAttrOfType(Identifier name) {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  template <typename AttrClass> AttrClass getAttrOfType(StringRef name) {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  /// If the an attribute exists with the specified name, change it to the new
  /// value.  Otherwise, add a new attribute with the specified name/value.
  void setAttr(Identifier name, Attribute value) {
    attrs.set(getContext(), name, value);
  }
  void setAttr(StringRef name, Attribute value) {
    setAttr(Identifier::get(name, getContext()), value);
  }

  /// Remove the attribute with the specified name if it exists.  The return
  /// value indicates whether the attribute was present or not.
  NamedAttributeList::RemoveResult removeAttr(Identifier name) {
    return attrs.remove(getContext(), name);
  }

  //===--------------------------------------------------------------------===//
  // Blocks
  //===--------------------------------------------------------------------===//

  /// Returns the number of regions held by this operation.
  unsigned getNumRegions() { return numRegions; }

  /// Returns the regions held by this operation.
  MutableArrayRef<Region> getRegions() {
    auto *regions = getTrailingObjects<Region>();
    return {regions, numRegions};
  }

  /// Returns the region held by this operation at position 'index'.
  Region &getRegion(unsigned index) {
    assert(index < numRegions && "invalid region index");
    return getRegions()[index];
  }

  //===--------------------------------------------------------------------===//
  // Terminators
  //===--------------------------------------------------------------------===//

  MutableArrayRef<BlockOperand> getBlockOperands() {
    return {getTrailingObjects<BlockOperand>(), numSuccs};
  }

  /// Return the operands of this operation that are *not* successor arguments.
  operand_range getNonSuccessorOperands();

  operand_range getSuccessorOperands(unsigned index);

  Value *getSuccessorOperand(unsigned succIndex, unsigned opIndex) {
    assert(!isKnownNonTerminator() && "only terminators may have successors");
    assert(opIndex < getNumSuccessorOperands(succIndex));
    return getOperand(getSuccessorOperandIndex(succIndex) + opIndex);
  }

  unsigned getNumSuccessors() { return numSuccs; }
  unsigned getNumSuccessorOperands(unsigned index) {
    assert(!isKnownNonTerminator() && "only terminators may have successors");
    assert(index < getNumSuccessors());
    return getTrailingObjects<unsigned>()[index];
  }

  Block *getSuccessor(unsigned index) {
    assert(index < getNumSuccessors());
    return getBlockOperands()[index].get();
  }
  void setSuccessor(Block *block, unsigned index);

  /// Erase a specific operand from the operand list of the successor at
  /// 'index'.
  void eraseSuccessorOperand(unsigned succIndex, unsigned opIndex) {
    assert(succIndex < getNumSuccessors());
    assert(opIndex < getNumSuccessorOperands(succIndex));
    getOperandStorage().eraseOperand(getSuccessorOperandIndex(succIndex) +
                                     opIndex);
    --getTrailingObjects<unsigned>()[succIndex];
  }

  /// Get the index of the first operand of the successor at the provided
  /// index.
  unsigned getSuccessorOperandIndex(unsigned index);

  //===--------------------------------------------------------------------===//
  // Accessors for various properties of operations
  //===--------------------------------------------------------------------===//

  /// Returns whether the operation is commutative.
  bool isCommutative() {
    if (auto *absOp = getAbstractOperation())
      return absOp->hasProperty(OperationProperty::Commutative);
    return false;
  }

  /// Returns whether the operation has side-effects.
  bool hasNoSideEffect() {
    if (auto *absOp = getAbstractOperation())
      return absOp->hasProperty(OperationProperty::NoSideEffect);
    return false;
  }

  /// Represents the status of whether an operation is a terminator. We
  /// represent an 'unknown' status because we want to support unregistered
  /// terminators.
  enum class TerminatorStatus { Terminator, NonTerminator, Unknown };

  /// Returns the status of whether this operation is a terminator or not.
  TerminatorStatus getTerminatorStatus() {
    if (auto *absOp = getAbstractOperation()) {
      return absOp->hasProperty(OperationProperty::Terminator)
                 ? TerminatorStatus::Terminator
                 : TerminatorStatus::NonTerminator;
    }
    return TerminatorStatus::Unknown;
  }

  /// Returns if the operation is known to be a terminator.
  bool isKnownTerminator() {
    return getTerminatorStatus() == TerminatorStatus::Terminator;
  }

  /// Returns if the operation is known to *not* be a terminator.
  bool isKnownNonTerminator() {
    return getTerminatorStatus() == TerminatorStatus::NonTerminator;
  }

  /// Attempt to constant fold this operation with the specified constant
  /// operand values - the elements in "operands" will correspond directly to
  /// the operands of the operation, but may be null if non-constant.  If
  /// constant folding is successful, this fills in the `results` vector.  If
  /// not, `results` is unspecified.
  LogicalResult constantFold(ArrayRef<Attribute> operands,
                             SmallVectorImpl<Attribute> &results);

  /// Attempt to fold this operation using the Op's registered foldHook.
  LogicalResult fold(SmallVectorImpl<Value *> &results);

  //===--------------------------------------------------------------------===//
  // Conversions to declared operations like DimOp
  //===--------------------------------------------------------------------===//

  // Return a null OpPointer for the specified type.
  template <typename OpClass> static OpPointer<OpClass> getNull() {
    return OpPointer<OpClass>(OpClass(nullptr));
  }

  /// The dyn_cast methods perform a dynamic cast from an Instruction to a typed
  /// Op like DimOp.  This returns a null OpPointer on failure.
  template <typename OpClass> OpPointer<OpClass> dyn_cast() {
    if (isa<OpClass>()) {
      return cast<OpClass>();
    } else {
      return OpPointer<OpClass>(OpClass(nullptr));
    }
  }

  /// The cast methods perform a cast from an Instruction to a typed Op like
  /// DimOp.  This aborts if the parameter to the template isn't an instance of
  /// the template type argument.
  template <typename OpClass> OpPointer<OpClass> cast() {
    assert(isa<OpClass>() && "cast<Ty>() argument of incompatible type!");
    return OpPointer<OpClass>(OpClass(this));
  }

  /// The is methods return true if the operation is a typed op (like DimOp) of
  /// of the given class.
  template <typename OpClass> bool isa() { return OpClass::isClassFor(this); }

  //===--------------------------------------------------------------------===//
  // Instruction Walkers
  //===--------------------------------------------------------------------===//

  /// Walk the instructions held by this instruction in preorder, calling the
  /// callback for each instruction.
  void walk(const std::function<void(Instruction *)> &callback);

  /// Specialization of walk to only visit operations of 'OpTy'.
  template <typename OpTy>
  void walk(std::function<void(OpPointer<OpTy>)> callback) {
    walk([&](Instruction *inst) {
      if (auto op = inst->dyn_cast<OpTy>())
        callback(op);
    });
  }

  /// Walk the instructions held by this function in postorder, calling the
  /// callback for each instruction.
  void walkPostOrder(const std::function<void(Instruction *)> &callback);

  /// Specialization of walkPostOrder to only visit operations of 'OpTy'.
  template <typename OpTy>
  void walkPostOrder(std::function<void(OpPointer<OpTy>)> callback) {
    walkPostOrder([&](Instruction *inst) {
      if (auto op = inst->dyn_cast<OpTy>())
        callback(op);
    });
  }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Emit an error with the op name prefixed, like "'dim' op " which is
  /// convenient for verifiers.  This function always returns true.
  bool emitOpError(const Twine &message);

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.  This function always
  /// returns true.  NOTE: This may terminate the containing application, only
  /// use when the IR is in an inconsistent state.
  bool emitError(const Twine &message);

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitWarning(const Twine &message);

  /// Emit a note about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitNote(const Twine &message);

private:
  Instruction(Location location, OperationName name, unsigned numResults,
              unsigned numSuccessors, unsigned numRegions,
              const NamedAttributeList &attributes, MLIRContext *context);

  // Instructions are deleted through the destroy() member because they are
  // allocated with malloc.
  ~Instruction();

  /// Returns the operand storage object.
  detail::OperandStorage &getOperandStorage() {
    return *getTrailingObjects<detail::OperandStorage>();
  }

  // Provide a 'getParent' method for ilist_node_with_parent methods.
  Block *getParent() { return getBlock(); }

  /// The instruction block that containts this instruction.
  Block *block = nullptr;

  /// This holds information about the source location the operation was defined
  /// or derived from.
  Location location;

  /// Relative order of this instruction in its parent block. Used for
  /// O(1) local dominance checks between instructions.
  mutable unsigned orderIndex = 0;

  const unsigned numResults, numSuccs, numRegions;

  /// This holds the name of the operation.
  OperationName name;

  /// This holds general named attributes for the operation.
  NamedAttributeList attrs;

  // allow ilist_traits access to 'block' field.
  friend struct llvm::ilist_traits<Instruction>;

  // allow block to access the 'orderIndex' field.
  friend class Block;

  // allow ilist_node_with_parent to access the 'getParent' method.
  friend class llvm::ilist_node_with_parent<Instruction, Block>;

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<Instruction, InstResult, BlockOperand, unsigned,
                               Region, detail::OperandStorage>;
  size_t numTrailingObjects(OverloadToken<InstResult>) const {
    return numResults;
  }
  size_t numTrailingObjects(OverloadToken<BlockOperand>) const {
    return numSuccs;
  }
  size_t numTrailingObjects(OverloadToken<Region>) const { return numRegions; }
  size_t numTrailingObjects(OverloadToken<unsigned>) const { return numSuccs; }
};

inline raw_ostream &operator<<(raw_ostream &os, Instruction &inst) {
  inst.print(os);
  return os;
}

/// This class implements the const/non-const operand iterators for the
/// Instruction class in terms of getOperand(idx).
class OperandIterator final
    : public IndexedAccessorIterator<OperandIterator, Instruction, Value> {
public:
  /// Initializes the operand iterator to the specified operand index.
  OperandIterator(Instruction *object, unsigned index)
      : IndexedAccessorIterator<OperandIterator, Instruction, Value>(object,
                                                                     index) {}

  Value *operator*() const { return this->object->getOperand(this->index); }
};

// Implement the inline operand iterator methods.
inline auto Instruction::operand_begin() -> operand_iterator {
  return operand_iterator(this, 0);
}

inline auto Instruction::operand_end() -> operand_iterator {
  return operand_iterator(this, getNumOperands());
}

inline auto Instruction::getOperands() -> operand_range {
  return {operand_begin(), operand_end()};
}

/// This class implements the result iterators for the Instruction class
/// in terms of getResult(idx).
class ResultIterator final
    : public IndexedAccessorIterator<ResultIterator, Instruction, Value> {
public:
  /// Initializes the result iterator to the specified index.
  ResultIterator(Instruction *object, unsigned index)
      : IndexedAccessorIterator<ResultIterator, Instruction, Value>(object,
                                                                    index) {}

  Value *operator*() const { return this->object->getResult(this->index); }
};

/// This class implements the result type iterators for the Instruction
/// class in terms of getResult(idx)->getType().
class ResultTypeIterator final
    : public IndexedAccessorIterator<ResultTypeIterator, Instruction, Value> {
public:
  /// Initializes the result type iterator to the specified index.
  ResultTypeIterator(Instruction *object, unsigned index)
      : IndexedAccessorIterator<ResultTypeIterator, Instruction, Value>(object,
                                                                        index) {
  }

  Type operator*() const {
    return this->object->getResult(this->index)->getType();
  }
};

// Implement the inline result iterator methods.
inline auto Instruction::result_begin() -> result_iterator {
  return result_iterator(this, 0);
}

inline auto Instruction::result_end() -> result_iterator {
  return result_iterator(this, getNumResults());
}

inline auto Instruction::getResults() -> llvm::iterator_range<result_iterator> {
  return {result_begin(), result_end()};
}

inline auto Instruction::result_type_begin() -> result_type_iterator {
  return result_type_iterator(this, 0);
}

inline auto Instruction::result_type_end() -> result_type_iterator {
  return result_type_iterator(this, getNumResults());
}

inline auto Instruction::getResultTypes()
    -> llvm::iterator_range<result_type_iterator> {
  return {result_type_begin(), result_type_end()};
}

} // end namespace mlir

#endif // MLIR_IR_INSTRUCTION_H
