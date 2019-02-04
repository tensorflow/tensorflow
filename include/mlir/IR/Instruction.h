//===- Instruction.h - MLIR ML Instruction Class --------------------*- C++
//-*-===//
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
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
class AttributeListStorage;
class BlockAndValueMapping;
template <typename OpType> class ConstOpPointer;
class Location;
class MLIRContext;
template <typename OpType> class OpPointer;
template <typename ObjectType, typename ElementType> class ResultIterator;
template <typename ObjectType, typename ElementType> class ResultTypeIterator;

/// Terminator operations can have Block operands to represent successors.
using BlockOperand = IROperandImpl<Block>;

} // namespace mlir

namespace mlir {
template <typename ObjectType, typename ElementType> class OperandIterator;

namespace detail {
/// A utility class holding the information necessary to dynamically resize
/// operands.
struct ResizableStorage {
  ResizableStorage(InstOperand *opBegin, unsigned numOperands)
      : firstOpAndIsDynamic(opBegin, false), capacity(numOperands) {}

  ~ResizableStorage() { cleanupStorage(); }

  /// Cleanup any allocated storage.
  void cleanupStorage() {
    // If the storage is dynamic, then we need to free the storage.
    if (isStorageDynamic())
      free(firstOpAndIsDynamic.getPointer());
  }

  /// Sets the storage pointer to a new dynamically allocated block.
  void setDynamicStorage(InstOperand *opBegin) {
    /// Cleanup the old storage if necessary.
    cleanupStorage();
    firstOpAndIsDynamic.setPointerAndInt(opBegin, true);
  }

  /// Returns the current storage pointer.
  InstOperand *getPointer() { return firstOpAndIsDynamic.getPointer(); }
  const InstOperand *getPointer() const {
    return firstOpAndIsDynamic.getPointer();
  }

  /// Returns if the current storage of operands is in the trailing objects is
  /// in a dynamically allocated memory block.
  bool isStorageDynamic() const { return firstOpAndIsDynamic.getInt(); }

  /// A pointer to the first operand element. This is either to the trailing
  /// objects storage, or a dynamically allocated block of memory.
  llvm::PointerIntPair<InstOperand *, 1, bool> firstOpAndIsDynamic;

  // The maximum number of operands that can be currently held by the storage.
  unsigned capacity;
};

/// This class handles the management of instruction operands. Operands are
/// stored similarly to the elements of a SmallVector except for two key
/// differences. The first is the inline storage, which is a trailing objects
/// array. The second is that being able to dynamically resize the operand list
/// is optional.
class OperandStorage final
    : private llvm::TrailingObjects<OperandStorage, ResizableStorage,
                                    InstOperand> {
public:
  OperandStorage(unsigned numOperands, bool resizable)
      : numOperands(numOperands), resizable(resizable) {
    // Initialize the resizable storage.
    if (resizable)
      new (&getResizableStorage())
          ResizableStorage(getTrailingObjects<InstOperand>(), numOperands);
  }

  ~OperandStorage() {
    // Manually destruct the operands.
    for (auto &operand : getInstOperands())
      operand.~InstOperand();

    // If the storage is resizable then destruct the utility.
    if (resizable)
      getResizableStorage().~ResizableStorage();
  }

  /// Replace the operands contained in the storage with the ones provided in
  /// 'operands'.
  void setOperands(Instruction *owner, ArrayRef<Value *> operands);

  /// Erase an operand held by the storage.
  void eraseOperand(unsigned index);

  /// Get the instruction operands held by the storage.
  ArrayRef<InstOperand> getInstOperands() const {
    return {getRawOperands(), size()};
  }
  MutableArrayRef<InstOperand> getInstOperands() {
    return {getRawOperands(), size()};
  }

  /// Return the number of operands held in the storage.
  unsigned size() const { return numOperands; }

  /// Returns the additional size necessary for allocating this object.
  static size_t additionalAllocSize(unsigned numOperands, bool resizable) {
    return additionalSizeToAlloc<ResizableStorage, InstOperand>(
        resizable ? 1 : 0, numOperands);
  }

  /// Returns if this storage is resizable.
  bool isResizable() const { return resizable; }

private:
  /// Clear the storage and destroy the current operands held by the storage.
  void clear() { numOperands = 0; }

  /// Returns the current pointer for the raw operands array.
  InstOperand *getRawOperands() {
    return resizable ? getResizableStorage().getPointer()
                     : getTrailingObjects<InstOperand>();
  }
  const InstOperand *getRawOperands() const {
    return resizable ? getResizableStorage().getPointer()
                     : getTrailingObjects<InstOperand>();
  }

  /// Returns the resizable operand utility class.
  ResizableStorage &getResizableStorage() {
    assert(resizable);
    return *getTrailingObjects<ResizableStorage>();
  }
  const ResizableStorage &getResizableStorage() const {
    assert(resizable);
    return *getTrailingObjects<ResizableStorage>();
  }

  /// Grow the internal resizable operand storage.
  void grow(ResizableStorage &resizeUtil, size_t minSize);

  /// The current number of operands, and the current max operand capacity.
  unsigned numOperands : 31;

  /// Whether this storage is resizable or not.
  bool resizable : 1;

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<OperandStorage, ResizableStorage, InstOperand>;
  size_t numTrailingObjects(OverloadToken<ResizableStorage>) const {
    return resizable ? 1 : 0;
  }
};
} // end namespace detail

/// Instruction is a basic unit of execution within a function. Instructions can
/// be nested within other instructions effectively forming a tree. Child
/// instructions are organized into instruction blocks represented by a 'Block'
/// class.
class Instruction final
    : public llvm::ilist_node_with_parent<Instruction, Block>,
      private llvm::TrailingObjects<Instruction, InstResult, BlockOperand,
                                    unsigned, BlockList,
                                    detail::OperandStorage> {
public:
  /// Create a new Instruction with the specific fields.
  static Instruction *
  create(Location location, OperationName name, ArrayRef<Value *> operands,
         ArrayRef<Type> resultTypes, ArrayRef<NamedAttribute> attributes,
         ArrayRef<Block *> successors, unsigned numBlockLists,
         bool resizableOperandList, MLIRContext *context);

  /// The name of an operation is the key identifier for it.
  OperationName getName() const { return name; }

  /// If this operation has a registered operation description, return it.
  /// Otherwise return null.
  const AbstractOperation *getAbstractOperation() const {
    return getName().getAbstractOperation();
  }

  /// Remove this instruction from its parent block and delete it.
  void erase();

  /// Create a deep copy of this instruction, remapping any operands that use
  /// values outside of the instruction using the map that is provided (leaving
  /// them alone if no entry is present).  Replaces references to cloned
  /// sub-instructions to the corresponding instruction that is copied, and adds
  /// those mappings to the map.
  Instruction *clone(BlockAndValueMapping &mapper, MLIRContext *context) const;
  Instruction *clone(MLIRContext *context) const;

  /// Returns the instruction block that contains this instruction.
  const Block *getBlock() const { return block; }
  Block *getBlock() { return block; }

  /// Return the context this operation is associated with.
  MLIRContext *getContext() const;

  /// The source location the operation was defined or derived from.
  Location getLoc() const { return location; }

  /// Set the source location the operation was defined or derived from.
  void setLoc(Location loc) { location = loc; }

  /// Returns the closest surrounding instruction that contains this instruction
  /// or nullptr if this is a top-level instruction.
  Instruction *getParentInst() const;

  /// Returns the function that this instruction is part of.
  /// The function is determined by traversing the chain of parent instructions.
  /// Returns nullptr if the instruction is unlinked.
  Function *getFunction() const;

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
  bool isBeforeInBlock(const Instruction *other) const;

  /// Check if this instruction is a return instruction.
  bool isReturn() const;

  void print(raw_ostream &os) const;
  void dump() const;

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  /// Returns if the operation has a resizable operation list, i.e. operands can
  /// be added.
  bool hasResizableOperandsList() const {
    return getOperandStorage().isResizable();
  }

  /// Replace the current operands of this operation with the ones provided in
  /// 'operands'. If the operands list is not resizable, the size of 'operands'
  /// must be less than or equal to the current number of operands.
  void setOperands(ArrayRef<Value *> operands) {
    getOperandStorage().setOperands(this, operands);
  }

  unsigned getNumOperands() const { return getOperandStorage().size(); }

  Value *getOperand(unsigned idx) { return getInstOperand(idx).get(); }
  const Value *getOperand(unsigned idx) const {
    return getInstOperand(idx).get();
  }
  void setOperand(unsigned idx, Value *value) {
    return getInstOperand(idx).set(value);
  }

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<Instruction, Value>;

  operand_iterator operand_begin();
  operand_iterator operand_end();

  /// Returns an iterator on the underlying Value's (Value *).
  llvm::iterator_range<operand_iterator> getOperands();

  // Support const operand iteration.
  using const_operand_iterator =
      OperandIterator<const Instruction, const Value>;

  const_operand_iterator operand_begin() const;
  const_operand_iterator operand_end() const;

  /// Returns a const iterator on the underlying Value's (Value *).
  llvm::iterator_range<const_operand_iterator> getOperands() const;

  ArrayRef<InstOperand> getInstOperands() const {
    return getOperandStorage().getInstOperands();
  }
  MutableArrayRef<InstOperand> getInstOperands() {
    return getOperandStorage().getInstOperands();
  }

  InstOperand &getInstOperand(unsigned idx) { return getInstOperands()[idx]; }
  const InstOperand &getInstOperand(unsigned idx) const {
    return getInstOperands()[idx];
  }

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  /// Return true if there are no users of any results of this operation.
  bool use_empty() const;

  unsigned getNumResults() const { return numResults; }

  Value *getResult(unsigned idx) { return &getInstResult(idx); }
  const Value *getResult(unsigned idx) const { return &getInstResult(idx); }

  // Support non-const result iteration.
  using result_iterator = ResultIterator<Instruction, Value>;
  result_iterator result_begin();
  result_iterator result_end();
  llvm::iterator_range<result_iterator> getResults();

  // Support const result iteration.
  using const_result_iterator = ResultIterator<const Instruction, const Value>;
  const_result_iterator result_begin() const;

  const_result_iterator result_end() const;

  llvm::iterator_range<const_result_iterator> getResults() const;

  ArrayRef<InstResult> getInstResults() const {
    return {getTrailingObjects<InstResult>(), numResults};
  }

  MutableArrayRef<InstResult> getInstResults() {
    return {getTrailingObjects<InstResult>(), numResults};
  }

  InstResult &getInstResult(unsigned idx) { return getInstResults()[idx]; }

  const InstResult &getInstResult(unsigned idx) const {
    return getInstResults()[idx];
  }

  // Support result type iteration.
  using result_type_iterator =
      ResultTypeIterator<const Instruction, const Value>;
  result_type_iterator result_type_begin() const;

  result_type_iterator result_type_end() const;

  llvm::iterator_range<result_type_iterator> getResultTypes() const;

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  // Operations may optionally carry a list of attributes that associate
  // constants to names.  Attributes may be dynamically added and removed over
  // the lifetime of an operation.
  //
  // We assume there will be relatively few attributes on a given operation
  // (maybe a dozen or so, but not hundreds or thousands) so we use linear
  // searches for everything.

  /// Return all of the attributes on this operation.
  ArrayRef<NamedAttribute> getAttrs() const;

  /// Return the specified attribute if present, null otherwise.
  Attribute getAttr(Identifier name) const {
    for (auto elt : getAttrs())
      if (elt.first == name)
        return elt.second;
    return nullptr;
  }

  Attribute getAttr(StringRef name) const {
    for (auto elt : getAttrs())
      if (elt.first.is(name))
        return elt.second;
    return nullptr;
  }

  template <typename AttrClass> AttrClass getAttrOfType(Identifier name) const {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  template <typename AttrClass> AttrClass getAttrOfType(StringRef name) const {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  /// If the an attribute exists with the specified name, change it to the new
  /// value.  Otherwise, add a new attribute with the specified name/value.
  void setAttr(Identifier name, Attribute value);

  enum class RemoveResult { Removed, NotFound };

  /// Remove the attribute with the specified name if it exists.  The return
  /// value indicates whether the attribute was present or not.
  RemoveResult removeAttr(Identifier name);

  //===--------------------------------------------------------------------===//
  // Blocks
  //===--------------------------------------------------------------------===//

  /// Returns the number of block lists held by this operation.
  unsigned getNumBlockLists() const { return numBlockLists; }

  /// Returns the block lists held by this operation.
  MutableArrayRef<BlockList> getBlockLists() {
    return {getTrailingObjects<BlockList>(), numBlockLists};
  }
  ArrayRef<BlockList> getBlockLists() const {
    return const_cast<Instruction *>(this)->getBlockLists();
  }

  /// Returns the block list held by this operation at position 'index'.
  BlockList &getBlockList(unsigned index) {
    assert(index < numBlockLists && "invalid block list index");
    return getBlockLists()[index];
  }
  const BlockList &getBlockList(unsigned index) const {
    return const_cast<Instruction *>(this)->getBlockList(index);
  }

  //===--------------------------------------------------------------------===//
  // Terminators
  //===--------------------------------------------------------------------===//

  MutableArrayRef<BlockOperand> getBlockOperands() {
    assert(isTerminator() && "Only terminators have a block operands list");
    return {getTrailingObjects<BlockOperand>(), numSuccs};
  }
  ArrayRef<BlockOperand> getBlockOperands() const {
    return const_cast<Instruction *>(this)->getBlockOperands();
  }

  /// Return the operands of this operation that are *not* successor arguments.
  llvm::iterator_range<const_operand_iterator> getNonSuccessorOperands() const;
  llvm::iterator_range<operand_iterator> getNonSuccessorOperands();

  llvm::iterator_range<const_operand_iterator>
  getSuccessorOperands(unsigned index) const;
  llvm::iterator_range<operand_iterator> getSuccessorOperands(unsigned index);

  Value *getSuccessorOperand(unsigned succIndex, unsigned opIndex) {
    assert(opIndex < getNumSuccessorOperands(succIndex));
    return getOperand(getSuccessorOperandIndex(succIndex) + opIndex);
  }
  const Value *getSuccessorOperand(unsigned succIndex, unsigned index) const {
    return const_cast<Instruction *>(this)->getSuccessorOperand(succIndex,
                                                                index);
  }

  unsigned getNumSuccessors() const { return numSuccs; }
  unsigned getNumSuccessorOperands(unsigned index) const {
    assert(isTerminator() && "Only terminators have successors");
    assert(index < getNumSuccessors());
    return getTrailingObjects<unsigned>()[index];
  }

  Block *getSuccessor(unsigned index) {
    assert(index < getNumSuccessors());
    return getBlockOperands()[index].get();
  }
  const Block *getSuccessor(unsigned index) const {
    return const_cast<Instruction *>(this)->getSuccessor(index);
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
  unsigned getSuccessorOperandIndex(unsigned index) const {
    assert(isTerminator() && "Only terminators have successors.");
    assert(index < getNumSuccessors());

    // Count the number of operands for each of the successors after, and
    // including, the one at 'index'. This is based upon the assumption that all
    // non successor operands are placed at the beginning of the operand list.
    auto *successorOpCountBegin = getTrailingObjects<unsigned>();
    unsigned postSuccessorOpCount =
        std::accumulate(successorOpCountBegin + index,
                        successorOpCountBegin + getNumSuccessors(), 0);
    return getNumOperands() - postSuccessorOpCount;
  }

  //===--------------------------------------------------------------------===//
  // Accessors for various properties of operations
  //===--------------------------------------------------------------------===//

  /// Returns whether the operation is commutative.
  bool isCommutative() const {
    if (auto *absOp = getAbstractOperation())
      return absOp->hasProperty(OperationProperty::Commutative);
    return false;
  }

  /// Returns whether the operation has side-effects.
  bool hasNoSideEffect() const {
    if (auto *absOp = getAbstractOperation())
      return absOp->hasProperty(OperationProperty::NoSideEffect);
    return false;
  }

  /// Returns whether the operation is a terminator.
  bool isTerminator() const {
    if (auto *absOp = getAbstractOperation())
      return absOp->hasProperty(OperationProperty::Terminator);
    return false;
  }

  /// Attempt to constant fold this operation with the specified constant
  /// operand values - the elements in "operands" will correspond directly to
  /// the operands of the operation, but may be null if non-constant.  If
  /// constant folding is successful, this returns false and fills in the
  /// `results` vector.  If not, this returns true and `results` is unspecified.
  bool constantFold(ArrayRef<Attribute> operands,
                    SmallVectorImpl<Attribute> &results) const;

  /// Attempt to fold this operation using the Op's registered foldHook.
  bool fold(SmallVectorImpl<Value *> &results);

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

  /// The dyn_cast methods perform a dynamic cast from an Instruction to a typed
  /// Op like DimOp.  This returns a null ConstOpPointer on failure.
  template <typename OpClass> ConstOpPointer<OpClass> dyn_cast() const {
    if (isa<OpClass>()) {
      return cast<OpClass>();
    } else {
      return ConstOpPointer<OpClass>(OpClass(nullptr));
    }
  }

  /// The cast methods perform a cast from an Instruction to a typed Op like
  /// DimOp.  This aborts if the parameter to the template isn't an instance of
  /// the template type argument.
  template <typename OpClass> OpPointer<OpClass> cast() {
    assert(isa<OpClass>() && "cast<Ty>() argument of incompatible type!");
    return OpPointer<OpClass>(OpClass(this));
  }

  /// The cast methods perform a cast from an Instruction to a typed Op like
  /// DimOp.  This aborts if the parameter to the template isn't an instance of
  /// the template type argument.
  template <typename OpClass> ConstOpPointer<OpClass> cast() const {
    assert(isa<OpClass>() && "cast<Ty>() argument of incompatible type!");
    return ConstOpPointer<OpClass>(OpClass(this));
  }

  /// The is methods return true if the operation is a typed op (like DimOp) of
  /// of the given class.
  template <typename OpClass> bool isa() const {
    return OpClass::isClassFor(this);
  }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Emit an error with the op name prefixed, like "'dim' op " which is
  /// convenient for verifiers.  This function always returns true.
  bool emitOpError(const Twine &message) const;

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.  This function always
  /// returns true.  NOTE: This may terminate the containing application, only
  /// use when the IR is in an inconsistent state.
  bool emitError(const Twine &message) const;

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitWarning(const Twine &message) const;

  /// Emit a note about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitNote(const Twine &message) const;

protected:
  Instruction(Location location, OperationName name, unsigned numResults,
              unsigned numSuccessors, unsigned numBlockLists,
              ArrayRef<NamedAttribute> attributes, MLIRContext *context);

  // Instructions are deleted through the destroy() member because they are
  // allocated with malloc.
  ~Instruction();

private:
  /// Returns the operand storage object.
  detail::OperandStorage &getOperandStorage() {
    return *getTrailingObjects<detail::OperandStorage>();
  }
  const detail::OperandStorage &getOperandStorage() const {
    return *getTrailingObjects<detail::OperandStorage>();
  }

  // Provide a 'getParent' method for ilist_node_with_parent methods.
  const Block *getParent() const { return getBlock(); }

  /// The instruction block that containts this instruction.
  Block *block = nullptr;

  /// This holds information about the source location the operation was defined
  /// or derived from.
  Location location;

  /// Relative order of this instruction in its parent block. Used for
  /// O(1) local dominance checks between instructions.
  mutable unsigned orderIndex = 0;

  const unsigned numResults, numSuccs, numBlockLists;

  /// This holds the name of the operation.
  OperationName name;

  /// This holds general named attributes for the operation.
  AttributeListStorage *attrs;

  // allow ilist_traits access to 'block' field.
  friend struct llvm::ilist_traits<Instruction>;

  // allow block to access the 'orderIndex' field.
  friend class Block;

  // allow ilist_node_with_parent to access the 'getParent' method.
  friend class llvm::ilist_node_with_parent<Instruction, Block>;

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<Instruction, InstResult, BlockOperand, unsigned,
                               BlockList, detail::OperandStorage>;
  size_t numTrailingObjects(OverloadToken<InstResult>) const {
    return numResults;
  }
  size_t numTrailingObjects(OverloadToken<BlockOperand>) const {
    return numSuccs;
  }
  size_t numTrailingObjects(OverloadToken<BlockList>) const {
    return numBlockLists;
  }
  size_t numTrailingObjects(OverloadToken<unsigned>) const { return numSuccs; }
};

inline raw_ostream &operator<<(raw_ostream &os, const Instruction &inst) {
  inst.print(os);
  return os;
}

/// This template implements the const/non-const operand iterators for the
/// Instruction class in terms of getOperand(idx).
template <typename ObjectType, typename ElementType>
class OperandIterator final
    : public IndexedAccessorIterator<OperandIterator<ObjectType, ElementType>,
                                     ObjectType, ElementType> {
public:
  /// Initializes the operand iterator to the specified operand index.
  OperandIterator(ObjectType *object, unsigned index)
      : IndexedAccessorIterator<OperandIterator<ObjectType, ElementType>,
                                ObjectType, ElementType>(object, index) {}

  /// Support converting to the const variant. This will be a no-op for const
  /// variant.
  operator OperandIterator<const ObjectType, const ElementType>() const {
    return OperandIterator<const ObjectType, const ElementType>(this->object,
                                                                this->index);
  }

  ElementType *operator*() const {
    return this->object->getOperand(this->index);
  }
};

// Implement the inline operand iterator methods.
inline auto Instruction::operand_begin() -> operand_iterator {
  return operand_iterator(this, 0);
}

inline auto Instruction::operand_end() -> operand_iterator {
  return operand_iterator(this, getNumOperands());
}

inline auto Instruction::getOperands()
    -> llvm::iterator_range<operand_iterator> {
  return {operand_begin(), operand_end()};
}

inline auto Instruction::operand_begin() const -> const_operand_iterator {
  return const_operand_iterator(this, 0);
}

inline auto Instruction::operand_end() const -> const_operand_iterator {
  return const_operand_iterator(this, getNumOperands());
}

inline auto Instruction::getOperands() const
    -> llvm::iterator_range<const_operand_iterator> {
  return {operand_begin(), operand_end()};
}

/// This template implements the result iterators for the Instruction class
/// in terms of getResult(idx).
template <typename ObjectType, typename ElementType>
class ResultIterator final
    : public IndexedAccessorIterator<ResultIterator<ObjectType, ElementType>,
                                     ObjectType, ElementType> {
public:
  /// Initializes the result iterator to the specified index.
  ResultIterator(ObjectType *object, unsigned index)
      : IndexedAccessorIterator<ResultIterator<ObjectType, ElementType>,
                                ObjectType, ElementType>(object, index) {}

  /// Support converting to the const variant. This will be a no-op for const
  /// variant.
  operator ResultIterator<const ObjectType, const ElementType>() const {
    return ResultIterator<const ObjectType, const ElementType>(this->object,
                                                               this->index);
  }

  ElementType *operator*() const {
    return this->object->getResult(this->index);
  }
};

/// This template implements the result type iterators for the Instruction
/// class in terms of getResult(idx)->getType().
template <typename ObjectType, typename ElementType>
class ResultTypeIterator final
    : public IndexedAccessorIterator<
          ResultTypeIterator<ObjectType, ElementType>, ObjectType,
          ElementType> {
public:
  /// Initializes the result type iterator to the specified index.
  ResultTypeIterator(ObjectType *object, unsigned index)
      : IndexedAccessorIterator<ResultTypeIterator<ObjectType, ElementType>,
                                ObjectType, ElementType>(object, index) {}

  /// Support converting to the const variant. This will be a no-op for const
  /// variant.
  operator ResultTypeIterator<const ObjectType, const ElementType>() const {
    return ResultTypeIterator<const ObjectType, const ElementType>(this->object,
                                                                   this->index);
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

inline auto Instruction::result_begin() const -> const_result_iterator {
  return const_result_iterator(this, 0);
}

inline auto Instruction::result_end() const -> const_result_iterator {
  return const_result_iterator(this, getNumResults());
}

inline auto Instruction::getResults() const
    -> llvm::iterator_range<const_result_iterator> {
  return {result_begin(), result_end()};
}

inline auto Instruction::result_type_begin() const -> result_type_iterator {
  return result_type_iterator(this, 0);
}

inline auto Instruction::result_type_end() const -> result_type_iterator {
  return result_type_iterator(this, getNumResults());
}

inline auto Instruction::getResultTypes() const
    -> llvm::iterator_range<result_type_iterator> {
  return {result_type_begin(), result_type_end()};
}

} // end namespace mlir

#endif // MLIR_IR_INSTRUCTION_H
