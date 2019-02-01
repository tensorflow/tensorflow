//===- Instructions.h - MLIR ML Instruction Classes -----------------*- C++
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
// This file defines classes for special kinds of ML Function instructions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INSTRUCTIONS_H
#define MLIR_IR_INSTRUCTIONS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Instruction.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
class AffineBound;
class IntegerSet;
class AffineCondition;
class AttributeListStorage;
template <typename OpType> class ConstOpPointer;
template <typename OpType> class OpPointer;
template <typename ObjectType, typename ElementType> class ResultIterator;
template <typename ObjectType, typename ElementType> class ResultTypeIterator;
class Function;

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

/// Operations represent all of the arithmetic and other basic computation in
/// MLIR.
///
class OperationInst final
    : public Instruction,
      private llvm::TrailingObjects<OperationInst, InstResult, BlockOperand,
                                    unsigned, BlockList,
                                    detail::OperandStorage> {
public:
  /// Create a new OperationInst with the specific fields.
  static OperationInst *
  create(Location location, OperationName name, ArrayRef<Value *> operands,
         ArrayRef<Type> resultTypes, ArrayRef<NamedAttribute> attributes,
         ArrayRef<Block *> successors, unsigned numBlockLists,
         bool resizableOperandList, MLIRContext *context);

  /// Return the context this operation is associated with.
  MLIRContext *getContext() const;

  /// The name of an operation is the key identifier for it.
  OperationName getName() const { return name; }

  /// If this operation has a registered operation description, return it.
  /// Otherwise return null.
  const AbstractOperation *getAbstractOperation() const {
    return getName().getAbstractOperation();
  }

  /// Check if this instruction is a return instruction.
  bool isReturn() const;

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  /// Returns if the operation has a resizable operation list, i.e. operands can
  /// be added.
  bool hasResizableOperandsList() const {
    return getOperandStorage().isResizable();
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
  using operand_iterator = OperandIterator<OperationInst, Value>;

  operand_iterator operand_begin() { return operand_iterator(this, 0); }

  operand_iterator operand_end() {
    return operand_iterator(this, getNumOperands());
  }

  /// Returns an iterator on the underlying Value's (Value *).
  llvm::iterator_range<operand_iterator> getOperands() {
    return {operand_begin(), operand_end()};
  }

  // Support const operand iteration.
  using const_operand_iterator =
      OperandIterator<const OperationInst, const Value>;

  const_operand_iterator operand_begin() const {
    return const_operand_iterator(this, 0);
  }

  const_operand_iterator operand_end() const {
    return const_operand_iterator(this, getNumOperands());
  }

  /// Returns a const iterator on the underlying Value's (Value *).
  llvm::iterator_range<const_operand_iterator> getOperands() const {
    return {operand_begin(), operand_end()};
  }

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
  using result_iterator = ResultIterator<OperationInst, Value>;
  result_iterator result_begin();
  result_iterator result_end();
  llvm::iterator_range<result_iterator> getResults();

  // Support const result iteration.
  using const_result_iterator =
      ResultIterator<const OperationInst, const Value>;
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
      ResultTypeIterator<const OperationInst, const Value>;
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
    return const_cast<OperationInst *>(this)->getBlockLists();
  }

  /// Returns the block list held by this operation at position 'index'.
  BlockList &getBlockList(unsigned index) {
    assert(index < numBlockLists && "invalid block list index");
    return getBlockLists()[index];
  }
  const BlockList &getBlockList(unsigned index) const {
    return const_cast<OperationInst *>(this)->getBlockList(index);
  }

  //===--------------------------------------------------------------------===//
  // Terminators
  //===--------------------------------------------------------------------===//

  MutableArrayRef<BlockOperand> getBlockOperands() {
    assert(isTerminator() && "Only terminators have a block operands list");
    return {getTrailingObjects<BlockOperand>(), numSuccs};
  }
  ArrayRef<BlockOperand> getBlockOperands() const {
    return const_cast<OperationInst *>(this)->getBlockOperands();
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
    return const_cast<OperationInst *>(this)->getSuccessorOperand(succIndex,
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
    return const_cast<OperationInst *>(this)->getSuccessor(index);
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

  /// The dyn_cast methods perform a dynamic cast from an OperationInst (like
  /// Instruction and OperationInst) to a typed Op like DimOp.  This returns
  /// a null OpPointer on failure.
  template <typename OpClass> OpPointer<OpClass> dyn_cast() {
    if (isa<OpClass>()) {
      return cast<OpClass>();
    } else {
      return OpPointer<OpClass>(OpClass(nullptr));
    }
  }

  /// The dyn_cast methods perform a dynamic cast from an OperationInst (like
  /// Instruction and OperationInst) to a typed Op like DimOp.  This returns
  /// a null ConstOpPointer on failure.
  template <typename OpClass> ConstOpPointer<OpClass> dyn_cast() const {
    if (isa<OpClass>()) {
      return cast<OpClass>();
    } else {
      return ConstOpPointer<OpClass>(OpClass(nullptr));
    }
  }

  /// The cast methods perform a cast from an OperationInst (like
  /// Instruction and OperationInst) to a typed Op like DimOp.  This aborts
  /// if the parameter to the template isn't an instance of the template type
  /// argument.
  template <typename OpClass> OpPointer<OpClass> cast() {
    assert(isa<OpClass>() && "cast<Ty>() argument of incompatible type!");
    return OpPointer<OpClass>(OpClass(this));
  }

  /// The cast methods perform a cast from an OperationInst (like
  /// Instruction and OperationInst) to a typed Op like DimOp.  This aborts
  /// if the parameter to the template isn't an instance of the template type
  /// argument.
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

  void destroy();

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const IROperandOwner *ptr) {
    return ptr->getKind() == IROperandOwner::Kind::OperationInst;
  }

private:
  const unsigned numResults, numSuccs, numBlockLists;

  /// This holds the name of the operation.
  OperationName name;

  /// This holds general named attributes for the operation.
  AttributeListStorage *attrs;

  OperationInst(Location location, OperationName name, unsigned numResults,
                unsigned numSuccessors, unsigned numBlockLists,
                ArrayRef<NamedAttribute> attributes, MLIRContext *context);
  ~OperationInst();

  /// Returns the operand storage object.
  detail::OperandStorage &getOperandStorage() {
    return *getTrailingObjects<detail::OperandStorage>();
  }
  const detail::OperandStorage &getOperandStorage() const {
    return *getTrailingObjects<detail::OperandStorage>();
  }

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<OperationInst, InstResult, BlockOperand,
                               unsigned, BlockList, detail::OperandStorage>;
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

/// This template implements the result iterators for the OperationInst class
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

/// This template implements the result type iterators for the OperationInst
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
inline auto OperationInst::result_begin() -> result_iterator {
  return result_iterator(this, 0);
}

inline auto OperationInst::result_end() -> result_iterator {
  return result_iterator(this, getNumResults());
}

inline auto OperationInst::getResults()
    -> llvm::iterator_range<result_iterator> {
  return {result_begin(), result_end()};
}

inline auto OperationInst::result_begin() const -> const_result_iterator {
  return const_result_iterator(this, 0);
}

inline auto OperationInst::result_end() const -> const_result_iterator {
  return const_result_iterator(this, getNumResults());
}

inline auto OperationInst::getResults() const
    -> llvm::iterator_range<const_result_iterator> {
  return {result_begin(), result_end()};
}

inline auto OperationInst::result_type_begin() const -> result_type_iterator {
  return result_type_iterator(this, 0);
}

inline auto OperationInst::result_type_end() const -> result_type_iterator {
  return result_type_iterator(this, getNumResults());
}

inline auto OperationInst::getResultTypes() const
    -> llvm::iterator_range<result_type_iterator> {
  return {result_type_begin(), result_type_end()};
}

/// For instruction represents an affine loop nest.
class ForInst final
    : public Instruction,
      private llvm::TrailingObjects<ForInst, detail::OperandStorage> {
public:
  static ForInst *create(Location location, ArrayRef<Value *> lbOperands,
                         AffineMap lbMap, ArrayRef<Value *> ubOperands,
                         AffineMap ubMap, int64_t step);

  /// Resolve base class ambiguity.
  using Instruction::getFunction;

  /// Operand iterators.
  using operand_iterator = OperandIterator<ForInst, Value>;
  using const_operand_iterator = OperandIterator<const ForInst, const Value>;

  /// Operand iterator range.
  using operand_range = llvm::iterator_range<operand_iterator>;
  using const_operand_range = llvm::iterator_range<const_operand_iterator>;

  /// Get the body of the ForInst.
  Block *getBody() { return &body.front(); }

  /// Get the body of the ForInst.
  const Block *getBody() const { return &body.front(); }

  //===--------------------------------------------------------------------===//
  // Bounds and step
  //===--------------------------------------------------------------------===//

  /// Returns information about the lower bound as a single object.
  const AffineBound getLowerBound() const;

  /// Returns information about the upper bound as a single object.
  const AffineBound getUpperBound() const;

  /// Returns loop step.
  int64_t getStep() const { return step; }

  /// Returns affine map for the lower bound.
  AffineMap getLowerBoundMap() const { return lbMap; }
  /// Returns affine map for the upper bound. The upper bound is exclusive.
  AffineMap getUpperBoundMap() const { return ubMap; }

  /// Set lower bound.
  void setLowerBound(ArrayRef<Value *> operands, AffineMap map);
  /// Set upper bound.
  void setUpperBound(ArrayRef<Value *> operands, AffineMap map);

  /// Set the lower bound map without changing operands.
  void setLowerBoundMap(AffineMap map);

  /// Set the upper bound map without changing operands.
  void setUpperBoundMap(AffineMap map);

  /// Set loop step.
  void setStep(int64_t step) {
    assert(step > 0 && "step has to be a positive integer constant");
    this->step = step;
  }

  /// Returns true if the lower bound is constant.
  bool hasConstantLowerBound() const;
  /// Returns true if the upper bound is constant.
  bool hasConstantUpperBound() const;
  /// Returns true if both bounds are constant.
  bool hasConstantBounds() const {
    return hasConstantLowerBound() && hasConstantUpperBound();
  }
  /// Returns the value of the constant lower bound.
  /// Fails assertion if the bound is non-constant.
  int64_t getConstantLowerBound() const;
  /// Returns the value of the constant upper bound. The upper bound is
  /// exclusive. Fails assertion if the bound is non-constant.
  int64_t getConstantUpperBound() const;
  /// Sets the lower bound to the given constant value.
  void setConstantLowerBound(int64_t value);
  /// Sets the upper bound to the given constant value.
  void setConstantUpperBound(int64_t value);

  /// Returns true if both the lower and upper bound have the same operand lists
  /// (same operands in the same order).
  bool matchingBoundOperandList() const;

  /// Walk the operation instructions in the 'for' instruction in preorder,
  /// calling the callback for each operation.
  void walkOps(std::function<void(OperationInst *)> callback);

  /// Walk the operation instructions in the 'for' instruction in postorder,
  /// calling the callback for each operation.
  void walkOpsPostOrder(std::function<void(OperationInst *)> callback);

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  unsigned getNumOperands() const { return getOperandStorage().size(); }

  Value *getOperand(unsigned idx) { return getInstOperand(idx).get(); }
  const Value *getOperand(unsigned idx) const {
    return getInstOperand(idx).get();
  }
  void setOperand(unsigned idx, Value *value) {
    getInstOperand(idx).set(value);
  }

  operand_iterator operand_begin() { return operand_iterator(this, 0); }
  operand_iterator operand_end() {
    return operand_iterator(this, getNumOperands());
  }

  const_operand_iterator operand_begin() const {
    return const_operand_iterator(this, 0);
  }
  const_operand_iterator operand_end() const {
    return const_operand_iterator(this, getNumOperands());
  }

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

  // TODO: provide iterators for the lower and upper bound operands
  // if the current access via getLowerBound(), getUpperBound() is too slow.

  /// Returns operands for the lower bound map.
  operand_range getLowerBoundOperands();
  const_operand_range getLowerBoundOperands() const;

  /// Returns operands for the upper bound map.
  operand_range getUpperBoundOperands();
  const_operand_range getUpperBoundOperands() const;

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Return the context this operation is associated with.
  MLIRContext *getContext() const {
    return getInductionVar()->getType().getContext();
  }

  using Instruction::dump;
  using Instruction::print;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const IROperandOwner *ptr) {
    return ptr->getKind() == IROperandOwner::Kind::ForInst;
  }

  /// Returns the induction variable for this loop.
  Value *getInductionVar();
  const Value *getInductionVar() const {
    return const_cast<ForInst *>(this)->getInductionVar();
  }

  void destroy();

private:
  // The Block for the body. By construction, this list always contains exactly
  // one block.
  BlockList body;

  // Affine map for the lower bound.
  AffineMap lbMap;
  // Affine map for the upper bound. The upper bound is exclusive.
  AffineMap ubMap;
  // Positive constant step. Since index is stored as an int64_t, we restrict
  // step to the set of positive integers that int64_t can represent.
  int64_t step;

  explicit ForInst(Location location, AffineMap lbMap, AffineMap ubMap,
                   int64_t step);
  ~ForInst();

  /// Returns the operand storage object.
  detail::OperandStorage &getOperandStorage() {
    return *getTrailingObjects<detail::OperandStorage>();
  }
  const detail::OperandStorage &getOperandStorage() const {
    return *getTrailingObjects<detail::OperandStorage>();
  }

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<ForInst, detail::OperandStorage>;
};

/// Returns if the provided value is the induction variable of a ForInst.
bool isForInductionVar(const Value *val);

/// Returns the loop parent of an induction variable. If the provided value is
/// not an induction variable, then return nullptr.
ForInst *getForInductionVarOwner(Value *val);
const ForInst *getForInductionVarOwner(const Value *val);

/// Extracts the induction variables from a list of ForInsts and returns them.
SmallVector<Value *, 8> extractForInductionVars(ArrayRef<ForInst *> forInsts);

/// AffineBound represents a lower or upper bound in the for instruction.
/// This class does not own the underlying operands. Instead, it refers
/// to the operands stored in the ForInst. Its life span should not exceed
/// that of the for instruction it refers to.
class AffineBound {
public:
  const ForInst *getForInst() const { return &inst; }
  AffineMap getMap() const { return map; }

  unsigned getNumOperands() const { return opEnd - opStart; }
  const Value *getOperand(unsigned idx) const {
    return inst.getOperand(opStart + idx);
  }
  const InstOperand &getInstOperand(unsigned idx) const {
    return inst.getInstOperand(opStart + idx);
  }

  using operand_iterator = ForInst::operand_iterator;
  using operand_range = ForInst::operand_range;

  operand_iterator operand_begin() const {
    // These are iterators over Value *. Not casting away const'ness would
    // require the caller to use const Value *.
    return operand_iterator(const_cast<ForInst *>(&inst), opStart);
  }
  operand_iterator operand_end() const {
    return operand_iterator(const_cast<ForInst *>(&inst), opEnd);
  }

  /// Returns an iterator on the underlying Value's (Value *).
  operand_range getOperands() const { return {operand_begin(), operand_end()}; }
  ArrayRef<InstOperand> getInstOperands() const {
    auto ops = inst.getInstOperands();
    return ArrayRef<InstOperand>(ops.begin() + opStart, ops.begin() + opEnd);
  }

private:
  // 'for' instruction that contains this bound.
  const ForInst &inst;
  // Start and end positions of this affine bound operands in the list of
  // the containing 'for' instruction operands.
  unsigned opStart, opEnd;
  // Affine map for this bound.
  AffineMap map;

  AffineBound(const ForInst &inst, unsigned opStart, unsigned opEnd,
              AffineMap map)
      : inst(inst), opStart(opStart), opEnd(opEnd), map(map) {}

  friend class ForInst;
};
} // end namespace mlir

#endif // MLIR_IR_INSTRUCTIONS_H
