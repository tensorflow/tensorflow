//===- OperationSupport.h ---------------------------------------*- C++ -*-===//
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
// This file defines a number of support types that Operation and related
// classes build on top of.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPERATION_SUPPORT_H
#define MLIR_IR_OPERATION_SUPPORT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/TrailingObjects.h"
#include <memory>

namespace mlir {
class Block;
class Dialect;
class Operation;
using Instruction = Operation;
struct OperationState;
class OpAsmParser;
class OpAsmParserResult;
class OpAsmPrinter;
class Pattern;
class Region;
class RewritePattern;
class Type;
class Value;

/// This is a vector that owns the patterns inside of it.
using OwningPatternList = std::vector<std::unique_ptr<Pattern>>;
using OwningRewritePatternList = std::vector<std::unique_ptr<RewritePattern>>;

enum class OperationProperty {
  /// This bit is set for an operation if it is a commutative operation: that
  /// is a binary operator (two inputs) where "a op b" and "b op a" produce the
  /// same results.
  Commutative = 0b01,

  /// This bit is set for operations that have no side effects: that means that
  /// they do not read or write memory, or access any hidden state.
  NoSideEffect = 0b10,

  /// This bit is set for an operation if it is a terminator: that means
  /// an operation at the end of a block.
  Terminator = 0b100,
};

/// This is a "type erased" representation of a registered operation.  This
/// should only be used by things like the AsmPrinter and other things that need
/// to be parameterized by generic operation hooks.  Most user code should use
/// the concrete operation types.
class AbstractOperation {
public:
  using OperationProperties = uint32_t;

  /// This is the name of the operation.
  const StringRef name;

  /// This is the dialect that this operation belongs to.
  Dialect &dialect;

  /// Return true if this "op class" can match against the specified operation.
  bool (&isClassFor)(Operation *op);

  /// Use the specified object to parse this ops custom assembly format.
  bool (&parseAssembly)(OpAsmParser *parser, OperationState *result);

  /// This hook implements the AsmPrinter for this operation.
  void (&printAssembly)(Operation *op, OpAsmPrinter *p);

  /// This hook implements the verifier for this operation.  It should emits an
  /// error message and returns true if a problem is detected, or returns false
  /// if everything is ok.
  bool (&verifyInvariants)(Operation *op);

  /// This hook implements a constant folder for this operation.  It fills in
  /// `results` on success.
  LogicalResult (&constantFoldHook)(Operation *op, ArrayRef<Attribute> operands,
                                    SmallVectorImpl<Attribute> &results);

  /// This hook implements a generalized folder for this operation.  Operations
  /// can implement this to provide simplifications rules that are applied by
  /// the FuncBuilder::foldOrCreate API and the canonicalization pass.
  ///
  /// This is an intentionally limited interface - implementations of this hook
  /// can only perform the following changes to the operation:
  ///
  ///  1. They can leave the operation alone and without changing the IR, and
  ///     return failure.
  ///  2. They can mutate the operation in place, without changing anything else
  ///     in the IR.  In this case, return success.
  ///  3. They can return a list of existing values that can be used instead of
  ///     the operation.  In this case, fill in the results list and return
  ///     success.  The caller will remove the operation and use those results
  ///     instead.
  ///
  /// This allows expression of some simple in-place canonicalizations (e.g.
  /// "x+0 -> x", "min(x,y,x,z) -> min(x,y,z)", "x+y-x -> y", etc), but does
  /// not allow for canonicalizations that need to introduce new operations, not
  /// even constants (e.g. "x-x -> 0" cannot be expressed).
  LogicalResult (&foldHook)(Operation *op, SmallVectorImpl<Value *> &results);

  /// This hook returns any canonicalization pattern rewrites that the operation
  /// supports, for use by the canonicalization pass.
  void (&getCanonicalizationPatterns)(OwningRewritePatternList &results,
                                      MLIRContext *context);

  /// Returns whether the operation has a particular property.
  bool hasProperty(OperationProperty property) const {
    return opProperties & static_cast<OperationProperties>(property);
  }

  /// Look up the specified operation in the specified MLIRContext and return a
  /// pointer to it if present.  Otherwise, return a null pointer.
  static const AbstractOperation *lookup(StringRef opName,
                                         MLIRContext *context);

  /// This constructor is used by Dialect objects when they register the list of
  /// operations they contain.
  template <typename T> static AbstractOperation get(Dialect &dialect) {
    return AbstractOperation(
        T::getOperationName(), dialect, T::getOperationProperties(),
        T::isClassFor, T::parseAssembly, T::printAssembly, T::verifyInvariants,
        T::constantFoldHook, T::foldHook, T::getCanonicalizationPatterns);
  }

private:
  AbstractOperation(
      StringRef name, Dialect &dialect, OperationProperties opProperties,
      bool (&isClassFor)(Operation *op),
      bool (&parseAssembly)(OpAsmParser *parser, OperationState *result),
      void (&printAssembly)(Operation *op, OpAsmPrinter *p),
      bool (&verifyInvariants)(Operation *op),
      LogicalResult (&constantFoldHook)(Operation *op,
                                        ArrayRef<Attribute> operands,
                                        SmallVectorImpl<Attribute> &results),
      LogicalResult (&foldHook)(Operation *op,
                                SmallVectorImpl<Value *> &results),
      void (&getCanonicalizationPatterns)(OwningRewritePatternList &results,
                                          MLIRContext *context))
      : name(name), dialect(dialect), isClassFor(isClassFor),
        parseAssembly(parseAssembly), printAssembly(printAssembly),
        verifyInvariants(verifyInvariants), constantFoldHook(constantFoldHook),
        foldHook(foldHook),
        getCanonicalizationPatterns(getCanonicalizationPatterns),
        opProperties(opProperties) {}

  /// The properties of the operation.
  const OperationProperties opProperties;
};

class OperationName {
public:
  using RepresentationUnion =
      llvm::PointerUnion<Identifier, const AbstractOperation *>;

  OperationName(AbstractOperation *op) : representation(op) {}
  OperationName(StringRef name, MLIRContext *context);

  /// Return the name of this operation.  This always succeeds.
  StringRef getStringRef() const;

  /// If this operation has a registered operation description, return it.
  /// Otherwise return null.
  const AbstractOperation *getAbstractOperation() const;

  void print(raw_ostream &os) const;
  void dump() const;

  void *getAsOpaquePointer() const {
    return static_cast<void *>(representation.getOpaqueValue());
  }
  static OperationName getFromOpaquePointer(void *pointer);

private:
  RepresentationUnion representation;
  OperationName(RepresentationUnion representation)
      : representation(representation) {}
};

inline raw_ostream &operator<<(raw_ostream &os, OperationName identifier) {
  identifier.print(os);
  return os;
}

inline bool operator==(OperationName lhs, OperationName rhs) {
  return lhs.getAsOpaquePointer() == rhs.getAsOpaquePointer();
}

inline bool operator!=(OperationName lhs, OperationName rhs) {
  return lhs.getAsOpaquePointer() != rhs.getAsOpaquePointer();
}

// Make operation names hashable.
inline llvm::hash_code hash_value(OperationName arg) {
  return llvm::hash_value(arg.getAsOpaquePointer());
}

/// This represents an operation in an abstracted form, suitable for use with
/// the builder APIs.  This object is a large and heavy weight object meant to
/// be used as a temporary object on the stack.  It is generally unwise to put
/// this in a collection.
struct OperationState {
  MLIRContext *const context;
  Location location;
  OperationName name;
  SmallVector<Value *, 4> operands;
  /// Types of the results of this operation.
  SmallVector<Type, 4> types;
  SmallVector<NamedAttribute, 4> attributes;
  /// Successors of this operation and their respective operands.
  SmallVector<Block *, 1> successors;
  /// Regions that the op will hold.
  SmallVector<std::unique_ptr<Region>, 1> regions;
  /// If the operation has a resizable operand list.
  bool resizableOperandList = false;

public:
  OperationState(MLIRContext *context, Location location, StringRef name);

  OperationState(MLIRContext *context, Location location, OperationName name);

  OperationState(MLIRContext *context, Location location, StringRef name,
                 ArrayRef<Value *> operands, ArrayRef<Type> types,
                 ArrayRef<NamedAttribute> attributes,
                 ArrayRef<Block *> successors = {},
                 MutableArrayRef<std::unique_ptr<Region>> regions = {},
                 bool resizableOperandList = false);

  void addOperands(ArrayRef<Value *> newOperands) {
    assert(successors.empty() &&
           "Non successor operands should be added first.");
    operands.append(newOperands.begin(), newOperands.end());
  }

  void addTypes(ArrayRef<Type> newTypes) {
    types.append(newTypes.begin(), newTypes.end());
  }

  /// Add an attribute with the specified name.
  void addAttribute(StringRef name, Attribute attr) {
    addAttribute(Identifier::get(name, context), attr);
  }

  /// Add an attribute with the specified name.
  void addAttribute(Identifier name, Attribute attr) {
    attributes.push_back({name, attr});
  }

  void addSuccessor(Block *successor, ArrayRef<Value *> succOperands) {
    successors.push_back(successor);
    // Insert a sentinal operand to mark a barrier between successor operands.
    operands.push_back(nullptr);
    operands.append(succOperands.begin(), succOperands.end());
  }

  /// Create a region that should be attached to the operation.  These regions
  /// can be filled in immediately without waiting for Operation to be
  /// created.  When it is, the region bodies will be transferred.
  Region *addRegion();

  /// Take a region that should be attached to the Operation.  The body of the
  /// region will be transferred when the Operation is constructed.  If the
  /// region is null, a new empty region will be attached to the Operation.
  void addRegion(std::unique_ptr<Region> &&region);

  /// Sets the operand list of the operation as resizable.
  void setOperandListToResizable(bool isResizable = true) {
    resizableOperandList = isResizable;
  }
};

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

/// This class handles the management of operation operands. Operands are
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
    if (resizable) {
      new (&getResizableStorage())
          ResizableStorage(getTrailingObjects<InstOperand>(), numOperands);
    }
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
  void setOperands(Operation *owner, ArrayRef<Value *> operands);

  /// Erase an operand held by the storage.
  void eraseOperand(unsigned index);

  /// Get the operation operands held by the storage.
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
} // end namespace mlir

namespace llvm {
// Identifiers hash just like pointers, there is no need to hash the bytes.
template <> struct DenseMapInfo<mlir::OperationName> {
  static mlir::OperationName getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::OperationName::getFromOpaquePointer(pointer);
  }
  static mlir::OperationName getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::OperationName::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::OperationName Val) {
    return DenseMapInfo<void *>::getHashValue(Val.getAsOpaquePointer());
  }
  static bool isEqual(mlir::OperationName LHS, mlir::OperationName RHS) {
    return LHS == RHS;
  }
};

/// The pointer inside of an identifier comes from a StringMap, so its alignment
/// is always at least 4 and probably 8 (on 64-bit machines).  Allow LLVM to
/// steal the low bits.
template <> struct PointerLikeTypeTraits<mlir::OperationName> {
public:
  static inline void *getAsVoidPointer(mlir::OperationName I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::OperationName getFromVoidPointer(void *P) {
    return mlir::OperationName::getFromOpaquePointer(P);
  }
  enum {
    NumLowBitsAvailable = PointerLikeTypeTraits<
        mlir::OperationName::RepresentationUnion>::NumLowBitsAvailable
  };
};

} // end namespace llvm

#endif
