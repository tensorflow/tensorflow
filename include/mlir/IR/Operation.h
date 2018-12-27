//===- Operation.h - MLIR Operation Class -----------------------*- C++ -*-===//
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

#ifndef MLIR_IR_OPERATION_H
#define MLIR_IR_OPERATION_H

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Statement.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
class AttributeListStorage;
template <typename OpType> class ConstOpPointer;
template <typename OpType> class OpPointer;
template <typename ObjectType, typename ElementType> class OperandIterator;
template <typename ObjectType, typename ElementType> class ResultIterator;
template <typename ObjectType, typename ElementType> class ResultTypeIterator;
class Function;
class IROperandOwner;
class Statement;
class OperationStmt;
using Instruction = Statement;

/// Operations represent all of the arithmetic and other basic computation in
/// MLIR.  This class is the common implementation details behind Instruction
/// and OperationStmt.
///
class Operation {
public:
  /// Return the context this operation is associated with.
  MLIRContext *getContext() const;

  /// The source location the operation was defined or derived from.
  Location getLoc() const;

  /// Set the source location the operation was defined or derived from.
  void setLoc(Location loc);

  /// Return the function this operation is defined in.  This has a verbose
  /// name to avoid name lookup ambiguities.
  Function *getOperationFunction();

  const Function *getOperationFunction() const {
    return const_cast<Operation *>(this)->getOperationFunction();
  }

  /// The name of an operation is the key identifier for it.
  OperationName getName() const { return nameAndIsInstruction.getPointer(); }

  /// Return the number of operands this operation has.
  unsigned getNumOperands() const;

  Value *getOperand(unsigned idx);
  const Value *getOperand(unsigned idx) const {
    return const_cast<Operation *>(this)->getOperand(idx);
  }
  void setOperand(unsigned idx, Value *value);

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<Operation, Value>;
  operand_iterator operand_begin();
  operand_iterator operand_end();
  llvm::iterator_range<operand_iterator> getOperands();

  // Support const operand iteration.
  using const_operand_iterator = OperandIterator<const Operation, const Value>;
  const_operand_iterator operand_begin() const;
  const_operand_iterator operand_end() const;
  llvm::iterator_range<const_operand_iterator> getOperands() const;

  /// Return the number of results this operation has.
  unsigned getNumResults() const;

  /// Return the indicated result.
  Value *getResult(unsigned idx);
  const Value *getResult(unsigned idx) const {
    return const_cast<Operation *>(this)->getResult(idx);
  }

  // Support non-const result iteration.
  using result_iterator = ResultIterator<Operation, Value>;
  result_iterator result_begin();
  result_iterator result_end();
  llvm::iterator_range<result_iterator> getResults();

  // Support const result iteration.
  using const_result_iterator = ResultIterator<const Operation, const Value>;
  const_result_iterator result_begin() const;
  const_result_iterator result_end() const;
  llvm::iterator_range<const_result_iterator> getResults() const;

  // Support for result type iteration.
  using result_type_iterator = ResultTypeIterator<const Operation, const Value>;
  result_type_iterator result_type_begin() const;
  result_type_iterator result_type_end() const;
  llvm::iterator_range<result_type_iterator> getResultTypes() const;

  // Support for successor querying.
  unsigned getNumSuccessors() const;
  unsigned getNumSuccessorOperands(unsigned index) const;
  BasicBlock *getSuccessor(unsigned index);
  BasicBlock *getSuccessor(unsigned index) const {
    return const_cast<Operation *>(this)->getSuccessor(index);
  }
  void setSuccessor(BasicBlock *block, unsigned index);
  void eraseSuccessorOperand(unsigned succIndex, unsigned opIndex);
  llvm::iterator_range<const_operand_iterator>
  getSuccessorOperands(unsigned index) const;
  llvm::iterator_range<operand_iterator> getSuccessorOperands(unsigned index);

  /// Return true if there are no users of any results of this operation.
  bool use_empty() const;

  /// Unlink this operation from its current block and insert it right before
  /// `existingOp` which may be in the same or another block of the same
  /// function.
  void moveBefore(Operation *existingOp);

  // Attributes.  Operations may optionally carry a list of attributes that
  // associate constants to names.  Attributes may be dynamically added and
  // removed over the lifetime of an operation.
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

  enum class RemoveResult {
    Removed, NotFound
  };

  /// Remove the attribute with the specified name if it exists.  The return
  /// value indicates whether the attribute was present or not.
  RemoveResult removeAttr(Identifier name);

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.  This function always
  /// returns true.  NOTE: This may terminate the containing application, only
  /// use when the IR is in an inconsistent state.
  bool emitError(const Twine &message) const;

  /// Emit an error with the op name prefixed, like "'dim' op " which is
  /// convenient for verifiers.  This function always returns true.
  bool emitOpError(const Twine &message) const;

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitWarning(const Twine &message) const;

  /// Emit a note about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitNote(const Twine &message) const;

  /// If this operation has a registered operation description, return it.
  /// Otherwise return null.
  const AbstractOperation *getAbstractOperation() const {
    return getName().getAbstractOperation();
  }

  // Return a null OpPointer for the specified type.
  template <typename OpClass> static OpPointer<OpClass> getNull() {
    return OpPointer<OpClass>(OpClass(nullptr));
  }

  /// The dyn_cast methods perform a dynamic cast from an Operation (like
  /// Instruction and OperationStmt) to a typed Op like DimOp.  This returns
  /// a null OpPointer on failure.
  template <typename OpClass> OpPointer<OpClass> dyn_cast() {
    if (isa<OpClass>()) {
      return cast<OpClass>();
    } else {
      return OpPointer<OpClass>(OpClass(nullptr));
    }
  }

  /// The dyn_cast methods perform a dynamic cast from an Operation (like
  /// Instruction and OperationStmt) to a typed Op like DimOp.  This returns
  /// a null ConstOpPointer on failure.
  template <typename OpClass> ConstOpPointer<OpClass> dyn_cast() const {
    if (isa<OpClass>()) {
      return cast<OpClass>();
    } else {
      return ConstOpPointer<OpClass>(OpClass(nullptr));
    }
  }

  /// The cast methods perform a cast from an Operation (like
  /// Instruction and OperationStmt) to a typed Op like DimOp.  This aborts
  /// if the parameter to the template isn't an instance of the template type
  /// argument.
  template <typename OpClass> OpPointer<OpClass> cast() {
    assert(isa<OpClass>() && "cast<Ty>() argument of incompatible type!");
    return OpPointer<OpClass>(OpClass(this));
  }

  /// The cast methods perform a cast from an Operation (like
  /// Instruction and OperationStmt) to a typed Op like DimOp.  This aborts
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

  enum class OperationKind { Instruction, Statement };
  // This is used to implement the dynamic casting logic, but you shouldn't
  // call it directly.  Use something like isa<Instruction>(someOp) instead.
  OperationKind getOperationKind() const {
    return nameAndIsInstruction.getInt() ? OperationKind::Instruction
                                         : OperationKind::Statement;
  }

  // Returns whether the operation is commutative.
  bool isCommutative() const {
    if (auto *absOp = getAbstractOperation())
      return absOp->hasProperty(OperationProperty::Commutative);
    return false;
  }

  // Returns whether the operation has side-effects.
  bool hasNoSideEffect() const {
    if (auto *absOp = getAbstractOperation())
      return absOp->hasProperty(OperationProperty::NoSideEffect);
    return false;
  }

  // Returns whether the operation is a terminator.
  bool isTerminator() const {
    if (auto *absOp = getAbstractOperation())
      return absOp->hasProperty(OperationProperty::Terminator);
    return false;
  }

  /// Remove this operation from its parent block and delete it.
  void erase();

  /// Attempt to constant fold this operation with the specified constant
  /// operand values - the elements in "operands" will correspond directly to
  /// the operands of the operation, but may be null if non-constant.  If
  /// constant folding is successful, this returns false and fills in the
  /// `results` vector.  If not, this returns true and `results` is unspecified.
  bool constantFold(ArrayRef<Attribute> operands,
                    SmallVectorImpl<Attribute> &results) const;

  void print(raw_ostream &os) const;
  void dump() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Statement *stmt);
  static bool classof(const IROperandOwner *ptr);

protected:
  Operation(bool isInstruction, OperationName name,
            ArrayRef<NamedAttribute> attrs, MLIRContext *context);
  ~Operation();

private:
  Operation(const Operation&) = delete;
  void operator=(const Operation&) = delete;

  /// This holds the name of the operation, and a bool.  The bool is true if
  /// this operation is an Instruction, false if it is a OperationStmt.
  llvm::PointerIntPair<OperationName, 1, bool> nameAndIsInstruction;

  /// This holds general named attributes for the operation.
  AttributeListStorage *attrs;
};

/// This template implements the result iterators for the various IR classes
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

/// This template implements the result type iterators for the various IR
/// classes in terms of getResult(idx)->getType().
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

// Implement the inline operand iterator methods.
inline auto Operation::operand_begin() -> operand_iterator {
  return operand_iterator(this, 0);
}

inline auto Operation::operand_end() -> operand_iterator {
  return operand_iterator(this, getNumOperands());
}

inline auto Operation::getOperands() -> llvm::iterator_range<operand_iterator> {
  return {operand_begin(), operand_end()};
}

inline auto Operation::operand_begin() const -> const_operand_iterator {
  return const_operand_iterator(this, 0);
}

inline auto Operation::operand_end() const -> const_operand_iterator {
  return const_operand_iterator(this, getNumOperands());
}

inline auto Operation::getOperands() const
    -> llvm::iterator_range<const_operand_iterator> {
  return {operand_begin(), operand_end()};
}

// Implement the inline result iterator methods.
inline auto Operation::result_begin() -> result_iterator {
  return result_iterator(this, 0);
}

inline auto Operation::result_end() -> result_iterator {
  return result_iterator(this, getNumResults());
}

inline auto Operation::getResults() -> llvm::iterator_range<result_iterator> {
  return {result_begin(), result_end()};
}

inline auto Operation::result_begin() const -> const_result_iterator {
  return const_result_iterator(this, 0);
}

inline auto Operation::result_end() const -> const_result_iterator {
  return const_result_iterator(this, getNumResults());
}

inline auto Operation::getResults() const
    -> llvm::iterator_range<const_result_iterator> {
  return {result_begin(), result_end()};
}

inline auto Operation::result_type_begin() const -> result_type_iterator {
  return result_type_iterator(this, 0);
}

inline auto Operation::result_type_end() const -> result_type_iterator {
  return result_type_iterator(this, getNumResults());
}

inline auto Operation::getResultTypes() const
    -> llvm::iterator_range<result_type_iterator> {
  return {result_type_begin(), result_type_end()};
}
} // end namespace mlir

/// We need to teach the LLVM cast/dyn_cast etc logic how to cast from an
/// IROperandOwner* to Operation*.  This can't be done with a simple pointer to
/// pointer cast because the pointer adjustment depends on whether the Owner is
/// dynamically an Instruction or Statement, because of multiple inheritance.
namespace llvm {
template <>
struct cast_convert_val<mlir::Operation, mlir::IROperandOwner *,
                        mlir::IROperandOwner *> {
  static mlir::Operation *doit(const mlir::IROperandOwner *value);
};
template <typename From>
struct cast_convert_val<mlir::Operation, From *, From *> {
  template <typename FromImpl,
            typename std::enable_if<std::is_base_of<
                mlir::IROperandOwner, FromImpl>::value>::type * = nullptr>
  static mlir::Operation *doit_impl(const FromImpl *value) {
    return cast_convert_val<mlir::Operation, mlir::IROperandOwner *,
                            mlir::IROperandOwner *>::doit(value);
  }

  static mlir::Operation *doit(const From *value) { return doit_impl(value); }
};
} // namespace llvm

#endif
