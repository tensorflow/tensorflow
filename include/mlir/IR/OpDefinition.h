//===- OpDefinition.h - Classes for defining concrete Op types --*- C++ -*-===//
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
// This file implements helper classes for implementing the "Op" types.  This
// includes the Op type, which is the base class for Op class definitions,
// as well as number of traits in the OpTrait namespace that provide a
// declarative way to specify properties of Ops.
//
// The purpose of these types are to allow light-weight implementation of
// concrete ops (like DimOp) with very little boilerplate.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPDEFINITION_H
#define MLIR_IR_OPDEFINITION_H

#include "mlir/IR/Statements.h"
#include <type_traits>

namespace mlir {
class Builder;

namespace OpTrait {
template <typename ConcreteType> class OneResult;
}

/// This type trait produces true if the specified type is in the specified
/// type list.
template <typename same, typename first, typename... more>
struct typelist_contains {
  static const bool value = std::is_same<same, first>::value ||
                            typelist_contains<same, more...>::value;
};
template <typename same, typename first>
struct typelist_contains<same, first> : std::is_same<same, first> {};

/// This type trait is used to determine if an operation has a single result.
template <typename OpType> struct IsSingleResult {
  static const bool value = std::is_convertible<
      OpType *, OpTrait::OneResult<typename OpType::ConcreteOpType> *>::value;
};

/// This pointer represents a notional "OperationInst*" but where the actual
/// storage of the pointer is maintained in the templated "OpType" class.
template <typename OpType>
class OpPointer {
public:
  explicit OpPointer() : value(OperationInst::getNull<OpType>().value) {}
  explicit OpPointer(OpType value) : value(value) {}

  OpType &operator*() { return value; }

  OpType *operator->() { return &value; }

  operator bool() const { return value.getOperation(); }

  /// OpPointer can be implicitly converted to OpType*.
  /// Return `nullptr` if there is no associated OperationInst*.
  operator OpType *() {
    if (!value.getOperation())
      return nullptr;
    return &value;
  }

  /// If the OpType operation includes the OneResult trait, then OpPointer can
  /// be implicitly converted to an Value*.  This yields the value of the
  /// only result.
  template <typename SFINAE = OpType>
  operator typename std::enable_if<IsSingleResult<SFINAE>::value,
                                   Value *>::type() {
    return value.getResult();
  }

private:
  OpType value;
};

/// This pointer represents a notional "const OperationInst*" but where the
/// actual storage of the pointer is maintained in the templated "OpType" class.
template <typename OpType>
class ConstOpPointer {
public:
  explicit ConstOpPointer() : value(OperationInst::getNull<OpType>().value) {}
  explicit ConstOpPointer(OpType value) : value(value) {}

  const OpType &operator*() const { return value; }

  const OpType *operator->() const { return &value; }

  /// Return true if non-null.
  operator bool() const { return value.getOperation(); }

  /// ConstOpPointer can always be implicitly converted to const OpType*.
  /// Return `nullptr` if there is no associated OperationInst*.
  operator const OpType *() const {
    if (!value.getOperation())
      return nullptr;
    return &value;
  }

  /// If the OpType operation includes the OneResult trait, then OpPointer can
  /// be implicitly converted to an const Value*.  This yields the value of
  /// the only result.
  template <typename SFINAE = OpType>
  operator typename std::enable_if<
      std::is_convertible<
          SFINAE *,
          OpTrait::OneResult<typename SFINAE::ConcreteOpType> *>::value,
      const Value *>::type() const {
    return value.getResult();
  }

private:
  const OpType value;
};

/// This is the concrete base class that holds the operation pointer and has
/// non-generic methods that only depend on State (to avoid having them
/// instantiated on template types that don't affect them.
///
/// This also has the fallback implementations of customization hooks for when
/// they aren't customized.
class OpState {
public:
  /// Return the operation that this refers to.
  const OperationInst *getOperation() const { return state; }
  OperationInst *getOperation() { return state; }

  /// The source location the operation was defined or derived from.
  Location getLoc() const { return state->getLoc(); }

  /// Return all of the attributes on this operation.
  ArrayRef<NamedAttribute> getAttrs() const { return state->getAttrs(); }

  /// Return an attribute with the specified name.
  Attribute getAttr(StringRef name) const { return state->getAttr(name); }

  /// If the operation has an attribute of the specified type, return it.
  template <typename AttrClass> AttrClass getAttrOfType(StringRef name) const {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  /// If the an attribute exists with the specified name, change it to the new
  /// value.  Otherwise, add a new attribute with the specified name/value.
  void setAttr(Identifier name, Attribute value) {
    state->setAttr(name, value);
  }

  /// Return true if there are no users of any results of this operation.
  bool use_empty() const { return state->use_empty(); }

  /// Remove this operation from its parent block and delete it.
  void erase() { state->erase(); }

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.  This function always
  /// returns true.  NOTE: This may terminate the containing application, only
  /// use when the IR is in an inconsistent state.
  bool emitError(const Twine &message) const;

  /// Emit an error with the op name prefixed, like "'dim' op " which is
  /// convenient for verifiers.  This always returns true.
  bool emitOpError(const Twine &message) const;

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitWarning(const Twine &message) const;

  /// Emit a note about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitNote(const Twine &message) const;

  // These are default implementations of customization hooks.
public:
  /// This hook returns any canonicalization pattern rewrites that the operation
  /// supports, for use by the canonicalization pass.
  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {}

protected:
  /// If the concrete type didn't implement a custom verifier hook, just fall
  /// back to this one which accepts everything.
  bool verify() const { return false; }

  /// Unless overridden, the short form of an op is always rejected.  Op
  /// implementations should implement this to return boolean true on failure.
  /// On success, they should return false and fill in result with the fields to
  /// use.
  static bool parse(OpAsmParser *parser, OperationState *result);

  // The fallback for the printer is to print it the longhand form.
  void print(OpAsmPrinter *p) const;

  /// Mutability management is handled by the OpWrapper/OpConstWrapper classes,
  /// so we can cast it away here.
  explicit OpState(const OperationInst *state)
      : state(const_cast<OperationInst *>(state)) {}

private:
  OperationInst *state;
};

/// This template defines the constantFoldHook as used by AbstractOperation.
/// The default implementation uses a general constantFold method that can be
/// defined on custom ops which can return multiple results.
template <typename ConcreteType, bool isSingleResult, typename = void>
class ConstFoldingHook {
public:
  /// This hook implements a constant folder for this operation.  It returns
  /// true if folding failed, or returns false and fills in `results` on
  /// success.
  static bool constantFoldHook(const OperationInst *op,
                               ArrayRef<Attribute> operands,
                               SmallVectorImpl<Attribute> &results) {
    return op->cast<ConcreteType>()->constantFold(operands, results,
                                                  op->getContext());
  }

  /// Op implementations can implement this hook.  It should attempt to constant
  /// fold this operation with the specified constant operand values - the
  /// elements in "operands" will correspond directly to the operands of the
  /// operation, but may be null if non-constant.  If constant folding is
  /// successful, this returns false and fills in the `results` vector.  If not,
  /// this returns true and `results` is unspecified.
  ///
  /// If not overridden, this fallback implementation always fails to fold.
  ///
  bool constantFold(ArrayRef<Attribute> operands,
                    SmallVectorImpl<Attribute> &results,
                    MLIRContext *context) const {
    return true;
  }
};

/// This template specialization defines the constantFoldHook as used by
/// AbstractOperation for single-result operations.  This gives the hook a nicer
/// signature that is easier to implement.
template <typename ConcreteType, bool isSingleResult>
class ConstFoldingHook<ConcreteType, isSingleResult,
                       typename std::enable_if<isSingleResult>::type> {
public:
  /// This hook implements a constant folder for this operation.  It returns
  /// true if folding failed, or returns false and fills in `results` on
  /// success.
  static bool constantFoldHook(const OperationInst *op,
                               ArrayRef<Attribute> operands,
                               SmallVectorImpl<Attribute> &results) {
    auto result =
        op->cast<ConcreteType>()->constantFold(operands, op->getContext());
    if (!result)
      return true;

    results.push_back(result);
    return false;
  }
};

//===----------------------------------------------------------------------===//
// OperationInst Trait Types
//===----------------------------------------------------------------------===//

namespace OpTrait {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes.  This avoids them being template
// instantiated/duplicated.
namespace impl {
bool verifyZeroOperands(const OperationInst *op);
bool verifyOneOperand(const OperationInst *op);
bool verifyNOperands(const OperationInst *op, unsigned numOperands);
bool verifyAtLeastNOperands(const OperationInst *op, unsigned numOperands);
bool verifyOperandsAreIntegerLike(const OperationInst *op);
bool verifySameTypeOperands(const OperationInst *op);
bool verifyZeroResult(const OperationInst *op);
bool verifyOneResult(const OperationInst *op);
bool verifyNResults(const OperationInst *op, unsigned numOperands);
bool verifyAtLeastNResults(const OperationInst *op, unsigned numOperands);
bool verifySameOperandsAndResultShape(const OperationInst *op);
bool verifySameOperandsAndResultType(const OperationInst *op);
bool verifyResultsAreBoolLike(const OperationInst *op);
bool verifyResultsAreFloatLike(const OperationInst *op);
bool verifyResultsAreIntegerLike(const OperationInst *op);
bool verifyIsTerminator(const OperationInst *op);
} // namespace impl

/// Helper class for implementing traits.  Clients are not expected to interact
/// with this directly, so its members are all protected.
template <typename ConcreteType, template <typename> class TraitType>
class TraitBase {
protected:
  /// Return the ultimate OperationInst being worked on.
  OperationInst *getOperation() {
    // We have to cast up to the trait type, then to the concrete type, then to
    // the BaseState class in explicit hops because the concrete type will
    // multiply derive from the (content free) TraitBase class, and we need to
    // be able to disambiguate the path for the C++ compiler.
    auto *trait = static_cast<TraitType<ConcreteType> *>(this);
    auto *concrete = static_cast<ConcreteType *>(trait);
    auto *base = static_cast<OpState *>(concrete);
    return base->getOperation();
  }
  const OperationInst *getOperation() const {
    return const_cast<TraitBase *>(this)->getOperation();
  }

  /// Provide default implementations of trait hooks.  This allows traits to
  /// provide exactly the overrides they care about.
  static bool verifyTrait(const OperationInst *op) { return false; }
  static AbstractOperation::OperationProperties getTraitProperties() {
    return 0;
  }
};

/// This class provides the API for ops that are known to have no
/// SSA operand.
template <typename ConcreteType>
class ZeroOperands : public TraitBase<ConcreteType, ZeroOperands> {
public:
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifyZeroOperands(op);
  }

private:
  // Disable these.
  void getOperand() const {}
  void setOperand() const {}
};

/// This class provides the API for ops that are known to have exactly one
/// SSA operand.
template <typename ConcreteType>
class OneOperand : public TraitBase<ConcreteType, OneOperand> {
public:
  const Value *getOperand() const {
    return this->getOperation()->getOperand(0);
  }

  Value *getOperand() { return this->getOperation()->getOperand(0); }

  void setOperand(Value *value) { this->getOperation()->setOperand(0, value); }

  static bool verifyTrait(const OperationInst *op) {
    return impl::verifyOneOperand(op);
  }
};

/// This class provides the API for ops that are known to have a specified
/// number of operands.  This is used as a trait like this:
///
///   class FooOp : public Op<FooOp, OpTrait::NOperands<2>::Impl> {
///
template <unsigned N> class NOperands {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, NOperands<N>::Impl> {
  public:
    const Value *getOperand(unsigned i) const {
      return this->getOperation()->getOperand(i);
    }

    Value *getOperand(unsigned i) {
      return this->getOperation()->getOperand(i);
    }

    void setOperand(unsigned i, Value *value) {
      this->getOperation()->setOperand(i, value);
    }

    static bool verifyTrait(const OperationInst *op) {
      return impl::verifyNOperands(op, N);
    }
  };
};

/// This class provides the API for ops that are known to have a at least a
/// specified number of operands.  This is used as a trait like this:
///
///   class FooOp : public Op<FooOp, OpTrait::AtLeastNOperands<2>::Impl> {
///
template <unsigned N> class AtLeastNOperands {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, AtLeastNOperands<N>::Impl> {
  public:
    unsigned getNumOperands() const {
      return this->getOperation()->getNumOperands();
    }
    const Value *getOperand(unsigned i) const {
      return this->getOperation()->getOperand(i);
    }

    Value *getOperand(unsigned i) {
      return this->getOperation()->getOperand(i);
    }

    void setOperand(unsigned i, Value *value) {
      this->getOperation()->setOperand(i, value);
    }

    // Support non-const operand iteration.
    using operand_iterator = OperationInst::operand_iterator;
    operand_iterator operand_begin() {
      return this->getOperation()->operand_begin();
    }
    operand_iterator operand_end() {
      return this->getOperation()->operand_end();
    }
    llvm::iterator_range<operand_iterator> getOperands() {
      return this->getOperation()->getOperands();
    }

    // Support const operand iteration.
    using const_operand_iterator = OperationInst::const_operand_iterator;
    const_operand_iterator operand_begin() const {
      return this->getOperation()->operand_begin();
    }
    const_operand_iterator operand_end() const {
      return this->getOperation()->operand_end();
    }
    llvm::iterator_range<const_operand_iterator> getOperands() const {
      return this->getOperation()->getOperands();
    }

    static bool verifyTrait(const OperationInst *op) {
      return impl::verifyAtLeastNOperands(op, N);
    }
  };
};

/// This class provides the API for ops which have an unknown number of
/// SSA operands.
template <typename ConcreteType>
class VariadicOperands : public TraitBase<ConcreteType, VariadicOperands> {
public:
  unsigned getNumOperands() const {
    return this->getOperation()->getNumOperands();
  }

  const Value *getOperand(unsigned i) const {
    return this->getOperation()->getOperand(i);
  }

  Value *getOperand(unsigned i) { return this->getOperation()->getOperand(i); }

  void setOperand(unsigned i, Value *value) {
    this->getOperation()->setOperand(i, value);
  }

  // Support non-const operand iteration.
  using operand_iterator = OperationInst::operand_iterator;
  operand_iterator operand_begin() {
    return this->getOperation()->operand_begin();
  }
  operand_iterator operand_end() { return this->getOperation()->operand_end(); }
  llvm::iterator_range<operand_iterator> getOperands() {
    return this->getOperation()->getOperands();
  }

  // Support const operand iteration.
  using const_operand_iterator = OperationInst::const_operand_iterator;
  const_operand_iterator operand_begin() const {
    return this->getOperation()->operand_begin();
  }
  const_operand_iterator operand_end() const {
    return this->getOperation()->operand_end();
  }
  llvm::iterator_range<const_operand_iterator> getOperands() const {
    return this->getOperation()->getOperands();
  }
};

/// This class provides return value APIs for ops that are known to have
/// zero results.
template <typename ConcreteType>
class ZeroResult : public TraitBase<ConcreteType, ZeroResult> {
public:
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifyZeroResult(op);
  }
};

/// This class provides return value APIs for ops that are known to have a
/// single result.
template <typename ConcreteType>
class OneResult : public TraitBase<ConcreteType, OneResult> {
public:
  Value *getResult() { return this->getOperation()->getResult(0); }
  const Value *getResult() const { return this->getOperation()->getResult(0); }

  Type getType() const { return getResult()->getType(); }

  /// Replace all uses of 'this' value with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  void replaceAllUsesWith(Value *newValue) {
    getResult()->replaceAllUsesWith(newValue);
  }

  static bool verifyTrait(const OperationInst *op) {
    return impl::verifyOneResult(op);
  }

  /// Op implementations can implement this hook.  It should attempt to constant
  /// fold this operation with the specified constant operand values - the
  /// elements in "operands" will correspond directly to the operands of the
  /// operation, but may be null if non-constant.  If constant folding is
  /// successful, this returns a non-null attribute, otherwise it returns null
  /// on failure.
  ///
  /// If not overridden, this fallback implementation always fails to fold.
  ///
  Attribute constantFold(ArrayRef<Attribute> operands,
                         MLIRContext *context) const {
    return nullptr;
  }
};

/// This class provides the API for ops that are known to have a specified
/// number of results.  This is used as a trait like this:
///
///   class FooOp : public Op<FooOp, OpTrait::NResults<2>::Impl> {
///
template <unsigned N> class NResults {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, NResults<N>::Impl> {
  public:
    static unsigned getNumResults() { return N; }

    const Value *getResult(unsigned i) const {
      return this->getOperation()->getResult(i);
    }

    Value *getResult(unsigned i) { return this->getOperation()->getResult(i); }

    Type getType(unsigned i) const { return getResult(i)->getType(); }

    static bool verifyTrait(const OperationInst *op) {
      return impl::verifyNResults(op, N);
    }
  };
};

/// This class provides the API for ops that are known to have at least a
/// specified number of results.  This is used as a trait like this:
///
///   class FooOp : public Op<FooOp, OpTrait::AtLeastNResults<2>::Impl> {
///
template <unsigned N> class AtLeastNResults {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, AtLeastNResults<N>::Impl> {
  public:
    const Value *getResult(unsigned i) const {
      return this->getOperation()->getResult(i);
    }

    Value *getResult(unsigned i) { return this->getOperation()->getResult(i); }

    Type getType(unsigned i) const { return getResult(i)->getType(); }

    static bool verifyTrait(const OperationInst *op) {
      return impl::verifyAtLeastNResults(op, N);
    }
  };
};

/// This class provides the API for ops which have an unknown number of
/// results.
template <typename ConcreteType>
class VariadicResults : public TraitBase<ConcreteType, VariadicResults> {
public:
  unsigned getNumResults() const {
    return this->getOperation()->getNumResults();
  }

  const Value *getResult(unsigned i) const {
    return this->getOperation()->getResult(i);
  }

  Value *getResult(unsigned i) { return this->getOperation()->getResult(i); }

  void setResult(unsigned i, Value *value) {
    this->getOperation()->setResult(i, value);
  }

  // Support non-const result iteration.
  using result_iterator = OperationInst::result_iterator;
  result_iterator result_begin() {
    return this->getOperation()->result_begin();
  }
  result_iterator result_end() { return this->getOperation()->result_end(); }
  llvm::iterator_range<result_iterator> getResults() {
    return this->getOperation()->getResults();
  }

  // Support const result iteration.
  using const_result_iterator = OperationInst::const_result_iterator;
  const_result_iterator result_begin() const {
    return this->getOperation()->result_begin();
  }
  const_result_iterator result_end() const {
    return this->getOperation()->result_end();
  }
  llvm::iterator_range<const_result_iterator> getResults() const {
    return this->getOperation()->getResults();
  }
};

/// This class provides verification for ops that are known to have the same
/// operand and result shape: both are scalars, vectors/tensors of the same
/// shape.
template <typename ConcreteType>
class SameOperandsAndResultShape
    : public TraitBase<ConcreteType, SameOperandsAndResultShape> {
public:
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifySameOperandsAndResultShape(op);
  }
};

/// This class provides verification for ops that are known to have the same
/// operand and result type.
///
/// Note: this trait subsumes the SameOperandsAndResultShape trait.
/// Additionally, it requires all operands and results should also have
/// the same element type.
template <typename ConcreteType>
class SameOperandsAndResultType
    : public TraitBase<ConcreteType, SameOperandsAndResultType> {
public:
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifySameOperandsAndResultType(op);
  }
};

/// This class verifies that any results of the specified op have a boolean
/// type, a vector thereof, or a tensor thereof.
template <typename ConcreteType>
class ResultsAreBoolLike : public TraitBase<ConcreteType, ResultsAreBoolLike> {
public:
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifyResultsAreBoolLike(op);
  }
};

/// This class verifies that any results of the specified op have a floating
/// point type, a vector thereof, or a tensor thereof.
template <typename ConcreteType>
class ResultsAreFloatLike
    : public TraitBase<ConcreteType, ResultsAreFloatLike> {
public:
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifyResultsAreFloatLike(op);
  }
};

/// This class verifies that any results of the specified op have an integer or
/// index type, a vector thereof, or a tensor thereof.
template <typename ConcreteType>
class ResultsAreIntegerLike
    : public TraitBase<ConcreteType, ResultsAreIntegerLike> {
public:
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifyResultsAreIntegerLike(op);
  }
};

/// This class adds property that the operation is commutative.
template <typename ConcreteType>
class IsCommutative : public TraitBase<ConcreteType, IsCommutative> {
public:
  static AbstractOperation::OperationProperties getTraitProperties() {
    return static_cast<AbstractOperation::OperationProperties>(
        OperationProperty::Commutative);
  }
};

/// This class adds property that the operation has no side effects.
template <typename ConcreteType>
class HasNoSideEffect : public TraitBase<ConcreteType, HasNoSideEffect> {
public:
  static AbstractOperation::OperationProperties getTraitProperties() {
    return static_cast<AbstractOperation::OperationProperties>(
        OperationProperty::NoSideEffect);
  }
};

/// This class verifies that all operands of the specified op have an integer or
/// index type, a vector thereof, or a tensor thereof.
template <typename ConcreteType>
class OperandsAreIntegerLike
    : public TraitBase<ConcreteType, OperandsAreIntegerLike> {
public:
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifyOperandsAreIntegerLike(op);
  }
};

/// This class verifies that all operands of the specified op have the same
/// type.
template <typename ConcreteType>
class SameTypeOperands : public TraitBase<ConcreteType, SameTypeOperands> {
public:
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifySameTypeOperands(op);
  }
};

/// This class provides the API for ops that are known to be terminators.
template <typename ConcreteType>
class IsTerminator : public TraitBase<ConcreteType, IsTerminator> {
public:
  static AbstractOperation::OperationProperties getTraitProperties() {
    return static_cast<AbstractOperation::OperationProperties>(
        OperationProperty::Terminator);
  }
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifyIsTerminator(op);
  }

  unsigned getNumSuccessors() const {
    return this->getOperation()->getNumSuccessors();
  }
  unsigned getNumSuccessorOperands(unsigned index) const {
    return this->getOperation()->getNumSuccessorOperands(index);
  }

  const BasicBlock *getSuccessor(unsigned index) const {
    return this->getOperation()->getSuccessor(index);
  }
  BasicBlock *getSuccessor(unsigned index) {
    return this->getOperation()->getSuccessor(index);
  }

  void setSuccessor(BasicBlock *block, unsigned index) {
    return this->getOperation()->setSuccessor(block, index);
  }

  void addSuccessorOperand(unsigned index, Value *value) {
    return this->getOperation()->addSuccessorOperand(index, value);
  }
  void addSuccessorOperands(unsigned index, ArrayRef<Value *> values) {
    return this->getOperation()->addSuccessorOperand(index, values);
  }
};

} // end namespace OpTrait

//===----------------------------------------------------------------------===//
// OperationInst Definition classes
//===----------------------------------------------------------------------===//

/// This provides public APIs that all operations should have.  The template
/// argument 'ConcreteType' should be the concrete type by CRTP and the others
/// are base classes by the policy pattern.
template <typename ConcreteType, template <typename T> class... Traits>
class Op : public OpState,
           public Traits<ConcreteType>...,
           public ConstFoldingHook<
               ConcreteType,
               typelist_contains<OpTrait::OneResult<ConcreteType>, OpState,
                                 Traits<ConcreteType>...>::value> {
public:
  /// Return the operation that this refers to.
  const OperationInst *getOperation() const { return OpState::getOperation(); }
  OperationInst *getOperation() { return OpState::getOperation(); }

  /// Return true if this "op class" can match against the specified operation.
  /// This hook can be overridden with a more specific implementation in
  /// the subclass of Base.
  ///
  static bool isClassFor(const OperationInst *op) {
    return op->getName().getStringRef() == ConcreteType::getOperationName();
  }

  /// This is the hook used by the AsmParser to parse the custom form of this
  /// op from an .mlir file.  Op implementations should provide a parse method,
  /// which returns boolean true on failure.  On success, they should return
  /// false and fill in result with the fields to use.
  static bool parseAssembly(OpAsmParser *parser, OperationState *result) {
    return ConcreteType::parse(parser, result);
  }

  /// This is the hook used by the AsmPrinter to emit this to the .mlir file.
  /// Op implementations should provide a print method.
  static void printAssembly(const OperationInst *op, OpAsmPrinter *p) {
    auto opPointer = op->dyn_cast<ConcreteType>();
    assert(opPointer &&
           "op's name does not match name of concrete type instantiated with");
    opPointer->print(p);
  }

  /// This is the hook that checks whether or not this instruction is well
  /// formed according to the invariants of its opcode.  It delegates to the
  /// Traits for their policy implementations, and allows the user to specify
  /// their own verify() method.
  ///
  /// On success this returns false; on failure it emits an error to the
  /// diagnostic subsystem and returns true.
  static bool verifyInvariants(const OperationInst *op) {
    return BaseVerifier<Traits<ConcreteType>...>::verifyTrait(op) ||
           op->cast<ConcreteType>()->verify();
  }

  // Returns the properties of an operation by combining the properties of the
  // traits of the op.
  static AbstractOperation::OperationProperties getOperationProperties() {
    return BaseProperties<Traits<ConcreteType>...>::getTraitProperties();
  }

  // TODO: Provide a dump() method.

  /// Expose the type we are instantiated on to template machinery that may want
  /// to introspect traits on this operation.
  using ConcreteOpType = ConcreteType;

protected:
  explicit Op(const OperationInst *state) : OpState(state) {}

private:
  template <typename... Types> struct BaseVerifier;

  template <typename First, typename... Rest>
  struct BaseVerifier<First, Rest...> {
    static bool verifyTrait(const OperationInst *op) {
      return First::verifyTrait(op) || BaseVerifier<Rest...>::verifyTrait(op);
    }
  };

  template <typename First> struct BaseVerifier<First> {
    static bool verifyTrait(const OperationInst *op) {
      return First::verifyTrait(op);
    }
  };

  template <> struct BaseVerifier<> {
    static bool verifyTrait(const OperationInst *op) { return false; }
  };

  template <typename... Types> struct BaseProperties;

  template <typename First, typename... Rest>
  struct BaseProperties<First, Rest...> {
    static AbstractOperation::OperationProperties getTraitProperties() {
      return First::getTraitProperties() |
             BaseProperties<Rest...>::getTraitProperties();
    }
  };

  template <typename First> struct BaseProperties<First> {
    static AbstractOperation::OperationProperties getTraitProperties() {
      return First::getTraitProperties();
    }
  };

  template <> struct BaseProperties<> {
    static AbstractOperation::OperationProperties getTraitProperties() {
      return 0;
    }
  };
};

// These functions are out-of-line implementations of the methods in BinaryOp,
// which avoids them being template instantiated/duplicated.
namespace impl {
void buildBinaryOp(Builder *builder, OperationState *result, Value *lhs,
                   Value *rhs);
bool parseBinaryOp(OpAsmParser *parser, OperationState *result);
void printBinaryOp(const OperationInst *op, OpAsmPrinter *p);
} // namespace impl

/// This template is used for operations that are simple binary ops that have
/// two input operands, one result, and whose operands and results all have
/// the same type.
///
/// From this structure, subclasses get a standard builder, parser and printer.
///
template <typename ConcreteType, template <typename T> class... Traits>
class BinaryOp
    : public Op<ConcreteType, OpTrait::NOperands<2>::Impl, OpTrait::OneResult,
                OpTrait::SameOperandsAndResultType, Traits...> {
public:
  static void build(Builder *builder, OperationState *result, Value *lhs,
                    Value *rhs) {
    impl::buildBinaryOp(builder, result, lhs, rhs);
  }
  static bool parse(OpAsmParser *parser, OperationState *result) {
    return impl::parseBinaryOp(parser, result);
  }
  void print(OpAsmPrinter *p) const {
    return impl::printBinaryOp(this->getOperation(), p);
  }

protected:
  explicit BinaryOp(const OperationInst *state)
      : Op<ConcreteType, OpTrait::NOperands<2>::Impl, OpTrait::OneResult,
           OpTrait::SameOperandsAndResultType, Traits...>(state) {}
};

// These functions are out-of-line implementations of the methods in CastOp,
// which avoids them being template instantiated/duplicated.
namespace impl {
void buildCastOp(Builder *builder, OperationState *result, Value *source,
                 Type destType);
bool parseCastOp(OpAsmParser *parser, OperationState *result);
void printCastOp(const OperationInst *op, OpAsmPrinter *p);
} // namespace impl

/// This template is used for operations that are cast operations, that have a
/// single operand and single results, whose source and destination types are
/// different.
///
/// From this structure, subclasses get a standard builder, parser and printer.
///
template <typename ConcreteType, template <typename T> class... Traits>
class CastOp : public Op<ConcreteType, OpTrait::OneOperand, OpTrait::OneResult,
                         OpTrait::HasNoSideEffect, Traits...> {
public:
  static void build(Builder *builder, OperationState *result, Value *source,
                    Type destType) {
    impl::buildCastOp(builder, result, source, destType);
  }
  static bool parse(OpAsmParser *parser, OperationState *result) {
    return impl::parseCastOp(parser, result);
  }
  void print(OpAsmPrinter *p) const {
    return impl::printCastOp(this->getOperation(), p);
  }

protected:
  explicit CastOp(const OperationInst *state)
      : Op<ConcreteType, OpTrait::OneOperand, OpTrait::OneResult,
           OpTrait::HasNoSideEffect, Traits...>(state) {}
};

} // end namespace mlir

#endif
