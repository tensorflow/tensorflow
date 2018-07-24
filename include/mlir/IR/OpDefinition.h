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
// This file implements helper classes for implementing the "Op" types.  Most of
// this goes into the mlir::OpImpl namespace since they are only used by code
// that is defining the op implementations, not by clients.
//
// The purpose of these types are to allow light-weight implementation of
// concrete ops (like DimOp) with very little boilerplate.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPDEFINITION_H
#define MLIR_IR_OPDEFINITION_H

#include "mlir/IR/Operation.h"

namespace mlir {
class Type;
class OpAsmPrinter;

/// This pointer represents a notional "Operation*" but where the actual
/// storage of the pointer is maintained in the templated "OpType" class.
template <typename OpType>
class OpPointer {
public:
  explicit OpPointer(OpType value) : value(value) {}

  OpType &operator*() { return value; }

  OpType *operator->() { return &value; }

  operator bool() const { return value.getOperation(); }

public:
  OpType value;
};

/// This pointer represents a notional "const Operation*" but where the actual
/// storage of the pointer is maintained in the templated "OpType" class.
template <typename OpType>
class ConstOpPointer {
public:
  explicit ConstOpPointer(OpType value) : value(value) {}

  const OpType &operator*() const { return value; }

  const OpType *operator->() const { return &value; }

  /// Return true if non-null.
  operator bool() const { return value.getOperation(); }

public:
  const OpType value;
};

//===----------------------------------------------------------------------===//
// OpImpl Types
//===----------------------------------------------------------------------===//

namespace OpImpl {

/// This is the concrete base class that holds the operation pointer and has
/// non-generic methods that only depend on State (to avoid having them
/// instantiated on template types that don't affect them.
///
/// This also has the fallback implementations of customization hooks for when
/// they aren't customized.
class BaseState {
public:
  /// Return the operation that this refers to.
  const Operation *getOperation() const { return state; }
  Operation *getOperation() { return state; }

  /// Return an attribute with the specified name.
  Attribute *getAttr(StringRef name) const { return state->getAttr(name); }

  /// If the operation has an attribute of the specified type, return it.
  template <typename AttrClass>
  AttrClass *getAttrOfType(StringRef name) const {
    return dyn_cast_or_null<AttrClass>(getAttr(name));
  }

  /// If the an attribute exists with the specified name, change it to the new
  /// value.  Otherwise, add a new attribute with the specified name/value.
  void setAttr(Identifier name, Attribute *value, MLIRContext *context) {
    state->setAttr(name, value, context);
  }

protected:
  // These are default implementations of customization hooks.

  /// If the concrete type didn't implement a custom verifier hook, just fall
  /// back to this one which accepts everything.
  const char *verify() const { return nullptr; }

  // The fallback for the printer is to print it the longhand form.
  void print(OpAsmPrinter *p) const;

  /// Mutability management is handled by the OpWrapper/OpConstWrapper classes,
  /// so we can cast it away here.
  explicit BaseState(const Operation *state)
      : state(const_cast<Operation *>(state)) {}

private:
  Operation *state;
};

/// This provides public APIs that all operations should have.  The template
/// argument 'ConcreteType' should be the concrete type by CRTP and the others
/// are base classes by the policy pattern.
template <typename ConcreteType, template <typename T> class... Traits>
class Base : public BaseState, public Traits<ConcreteType>... {
public:
  /// Return the operation that this refers to.
  const Operation *getOperation() const { return BaseState::getOperation(); }
  Operation *getOperation() { return BaseState::getOperation(); }

  /// Return true if this "op class" can match against the specified operation.
  /// This hook can be overridden with a more specific implementation in
  /// the subclass of Base.
  ///
  static bool isClassFor(const Operation *op) {
    return op->getName().is(ConcreteType::getOperationName());
  }

  /// This is the hook used by the AsmPrinter to emit this to the .mlir file.
  /// Op implementations should provide a print method.
  static void printAssembly(const Operation *op, OpAsmPrinter *p) {
    op->getAs<ConcreteType>()->print(p);
  }

  /// This is the hook used by the Verifier to check out this instruction.  It
  /// delegates to the Traits for their policy implementations, and allows the
  /// user to specify their own verify() method.
  static const char *verifyInvariants(const Operation *op) {
    if (auto error = BaseVerifier<Traits<ConcreteType>...>::verifyTrait(op))
      return error;
    return op->getAs<ConcreteType>()->verify();
  }

  // TODO: Provide a dump() method.

protected:
  explicit Base(const Operation *state) : BaseState(state) {}

private:
  template <typename... Types>
  struct BaseVerifier;

  template <typename First, typename... Rest>
  struct BaseVerifier<First, Rest...> {
    static const char *verifyTrait(const Operation *op) {
      if (auto error = First::verifyTrait(op))
        return error;
      return BaseVerifier<Rest...>::verifyTrait(op);
    }
  };

  template <typename First>
  struct BaseVerifier<First> {
    static const char *verifyTrait(const Operation *op) {
      return First::verifyTrait(op);
    }
  };

  template <>
  struct BaseVerifier<> {
    static const char *verifyTrait(const Operation *op) { return nullptr; }
  };
};

/// Helper class for implementing traits.  Clients are not expected to interact
/// with this directly, so its members are all protected.
template <typename ConcreteType, template <typename> class TraitType>
class TraitImpl {
protected:
  /// Return the ultimate Operation being worked on.
  Operation *getOperation() {
    // We have to cast up to the trait type, then to the concrete type, then to
    // the BaseState class in explicit hops because the concrete type will
    // multiply derive from the (content free) TraitImpl class, and we need to
    // be able to disambiguate the path for the C++ compiler.
    auto *trait = static_cast<TraitType<ConcreteType> *>(this);
    auto *concrete = static_cast<ConcreteType *>(trait);
    auto *base = static_cast<BaseState *>(concrete);
    return base->getOperation();
  }
  const Operation *getOperation() const {
    return const_cast<TraitImpl *>(this)->getOperation();
  }

  /// Provide default implementations of trait hooks.  This allows traits to
  /// provide exactly the overrides they care about.
  static const char *verifyTrait(const Operation *op) { return nullptr; }
};

/// This class provides the API for ops that are known to have exactly one
/// SSA operand.
template <typename ConcreteType>
class ZeroOperands : public TraitImpl<ConcreteType, ZeroOperands> {
public:
  static const char *verifyTrait(const Operation *op) {
    if (op->getNumOperands() != 0)
      return "requires zero operands";
    return nullptr;
  }

private:
  // Disable these.
  void getOperand() const {}
  void setOperand() const {}
};

/// This class provides the API for ops that are known to have exactly one
/// SSA operand.
template <typename ConcreteType>
class OneOperand : public TraitImpl<ConcreteType, OneOperand> {
public:
  const SSAValue *getOperand() const {
    return this->getOperation()->getOperand(0);
  }

  SSAValue *getOperand() { return this->getOperation()->getOperand(0); }

  void setOperand(SSAValue *value) {
    this->getOperation()->setOperand(0, value);
  }

  static const char *verifyTrait(const Operation *op) {
    if (op->getNumOperands() != 1)
      return "requires a single operand";
    return nullptr;
  }
};

/// This class provides the API for ops that are known to have exactly two
/// SSA operands.
template <typename ConcreteType>
class TwoOperands : public TraitImpl<ConcreteType, TwoOperands> {
public:
  const SSAValue *getOperand(unsigned i) const {
    return this->getOperation()->getOperand(i);
  }

  SSAValue *getOperand(unsigned i) {
    return this->getOperation()->getOperand(i);
  }

  void setOperand(unsigned i, SSAValue *value) {
    this->getOperation()->setOperand(i, value);
  }

  static const char *verifyTrait(const Operation *op) {
    if (op->getNumOperands() != 2)
      return "requires two operands";
    return nullptr;
  }
};

/// This class provides return value APIs for ops that are known to have a
/// single result.
template <typename ConcreteType>
class OneResult : public TraitImpl<ConcreteType, OneResult> {
public:
  SSAValue *getResult() { return this->getOperation()->getResult(0); }
  const SSAValue *getResult() const {
    return this->getOperation()->getResult(0);
  }

  Type *getType() const { return getResult()->getType(); }

  static const char *verifyTrait(const Operation *op) {
    if (op->getNumResults() != 1)
      return "requires one result";
    return nullptr;
  }
};

} // end namespace OpImpl

} // end namespace mlir

#endif
