//===- Dialect.h - IR Dialect Description -----------------------*- C++ -*-===//
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
// This file defines the 'dialect' abstraction.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECT_H
#define MLIR_IR_DIALECT_H

#include "mlir/IR/OperationSupport.h"

namespace mlir {
class AffineMap;
class IntegerSet;
class Type;

using DialectConstantDecodeHook =
    std::function<bool(const OpaqueElementsAttr, ElementsAttr &)>;
using DialectConstantFoldHook = std::function<LogicalResult(
    const Instruction *, ArrayRef<Attribute>, SmallVectorImpl<Attribute> &)>;
using DialectExtractElementHook =
    std::function<Attribute(const OpaqueElementsAttr, ArrayRef<uint64_t>)>;

/// Dialects are groups of MLIR operations and behavior associated with the
/// entire group.  For example, hooks into other systems for constant folding,
/// default named types for asm printing, etc.
///
/// Instances of the dialect object are global across all MLIRContext's that may
/// be active in the process.
///
class Dialect {
public:
  MLIRContext *getContext() const { return context; }

  StringRef getNamespace() const { return namePrefix; }

  /// Registered fallback constant fold hook for the dialect. Like the constant
  /// fold hook of each operation, it attempts to constant fold the operation
  /// with the specified constant operand values - the elements in "operands"
  /// will correspond directly to the operands of the operation, but may be null
  /// if non-constant.  If constant folding is successful, this fills in the
  /// `results` vector.  If not, this returns failure and `results` is
  /// unspecified.
  DialectConstantFoldHook constantFoldHook =
      [](const Instruction *op, ArrayRef<Attribute> operands,
         SmallVectorImpl<Attribute> &results) { return failure(); };

  /// Registered hook to decode opaque constants associated with this
  /// dialect. The hook function attempts to decode an opaque constant tensor
  /// into a tensor with non-opaque content. If decoding is successful, this
  /// method returns false and sets 'output' attribute. If not, it returns true
  /// and leaves 'output' unspecified. The default hook fails to decode.
  DialectConstantDecodeHook decodeHook =
      [](const OpaqueElementsAttr input, ElementsAttr &output) { return true; };

  /// Registered hook to extract an element from an opaque constant associated
  /// with this dialect. If element has been successfully extracted, this
  /// method returns that element. If not, it returns an empty attribute.
  /// The default hook fails to extract an element.
  DialectExtractElementHook extractElementHook =
      [](const OpaqueElementsAttr input, ArrayRef<uint64_t> index) {
        return Attribute();
      };

  /// Parse a type registered to this dialect.
  virtual Type parseType(StringRef tyData, Location loc,
                         MLIRContext *context) const;

  /// Print a type registered to this dialect.
  /// Note: The data printed for the provided type must not include any '"'
  /// characters.
  virtual void printType(Type, raw_ostream &) const {
    assert(0 && "dialect has no registered type printing hook");
  }

  /// Registered hooks for getting identifier aliases for symbols. The
  /// identifier is used in place of the symbol when printing textual IR.
  ///
  /// Hook for defining AffineMap aliases.
  virtual void getAffineMapAliases(
      SmallVectorImpl<std::pair<StringRef, AffineMap>> &aliases) {}
  /// Hook for defining IntegerSet aliases.
  virtual void getIntegerSetAliases(
      SmallVectorImpl<std::pair<StringRef, IntegerSet>> &aliases) {}
  /// Hook for defining Type aliases.
  virtual void
  getTypeAliases(SmallVectorImpl<std::pair<StringRef, Type>> &aliases) {}

  /// Verify an attribute from this dialect on the given function. Returns true
  /// if the verification failed, false otherwise.
  virtual bool verifyFunctionAttribute(Function *, NamedAttribute) {
    return false;
  }

  /// Verify an attribute from this dialect on the argument at 'argIndex' for
  /// the given function. Returns true if the verification failed, false
  /// otherwise.
  virtual bool verifyFunctionArgAttribute(Function *, unsigned argIndex,
                                          NamedAttribute) {
    return false;
  }

  /// Verify an attribute from this dialect on the given instruction. Returns
  /// true if the verification failed, false otherwise.
  virtual bool verifyInstructionAttribute(const Instruction *, NamedAttribute) {
    return false;
  }

  virtual ~Dialect();

  /// Utility function that returns if the given string is a valid dialect
  /// namespace.
  static bool isValidNamespace(StringRef str);

protected:
  /// Note: The namePrefix can be empty, but it must not contain '.' characters.
  /// Note: If the name is non empty, then all operations belonging to this
  /// dialect will need to start with the namePrefix followed by a '.'.
  /// Example:
  ///       - "" for the standard operation set.
  ///       - "tf" for the TensorFlow ops like "tf.add".
  Dialect(StringRef namePrefix, MLIRContext *context);

  /// This method is used by derived classes to add their operations to the set.
  ///
  template <typename... Args> void addOperations() {
    VariadicOperationAdder<Args...>::addToSet(*this);
  }

  // It would be nice to define this as variadic functions instead of a nested
  // variadic type, but we can't do that: function template partial
  // specialization is not allowed, and we can't define an overload set because
  // we don't have any arguments of the types we are pushing around.
  template <typename First, typename... Rest> class VariadicOperationAdder {
  public:
    static void addToSet(Dialect &dialect) {
      dialect.addOperation(AbstractOperation::get<First>(dialect));
      VariadicOperationAdder<Rest...>::addToSet(dialect);
    }
  };

  template <typename First> class VariadicOperationAdder<First> {
  public:
    static void addToSet(Dialect &dialect) {
      dialect.addOperation(AbstractOperation::get<First>(dialect));
    }
  };

  void addOperation(AbstractOperation opInfo);

  /// This method is used by derived classes to add their types to the set.
  template <typename... Args> void addTypes() {
    VariadicTypeAdder<Args...>::addToSet(*this);
  }

  // It would be nice to define this as variadic functions instead of a nested
  // variadic type, but we can't do that: function template partial
  // specialization is not allowed, and we can't define an overload set
  // because we don't have any arguments of the types we are pushing around.
  template <typename First, typename... Rest> struct VariadicTypeAdder {
    static void addToSet(Dialect &dialect) {
      VariadicTypeAdder<First>::addToSet(dialect);
      VariadicTypeAdder<Rest...>::addToSet(dialect);
    }
  };

  template <typename First> struct VariadicTypeAdder<First> {
    static void addToSet(Dialect &dialect) {
      dialect.addType(First::getTypeID());
    }
  };

  // Register a type with its given unqiue type identifer.
  void addType(const TypeID *const typeID);

private:
  Dialect(const Dialect &) = delete;
  void operator=(Dialect &) = delete;

  /// Register this dialect object with the specified context.  The context
  /// takes ownership of the heap allocated dialect.
  void registerDialect(MLIRContext *context);

  /// This is the namespace used as a prefix for IR defined by this dialect.
  StringRef namePrefix;

  /// This is the context that owns this Dialect object.
  MLIRContext *context;
};

using DialectAllocatorFunction = std::function<void(MLIRContext *)>;

/// Registers a specific dialect creation function with the system, typically
/// used through the DialectRegistration template.
void registerDialectAllocator(const DialectAllocatorFunction &function);

/// Registers all dialects with the specified MLIRContext.
void registerAllDialects(MLIRContext *context);

/// DialectRegistration provides a global initialiser that registers a Dialect
/// allocation routine.
///
/// Usage:
///
///   // At namespace scope.
///   static DialectRegistration<MyDialect> Unused;
template <typename ConcreteDialect> struct DialectRegistration {
  DialectRegistration() {
    registerDialectAllocator([&](MLIRContext *ctx) {
      // Just allocate the dialect, the context takes ownership of it.
      new ConcreteDialect(ctx);
    });
  }
};

} // namespace mlir

#endif
