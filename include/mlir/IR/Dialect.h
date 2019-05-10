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
    Operation *, ArrayRef<Attribute>, SmallVectorImpl<Attribute> &)>;
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

  StringRef getNamespace() const { return name; }

  /// Returns true if this dialect allows for unregistered operations, i.e.
  /// operations prefixed with the dialect namespace but not registered with
  /// addOperation.
  bool allowsUnknownOperations() const { return allowUnknownOps; }

  /// Registered fallback constant fold hook for the dialect. Like the constant
  /// fold hook of each operation, it attempts to constant fold the operation
  /// with the specified constant operand values - the elements in "operands"
  /// will correspond directly to the operands of the operation, but may be null
  /// if non-constant.  If constant folding is successful, this fills in the
  /// `results` vector.  If not, this returns failure and `results` is
  /// unspecified.
  DialectConstantFoldHook constantFoldHook =
      [](Operation *op, ArrayRef<Attribute> operands,
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
  virtual Type parseType(StringRef tyData, Location loc) const;

  /// Print a type registered to this dialect.
  /// Note: The data printed for the provided type must not include any '"'
  /// characters.
  virtual void printType(Type, raw_ostream &) const {
    assert(0 && "dialect has no registered type printing hook");
  }

  /// Registered hooks for getting identifier aliases for symbols. The
  /// identifier is used in place of the symbol when printing textual IR.
  ///
  /// Hook for defining Attribute kind aliases. This will generate an alias for
  /// all attributes of the given kind in the form : <alias>[0-9]+. These
  /// aliases must not contain `.`.
  virtual void getAttributeKindAliases(
      SmallVectorImpl<std::pair<unsigned, StringRef>> &aliases) {}
  /// Hook for defining Attribute aliases. These aliases must not contain `.` or
  /// end with a numeric digit([0-9]+).
  virtual void getAttributeAliases(
      SmallVectorImpl<std::pair<Attribute, StringRef>> &aliases) {}
  /// Hook for defining Type aliases.
  virtual void
  getTypeAliases(SmallVectorImpl<std::pair<Type, StringRef>> &aliases) {}

  /// Verify an attribute from this dialect on the given function. Returns
  /// failure if the verification failed, success otherwise.
  virtual LogicalResult verifyFunctionAttribute(Function *, NamedAttribute) {
    return success();
  }

  /// Verify an attribute from this dialect on the argument at 'argIndex' for
  /// the given function. Returns failure if the verification failed, success
  /// otherwise.
  virtual LogicalResult
  verifyFunctionArgAttribute(Function *, unsigned argIndex, NamedAttribute) {
    return success();
  }

  /// Verify an attribute from this dialect on the given operation. Returns
  /// failure if the verification failed, success otherwise.
  virtual LogicalResult verifyOperationAttribute(Operation *, NamedAttribute) {
    return success();
  }

  virtual ~Dialect();

  /// Utility function that returns if the given string is a valid dialect
  /// namespace.
  static bool isValidNamespace(StringRef str);

protected:
  /// The constructor takes a unique namespace for this dialect as well as the
  /// context to bind to.
  /// Note: The namespace must not contain '.' characters.
  /// Note: All operations belonging to this dialect must have names starting
  ///       with the namespace followed by '.'.
  /// Example:
  ///       - "tf" for the TensorFlow ops like "tf.add".
  Dialect(StringRef name, MLIRContext *context);

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
    VariadicSymbolAdder<Args...>::addToSet(*this);
  }

  /// This method is used by derived classes to add their attributes to the set.
  template <typename... Args> void addAttributes() {
    VariadicSymbolAdder<Args...>::addToSet(*this);
  }

  // It would be nice to define this as variadic functions instead of a nested
  // variadic type, but we can't do that: function template partial
  // specialization is not allowed, and we can't define an overload set
  // because we don't have any arguments of the types we are pushing around.
  template <typename First, typename... Rest> struct VariadicSymbolAdder {
    static void addToSet(Dialect &dialect) {
      VariadicSymbolAdder<First>::addToSet(dialect);
      VariadicSymbolAdder<Rest...>::addToSet(dialect);
    }
  };

  template <typename First> struct VariadicSymbolAdder<First> {
    static void addToSet(Dialect &dialect) {
      dialect.addSymbol(First::getClassID());
    }
  };

  // Enable support for unregistered operations.
  void allowUnknownOperations(bool allow = true) { allowUnknownOps = allow; }

private:
  // Register a symbol(e.g. type) with its given unique class identifier.
  void addSymbol(const ClassID *const classID);

  Dialect(const Dialect &) = delete;
  void operator=(Dialect &) = delete;

  /// Register this dialect object with the specified context.  The context
  /// takes ownership of the heap allocated dialect.
  void registerDialect(MLIRContext *context);

  /// The namespace of this dialect.
  StringRef name;

  /// This is the context that owns this Dialect object.
  MLIRContext *context;

  /// Flag that toggles if this dialect supports unregistered operations, i.e.
  /// operations prefixed with the dialect namespace but not registered with
  /// addOperation.
  bool allowUnknownOps;
};

using DialectAllocatorFunction = std::function<void(MLIRContext *)>;

/// Registers a specific dialect creation function with the system, typically
/// used through the DialectRegistration template.
void registerDialectAllocator(const DialectAllocatorFunction &function);

/// Registers all dialects with the specified MLIRContext.
void registerAllDialects(MLIRContext *context);

/// Utility to register a dialect. Client can register their dialect with the
/// global registry by calling registerDialect<MyDialect>();
template <typename ConcreteDialect> void registerDialect() {
  registerDialectAllocator([](MLIRContext *ctx) {
    // Just allocate the dialect, the context takes ownership of it.
    new ConcreteDialect(ctx);
  });
}

/// DialectRegistration provides a global initialiser that registers a Dialect
/// allocation routine.
///
/// Usage:
///
///   // At namespace scope.
///   static DialectRegistration<MyDialect> Unused;
template <typename ConcreteDialect> struct DialectRegistration {
  DialectRegistration() { registerDialect<ConcreteDialect>(); }
};

} // namespace mlir

#endif
