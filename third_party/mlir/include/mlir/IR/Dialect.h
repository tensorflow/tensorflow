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
class DialectAsmParser;
class DialectAsmPrinter;
class DialectInterface;
class OpBuilder;
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
  virtual ~Dialect();

  /// Utility function that returns if the given string is a valid dialect
  /// namespace.
  static bool isValidNamespace(StringRef str);

  MLIRContext *getContext() const { return context; }

  StringRef getNamespace() const { return name; }

  /// Returns true if this dialect allows for unregistered operations, i.e.
  /// operations prefixed with the dialect namespace but not registered with
  /// addOperation.
  bool allowsUnknownOperations() const { return unknownOpsAllowed; }

  /// Return true if this dialect allows for unregistered types, i.e., types
  /// prefixed with the dialect namespace but not registered with addType.
  /// These are represented with OpaqueType.
  bool allowsUnknownTypes() const { return unknownTypesAllowed; }

  //===--------------------------------------------------------------------===//
  // Constant Hooks
  //===--------------------------------------------------------------------===//

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

  /// Registered hook to materialize a single constant operation from a given
  /// attribute value with the desired resultant type. This method should use
  /// the provided builder to create the operation without changing the
  /// insertion position. The generated operation is expected to be constant
  /// like, i.e. single result, zero operands, non side-effecting, etc. On
  /// success, this hook should return the value generated to represent the
  /// constant value. Otherwise, it should return null on failure.
  virtual Operation *materializeConstant(OpBuilder &builder, Attribute value,
                                         Type type, Location loc) {
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Parsing Hooks
  //===--------------------------------------------------------------------===//

  /// Parse an attribute registered to this dialect. If 'type' is nonnull, it
  /// refers to the expected type of the attribute.
  virtual Attribute parseAttribute(DialectAsmParser &parser, Type type) const;

  /// Print an attribute registered to this dialect. Note: The type of the
  /// attribute need not be printed by this method as it is always printed by
  /// the caller.
  virtual void printAttribute(Attribute, DialectAsmPrinter &) const {
    llvm_unreachable("dialect has no registered attribute printing hook");
  }

  /// Parse a type registered to this dialect.
  virtual Type parseType(DialectAsmParser &parser) const;

  /// Print a type registered to this dialect.
  virtual void printType(Type, DialectAsmPrinter &) const {
    llvm_unreachable("dialect has no registered type printing hook");
  }

  //===--------------------------------------------------------------------===//
  // Verification Hooks
  //===--------------------------------------------------------------------===//

  /// Verify an attribute from this dialect on the argument at 'argIndex' for
  /// the region at 'regionIndex' on the given operation. Returns failure if
  /// the verification failed, success otherwise. This hook may optionally be
  /// invoked from any operation containing a region.
  virtual LogicalResult verifyRegionArgAttribute(Operation *,
                                                 unsigned regionIndex,
                                                 unsigned argIndex,
                                                 NamedAttribute);

  /// Verify an attribute from this dialect on the result at 'resultIndex' for
  /// the region at 'regionIndex' on the given operation. Returns failure if
  /// the verification failed, success otherwise. This hook may optionally be
  /// invoked from any operation containing a region.
  virtual LogicalResult verifyRegionResultAttribute(Operation *,
                                                    unsigned regionIndex,
                                                    unsigned resultIndex,
                                                    NamedAttribute);

  /// Verify an attribute from this dialect on the given operation. Returns
  /// failure if the verification failed, success otherwise.
  virtual LogicalResult verifyOperationAttribute(Operation *, NamedAttribute) {
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Interfaces
  //===--------------------------------------------------------------------===//

  /// Lookup an interface for the given ID if one is registered, otherwise
  /// nullptr.
  const DialectInterface *getRegisteredInterface(ClassID *interfaceID) {
    auto it = registeredInterfaces.find(interfaceID);
    return it != registeredInterfaces.end() ? it->getSecond().get() : nullptr;
  }
  template <typename InterfaceT> const InterfaceT *getRegisteredInterface() {
    return static_cast<const InterfaceT *>(
        getRegisteredInterface(InterfaceT::getInterfaceID()));
  }

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

  /// Enable support for unregistered operations.
  void allowUnknownOperations(bool allow = true) { unknownOpsAllowed = allow; }

  /// Enable support for unregistered types.
  void allowUnknownTypes(bool allow = true) { unknownTypesAllowed = allow; }

  /// Register a dialect interface with this dialect instance.
  void addInterface(std::unique_ptr<DialectInterface> interface);

  /// Register a set of dialect interfaces with this dialect instance.
  template <typename T, typename T2, typename... Tys> void addInterfaces() {
    addInterfaces<T>();
    addInterfaces<T2, Tys...>();
  }
  template <typename T> void addInterfaces() {
    addInterface(std::make_unique<T>(this));
  }

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

  /// Flag that specifies whether this dialect supports unregistered operations,
  /// i.e. operations prefixed with the dialect namespace but not registered
  /// with addOperation.
  bool unknownOpsAllowed = false;

  /// Flag that specifies whether this dialect allows unregistered types, i.e.
  /// types prefixed with the dialect namespace but not registered with addType.
  /// These types are represented with OpaqueType.
  bool unknownTypesAllowed = false;

  /// A collection of registered dialect interfaces.
  DenseMap<ClassID *, std::unique_ptr<DialectInterface>> registeredInterfaces;
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

/// DialectRegistration provides a global initializer that registers a Dialect
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
