//===- OpImplementation.h - Classes for implementing Op types ---*- C++ -*-===//
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
// This classes used by the implementation details of Op types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPIMPLEMENTATION_H
#define MLIR_IR_OPIMPLEMENTATION_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class AffineMap;
class AffineExpr;
class Builder;

//===----------------------------------------------------------------------===//
// OpAsmPrinter
//===----------------------------------------------------------------------===//

/// This is a pure-virtual base class that exposes the asmprinter hooks
/// necessary to implement a custom print() method.
class OpAsmPrinter {
public:
  OpAsmPrinter() {}
  virtual ~OpAsmPrinter();
  virtual raw_ostream &getStream() const = 0;

  /// Print implementations for various things an operation contains.
  virtual void printOperand(const SSAValue *value) = 0;

  /// Print a comma separated list of operands.
  template <typename ContainerType>
  void printOperands(const ContainerType &container) {
    printOperands(container.begin(), container.end());
  }

  /// Print a comma separated list of operands.
  template <typename IteratorType>
  void printOperands(IteratorType it, IteratorType end) {
    if (it == end)
      return;
    printOperand(*it);
    for (++it; it != end; ++it) {
      getStream() << ", ";
      printOperand(*it);
    }
  }
  virtual void printType(const Type *type) = 0;
  virtual void printAttribute(const Attribute *attr) = 0;
  virtual void printAffineMap(const AffineMap *map) = 0;
  virtual void printAffineExpr(const AffineExpr *expr) = 0;

  /// Print the entire operation with the default verbose formatting.
  virtual void printDefaultOp(const Operation *op) = 0;

private:
  OpAsmPrinter(const OpAsmPrinter &) = delete;
  void operator=(const OpAsmPrinter &) = delete;
};

// Make the implementations convenient to use.
inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const SSAValue &value) {
  p.printOperand(&value);
  return p;
}

inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const Type &type) {
  p.printType(&type);
  return p;
}

inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const Attribute &attr) {
  p.printAttribute(&attr);
  return p;
}

inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const AffineMap &map) {
  p.printAffineMap(&map);
  return p;
}

// Support printing anything that isn't convertible to one of the above types,
// even if it isn't exactly one of them.  For example, we want to print
// FunctionType with the Type& version above, not have it match this.
template <typename T, typename std::enable_if<
                          !std::is_convertible<T &, SSAValue &>::value &&
                              !std::is_convertible<T &, Type &>::value &&
                              !std::is_convertible<T &, Attribute &>::value &&
                              !std::is_convertible<T &, AffineMap &>::value,
                          T>::type * = nullptr>
inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const T &other) {
  p.getStream() << other;
  return p;
}

//===----------------------------------------------------------------------===//
// OpAsmParser
//===----------------------------------------------------------------------===//

/// The OpAsmParser has methods for interacting with the asm parser: parsing
/// things from it, emitting errors etc.  It has an intentionally high-level API
/// that is designed to reduce/constrain syntax innovation in individual
/// operations.
///
/// For example, consider an op like this:
///
///    %x = load %p[%1, %2] : memref<...>
///
/// The "%x = load" tokens are already parsed and therefore invisible to the
/// custom op parser.  This can be supported by calling `parseOperandList` to
/// parse the %p, then calling `parseOperandList` with a `SquareDelimeter` to
/// parse the indices, then calling `parseColonTypeList` to parse the result
/// type.
///
class OpAsmParser {
public:
  virtual ~OpAsmParser();

  //===--------------------------------------------------------------------===//
  // High level parsing methods.
  //===--------------------------------------------------------------------===//

  // These return void if they always succeed.  If they can fail, they emit an
  // error and return "true".  On success, they can optionally provide location
  // information for clients who want it.

  /// This parses... a comma!
  virtual bool parseComma(llvm::SMLoc *loc = nullptr) = 0;

  /// Parse a colon followed by a type.
  virtual bool parseColonType(Type *&result, llvm::SMLoc *loc = nullptr) = 0;

  /// Parse a type of a specific kind, e.g. a FunctionType.
  template <typename TypeType>
  bool parseColonType(TypeType *&result, llvm::SMLoc *loc = nullptr) {
    // Parse any kind of type.
    Type *type;
    llvm::SMLoc tmpLoc;
    if (parseColonType(type, &tmpLoc))
      return true;
    if (loc)
      *loc = tmpLoc;

    // Check for the right kind of attribute.
    result = dyn_cast<TypeType>(type);
    if (!result) {
      emitError(tmpLoc, "invalid kind of type specified");
      return true;
    }

    return false;
  }

  /// Parse a colon followed by a type list, which must have at least one type.
  virtual bool parseColonTypeList(SmallVectorImpl<Type *> &result,
                                  llvm::SMLoc *loc = nullptr) = 0;

  /// Parse an attribute.
  virtual bool parseAttribute(Attribute *&result,
                              llvm::SMLoc *loc = nullptr) = 0;

  /// Parse an attribute of a specific kind.
  template <typename AttrType>
  bool parseAttribute(AttrType *&result, llvm::SMLoc *loc = nullptr) {
    // Parse any kind of attribute.
    Attribute *attr;
    llvm::SMLoc tmpLoc;
    if (parseAttribute(attr, &tmpLoc))
      return true;
    if (loc)
      *loc = tmpLoc;

    // Check for the right kind of attribute.
    result = dyn_cast<AttrType>(attr);
    if (!result) {
      emitError(tmpLoc, "invalid kind of constant specified");
      return true;
    }

    return false;
  }

  /// This is the representation of an operand reference.
  struct OperandType {
    llvm::SMLoc location; // Location of the token.
    StringRef name;       // Value name, e.g. %42 or %abc
    unsigned number;      // Number, e.g. 12 for an operand like %xyz#12
  };

  /// Parse a single operand.
  virtual bool parseOperand(OperandType &result) = 0;

  /// These are the supported delimeters around operand lists, used by
  /// parseOperandList.
  enum Delimeter {
    /// Zero or more operands with no delimeters.
    NoDelimeter,
    /// Parens surrounding zero or more operands.
    ParenDelimeter,
    /// Square brackets surrounding zero or more operands.
    SquareDelimeter,
    /// Parens supporting zero or more operands, or nothing.
    OptionalParenDelimeter,
    /// Square brackets supporting zero or more ops, or nothing.
    OptionalSquareDelimeter,
  };

  /// Parse zero or more SSA comma-separated operand references with a specified
  /// surrounding delimeter, and an optional required operand count.
  virtual bool
  parseOperandList(SmallVectorImpl<OperandType> &result,
                   int requiredOperandCount = -1,
                   Delimeter delimeter = Delimeter::NoDelimeter) = 0;

  //===--------------------------------------------------------------------===//
  // Methods for interacting with the parser
  //===--------------------------------------------------------------------===//

  /// Return a builder which provides useful access to MLIRContext, global
  /// objects like types and attributes.
  virtual Builder &getBuilder() const = 0;

  /// Return the location of the original name token.
  virtual llvm::SMLoc getNameLoc() const = 0;

  /// Resolve an operand to an SSA value, emitting an error and returning true
  /// on failure.
  virtual bool resolveOperand(OperandType operand, Type *type,
                              SSAValue *&result) = 0;

  /// Resolve a list of operands to SSA values, emitting an error and returning
  /// true on failure, or appending the results to the list on success.
  virtual bool resolveOperands(ArrayRef<OperandType> operand, Type *type,
                               SmallVectorImpl<SSAValue *> &result) {
    for (auto elt : operand) {
      SSAValue *value;
      if (resolveOperand(elt, type, value))
        return true;
      result.push_back(value);
    }
    return false;
  }

  /// Emit a diagnostic at the specified location.
  virtual void emitError(llvm::SMLoc loc, const Twine &message) = 0;
};

} // end namespace mlir

#endif
