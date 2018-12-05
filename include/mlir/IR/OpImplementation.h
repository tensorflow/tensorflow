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

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class AffineExpr;
class AffineMap;
class Builder;
class Function;

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
  virtual void printType(Type type) = 0;
  virtual void printFunctionReference(const Function *func) = 0;
  virtual void printAttribute(Attribute attr) = 0;
  virtual void printAffineMap(AffineMap map) = 0;
  virtual void printAffineExpr(AffineExpr expr) = 0;

  /// Print a successor, and use list, of a terminator operation given the
  /// terminator and the successor index.
  virtual void printSuccessorAndUseList(const Operation *term,
                                        unsigned index) = 0;

  /// If the specified operation has attributes, print out an attribute
  /// dictionary with their values.  elidedAttrs allows the client to ignore
  /// specific well known attributes, commonly used if the attribute value is
  /// printed some other way (like as a fixed operand).
  virtual void
  printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                        ArrayRef<const char *> elidedAttrs = {}) = 0;

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

inline OpAsmPrinter &operator<<(OpAsmPrinter &p, Type type) {
  p.printType(type);
  return p;
}

inline OpAsmPrinter &operator<<(OpAsmPrinter &p, Attribute attr) {
  p.printAttribute(attr);
  return p;
}

inline OpAsmPrinter &operator<<(OpAsmPrinter &p, AffineMap map) {
  p.printAffineMap(map);
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
/// parse the %p, then calling `parseOperandList` with a `SquareDelimiter` to
/// parse the indices, then calling `parseColonTypeList` to parse the result
/// type.
///
class OpAsmParser {
public:
  virtual ~OpAsmParser();

  //===--------------------------------------------------------------------===//
  // High level parsing methods.
  //===--------------------------------------------------------------------===//

  // These emit an error and return true on failure, or return false on success.
  // This allows these to be chained together into a linear sequence of ||
  // expressions in many cases.

  /// Get the location of the next token and store it into the argument.  This
  /// always succeeds.
  virtual bool getCurrentLocation(llvm::SMLoc *loc) = 0;

  /// This parses... a comma!
  virtual bool parseComma() = 0;

  /// Parse a type.
  virtual bool parseType(Type &result) = 0;

  /// Parse a colon followed by a type.
  virtual bool parseColonType(Type &result) = 0;

  /// Parse a type of a specific kind, e.g. a FunctionType.
  template <typename TypeType> bool parseColonType(TypeType &result) {
    llvm::SMLoc loc;
    getCurrentLocation(&loc);

    // Parse any kind of type.
    Type type;
    if (parseColonType(type))
      return true;

    // Check for the right kind of attribute.
    result = type.dyn_cast<TypeType>();
    if (!result) {
      emitError(loc, "invalid kind of type specified");
      return true;
    }

    return false;
  }

  /// Parse a colon followed by a type list, which must have at least one type.
  virtual bool parseColonTypeList(SmallVectorImpl<Type> &result) = 0;

  /// Parse a keyword followed by a type.
  virtual bool parseKeywordType(const char *keyword, Type &result) = 0;

  /// Add the specified type to the end of the specified type list and return
  /// false.  This is a helper designed to allow parse methods to be simple and
  /// chain through || operators.
  bool addTypeToList(Type type, SmallVectorImpl<Type> &result) {
    result.push_back(type);
    return false;
  }

  /// Add the specified types to the end of the specified type list and return
  /// false.  This is a helper designed to allow parse methods to be simple and
  /// chain through || operators.
  bool addTypesToList(ArrayRef<Type> types, SmallVectorImpl<Type> &result) {
    result.append(types.begin(), types.end());
    return false;
  }

  /// Parse an arbitrary attribute and return it in result.  This also adds the
  /// attribute to the specified attribute list with the specified name.
  virtual bool parseAttribute(Attribute &result, const char *attrName,
                              SmallVectorImpl<NamedAttribute> &attrs) = 0;

  /// Parse an arbitrary attribute of a given type and return it in result. This
  /// also adds the attribute to the specified attribute list with the specified
  /// name.
  virtual bool parseAttribute(Attribute &result, Type type,
                              const char *attrName,
                              SmallVectorImpl<NamedAttribute> &attrs) = 0;

  /// Parse an attribute of a specific kind and type.
  template <typename AttrType>
  bool parseAttribute(AttrType &result, Type type, const char *attrName,
                      SmallVectorImpl<NamedAttribute> &attrs) {
    llvm::SMLoc loc;
    getCurrentLocation(&loc);

    // Parse any kind of attribute.
    Attribute attr;
    if (parseAttribute(attr, type, attrName, attrs))
      return true;

    // Check for the right kind of attribute.
    result = attr.dyn_cast<AttrType>();
    if (!result) {
      emitError(loc, "invalid kind of constant specified");
      return true;
    }

    return false;
  }

  /// If a named attribute dictionary is present, parse it into result.
  virtual bool
  parseOptionalAttributeDict(SmallVectorImpl<NamedAttribute> &result) = 0;

  /// Parse a function name like '@foo' and return the name in a form that can
  /// be passed to resolveFunctionName when a function type is available.
  virtual bool parseFunctionName(StringRef &result, llvm::SMLoc &loc) = 0;

  /// This is the representation of an operand reference.
  struct OperandType {
    llvm::SMLoc location; // Location of the token.
    StringRef name;       // Value name, e.g. %42 or %abc
    unsigned number;      // Number, e.g. 12 for an operand like %xyz#12
  };

  /// Parse a single operand.
  virtual bool parseOperand(OperandType &result) = 0;

  /// Parse a single operation successor and it's operand list.
  virtual bool
  parseSuccessorAndUseList(BasicBlock *&dest,
                           SmallVectorImpl<SSAValue *> &operands) = 0;

  /// These are the supported delimiters around operand lists, used by
  /// parseOperandList.
  enum Delimiter {
    /// Zero or more operands with no delimiters.
    None,
    /// Parens surrounding zero or more operands.
    Paren,
    /// Square brackets surrounding zero or more operands.
    Square,
    /// Parens supporting zero or more operands, or nothing.
    OptionalParen,
    /// Square brackets supporting zero or more ops, or nothing.
    OptionalSquare,
  };

  /// Parse zero or more SSA comma-separated operand references with a specified
  /// surrounding delimiter, and an optional required operand count.
  virtual bool parseOperandList(SmallVectorImpl<OperandType> &result,
                                int requiredOperandCount = -1,
                                Delimiter delimiter = Delimiter::None) = 0;

  /// Parse zero or more trailing SSA comma-separated trailing operand
  /// references with a specified surrounding delimiter, and an optional
  /// required operand count. A leading comma is expected before the operands.
  virtual bool
  parseTrailingOperandList(SmallVectorImpl<OperandType> &result,
                           int requiredOperandCount = -1,
                           Delimiter delimiter = Delimiter::None) = 0;

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
  virtual bool resolveOperand(const OperandType &operand, Type type,
                              SmallVectorImpl<SSAValue *> &result) = 0;

  /// Resolve a list of operands to SSA values, emitting an error and returning
  /// true on failure, or appending the results to the list on success.
  /// This method should be used when all operands have the same type.
  virtual bool resolveOperands(ArrayRef<OperandType> operands, Type type,
                               SmallVectorImpl<SSAValue *> &result) {
    for (auto elt : operands)
      if (resolveOperand(elt, type, result))
        return true;
    return false;
  }

  /// Resolve a list of operands and a list of operand types to SSA values,
  /// emitting an error and returning true on failure, or appending the results
  /// to the list on success.
  virtual bool resolveOperands(ArrayRef<OperandType> operands,
                               ArrayRef<Type> types, llvm::SMLoc loc,
                               SmallVectorImpl<SSAValue *> &result) {
    if (operands.size() != types.size())
      return emitError(loc, Twine(operands.size()) +
                                " operands present, but expected " +
                                Twine(types.size()));

    for (unsigned i = 0, e = operands.size(); i != e; ++i) {
      if (resolveOperand(operands[i], types[i], result))
        return true;
    }
    return false;
  }

  /// Resolve a parse function name and a type into a function reference.
  virtual bool resolveFunctionName(StringRef name, FunctionType type,
                                   llvm::SMLoc loc, Function *&result) = 0;

  /// Emit a diagnostic at the specified location and return true.
  virtual bool emitError(llvm::SMLoc loc, const Twine &message) = 0;
};

} // end namespace mlir

#endif
