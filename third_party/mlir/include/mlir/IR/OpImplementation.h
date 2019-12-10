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

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

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
  virtual void printOperand(Value *value) = 0;

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
  virtual void printAttribute(Attribute attr) = 0;

  /// Print a successor, and use list, of a terminator operation given the
  /// terminator and the successor index.
  virtual void printSuccessorAndUseList(Operation *term, unsigned index) = 0;

  /// If the specified operation has attributes, print out an attribute
  /// dictionary with their values.  elidedAttrs allows the client to ignore
  /// specific well known attributes, commonly used if the attribute value is
  /// printed some other way (like as a fixed operand).
  virtual void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                                     ArrayRef<StringRef> elidedAttrs = {}) = 0;

  /// If the specified operation has attributes, print out an attribute
  /// dictionary prefixed with 'attributes'.
  virtual void
  printOptionalAttrDictWithKeyword(ArrayRef<NamedAttribute> attrs,
                                   ArrayRef<StringRef> elidedAttrs = {}) = 0;

  /// Print the entire operation with the default generic assembly form.
  virtual void printGenericOp(Operation *op) = 0;

  /// Prints a region.
  virtual void printRegion(Region &blocks, bool printEntryBlockArgs = true,
                           bool printBlockTerminators = true) = 0;

  /// Renumber the arguments for the specified region to the same names as the
  /// SSA values in namesToUse.  This may only be used for IsolatedFromAbove
  /// operations.  If any entry in namesToUse is null, the corresponding
  /// argument name is left alone.
  virtual void shadowRegionArgs(Region &region, ValueRange namesToUse) = 0;

  /// Prints an affine map of SSA ids, where SSA id names are used in place
  /// of dims/symbols.
  /// Operand values must come from single-result sources, and be valid
  /// dimensions/symbol identifiers according to mlir::isValidDim/Symbol.
  virtual void printAffineMapOfSSAIds(AffineMapAttr mapAttr,
                                      ValueRange operands) = 0;

  /// Print an optional arrow followed by a type list.
  void printOptionalArrowTypeList(ArrayRef<Type> types) {
    if (types.empty())
      return;
    auto &os = getStream() << " -> ";
    bool wrapped = types.size() != 1 || types[0].isa<FunctionType>();
    if (wrapped)
      os << '(';
    interleaveComma(types, *this);
    if (wrapped)
      os << ')';
  }

  /// Print the complete type of an operation in functional form.
  void printFunctionalType(Operation *op) {
    auto &os = getStream();
    os << "(";
    interleaveComma(op->getNonSuccessorOperands(), os, [&](Value *operand) {
      if (operand)
        printType(operand->getType());
      else
        os << "<<NULL>";
    });
    os << ") -> ";
    if (op->getNumResults() == 1 &&
        !op->getResult(0)->getType().isa<FunctionType>()) {
      printType(op->getResult(0)->getType());
    } else {
      os << '(';
      interleaveComma(op->getResultTypes(), os);
      os << ')';
    }
  }

  /// Print the given string as a symbol reference, i.e. a form representable by
  /// a SymbolRefAttr. A symbol reference is represented as a string prefixed
  /// with '@'. The reference is surrounded with ""'s and escaped if it has any
  /// special or non-printable characters in it.
  virtual void printSymbolName(StringRef symbolRef) = 0;

private:
  OpAsmPrinter(const OpAsmPrinter &) = delete;
  void operator=(const OpAsmPrinter &) = delete;
};

// Make the implementations convenient to use.
inline OpAsmPrinter &operator<<(OpAsmPrinter &p, Value &value) {
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

// Support printing anything that isn't convertible to one of the above types,
// even if it isn't exactly one of them.  For example, we want to print
// FunctionType with the Type version above, not have it match this.
template <typename T, typename std::enable_if<
                          !std::is_convertible<T &, Value &>::value &&
                              !std::is_convertible<T &, Type &>::value &&
                              !std::is_convertible<T &, Attribute &>::value,
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

  /// Emit a diagnostic at the specified location and return failure.
  virtual InFlightDiagnostic emitError(llvm::SMLoc loc,
                                       const Twine &message = {}) = 0;

  /// Return a builder which provides useful access to MLIRContext, global
  /// objects like types and attributes.
  virtual Builder &getBuilder() const = 0;

  /// Get the location of the next token and store it into the argument.  This
  /// always succeeds.
  virtual llvm::SMLoc getCurrentLocation() = 0;
  ParseResult getCurrentLocation(llvm::SMLoc *loc) {
    *loc = getCurrentLocation();
    return success();
  }

  /// Return the location of the original name token.
  virtual llvm::SMLoc getNameLoc() const = 0;

  // These methods emit an error and return failure or success. This allows
  // these to be chained together into a linear sequence of || expressions in
  // many cases.

  /// Parse an operation in its generic form.
  /// The parsed operation is parsed in the current context and inserted in the
  /// provided block and insertion point. The results produced by this operation
  /// aren't mapped to any named value in the parser. Returns nullptr on
  /// failure.
  virtual Operation *parseGenericOperation(Block *insertBlock,
                                           Block::iterator insertPt) = 0;

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a '->' token.
  virtual ParseResult parseArrow() = 0;

  /// Parse a '->' token if present
  virtual ParseResult parseOptionalArrow() = 0;

  /// Parse a `:` token.
  virtual ParseResult parseColon() = 0;

  /// Parse a `:` token if present.
  virtual ParseResult parseOptionalColon() = 0;

  /// Parse a `,` token.
  virtual ParseResult parseComma() = 0;

  /// Parse a `,` token if present.
  virtual ParseResult parseOptionalComma() = 0;

  /// Parse a `=` token.
  virtual ParseResult parseEqual() = 0;

  /// Parse a given keyword.
  ParseResult parseKeyword(StringRef keyword, const Twine &msg = "") {
    auto loc = getCurrentLocation();
    if (parseOptionalKeyword(keyword))
      return emitError(loc, "expected '") << keyword << "'" << msg;
    return success();
  }

  /// Parse a keyword into 'keyword'.
  ParseResult parseKeyword(StringRef *keyword) {
    auto loc = getCurrentLocation();
    if (parseOptionalKeyword(keyword))
      return emitError(loc, "expected valid keyword");
    return success();
  }

  /// Parse the given keyword if present.
  virtual ParseResult parseOptionalKeyword(StringRef keyword) = 0;

  /// Parse a keyword, if present, into 'keyword'.
  virtual ParseResult parseOptionalKeyword(StringRef *keyword) = 0;

  /// Parse a `(` token.
  virtual ParseResult parseLParen() = 0;

  /// Parse a `(` token if present.
  virtual ParseResult parseOptionalLParen() = 0;

  /// Parse a `)` token.
  virtual ParseResult parseRParen() = 0;

  /// Parse a `)` token if present.
  virtual ParseResult parseOptionalRParen() = 0;

  /// Parse a `[` token.
  virtual ParseResult parseLSquare() = 0;

  /// Parse a `[` token if present.
  virtual ParseResult parseOptionalLSquare() = 0;

  /// Parse a `]` token.
  virtual ParseResult parseRSquare() = 0;

  /// Parse a `]` token if present.
  virtual ParseResult parseOptionalRSquare() = 0;

  /// Parse a `...` token if present;
  virtual ParseResult parseOptionalEllipsis() = 0;

  //===--------------------------------------------------------------------===//
  // Attribute Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an arbitrary attribute and return it in result.  This also adds the
  /// attribute to the specified attribute list with the specified name.
  ParseResult parseAttribute(Attribute &result, StringRef attrName,
                             SmallVectorImpl<NamedAttribute> &attrs) {
    return parseAttribute(result, Type(), attrName, attrs);
  }

  /// Parse an attribute of a specific kind and type.
  template <typename AttrType>
  ParseResult parseAttribute(AttrType &result, StringRef attrName,
                             SmallVectorImpl<NamedAttribute> &attrs) {
    return parseAttribute(result, Type(), attrName, attrs);
  }

  /// Parse an arbitrary attribute of a given type and return it in result. This
  /// also adds the attribute to the specified attribute list with the specified
  /// name.
  virtual ParseResult
  parseAttribute(Attribute &result, Type type, StringRef attrName,
                 SmallVectorImpl<NamedAttribute> &attrs) = 0;

  /// Parse an attribute of a specific kind and type.
  template <typename AttrType>
  ParseResult parseAttribute(AttrType &result, Type type, StringRef attrName,
                             SmallVectorImpl<NamedAttribute> &attrs) {
    llvm::SMLoc loc = getCurrentLocation();

    // Parse any kind of attribute.
    Attribute attr;
    if (parseAttribute(attr, type, attrName, attrs))
      return failure();

    // Check for the right kind of attribute.
    result = attr.dyn_cast<AttrType>();
    if (!result)
      return emitError(loc, "invalid kind of attribute specified");

    return success();
  }

  /// Parse a named dictionary into 'result' if it is present.
  virtual ParseResult
  parseOptionalAttrDict(SmallVectorImpl<NamedAttribute> &result) = 0;

  /// Parse a named dictionary into 'result' if the `attributes` keyword is
  /// present.
  virtual ParseResult
  parseOptionalAttrDictWithKeyword(SmallVectorImpl<NamedAttribute> &result) = 0;

  //===--------------------------------------------------------------------===//
  // Identifier Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an @-identifier and store it (without the '@' symbol) in a string
  /// attribute named 'attrName'.
  ParseResult parseSymbolName(StringAttr &result, StringRef attrName,
                              SmallVectorImpl<NamedAttribute> &attrs) {
    if (failed(parseOptionalSymbolName(result, attrName, attrs)))
      return emitError(getCurrentLocation())
             << "expected valid '@'-identifier for symbol name";
    return success();
  }

  /// Parse an optional @-identifier and store it (without the '@' symbol) in a
  /// string attribute named 'attrName'.
  virtual ParseResult
  parseOptionalSymbolName(StringAttr &result, StringRef attrName,
                          SmallVectorImpl<NamedAttribute> &attrs) = 0;

  //===--------------------------------------------------------------------===//
  // Operand Parsing
  //===--------------------------------------------------------------------===//

  /// This is the representation of an operand reference.
  struct OperandType {
    llvm::SMLoc location; // Location of the token.
    StringRef name;       // Value name, e.g. %42 or %abc
    unsigned number;      // Number, e.g. 12 for an operand like %xyz#12
  };

  /// Parse a single operand.
  virtual ParseResult parseOperand(OperandType &result) = 0;

  /// These are the supported delimiters around operand lists and region
  /// argument lists, used by parseOperandList and parseRegionArgumentList.
  enum class Delimiter {
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
  virtual ParseResult
  parseOperandList(SmallVectorImpl<OperandType> &result,
                   int requiredOperandCount = -1,
                   Delimiter delimiter = Delimiter::None) = 0;
  ParseResult parseOperandList(SmallVectorImpl<OperandType> &result,
                               Delimiter delimiter) {
    return parseOperandList(result, /*requiredOperandCount=*/-1, delimiter);
  }

  /// Parse zero or more trailing SSA comma-separated trailing operand
  /// references with a specified surrounding delimiter, and an optional
  /// required operand count. A leading comma is expected before the operands.
  virtual ParseResult
  parseTrailingOperandList(SmallVectorImpl<OperandType> &result,
                           int requiredOperandCount = -1,
                           Delimiter delimiter = Delimiter::None) = 0;
  ParseResult parseTrailingOperandList(SmallVectorImpl<OperandType> &result,
                                       Delimiter delimiter) {
    return parseTrailingOperandList(result, /*requiredOperandCount=*/-1,
                                    delimiter);
  }

  /// Resolve an operand to an SSA value, emitting an error on failure.
  virtual ParseResult resolveOperand(const OperandType &operand, Type type,
                                     SmallVectorImpl<Value *> &result) = 0;

  /// Resolve a list of operands to SSA values, emitting an error on failure, or
  /// appending the results to the list on success. This method should be used
  /// when all operands have the same type.
  ParseResult resolveOperands(ArrayRef<OperandType> operands, Type type,
                              SmallVectorImpl<Value *> &result) {
    for (auto elt : operands)
      if (resolveOperand(elt, type, result))
        return failure();
    return success();
  }

  /// Resolve a list of operands and a list of operand types to SSA values,
  /// emitting an error and returning failure, or appending the results
  /// to the list on success.
  ParseResult resolveOperands(ArrayRef<OperandType> operands,
                              ArrayRef<Type> types, llvm::SMLoc loc,
                              SmallVectorImpl<Value *> &result) {
    if (operands.size() != types.size())
      return emitError(loc)
             << operands.size() << " operands present, but expected "
             << types.size();

    for (unsigned i = 0, e = operands.size(); i != e; ++i)
      if (resolveOperand(operands[i], types[i], result))
        return failure();
    return success();
  }

  /// Parses an affine map attribute where dims and symbols are SSA operands.
  /// Operand values must come from single-result sources, and be valid
  /// dimensions/symbol identifiers according to mlir::isValidDim/Symbol.
  virtual ParseResult
  parseAffineMapOfSSAIds(SmallVectorImpl<OperandType> &operands, Attribute &map,
                         StringRef attrName,
                         SmallVectorImpl<NamedAttribute> &attrs) = 0;

  //===--------------------------------------------------------------------===//
  // Region Parsing
  //===--------------------------------------------------------------------===//

  /// Parses a region. Any parsed blocks are appended to "region" and must be
  /// moved to the op regions after the op is created. The first block of the
  /// region takes "arguments" of types "argTypes". If "enableNameShadowing" is
  /// set to true, the argument names are allowed to shadow the names of other
  /// existing SSA values defined above the region scope. "enableNameShadowing"
  /// can only be set to true for regions attached to operations that are
  /// "IsolatedFromAbove".
  virtual ParseResult parseRegion(Region &region,
                                  ArrayRef<OperandType> arguments,
                                  ArrayRef<Type> argTypes,
                                  bool enableNameShadowing = false) = 0;

  /// Parses a region if present.
  virtual ParseResult parseOptionalRegion(Region &region,
                                          ArrayRef<OperandType> arguments,
                                          ArrayRef<Type> argTypes,
                                          bool enableNameShadowing = false) = 0;

  /// Parse a region argument, this argument is resolved when calling
  /// 'parseRegion'.
  virtual ParseResult parseRegionArgument(OperandType &argument) = 0;

  /// Parse zero or more region arguments with a specified surrounding
  /// delimiter, and an optional required argument count. Region arguments
  /// define new values; so this also checks if values with the same names have
  /// not been defined yet.
  virtual ParseResult
  parseRegionArgumentList(SmallVectorImpl<OperandType> &result,
                          int requiredOperandCount = -1,
                          Delimiter delimiter = Delimiter::None) = 0;
  virtual ParseResult
  parseRegionArgumentList(SmallVectorImpl<OperandType> &result,
                          Delimiter delimiter) {
    return parseRegionArgumentList(result, /*requiredOperandCount=*/-1,
                                   delimiter);
  }

  /// Parse a region argument if present.
  virtual ParseResult parseOptionalRegionArgument(OperandType &argument) = 0;

  //===--------------------------------------------------------------------===//
  // Successor Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a single operation successor and its operand list.
  virtual ParseResult
  parseSuccessorAndUseList(Block *&dest,
                           SmallVectorImpl<Value *> &operands) = 0;

  //===--------------------------------------------------------------------===//
  // Type Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a type.
  virtual ParseResult parseType(Type &result) = 0;

  /// Parse an optional arrow followed by a type list.
  virtual ParseResult
  parseOptionalArrowTypeList(SmallVectorImpl<Type> &result) = 0;

  /// Parse a colon followed by a type.
  virtual ParseResult parseColonType(Type &result) = 0;

  /// Parse a colon followed by a type of a specific kind, e.g. a FunctionType.
  template <typename TypeType> ParseResult parseColonType(TypeType &result) {
    llvm::SMLoc loc = getCurrentLocation();

    // Parse any kind of type.
    Type type;
    if (parseColonType(type))
      return failure();

    // Check for the right kind of attribute.
    result = type.dyn_cast<TypeType>();
    if (!result)
      return emitError(loc, "invalid kind of type specified");

    return success();
  }

  /// Parse a colon followed by a type list, which must have at least one type.
  virtual ParseResult parseColonTypeList(SmallVectorImpl<Type> &result) = 0;

  /// Parse an optional colon followed by a type list, which if present must
  /// have at least one type.
  virtual ParseResult
  parseOptionalColonTypeList(SmallVectorImpl<Type> &result) = 0;

  /// Parse a keyword followed by a type.
  ParseResult parseKeywordType(const char *keyword, Type &result) {
    return failure(parseKeyword(keyword) || parseType(result));
  }

  /// Add the specified type to the end of the specified type list and return
  /// success.  This is a helper designed to allow parse methods to be simple
  /// and chain through || operators.
  ParseResult addTypeToList(Type type, SmallVectorImpl<Type> &result) {
    result.push_back(type);
    return success();
  }

  /// Add the specified types to the end of the specified type list and return
  /// success.  This is a helper designed to allow parse methods to be simple
  /// and chain through || operators.
  ParseResult addTypesToList(ArrayRef<Type> types,
                             SmallVectorImpl<Type> &result) {
    result.append(types.begin(), types.end());
    return success();
  }

private:
  /// Parse either an operand list or a region argument list depending on
  /// whether isOperandList is true.
  ParseResult parseOperandOrRegionArgList(SmallVectorImpl<OperandType> &result,
                                          bool isOperandList,
                                          int requiredOperandCount,
                                          Delimiter delimiter);
};

//===--------------------------------------------------------------------===//
// Dialect OpAsm interface.
//===--------------------------------------------------------------------===//

/// A functor used to set the name of the start of a result group of an
/// operation. See 'getAsmResultNames' below for more details.
using OpAsmSetValueNameFn = function_ref<void(Value *, StringRef)>;

class OpAsmDialectInterface
    : public DialectInterface::Base<OpAsmDialectInterface> {
public:
  OpAsmDialectInterface(Dialect *dialect) : Base(dialect) {}

  /// Hooks for getting identifier aliases for symbols. The identifier is used
  /// in place of the symbol when printing textual IR.
  ///
  /// Hook for defining Attribute kind aliases. This will generate an alias for
  /// all attributes of the given kind in the form : <alias>[0-9]+. These
  /// aliases must not contain `.`.
  virtual void getAttributeKindAliases(
      SmallVectorImpl<std::pair<unsigned, StringRef>> &aliases) const {}
  /// Hook for defining Attribute aliases. These aliases must not contain `.` or
  /// end with a numeric digit([0-9]+).
  virtual void getAttributeAliases(
      SmallVectorImpl<std::pair<Attribute, StringRef>> &aliases) const {}
  /// Hook for defining Type aliases.
  virtual void
  getTypeAliases(SmallVectorImpl<std::pair<Type, StringRef>> &aliases) const {}

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  virtual void getAsmResultNames(Operation *op,
                                 OpAsmSetValueNameFn setNameFn) const {}
};

//===--------------------------------------------------------------------===//
// Operation OpAsm interface.
//===--------------------------------------------------------------------===//

/// The OpAsmOpInterface, see OpAsmInterface.td for more details.
#include "mlir/IR/OpAsmInterface.h.inc"

} // end namespace mlir

#endif
