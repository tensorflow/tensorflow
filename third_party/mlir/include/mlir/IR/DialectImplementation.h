//===- DialectImplementation.h ----------------------------------*- C++ -*-===//
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
// This file contains utilities classes for implementing dialect attributes and
// types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECTIMPLEMENTATION_H
#define MLIR_IR_DIALECTIMPLEMENTATION_H

#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class Builder;

//===----------------------------------------------------------------------===//
// DialectAsmPrinter
//===----------------------------------------------------------------------===//

/// This is a pure-virtual base class that exposes the asmprinter hooks
/// necessary to implement a custom printAttribute/printType() method on a
/// dialect.
class DialectAsmPrinter {
public:
  DialectAsmPrinter() {}
  virtual ~DialectAsmPrinter();
  virtual raw_ostream &getStream() const = 0;

  /// Print the given attribute to the stream.
  virtual void printAttribute(Attribute attr) = 0;

  /// Print the given floating point value in a stabilized form that can be
  /// roundtripped through the IR. This is the companion to the 'parseFloat'
  /// hook on the DialectAsmParser.
  virtual void printFloat(const APFloat &value) = 0;

  /// Print the given type to the stream.
  virtual void printType(Type type) = 0;

private:
  DialectAsmPrinter(const DialectAsmPrinter &) = delete;
  void operator=(const DialectAsmPrinter &) = delete;
};

// Make the implementations convenient to use.
inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p, Attribute attr) {
  p.printAttribute(attr);
  return p;
}

inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p,
                                     const APFloat &value) {
  p.printFloat(value);
  return p;
}
inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p, float value) {
  return p << APFloat(value);
}
inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p, double value) {
  return p << APFloat(value);
}

inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p, Type type) {
  p.printType(type);
  return p;
}

// Support printing anything that isn't convertible to one of the above types,
// even if it isn't exactly one of them.  For example, we want to print
// FunctionType with the Type version above, not have it match this.
template <typename T, typename std::enable_if<
                          !std::is_convertible<T &, Attribute &>::value &&
                              !std::is_convertible<T &, Type &>::value &&
                              !std::is_convertible<T &, APFloat &>::value &&
                              !llvm::is_one_of<T, double, float>::value,
                          T>::type * = nullptr>
inline DialectAsmPrinter &operator<<(DialectAsmPrinter &p, const T &other) {
  p.getStream() << other;
  return p;
}

//===----------------------------------------------------------------------===//
// DialectAsmParser
//===----------------------------------------------------------------------===//

/// The DialectAsmParser has methods for interacting with the asm parser:
/// parsing things from it, emitting errors etc.  It has an intentionally
/// high-level API that is designed to reduce/constrain syntax innovation in
/// individual attributes or types.
class DialectAsmParser {
public:
  virtual ~DialectAsmParser();

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

  /// Re-encode the given source location as an MLIR location and return it.
  virtual Location getEncodedSourceLoc(llvm::SMLoc loc) = 0;

  /// Returns the full specification of the symbol being parsed. This allows for
  /// using a separate parser if necessary.
  virtual StringRef getFullSymbolSpec() const = 0;
};

} // end namespace mlir

#endif
