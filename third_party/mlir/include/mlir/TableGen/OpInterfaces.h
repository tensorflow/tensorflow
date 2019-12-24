//===- OpInterfaces.h - OpInterfaces wrapper class --------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpInterfaces wrapper to simplify using TableGen OpInterfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_OPINTERFACES_H_
#define MLIR_TABLEGEN_OPINTERFACES_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Init;
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class with helper methods for accessing OpInterfaceMethod defined
// in TableGen.
class OpInterfaceMethod {
public:
  // This struct represents a single method argument.
  struct Argument {
    StringRef type;
    StringRef name;
  };

  explicit OpInterfaceMethod(const llvm::Record *def);

  // Return the return type of this method.
  StringRef getReturnType() const;

  // Return the name of this method.
  StringRef getName() const;

  // Return if this method is static.
  bool isStatic() const;

  // Return the body for this method if it has one.
  llvm::Optional<StringRef> getBody() const;

  // Return the default implementation for this method if it has one.
  llvm::Optional<StringRef> getDefaultImplementation() const;

  // Return the description of this method if it has one.
  llvm::Optional<StringRef> getDescription() const;

  // Arguments.
  ArrayRef<Argument> getArguments() const;
  bool arg_empty() const;

private:
  // The TableGen definition of this method.
  const llvm::Record *def;

  // The arguments of this method.
  SmallVector<Argument, 2> arguments;
};

//===----------------------------------------------------------------------===//
// OpInterface
//===----------------------------------------------------------------------===//

// Wrapper class with helper methods for accessing OpInterfaces defined in
// TableGen.
class OpInterface {
public:
  explicit OpInterface(const llvm::Record *def);

  // Return the name of this interface.
  StringRef getName() const;

  // Return the methods of this interface.
  ArrayRef<OpInterfaceMethod> getMethods() const;

  // Return the description of this method if it has one.
  llvm::Optional<StringRef> getDescription() const;

private:
  // The TableGen definition of this interface.
  const llvm::Record *def;

  // The methods of this interface.
  SmallVector<OpInterfaceMethod, 8> methods;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_OPINTERFACES_H_
