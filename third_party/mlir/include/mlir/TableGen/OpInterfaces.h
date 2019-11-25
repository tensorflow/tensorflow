//===- OpInterfaces.h - OpInterfaces wrapper class --------------*- C++ -*-===//
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
