//===- OpTrait.h - OpTrait wrapper class ------------------------*- C++ -*-===//
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
// OpTrait wrapper to simplify using TableGen Record defining an MLIR OpTrait.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_OPTRAIT_H_
#define MLIR_TABLEGEN_OPTRAIT_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Init;
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

class OpInterface;

// Wrapper class with helper methods for accessing OpTrait constraints defined
// in TableGen.
class OpTrait {
public:
  // Discriminator for kinds of op traits.
  enum class Kind {
    // OpTrait corresponding to C++ class.
    Native,
    // OpTrait corresponding to predicate on operation.
    Pred,
    // OpTrait controlling op definition generator internals.
    Internal,
    // OpTrait corresponding to OpInterface.
    Interface
  };

  explicit OpTrait(Kind kind, const llvm::Record *def);

  // Returns an OpTrait corresponding to the init provided.
  static OpTrait create(const llvm::Init *init);

  Kind getKind() const { return kind; }

protected:
  // The TableGen definition of this trait.
  const llvm::Record *def;
  Kind kind;
};

// OpTrait corresponding to a native C++ OpTrait.
class NativeOpTrait : public OpTrait {
public:
  // Returns the trait corresponding to a C++ trait class.
  StringRef getTrait() const;

  static bool classof(const OpTrait *t) { return t->getKind() == Kind::Native; }
};

// OpTrait corresponding to a predicate on the operation.
class PredOpTrait : public OpTrait {
public:
  // Returns the template for constructing the predicate.
  std::string getPredTemplate() const;

  // Returns the description of what the predicate is verifying.
  StringRef getDescription() const;

  static bool classof(const OpTrait *t) { return t->getKind() == Kind::Pred; }
};

// OpTrait controlling op definition generator internals.
class InternalOpTrait : public OpTrait {
public:
  // Returns the trait controlling op definition generator internals.
  StringRef getTrait() const;

  static bool classof(const OpTrait *t) {
    return t->getKind() == Kind::Internal;
  }
};

// OpTrait corresponding to an OpInterface on the operation.
class InterfaceOpTrait : public OpTrait {
public:
  // Returns member function definitions corresponding to the trait,
  OpInterface getOpInterface() const;

  // Returns the trait corresponding to a C++ trait class.
  StringRef getTrait() const;

  static bool classof(const OpTrait *t) {
    return t->getKind() == Kind::Interface;
  }

  // Whether the declaration of methods for this trait should be emitted.
  bool shouldDeclareMethods() const;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_OPTRAIT_H_
