//===- Attribute.h - Attribute wrapper class --------------------*- C++ -*-===//
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
// Attribute wrapper to simplify using TableGen Record defining a MLIR
// Attribute.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_ATTRIBUTE_H_
#define MLIR_TABLEGEN_ATTRIBUTE_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class DefInit;
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class providing helper methods for accessing MLIR Attribute defined
// in TableGen. This class should closely reflect what is defined as class
// `Attr` in TableGen.
class Attribute {
public:
  explicit Attribute(const llvm::Record &def);
  explicit Attribute(const llvm::Record *def);
  explicit Attribute(const llvm::DefInit *init);

  // Returns true if this attribute is a derived attribute (i.e., a subclass
  // of `DrivedAttr`).
  bool isDerivedAttr() const;

  // Returns true if this attribute has storage type set.
  bool hasStorageType() const;

  // Returns the storage type if set. Returns the default storage type
  // ("Attribute") otherwise.
  StringRef getStorageType() const;

  // Returns the return type for this attribute.
  StringRef getReturnType() const;

  // Returns the template getter method call which reads this attribute's
  // storage and returns the value as of the desired return type.
  // The call will contain a `{0}` which will be expanded to this attribute.
  StringRef getConvertFromStorageCall() const;

  // Returns true if this attribute can be built from a constant value.
  bool isConstBuildable() const;

  // Returns the template that can be used to produce an instance of the
  // attribute.
  // Syntax: {0} should be replaced with a builder, {1} should be replaced with
  // the constant value.
  StringRef getConstBuilderTemplate() const;

  StringRef getTableGenDefName() const;

  // Returns the code body for derived attribute. Aborts if this is not a
  // derived attribute.
  StringRef getDerivedCodeBody() const;

  // Returns the predicate that can be used to check if a attribute satisfies
  // this attribute's constraint.
  Pred getPredicate() const;

  // Returns the template that can be used to verify that an attribute satisfies
  // the constraints for its declared attribute type.
  // Syntax: {0} should be replaced with the attribute.
  std::string getConditionTemplate() const;

private:
  // The TableGen definition of this attribute.
  const llvm::Record *def;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_ATTRIBUTE_H_
