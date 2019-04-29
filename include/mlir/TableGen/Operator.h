//===- Operator.h - Operator class ------------------------------*- C++ -*-===//
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
// Operator wrapper to simplify using TableGen Record defining a MLIR Op.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_OPERATOR_H_
#define MLIR_TABLEGEN_OPERATOR_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {
class CodeInit;
class DefInit;
class Record;
class StringInit;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class that contains a MLIR op's information (e.g., operands,
// atributes) defined in TableGen and provides helper methods for
// accessing them.
class Operator {
public:
  explicit Operator(const llvm::Record &def);
  explicit Operator(const llvm::Record *def) : Operator(*def) {}

  // Returns the operation name.
  std::string getOperationName() const;

  // Returns this op's dialect name.
  StringRef getDialectName() const;

  // Returns this op's C++ class name.
  StringRef getCppClassName() const;

  // Returns the qualified C++ class name for the given TableGen def `name`.
  // The first `_` in `name` is treated as separating the dialect namespace
  // and the op class name if the dialect namespace is not empty. Otherwise,
  // if `name` starts with a `_`, the `_` is considered as part the class name.
  static std::string getQualCppClassName(StringRef name);

  // Returns this op's C++ class name prefixed with dialect namespace.
  std::string getQualCppClassName() const;

  // Returns the number of results this op produces.
  int getNumResults() const;

  // Returns the op result at the given `index`.
  NamedTypeConstraint &getResult(int index) { return results[index]; }
  const NamedTypeConstraint &getResult(int index) const {
    return results[index];
  }

  // Returns the `index`-th result's type constraint.
  TypeConstraint getResultTypeConstraint(int index) const;
  // Returns the `index`-th result's name.
  StringRef getResultName(int index) const;

  // Returns the number of variadic results in this operation.
  unsigned getNumVariadicResults() const;

  // Op attribute interators.
  using attribute_iterator = const NamedAttribute *;
  attribute_iterator attribute_begin() const;
  attribute_iterator attribute_end() const;
  llvm::iterator_range<attribute_iterator> getAttributes() const;

  int getNumAttributes() const { return attributes.size(); }
  // Returns the total number of native attributes.
  int getNumNativeAttributes() const;
  int getNumDerivedAttributes() const;

  // Op attribute accessors.
  NamedAttribute &getAttribute(int index) { return attributes[index]; }
  const NamedAttribute &getAttribute(int index) const;

  // Op operand iterators.
  using operand_iterator = NamedTypeConstraint *;
  operand_iterator operand_begin();
  operand_iterator operand_end();
  llvm::iterator_range<operand_iterator> getOperands();

  int getNumOperands() const { return operands.size(); }
  NamedTypeConstraint &getOperand(int index) { return operands[index]; }
  const NamedTypeConstraint &getOperand(int index) const {
    return operands[index];
  }

  // Returns the number of variadic operands in this operation.
  unsigned getNumVariadicOperands() const;

  // Returns the total number of arguments.
  int getNumArgs() const { return arguments.size(); }

  // Op argument (attribute or operand) accessors.
  Argument getArg(int index);
  StringRef getArgName(int index) const;

  // Returns the number of `PredOpTrait` traits.
  int getNumPredOpTraits() const;

  // Returns true if this op has the given MLIR C++ `trait`.
  // TODO: We should add a C++ wrapper class for TableGen OpTrait instead of
  // requiring the raw MLIR trait here.
  bool hasTrait(llvm::StringRef trait) const;

  // Trait.
  using const_trait_iterator = const OpTrait *;
  const_trait_iterator trait_begin() const;
  const_trait_iterator trait_end() const;
  llvm::iterator_range<const_trait_iterator> getTraits() const;

  ArrayRef<llvm::SMLoc> getLoc() const;

  // Query functions for the documentation of the operator.
  bool hasDescription() const;
  StringRef getDescription() const;
  bool hasSummary() const;
  StringRef getSummary() const;

private:
  // Populates the vectors containing operands, attributes, results and traits.
  void populateOpStructure();

  // The dialect name of the op.
  StringRef dialectName;

  // The unqualified C++ class name of the op.
  StringRef cppClassName;

  // The operands of the op.
  SmallVector<NamedTypeConstraint, 4> operands;

  // The attributes of the op.  Contains native attributes (corresponding to the
  // actual stored attributed of the operation) followed by derived attributes
  // (corresponding to dynamic properties of the operation that are computed
  // upon request).
  SmallVector<NamedAttribute, 4> attributes;

  // The arguments of the op (operands and native attributes).
  SmallVector<Argument, 4> arguments;

  // The results of the op.
  SmallVector<NamedTypeConstraint, 4> results;

  // The traits of the op.
  SmallVector<OpTrait, 4> traits;

  // The number of native attributes stored in the leading positions of
  // `attributes`.
  int numNativeAttributes;

  // The TableGen definition of this op.
  const llvm::Record &def;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_OPERATOR_H_
