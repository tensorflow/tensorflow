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
#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Region.h"
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
// attributes) defined in TableGen and provides helper methods for
// accessing them.
class Operator {
public:
  explicit Operator(const llvm::Record &def);
  explicit Operator(const llvm::Record *def) : Operator(*def) {}

  // Returns this op's dialect name.
  StringRef getDialectName() const;

  // Returns the operation name. The name will follow the "<dialect>.<op-name>"
  // format if its dialect name is not empty.
  std::string getOperationName() const;

  // Returns this op's C++ class name.
  StringRef getCppClassName() const;

  // Returns this op's C++ class name prefixed with namespaces.
  std::string getQualCppClassName() const;

  using value_iterator = NamedTypeConstraint *;
  using value_range = llvm::iterator_range<value_iterator>;

  // Returns true if this op has variadic operands or results.
  bool isVariadic() const;

  // Returns true if default builders should not be generated.
  bool skipDefaultBuilders() const;

  // Op result iterators.
  value_iterator result_begin();
  value_iterator result_end();
  value_range getResults();

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

  // Op attribute iterators.
  using attribute_iterator = const NamedAttribute *;
  attribute_iterator attribute_begin() const;
  attribute_iterator attribute_end() const;
  llvm::iterator_range<attribute_iterator> getAttributes() const;

  int getNumAttributes() const { return attributes.size(); }

  // Op attribute accessors.
  NamedAttribute &getAttribute(int index) { return attributes[index]; }

  // Op operand iterators.
  value_iterator operand_begin();
  value_iterator operand_end();
  value_range getOperands();

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
  Argument getArg(int index) const;
  StringRef getArgName(int index) const;

  // Returns true if this op has the given MLIR C++ `trait`.
  // TODO: We should add a C++ wrapper class for TableGen OpTrait instead of
  // requiring the raw MLIR trait here.
  bool hasTrait(llvm::StringRef trait) const;

  // Returns the number of regions.
  unsigned getNumRegions() const;
  // Returns the `index`-th region.
  const NamedRegion &getRegion(unsigned index) const;

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

  // Returns this op's extra class declaration code.
  StringRef getExtraClassDeclaration() const;

  // Returns the Tablegen definition this operator was constructed from.
  // TODO(antiagainst,zinenko): do not expose the TableGen record, this is a
  // temporary solution to OpEmitter requiring a Record because Operator does
  // not provide enough methods.
  const llvm::Record &getDef() const;

  // Returns the dialect of the op.
  const Dialect &getDialect() const { return dialect; }

private:
  // Populates the vectors containing operands, attributes, results and traits.
  void populateOpStructure();

  // The dialect of this op.
  Dialect dialect;

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

  // The regions of this op.
  SmallVector<NamedRegion, 1> regions;

  // The number of native attributes stored in the leading positions of
  // `attributes`.
  int numNativeAttributes;

  // The TableGen definition of this op.
  const llvm::Record &def;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_OPERATOR_H_
