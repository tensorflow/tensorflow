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
  StringRef getOperationName() const;

  // Returns the TableGen definition name split around '_'.
  const SmallVectorImpl<StringRef> &getSplitDefName() const;

  // Returns dialect name of the op.
  StringRef getDialectName() const;

  // Returns the C++ class name of the op.
  StringRef getCppClassName() const;

  // Returns the C++ class name of the op with namespace added.
  std::string getQualCppClassName() const;

  // Returns the number of results this op produces.
  int getNumResults() const;

  // Returns the `index`-th result's type.
  Type getResultType(int index) const;
  // Returns the `index`-th result's name.
  StringRef getResultName(int index) const;

  // Op attribute interators.
  using attribute_iterator = const NamedAttribute *;
  attribute_iterator attribute_begin() const;
  attribute_iterator attribute_end() const;
  llvm::iterator_range<attribute_iterator> getAttributes() const;

  // Op attribute accessors.
  int getNumAttributes() const { return attributes.size(); }
  // Returns the total number of native attributes.
  int getNumNativeAttributes() const;
  int getNumDerivedAttributes() const;
  NamedAttribute &getAttribute(int index) { return attributes[index]; }
  const NamedAttribute &getAttribute(int index) const;

  // Op operand iterators.
  using operand_iterator = Operand *;
  operand_iterator operand_begin();
  operand_iterator operand_end();
  llvm::iterator_range<operand_iterator> getOperands();

  // Op operand accessors.
  int getNumOperands() const { return operands.size(); }
  Operand &getOperand(int index) { return operands[index]; }
  const Operand &getOperand(int index) const { return operands[index]; }

  // Returns true if this operation has a variadic operand.
  bool hasVariadicOperand() const;

  // Op argument (attribute or operand) accessors.
  Argument getArg(int index);
  StringRef getArgName(int index) const;
  // Returns the total number of arguments.
  int getNumArgs() const { return getNumOperands() + getNumNativeAttributes(); }

  ArrayRef<llvm::SMLoc> getLoc() const;

  // Query functions for the documentation of the operator.
  bool hasDescription() const;
  StringRef getDescription() const;
  bool hasSummary() const;
  StringRef getSummary() const;

private:
  // Populates the operands and attributes.
  void populateOperandsAndAttributes();

  // The name of the op split around '_'.
  SmallVector<StringRef, 2> splittedDefName;

  // The operands of the op.
  SmallVector<Operand, 4> operands;

  // The attributes of the op.
  SmallVector<NamedAttribute, 4> attributes;

  // The start of native attributes, which are specified when creating the op
  // as a part of the op's definition.
  int nativeAttrStart;

  // The start of derived attributes, which are computed from properties of
  // the op.
  int derivedAttrStart;

  // The TableGen definition of this op.
  const llvm::Record &def;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_OPERATOR_H_
