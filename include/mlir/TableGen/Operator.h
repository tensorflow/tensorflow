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
// Operator wrapper to simplifying using Record corresponding to Operator.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_OPERATOR_H_
#define MLIR_TABLEGEN_OPERATOR_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class DefInit;
class Record;
class StringInit;
} // end namespace llvm

namespace mlir {

class Operator {
public:
  explicit Operator(const llvm::Record &def);
  explicit Operator(const llvm::Record *def) : Operator(*def) {}

  // Returns the operation name.
  StringRef getOperationName() const;

  // Returns the def name split around '_'.
  const SmallVectorImpl<StringRef> &getSplitDefName();

  // Returns the class name of the op.
  StringRef cppClassName();

  // Returns the class name of the op with namespace added.
  std::string qualifiedCppClassName();

  // Operations attribute accessors.
  struct Attribute {
    llvm::StringInit *name;
    llvm::Record *record;
    bool isDerived;
  };

  using attribute_iterator = Attribute *;
  attribute_iterator attribute_begin();
  attribute_iterator attribute_end();
  llvm::iterator_range<attribute_iterator> getAttributes();
  int getNumAttributes() { return attributes.size(); }
  Attribute &getAttribute(int index) { return attributes[index]; }
  const Attribute &getAttribute(int index) const { return attributes[index]; }

  // Operations operand accessors.
  struct Operand {
    llvm::StringInit *name;
    llvm::DefInit *defInit;
  };

  using operand_iterator = Operand *;
  operand_iterator operand_begin();
  operand_iterator operand_end();
  llvm::iterator_range<operand_iterator> getOperands();
  Operand &getOperand(int index) { return operands[index]; }
  const Operand &getOperand(int index) const { return operands[index]; }
  int getNumOperands() { return operands.size(); }

  // Operations argument accessors.
  using Argument = llvm::PointerUnion<Attribute *, Operand *>;
  Argument getArg(int index);
  StringRef getArgName(int index) const;

private:
  // Populates the operands and attributes.
  void populateOperandsAndAttributes();

  // The name of the op split around '_'.
  SmallVector<StringRef, 2> splittedDefName;

  // The operands of the op.
  SmallVector<Operand, 4> operands;

  // The attributes of the op.
  SmallVector<Attribute, 4> attributes;

  // The start of attributes.
  int attrStart;

  // The start of the derived attributes.
  int derivedAttrStart;

  const llvm::Record &def;
};

} // end namespace mlir

#endif // MLIR_TABLEGEN_OPERATOR_H_
