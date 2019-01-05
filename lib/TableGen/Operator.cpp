//===- Operator.cpp - Operator class --------------------------------------===//
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

#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using llvm::DagInit;
using llvm::DefInit;
using llvm::Record;

Operator::Operator(const llvm::Record &def) : def(def) {
  SplitString(def.getName(), splittedDefName, "_");
  populateOperandsAndAttributes();
}

const SmallVectorImpl<StringRef> &Operator::getSplitDefName() {
  return splittedDefName;
}

StringRef Operator::getOperationName() const {
  return def.getValueAsString("opName");
}

StringRef Operator::cppClassName() { return getSplitDefName().back(); }
std::string Operator::qualifiedCppClassName() {
  return llvm::join(getSplitDefName(), "::");
}

StringRef Operator::getArgName(int index) const {
  DagInit *argumentValues = def.getValueAsDag("arguments");
  return argumentValues->getArgName(index)->getValue();
}

auto Operator::attribute_begin() -> attribute_iterator {
  return attributes.begin();
}
auto Operator::attribute_end() -> attribute_iterator {
  return attributes.end();
}
auto Operator::getAttributes() -> llvm::iterator_range<attribute_iterator> {
  return {attribute_begin(), attribute_end()};
}

auto Operator::operand_begin() -> operand_iterator { return operands.begin(); }
auto Operator::operand_end() -> operand_iterator { return operands.end(); }
auto Operator::getOperands() -> llvm::iterator_range<operand_iterator> {
  return {operand_begin(), operand_end()};
}

auto Operator::getArg(int index) -> Argument {
  if (index < attrStart)
    return {&operands[index]};
  return {&attributes[index - attrStart]};
}

void Operator::populateOperandsAndAttributes() {
  auto &recordKeeper = def.getRecords();
  auto attrClass = recordKeeper.getClass("Attr");
  auto derivedAttrClass = recordKeeper.getClass("DerivedAttr");
  derivedAttrStart = -1;

  // The argument ordering is operands, non-derived attributes, derived
  // attributes.
  DagInit *argumentValues = def.getValueAsDag("arguments");
  unsigned i = 0;
  // Handle operands.
  for (unsigned e = argumentValues->getNumArgs(); i != e; ++i) {
    auto arg = argumentValues->getArg(i);
    auto givenName = argumentValues->getArgName(i);
    auto argDefInit = dyn_cast<DefInit>(arg);
    if (!argDefInit)
      PrintFatalError(def.getLoc(),
                      Twine("undefined type for argument ") + Twine(i));
    Record *argDef = argDefInit->getDef();
    if (argDef->isSubClassOf(attrClass))
      break;
    operands.push_back(Operand{givenName, argDefInit});
  }

  // Handle attribute.
  attrStart = i;
  for (unsigned e = argumentValues->getNumArgs(); i != e; ++i) {
    auto arg = argumentValues->getArg(i);
    auto givenName = argumentValues->getArgName(i);
    Record *argDef = cast<DefInit>(arg)->getDef();
    if (!argDef->isSubClassOf(attrClass))
      PrintFatalError(def.getLoc(),
                      Twine("expected attribute as argument ") + Twine(i));

    if (!givenName)
      PrintFatalError(argDef->getLoc(), "attributes must be named");
    bool isDerived = argDef->isSubClassOf(derivedAttrClass);
    if (isDerived)
      PrintFatalError(def.getLoc(),
                      "derived attributes not allowed in argument list");
    attributes.push_back({givenName, argDef, isDerived});
  }

  // Derived attributes.
  derivedAttrStart = i;
  for (const auto &val : def.getValues()) {
    if (auto *record = dyn_cast<llvm::RecordRecTy>(val.getType())) {
      if (!record->isSubClassOf(attrClass))
        continue;
      if (!record->isSubClassOf(derivedAttrClass))
        PrintFatalError(def.getLoc(),
                        "unexpected Attr where only DerivedAttr is allowed");

      if (record->getClasses().size() != 1) {
        PrintFatalError(
            def.getLoc(),
            "unsupported attribute modelling, only single class expected");
      }
      attributes.push_back({cast<llvm::StringInit>(val.getNameInit()),
                            cast<DefInit>(val.getValue())->getDef(),
                            /*isDerived=*/true});
    }
  }
}

bool mlir::Operator::Operand::hasMatcher() const {
  llvm::Init *matcher = defInit->getDef()->getValue("predicate")->getValue();
  return !isa<llvm::UnsetInit>(matcher);
}

std::string mlir::Operator::Operand::createTypeMatcherTemplate() const {
  auto predicate = defInit->getDef()->getValue("predicate")->getValue();
  auto predCnf = cast<llvm::DefInit>(predicate);
  PredCNF pred(predCnf->getDef()->getValueAsListInit("conditions"));
  return pred.createTypeMatcherTemplate();
}
