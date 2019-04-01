//===- Attribute.cpp - Attribute wrapper class ------------------*- C++ -*-===//
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

#include "mlir/TableGen/Operator.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

// Returns the initializer's value as string if the given TableGen initializer
// is a code or string initializer. Returns the empty StringRef otherwise.
static StringRef getValueAsString(const llvm::Init *init) {
  if (const auto *code = dyn_cast<llvm::CodeInit>(init))
    return code->getValue().trim();
  else if (const auto *str = dyn_cast<llvm::StringInit>(init))
    return str->getValue().trim();
  return {};
}

tblgen::AttrConstraint::AttrConstraint(const llvm::Record *record)
    : Constraint(Constraint::CK_Attr, record) {
  assert(def->isSubClassOf("AttrConstraint") &&
         "must be subclass of TableGen 'AttrConstraint' class");
}

tblgen::Attribute::Attribute(const llvm::Record *record)
    : AttrConstraint(record) {
  assert(record->isSubClassOf("Attr") &&
         "must be subclass of TableGen 'Attr' class");
}

tblgen::Attribute::Attribute(const llvm::DefInit *init)
    : Attribute(init->getDef()) {}

bool tblgen::Attribute::isDerivedAttr() const {
  return def->isSubClassOf("DerivedAttr");
}

bool tblgen::Attribute::isTypeAttr() const {
  return def->isSubClassOf("TypeAttrBase");
}

bool tblgen::Attribute::hasStorageType() const {
  const auto *init = def->getValueInit("storageType");
  return !getValueAsString(init).empty();
}

StringRef tblgen::Attribute::getStorageType() const {
  const auto *init = def->getValueInit("storageType");
  auto type = getValueAsString(init);
  if (type.empty())
    return "Attribute";
  return type;
}

StringRef tblgen::Attribute::getReturnType() const {
  const auto *init = def->getValueInit("returnType");
  return getValueAsString(init);
}

StringRef tblgen::Attribute::getConvertFromStorageCall() const {
  const auto *init = def->getValueInit("convertFromStorage");
  return getValueAsString(init);
}

bool tblgen::Attribute::isConstBuildable() const {
  const auto *init = def->getValueInit("constBuilderCall");
  return !getValueAsString(init).empty();
}

StringRef tblgen::Attribute::getConstBuilderTemplate() const {
  const auto *init = def->getValueInit("constBuilderCall");
  return getValueAsString(init);
}

bool tblgen::Attribute::hasDefaultValue() const {
  const auto *init = def->getValueInit("defaultValue");
  return !getValueAsString(init).empty();
}

bool tblgen::Attribute::isOptional() const {
  return def->getValueAsBit("isOptional");
}

std::string tblgen::Attribute::getDefaultValueTemplate() const {
  assert(isConstBuildable() && "requiers constBuilderCall");
  const auto *init = def->getValueInit("defaultValue");
  return llvm::formatv(getConstBuilderTemplate().str().c_str(), "{0}",
                       getValueAsString(init));
}

StringRef tblgen::Attribute::getTableGenDefName() const {
  return def->getName();
}

StringRef tblgen::Attribute::getDerivedCodeBody() const {
  assert(isDerivedAttr() && "only derived attribute has 'body' field");
  return def->getValueAsString("body");
}

tblgen::ConstantAttr::ConstantAttr(const llvm::DefInit *init)
    : def(init->getDef()) {
  assert(def->isSubClassOf("ConstantAttr") &&
         "must be subclass of TableGen 'ConstantAttr' class");
}

tblgen::Attribute tblgen::ConstantAttr::getAttribute() const {
  return Attribute(def->getValueAsDef("attr"));
}

StringRef tblgen::ConstantAttr::getConstantValue() const {
  return def->getValueAsString("value");
}

tblgen::EnumAttrCase::EnumAttrCase(const llvm::DefInit *init)
    : Attribute(init) {
  assert(def->isSubClassOf("EnumAttrCase") &&
         "must be subclass of TableGen 'EnumAttrCase' class");
}

StringRef tblgen::EnumAttrCase::getSymbol() const {
  return def->getValueAsString("symbol");
}

tblgen::EnumAttr::EnumAttr(const llvm::Record *record) : Attribute(record) {
  assert(def->isSubClassOf("EnumAttr") &&
         "must be subclass of TableGen 'EnumAttr' class");
}

tblgen::EnumAttr::EnumAttr(const llvm::DefInit *init)
    : EnumAttr(init->getDef()) {}

StringRef tblgen::EnumAttr::getEnumClassName() const {
  return def->getValueAsString("className");
}

std::vector<tblgen::EnumAttrCase> tblgen::EnumAttr::getAllCases() const {
  const auto *inits = def->getValueAsListInit("enumerants");

  std::vector<tblgen::EnumAttrCase> cases;
  cases.reserve(inits->size());

  for (const llvm::Init *init : *inits) {
    cases.push_back(tblgen::EnumAttrCase(cast<llvm::DefInit>(init)));
  }

  return cases;
}
