//===- Attribute.cpp - Attribute wrapper class ----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attribute wrapper to simplify using TableGen Record defining a MLIR
// Attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

using llvm::CodeInit;
using llvm::DefInit;
using llvm::Init;
using llvm::Record;
using llvm::StringInit;

// Returns the initializer's value as string if the given TableGen initializer
// is a code or string initializer. Returns the empty StringRef otherwise.
static StringRef getValueAsString(const Init *init) {
  if (const auto *code = dyn_cast<CodeInit>(init))
    return code->getValue().trim();
  else if (const auto *str = dyn_cast<StringInit>(init))
    return str->getValue().trim();
  return {};
}

tblgen::AttrConstraint::AttrConstraint(const Record *record)
    : Constraint(Constraint::CK_Attr, record) {
  assert(isSubClassOf("AttrConstraint") &&
         "must be subclass of TableGen 'AttrConstraint' class");
}

bool tblgen::AttrConstraint::isSubClassOf(StringRef className) const {
  return def->isSubClassOf(className);
}

tblgen::Attribute::Attribute(const Record *record) : AttrConstraint(record) {
  assert(record->isSubClassOf("Attr") &&
         "must be subclass of TableGen 'Attr' class");
}

tblgen::Attribute::Attribute(const DefInit *init) : Attribute(init->getDef()) {}

bool tblgen::Attribute::isDerivedAttr() const {
  return isSubClassOf("DerivedAttr");
}

bool tblgen::Attribute::isTypeAttr() const {
  return isSubClassOf("TypeAttrBase");
}

bool tblgen::Attribute::isEnumAttr() const {
  return isSubClassOf("EnumAttrInfo");
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

tblgen::Attribute tblgen::Attribute::getBaseAttr() const {
  if (const auto *defInit =
          llvm::dyn_cast<llvm::DefInit>(def->getValueInit("baseAttr"))) {
    return Attribute(defInit).getBaseAttr();
  }
  return *this;
}

bool tblgen::Attribute::hasDefaultValue() const {
  const auto *init = def->getValueInit("defaultValue");
  return !getValueAsString(init).empty();
}

StringRef tblgen::Attribute::getDefaultValue() const {
  const auto *init = def->getValueInit("defaultValue");
  return getValueAsString(init);
}

bool tblgen::Attribute::isOptional() const {
  return def->getValueAsBit("isOptional");
}

StringRef tblgen::Attribute::getAttrDefName() const {
  if (def->isAnonymous()) {
    return getBaseAttr().def->getName();
  }
  return def->getName();
}

StringRef tblgen::Attribute::getDerivedCodeBody() const {
  assert(isDerivedAttr() && "only derived attribute has 'body' field");
  return def->getValueAsString("body");
}

tblgen::ConstantAttr::ConstantAttr(const DefInit *init) : def(init->getDef()) {
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
  assert(isSubClassOf("EnumAttrCaseInfo") &&
         "must be subclass of TableGen 'EnumAttrInfo' class");
}

bool tblgen::EnumAttrCase::isStrCase() const {
  return isSubClassOf("StrEnumAttrCase");
}

StringRef tblgen::EnumAttrCase::getSymbol() const {
  return def->getValueAsString("symbol");
}

int64_t tblgen::EnumAttrCase::getValue() const {
  return def->getValueAsInt("value");
}

tblgen::EnumAttr::EnumAttr(const llvm::Record *record) : Attribute(record) {
  assert(isSubClassOf("EnumAttrInfo") &&
         "must be subclass of TableGen 'EnumAttr' class");
}

tblgen::EnumAttr::EnumAttr(const llvm::Record &record) : Attribute(&record) {}

tblgen::EnumAttr::EnumAttr(const llvm::DefInit *init)
    : EnumAttr(init->getDef()) {}

bool tblgen::EnumAttr::isBitEnum() const { return isSubClassOf("BitEnumAttr"); }

StringRef tblgen::EnumAttr::getEnumClassName() const {
  return def->getValueAsString("className");
}

StringRef tblgen::EnumAttr::getCppNamespace() const {
  return def->getValueAsString("cppNamespace");
}

StringRef tblgen::EnumAttr::getUnderlyingType() const {
  return def->getValueAsString("underlyingType");
}

StringRef tblgen::EnumAttr::getUnderlyingToSymbolFnName() const {
  return def->getValueAsString("underlyingToSymbolFnName");
}

StringRef tblgen::EnumAttr::getStringToSymbolFnName() const {
  return def->getValueAsString("stringToSymbolFnName");
}

StringRef tblgen::EnumAttr::getSymbolToStringFnName() const {
  return def->getValueAsString("symbolToStringFnName");
}

StringRef tblgen::EnumAttr::getSymbolToStringFnRetType() const {
  return def->getValueAsString("symbolToStringFnRetType");
}

StringRef tblgen::EnumAttr::getMaxEnumValFnName() const {
  return def->getValueAsString("maxEnumValFnName");
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

tblgen::StructFieldAttr::StructFieldAttr(const llvm::Record *record)
    : def(record) {
  assert(def->isSubClassOf("StructFieldAttr") &&
         "must be subclass of TableGen 'StructFieldAttr' class");
}

tblgen::StructFieldAttr::StructFieldAttr(const llvm::Record &record)
    : StructFieldAttr(&record) {}

tblgen::StructFieldAttr::StructFieldAttr(const llvm::DefInit *init)
    : StructFieldAttr(init->getDef()) {}

StringRef tblgen::StructFieldAttr::getName() const {
  return def->getValueAsString("name");
}

tblgen::Attribute tblgen::StructFieldAttr::getType() const {
  auto init = def->getValueInit("type");
  return tblgen::Attribute(cast<llvm::DefInit>(init));
}

tblgen::StructAttr::StructAttr(const llvm::Record *record) : Attribute(record) {
  assert(isSubClassOf("StructAttr") &&
         "must be subclass of TableGen 'StructAttr' class");
}

tblgen::StructAttr::StructAttr(const llvm::DefInit *init)
    : StructAttr(init->getDef()) {}

StringRef tblgen::StructAttr::getStructClassName() const {
  return def->getValueAsString("className");
}

StringRef tblgen::StructAttr::getCppNamespace() const {
  Dialect dialect(def->getValueAsDef("structDialect"));
  return dialect.getCppNamespace();
}

std::vector<mlir::tblgen::StructFieldAttr>
tblgen::StructAttr::getAllFields() const {
  std::vector<mlir::tblgen::StructFieldAttr> attributes;

  const auto *inits = def->getValueAsListInit("fields");
  attributes.reserve(inits->size());

  for (const llvm::Init *init : *inits) {
    attributes.emplace_back(cast<llvm::DefInit>(init));
  }

  return attributes;
}
