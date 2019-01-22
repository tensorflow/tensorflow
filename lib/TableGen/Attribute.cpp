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

tblgen::Attribute::Attribute(const llvm::Record *def) : def(def) {
  assert(def->isSubClassOf("Attr") &&
         "must be subclass of TableGen 'Attr' class");
}

tblgen::Attribute::Attribute(const llvm::Record &def) : Attribute(&def) {}

tblgen::Attribute::Attribute(const llvm::DefInit *init)
    : Attribute(*init->getDef()) {}

bool tblgen::Attribute::isDerivedAttr() const {
  return def->isSubClassOf("DerivedAttr");
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

tblgen::Pred tblgen::Attribute::getPredicate() const {
  auto *val = def->getValue("predicate");
  // If no predicate is specified, then return the null predicate (which
  // corresponds to true).
  if (!val)
    return Pred();

  const auto *pred = dyn_cast<llvm::DefInit>(val->getValue());
  return Pred(pred);
}

std::string tblgen::Attribute::getConditionTemplate() const {
  return getPredicate().getCondition();
}
