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
// Operator wrapper to simplify using TableGen Record defining a MLIR Op.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Predicate.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#define DEBUG_TYPE "mlir-tblgen-operator"

using namespace mlir;

using llvm::DagInit;
using llvm::DefInit;
using llvm::Record;

tblgen::Operator::Operator(const llvm::Record &def)
    : dialect(def.getValueAsDef("opDialect")), def(def) {
  // The first `_` in the op's TableGen def name is treated as separating the
  // dialect prefix and the op class name. The dialect prefix will be ignored if
  // not empty. Otherwise, if def name starts with a `_`, the `_` is considered
  // as part of the class name.
  StringRef prefix;
  std::tie(prefix, cppClassName) = def.getName().split('_');
  if (prefix.empty()) {
    // Class name with a leading underscore and without dialect prefix
    cppClassName = def.getName();
  } else if (cppClassName.empty()) {
    // Class name without dialect prefix
    cppClassName = prefix;
  }

  populateOpStructure();
}

std::string tblgen::Operator::getOperationName() const {
  auto prefix = dialect.getName();
  auto opName = def.getValueAsString("opName");
  if (prefix.empty())
    return opName;
  return llvm::formatv("{0}.{1}", prefix, opName);
}

StringRef tblgen::Operator::getDialectName() const { return dialect.getName(); }

StringRef tblgen::Operator::getCppClassName() const { return cppClassName; }

std::string tblgen::Operator::getQualCppClassName() const {
  auto prefix = dialect.getCppNamespace();
  if (prefix.empty())
    return cppClassName;
  return llvm::formatv("{0}::{1}", prefix, cppClassName);
}

int tblgen::Operator::getNumResults() const {
  DagInit *results = def.getValueAsDag("results");
  return results->getNumArgs();
}

StringRef tblgen::Operator::getExtraClassDeclaration() const {
  constexpr auto attr = "extraClassDeclaration";
  if (def.isValueUnset(attr))
    return {};
  return def.getValueAsString(attr);
}

const llvm::Record &tblgen::Operator::getDef() const { return def; }

bool tblgen::Operator::isVariadic() const {
  return getNumVariadicOperands() != 0 || getNumVariadicResults() != 0;
}

bool tblgen::Operator::skipDefaultBuilders() const {
  return def.getValueAsBit("skipDefaultBuilders");
}

auto tblgen::Operator::result_begin() -> value_iterator {
  return results.begin();
}

auto tblgen::Operator::result_end() -> value_iterator { return results.end(); }

auto tblgen::Operator::getResults() -> value_range {
  return {result_begin(), result_end()};
}

tblgen::TypeConstraint
tblgen::Operator::getResultTypeConstraint(int index) const {
  DagInit *results = def.getValueAsDag("results");
  return TypeConstraint(cast<DefInit>(results->getArg(index)));
}

StringRef tblgen::Operator::getResultName(int index) const {
  DagInit *results = def.getValueAsDag("results");
  return results->getArgNameStr(index);
}

unsigned tblgen::Operator::getNumVariadicResults() const {
  return std::count_if(
      results.begin(), results.end(),
      [](const NamedTypeConstraint &c) { return c.constraint.isVariadic(); });
}

unsigned tblgen::Operator::getNumVariadicOperands() const {
  return std::count_if(
      operands.begin(), operands.end(),
      [](const NamedTypeConstraint &c) { return c.constraint.isVariadic(); });
}

tblgen::Operator::arg_iterator tblgen::Operator::arg_begin() const {
  return arguments.begin();
}

tblgen::Operator::arg_iterator tblgen::Operator::arg_end() const {
  return arguments.end();
}

tblgen::Operator::arg_range tblgen::Operator::getArgs() const {
  return {arg_begin(), arg_end()};
}

StringRef tblgen::Operator::getArgName(int index) const {
  DagInit *argumentValues = def.getValueAsDag("arguments");
  return argumentValues->getArgName(index)->getValue();
}

const tblgen::OpTrait *tblgen::Operator::getTrait(StringRef trait) const {
  for (const auto &t : traits) {
    if (auto opTrait = dyn_cast<tblgen::NativeOpTrait>(&t)) {
      if (opTrait->getTrait() == trait)
        return opTrait;
    } else if (auto opTrait = dyn_cast<tblgen::InternalOpTrait>(&t)) {
      if (opTrait->getTrait() == trait)
        return opTrait;
    } else if (auto opTrait = dyn_cast<tblgen::InterfaceOpTrait>(&t)) {
      if (opTrait->getTrait() == trait)
        return opTrait;
    }
  }
  return nullptr;
}

unsigned tblgen::Operator::getNumRegions() const { return regions.size(); }

const tblgen::NamedRegion &tblgen::Operator::getRegion(unsigned index) const {
  return regions[index];
}

auto tblgen::Operator::trait_begin() const -> const_trait_iterator {
  return traits.begin();
}
auto tblgen::Operator::trait_end() const -> const_trait_iterator {
  return traits.end();
}
auto tblgen::Operator::getTraits() const
    -> llvm::iterator_range<const_trait_iterator> {
  return {trait_begin(), trait_end()};
}

auto tblgen::Operator::attribute_begin() const -> attribute_iterator {
  return attributes.begin();
}
auto tblgen::Operator::attribute_end() const -> attribute_iterator {
  return attributes.end();
}
auto tblgen::Operator::getAttributes() const
    -> llvm::iterator_range<attribute_iterator> {
  return {attribute_begin(), attribute_end()};
}

auto tblgen::Operator::operand_begin() -> value_iterator {
  return operands.begin();
}
auto tblgen::Operator::operand_end() -> value_iterator {
  return operands.end();
}
auto tblgen::Operator::getOperands() -> value_range {
  return {operand_begin(), operand_end()};
}

auto tblgen::Operator::getArg(int index) const -> Argument {
  return arguments[index];
}

void tblgen::Operator::populateOpStructure() {
  auto &recordKeeper = def.getRecords();
  auto typeConstraintClass = recordKeeper.getClass("TypeConstraint");
  auto attrClass = recordKeeper.getClass("Attr");
  auto derivedAttrClass = recordKeeper.getClass("DerivedAttr");
  numNativeAttributes = 0;

  DagInit *argumentValues = def.getValueAsDag("arguments");
  unsigned numArgs = argumentValues->getNumArgs();

  // Handle operands and native attributes.
  for (unsigned i = 0; i != numArgs; ++i) {
    auto arg = argumentValues->getArg(i);
    auto givenName = argumentValues->getArgNameStr(i);
    auto argDefInit = dyn_cast<DefInit>(arg);
    if (!argDefInit)
      PrintFatalError(def.getLoc(),
                      Twine("undefined type for argument #") + Twine(i));
    Record *argDef = argDefInit->getDef();

    if (argDef->isSubClassOf(typeConstraintClass)) {
      operands.push_back(
          NamedTypeConstraint{givenName, TypeConstraint(argDefInit)});
    } else if (argDef->isSubClassOf(attrClass)) {
      if (givenName.empty())
        PrintFatalError(argDef->getLoc(), "attributes must be named");
      if (argDef->isSubClassOf(derivedAttrClass))
        PrintFatalError(argDef->getLoc(),
                        "derived attributes not allowed in argument list");
      attributes.push_back({givenName, Attribute(argDef)});
      ++numNativeAttributes;
    } else {
      PrintFatalError(def.getLoc(), "unexpected def type; only defs deriving "
                                    "from TypeConstraint or Attr are allowed");
    }
  }

  // Handle derived attributes.
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
      attributes.push_back(
          {cast<llvm::StringInit>(val.getNameInit())->getValue(),
           Attribute(cast<DefInit>(val.getValue()))});
    }
  }

  // Populate `arguments`. This must happen after we've finalized `operands` and
  // `attributes` because we will put their elements' pointers in `arguments`.
  // SmallVector may perform re-allocation under the hood when adding new
  // elements.
  int operandIndex = 0, attrIndex = 0;
  for (unsigned i = 0; i != numArgs; ++i) {
    Record *argDef = dyn_cast<DefInit>(argumentValues->getArg(i))->getDef();

    if (argDef->isSubClassOf(typeConstraintClass)) {
      arguments.emplace_back(&operands[operandIndex++]);
    } else {
      assert(argDef->isSubClassOf(attrClass));
      arguments.emplace_back(&attributes[attrIndex++]);
    }
  }

  auto *resultsDag = def.getValueAsDag("results");
  auto *outsOp = dyn_cast<DefInit>(resultsDag->getOperator());
  if (!outsOp || outsOp->getDef()->getName() != "outs") {
    PrintFatalError(def.getLoc(), "'results' must have 'outs' directive");
  }

  // Handle results.
  for (unsigned i = 0, e = resultsDag->getNumArgs(); i < e; ++i) {
    auto name = resultsDag->getArgNameStr(i);
    auto *resultDef = dyn_cast<DefInit>(resultsDag->getArg(i));
    if (!resultDef) {
      PrintFatalError(def.getLoc(),
                      Twine("undefined type for result #") + Twine(i));
    }
    results.push_back({name, TypeConstraint(resultDef)});
  }

  // Create list of traits, skipping over duplicates: appending to lists in
  // tablegen is easy, making them unique less so, so dedupe here.
  if (auto traitList = def.getValueAsListInit("traits")) {
    // This is uniquing based on pointers of the trait.
    SmallPtrSet<const llvm::Init *, 32> traitSet;
    traits.reserve(traitSet.size());
    for (auto traitInit : *traitList) {
      // Keep traits in the same order while skipping over duplicates.
      if (traitSet.insert(traitInit).second)
        traits.push_back(OpTrait::create(traitInit));
    }
  }

  // Handle regions
  auto *regionsDag = def.getValueAsDag("regions");
  auto *regionsOp = dyn_cast<DefInit>(regionsDag->getOperator());
  if (!regionsOp || regionsOp->getDef()->getName() != "region") {
    PrintFatalError(def.getLoc(), "'regions' must have 'region' directive");
  }

  for (unsigned i = 0, e = regionsDag->getNumArgs(); i < e; ++i) {
    auto name = regionsDag->getArgNameStr(i);
    auto *regionInit = dyn_cast<DefInit>(regionsDag->getArg(i));
    if (!regionInit) {
      PrintFatalError(def.getLoc(),
                      Twine("undefined kind for region #") + Twine(i));
    }
    regions.push_back({name, Region(regionInit->getDef())});
  }

  LLVM_DEBUG(print(llvm::dbgs()));
}

ArrayRef<llvm::SMLoc> tblgen::Operator::getLoc() const { return def.getLoc(); }

bool tblgen::Operator::hasDescription() const {
  return def.getValue("description") != nullptr;
}

StringRef tblgen::Operator::getDescription() const {
  return def.getValueAsString("description");
}

bool tblgen::Operator::hasSummary() const {
  return def.getValue("summary") != nullptr;
}

StringRef tblgen::Operator::getSummary() const {
  return def.getValueAsString("summary");
}

void tblgen::Operator::print(llvm::raw_ostream &os) const {
  os << "op '" << getOperationName() << "'\n";
  for (Argument arg : arguments) {
    if (auto *attr = arg.dyn_cast<NamedAttribute *>())
      os << "[attribute] " << attr->name << '\n';
    else
      os << "[operand] " << arg.get<NamedTypeConstraint *>()->name << '\n';
  }
}
