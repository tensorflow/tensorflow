//===- Predicate.cpp - Predicate class ------------------------------------===//
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
// Wrapper around predicates defined in TableGen.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

tblgen::PredCNF::PredCNF(const llvm::Init *init) : def(nullptr) {
  if (const auto *defInit = dyn_cast<llvm::DefInit>(init)) {
    def = defInit->getDef();
    assert(def->isSubClassOf("PredCNF") &&
           "must be subclass of TableGen 'PredCNF' class");
  }
}

const llvm::ListInit *tblgen::PredCNF::getConditions() const {
  if (!def)
    return nullptr;

  return def->getValueAsListInit("conditions");
}

std::string
tblgen::PredCNF::createTypeMatcherTemplate(PredCNF predsKnownToHold) const {
  const auto *conjunctiveList = getConditions();
  if (!conjunctiveList)
    return "true";

  // Create a set of all the disjunctive conditions that hold. This is taking
  // advantage of uniquieing of lists to discard based on the pointer
  // below. This is not perfect but this will also be moved to FSM matching in
  // future and gets rid of trivial redundant checking.
  llvm::SmallSetVector<const llvm::Init *, 4> existingConditions;
  auto existingList = predsKnownToHold.getConditions();
  if (existingList) {
    for (auto disjunctiveInit : *existingList)
      existingConditions.insert(disjunctiveInit);
  }

  std::string outString;
  llvm::raw_string_ostream ss(outString);
  bool firstDisjunctive = true;
  for (auto disjunctiveInit : *conjunctiveList) {
    if (existingConditions.count(disjunctiveInit) != 0)
      continue;
    ss << (firstDisjunctive ? "(" : " && (");
    firstDisjunctive = false;
    bool firstConjunctive = true;
    for (auto atom : *cast<llvm::ListInit>(disjunctiveInit)) {
      auto predAtom = cast<llvm::DefInit>(atom)->getDef();
      ss << (firstConjunctive ? "" : " || ")
         << (predAtom->getValueAsBit("negated") ? "!" : "")
         << predAtom->getValueAsString("predCall");
      firstConjunctive = false;
    }
    ss << ")";
  }
  if (firstDisjunctive)
    return "true";
  ss.flush();
  return outString;
}
