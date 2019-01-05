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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

std::string mlir::PredCNF::createTypeMatcherTemplate() const {
  std::string outString;
  llvm::raw_string_ostream ss(outString);
  auto conjunctiveList = conditions;
  bool firstDisjunctive = true;
  for (auto disjunctiveInit : *conjunctiveList) {
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
  ss.flush();
  return outString;
}
