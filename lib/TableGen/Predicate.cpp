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

// Construct a Predicate from a record.
tblgen::Pred::Pred(const llvm::Record *def) : def(def) {
  assert(def->isSubClassOf("Pred") &&
         "must be a subclass of TableGen 'Pred' class");
}

// Construct a Predicate from an initializer.
tblgen::Pred::Pred(const llvm::Init *init) : def(nullptr) {
  if (const auto *defInit = dyn_cast_or_null<llvm::DefInit>(init))
    def = defInit->getDef();
}

// Get condition of the Predicate.
StringRef tblgen::Pred::getCondition() const {
  assert(!isNull() && "null predicate does not have a condition");
  return def->getValueAsString("predCall");
}
