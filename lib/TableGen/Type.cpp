//===- Type.cpp - Type class ------------------------------------*- C++ -*-===//
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
// Type wrapper to simplify using TableGen Record defining a MLIR Type.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Type.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

tblgen::Type::Type(const llvm::Record &def) : def(def) {
  assert(def.isSubClassOf("Type") &&
         "must be subclass of TableGen 'Type' class");
}

tblgen::Type::Type(const llvm::DefInit *init) : Type(*init->getDef()) {}

StringRef tblgen::Type::getTableGenDefName() const { return def.getName(); }

StringRef tblgen::Type::getBuilderCall() const {
  const auto *val = def.getValue("builderCall");
  assert(val && "TableGen 'Type' class should have 'builderCall' field");

  if (const auto *builder = dyn_cast<llvm::CodeInit>(val->getValue()))
    return builder->getValue();
  return {};
}

tblgen::PredCNF tblgen::Type::getPredicate() const {
  auto *val = def.getValue("predicate");
  assert(val && "TableGen 'Type' class should have 'predicate' field");

  const auto *pred = dyn_cast<llvm::DefInit>(val->getValue());
  return PredCNF(pred);
}
