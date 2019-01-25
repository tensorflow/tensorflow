//===- Argument.cpp - Argument definitions ----------------------*- C++ -*-===//
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

#include "mlir/TableGen/Argument.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

std::string tblgen::NamedAttribute::getName() const {
  std::string ret = name->getAsUnquotedString();
  // TODO(jpienaar): Revise this post dialect prefixing attribute discussion.
  auto split = StringRef(ret).split("__");
  if (split.second.empty())
    return ret;
  return llvm::join_items("$", split.first, split.second);
}

bool tblgen::Operand::hasMatcher() const {
  return !tblgen::TypeConstraint(*defInit).getPredicate().isNull();
}

tblgen::TypeConstraint tblgen::Operand::getTypeConstraint() const {
  return tblgen::TypeConstraint(*defInit);
}
