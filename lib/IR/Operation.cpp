//===- Operation.cpp - MLIR Operation Class -------------------------------===//
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

#include "mlir/IR/Operation.h"
using namespace mlir;

Operation::Operation(Identifier name, ArrayRef<NamedAttribute> attrs)
  : name(name), attrs(attrs.begin(), attrs.end()) {
#ifndef NDEBUG
  for (auto elt : attrs)
    assert(elt.second != nullptr && "Attributes cannot have null entries");
#endif
}

Operation::~Operation() {
}

auto Operation::removeAttr(Identifier name) -> RemoveResult {
  for (unsigned i = 0, e = attrs.size(); i != e; ++i) {
    if (attrs[i].first == name) {
      attrs.erase(attrs.begin()+i);
      return RemoveResult::Removed;
    }
  }
  return RemoveResult::NotFound;
}
