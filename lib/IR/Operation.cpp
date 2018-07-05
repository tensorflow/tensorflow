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

/// If an attribute exists with the specified name, change it to the new
/// value.  Otherwise, add a new attribute with the specified name/value.
void Operation::setAttr(Identifier name, Attribute *value) {
  assert(value && "attributes may never be null");
  // If we already have this attribute, replace it.
  for (auto &elt : attrs)
    if (elt.first == name) {
      elt.second = value;
      return;
    }

  // Otherwise, add it.
  attrs.push_back({name, value});
}

/// Remove the attribute with the specified name if it exists.  The return
/// value indicates whether the attribute was present or not.
auto Operation::removeAttr(Identifier name) -> RemoveResult {
  for (unsigned i = 0, e = attrs.size(); i != e; ++i) {
    if (attrs[i].first == name) {
      attrs.erase(attrs.begin()+i);
      return RemoveResult::Removed;
    }
  }
  return RemoveResult::NotFound;
}
