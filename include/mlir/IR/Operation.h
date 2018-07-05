//===- Operation.h - MLIR Operation Class -----------------------*- C++ -*-===//
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

#ifndef MLIR_IR_OPERATION_H
#define MLIR_IR_OPERATION_H

#include "mlir/IR/Identifier.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
  class Attribute;
  typedef std::pair<Identifier, Attribute*> NamedAttribute;

/// Operations represent all of the arithmetic and other basic computation in
/// MLIR.  This class is the common implementation details behind OperationInst
/// and OperationStmt.
///
class Operation {
public:
  Identifier getName() const { return name; }

  // TODO: Need to have results and operands.


  // Attributes.  Operations may optionally carry a list of attributes that
  // associate constants to names.  Attributes may be dynamically added and
  // removed over the lifetime of an operation.
  //
  // We assume there will be relatively few attributes on a given operation
  // (maybe a dozen or so, but not hundreds or thousands) so we use linear
  // searches for everything.

  ArrayRef<NamedAttribute> getAttrs() const {
    return attrs;
  }

  /// Return the specified attribute if present, null otherwise.
  Attribute *getAttr(Identifier name) const {
    for (auto elt : attrs)
      if (elt.first == name)
        return elt.second;
    return nullptr;
  }

  Attribute *getAttr(StringRef name) const {
    for (auto elt : attrs)
      if (elt.first.is(name))
        return elt.second;
    return nullptr;
  }

  void addAttr(Identifier name, Attribute *value) {
    assert(value && "attributes may never be null");
    attrs.push_back({name, value});
  }

  /// Indicate whether removal found a value to remove or not.
  enum class RemoveResult {
    Removed, NotFound
  };

  RemoveResult removeAttr(Identifier name);

protected:
  Operation(Identifier name, ArrayRef<NamedAttribute> attrs);
  ~Operation();
private:
  Operation(const Operation&) = delete;
  void operator=(const Operation&) = delete;

  Identifier name;
  std::vector<NamedAttribute> attrs;
};

} // end namespace mlir

#endif
