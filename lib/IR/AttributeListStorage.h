//===- AttributeListStorage.h - Attr representation for ops -----*- C++ -*-===//
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

#ifndef ATTRIBUTELISTSTORAGE_H
#define ATTRIBUTELISTSTORAGE_H

#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {

class AttributeListStorage final
    : private llvm::TrailingObjects<AttributeListStorage, NamedAttribute> {
  friend class llvm::TrailingObjects<AttributeListStorage, NamedAttribute>;

public:
  /// Given a list of NamedAttribute's, canonicalize the list (sorting
  /// by name) and return the unique'd result.  Note that the empty list is
  /// represented with a null pointer.
  static AttributeListStorage *get(ArrayRef<NamedAttribute> attrs,
                                   MLIRContext *context);

  /// Return the element constants for this aggregate constant.  These are
  /// known to all be constants.
  ArrayRef<NamedAttribute> getElements() const {
    return {getTrailingObjects<NamedAttribute>(), numElements};
  }

private:
  // This is used by the llvm::TrailingObjects base class.
  size_t numTrailingObjects(OverloadToken<NamedAttribute>) const {
    return numElements;
  }
  AttributeListStorage() = delete;
  AttributeListStorage(const AttributeListStorage &) = delete;
  AttributeListStorage(unsigned numElements) : numElements(numElements) {}

  /// This is the number of attributes.
  const unsigned numElements;
};

} // end namespace mlir

#endif
