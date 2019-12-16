//===- StructuredOpsUtils.h - Utilities used by structured ops --*- C++ -*-===//
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
// This header file define utilities that operate on standard types and are
// useful across multiple dialects that use structured ops abstractions. These
// abstractions consist of define custom operations that encode and transport
// information about their semantics (e.g. type of iterators like parallel,
// reduction, etc..) as attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_STRUCTUREDOPSUTILS_H
#define MLIR_DIALECT_UTILS_STRUCTUREDOPSUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
/// Attribute name for the AffineArrayAttr which encodes the relationship
/// between a structured op iterators' and its operands.
static constexpr StringLiteral getIndexingMapsAttrName() {
  return StringLiteral("indexing_maps");
}

/// Attribute name for the StrArrayAttr which encodes the type of a structured
/// op's iterators.
static constexpr StringLiteral getIteratorTypesAttrName() {
  return StringLiteral("iterator_types");
}

/// Attribute name for the IntegerAttr which encodes the number of input buffer
/// arguments.
static constexpr StringLiteral getArgsInAttrName() {
  return StringLiteral("args_in");
}

/// Attribute name for the IntegerAttr which encodes the number of input buffer
/// arguments.
static constexpr StringLiteral getArgsOutAttrName() {
  return StringLiteral("args_out");
}

/// Attribute name for the StringAttr which encodes an optional documentation
/// string of the structured op.
static constexpr StringLiteral getDocAttrName() { return StringLiteral("doc"); }

/// Attribute name for the StrArrayAttr which encodes the SymbolAttr for the
/// MLIR function that implements the body of the structured op.
static constexpr StringLiteral getFunAttrName() { return StringLiteral("fun"); }

/// Attribute name for the StrArrayAttr which encodes the external library
/// function that implements the structured op.
static constexpr StringLiteral getLibraryCallAttrName() {
  return StringLiteral("library_call");
}

/// Use to encode that a particular iterator type has parallel semantics.
inline static constexpr StringLiteral getParallelIteratorTypeName() {
  return StringLiteral("parallel");
}

/// Use to encode that a particular iterator type has reduction semantics.
inline static constexpr StringLiteral getReductionIteratorTypeName() {
  return StringLiteral("reduction");
}

/// Use to encode that a particular iterator type has window semantics.
inline static constexpr StringLiteral getWindowIteratorTypeName() {
  return StringLiteral("window");
}

/// Use to encode that a particular iterator type has window semantics.
inline static ArrayRef<StringRef> getAllIteratorTypeNames() {
  static StringRef names[3] = {getParallelIteratorTypeName(),
                               getReductionIteratorTypeName(),
                               getWindowIteratorTypeName()};
  return llvm::makeArrayRef(names);
}

/// Returns the iterator of a certain type.
inline unsigned getNumIterators(StringRef name, ArrayAttr iteratorTypes) {
  auto names = getAllIteratorTypeNames();
  (void)names;
  assert(llvm::is_contained(names, name));
  return llvm::count_if(iteratorTypes, [name](Attribute a) {
    return a.cast<StringAttr>().getValue() == name;
  });
}

inline unsigned getNumIterators(ArrayAttr iteratorTypes) {
  unsigned res = 0;
  for (auto n : getAllIteratorTypeNames())
    res += getNumIterators(n, iteratorTypes);
  return res;
}

} // end namespace mlir

#endif // MLIR_UTILS_STRUCTUREDOPSUTILS_H
