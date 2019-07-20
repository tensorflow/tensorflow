//===- TypeUtilities.cpp - Helper function for type queries ---------------===//
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
// This file defines generic type utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;

Type mlir::getElementTypeOrSelf(Type type) {
  if (auto st = type.dyn_cast<ShapedType>())
    return st.getElementType();
  return type;
}

Type mlir::getElementTypeOrSelf(Value *val) {
  return getElementTypeOrSelf(val->getType());
}

Type mlir::getElementTypeOrSelf(Value &val) {
  return getElementTypeOrSelf(val.getType());
}

Type mlir::getElementTypeOrSelf(Attribute attr) {
  return getElementTypeOrSelf(attr.getType());
}

SmallVector<Type, 10> mlir::getFlattenedTypes(TupleType t) {
  SmallVector<Type, 10> fTypes;
  t.getFlattenedTypes(fTypes);
  return fTypes;
}

OperandElementTypeIterator::OperandElementTypeIterator(OperandIterator it)
    : llvm::mapped_iterator<OperandIterator, Type (*)(Value *)>(it, &unwrap) {}

Type OperandElementTypeIterator::unwrap(Value *value) {
  return value->getType().cast<ShapedType>().getElementType();
}

ResultElementTypeIterator::ResultElementTypeIterator(ResultIterator it)
    : llvm::mapped_iterator<ResultIterator, Type (*)(Value *)>(it, &unwrap) {}

Type ResultElementTypeIterator::unwrap(Value *value) {
  return value->getType().cast<ShapedType>().getElementType();
}
