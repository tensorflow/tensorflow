/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "gml_st/transforms/transforms.h"

#include <utility>

#include "mlir/IR/Matchers.h"

namespace mlir {
namespace gml_st {

bool isZero(Value v) { return matchPattern(v, m_Zero()); }
bool isOne(Value v) { return matchPattern(v, m_One()); }

bool hasSingleElementOperandsAndResults(Operation *op) {
  auto isScalar = [](Type type) {
    return !type.isa<mlir::ShapedType>() ||
           (type.isa<TensorType>() &&
            hasSingleElement(type.cast<TensorType>()));
  };
  return llvm::all_of(op->getOperandTypes(), isScalar) &&
         llvm::all_of(op->getResultTypes(), isScalar);
}

bool isIdentitySlice(ValueRange offsets, ValueRange strides) {
  // Offsets must be all 0s and strides must be all 1s.
  return llvm::all_of(offsets, [](Value v) { return isZero(v); }) &&
         llvm::all_of(strides, [](Value v) { return isOne(v); });
}

bool haveSameStaticShape(Value lhs, Value rhs) {
  auto lhsType = lhs.getType().cast<ShapedType>();
  auto rhsType = rhs.getType().cast<ShapedType>();
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) return false;
  return lhsType == rhsType;
}

void setLabel(Operation *op, StringRef name) {
  op->setAttr(name, UnitAttr::get(op->getContext()));
}

void removeLabel(Operation *op, StringRef name) { op->removeAttr(name); }

bool hasLabel(Operation *op, StringRef name) { return op->hasAttr(name); }

constexpr llvm::StringLiteral kOpLabel = "op_label";

bool hasMatchingLabel(Operation *op, StringRef label) {
  auto opLabelAttr = op->getAttr(kOpLabel);
  if (!opLabelAttr) return false;

  return opLabelAttr.cast<StringAttr>().getValue() == label;
}

}  // namespace gml_st
}  // namespace mlir
