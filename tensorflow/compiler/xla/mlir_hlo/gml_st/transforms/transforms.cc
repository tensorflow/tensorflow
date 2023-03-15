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

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace gml_st {

bool hasSingleElementOperandsAndResults(Operation *op) {
  auto isScalar = [](Type type) {
    return !type.isa<mlir::ShapedType>() ||
           (type.isa<TensorType>() &&
            hasSingleElement(type.cast<TensorType>()));
  };
  return llvm::all_of(op->getOperandTypes(), isScalar) &&
         llvm::all_of(op->getResultTypes(), isScalar);
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
