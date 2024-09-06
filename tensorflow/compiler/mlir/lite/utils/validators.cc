/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/validators.h"

#include <algorithm>

#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {

// Returns true if the given `op`
//   * has an attribute with the given `name`,
//   * and the attribute is an integer list of the form [1, X, Y, 1],
// and writes X, Y as 32-bit integer attribute to `x`, `y`.
bool TFIntListIs1XY1(Operation *op, StringRef name, IntegerAttr *x,
                     IntegerAttr *y) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr) return false;

  auto elements = attr.getValue();
  if (elements.size() != 4 ||
      std::any_of(elements.begin(), elements.end(),
                  [](Attribute e) { return !mlir::isa<IntegerAttr>(e); }))
    return false;

  if (mlir::cast<IntegerAttr>(elements.front()).getInt() != 1 ||
      mlir::cast<IntegerAttr>(elements.back()).getInt() != 1)
    return false;

  Builder b(op->getContext());
  *x = b.getI32IntegerAttr(mlir::cast<IntegerAttr>(elements[1]).getInt());
  *y = b.getI32IntegerAttr(mlir::cast<IntegerAttr>(elements[2]).getInt());

  return true;
}

// Returns true if the attribute is an integer list of the form [1, X, Y, 1].
bool TFIntListIs1XY1(const Attribute attr) {
  const auto &elements = mlir::cast<ArrayAttr>(attr).getValue();
  if (elements.size() != 4 ||
      std::any_of(elements.begin(), elements.end(),
                  [](Attribute e) { return !mlir::isa<IntegerAttr>(e); }))
    return false;

  if (mlir::cast<IntegerAttr>(elements.front()).getValue() != 1 ||
      mlir::cast<IntegerAttr>(elements.back()).getValue() != 1)
    return false;
  return true;
}

// Returns true if the attribute is an integer list of the form [1, 1, X, Y].
bool TFIntListIs11XY(const Attribute attr) {
  const auto &elements = mlir::cast<ArrayAttr>(attr).getValue();
  if (elements.size() != 4 ||
      std::any_of(elements.begin(), elements.end(),
                  [](Attribute e) { return !mlir::isa<IntegerAttr>(e); }))
    return false;

  const Attribute *data = elements.data();
  if (mlir::cast<IntegerAttr>(data[0]).getValue() != 1 ||
      mlir::cast<IntegerAttr>(data[1]).getValue() != 1)
    return false;
  return true;
}

// Returns true if the given `op`
//   * has an attribute with the given `name`,
//   * and the attribute is an integer list of the form [1, X, Y, Z, 1],
// and writes X, Y as 32-bit integer attribute to `x`, `y`, z.
bool TFIntListIs1XYZ1(Operation *op, StringRef name, IntegerAttr *x,
                      IntegerAttr *y, IntegerAttr *z) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr) return false;

  auto elements = attr.getValue();
  if (elements.size() != 5 ||
      std::any_of(elements.begin(), elements.end(),
                  [](Attribute e) { return !mlir::isa<IntegerAttr>(e); }))
    return false;

  if (mlir::cast<IntegerAttr>(elements.front()).getInt() != 1 ||
      mlir::cast<IntegerAttr>(elements.back()).getInt() != 1)
    return false;

  Builder b(op->getContext());
  *x = b.getI32IntegerAttr(mlir::cast<IntegerAttr>(elements[1]).getInt());
  *y = b.getI32IntegerAttr(mlir::cast<IntegerAttr>(elements[2]).getInt());
  *z = b.getI32IntegerAttr(mlir::cast<IntegerAttr>(elements[3]).getInt());

  return true;
}

// Returns true if every element of the attribute is 1. All elements of `attr`
// must be `IntegerAttr`.
bool TFIntListIsAllOnes(const Attribute attr) {
  const auto &elements = mlir::cast<ArrayAttr>(attr).getValue();

  return !std::any_of(elements.begin(), elements.end(), [](Attribute e) {
    return mlir::cast<IntegerAttr>(e).getValue() != 1;
  });
}

bool IsBroadcastableElementsAttrs(mlir::TypedAttr a, mlir::TypedAttr b) {
  // This would return false if we had unranked tensors (where they should
  // probably be considered as broadcastable), but given we are working with
  // attributes here that shouldn't be an issue,
  return OpTrait::util::getBroadcastedType(a.getType(), b.getType()) != Type();
}

bool IsDimensionsDegenerateExceptLastOne(ArrayRef<int64_t> elements_shape) {
  if (elements_shape.empty()) return true;

  for (auto dim : elements_shape.drop_back(1)) {
    if (dim != 1) return false;
  }
  return true;
}

bool IsDimensionsDegenerateExceptLastOne(TypedAttr val) {
  if (auto ranked_type = mlir::dyn_cast<RankedTensorType>(val.getType())) {
    return IsDimensionsDegenerateExceptLastOne(ranked_type.getShape());
  }
  return false;
}

}  // namespace TFL
}  // namespace mlir
