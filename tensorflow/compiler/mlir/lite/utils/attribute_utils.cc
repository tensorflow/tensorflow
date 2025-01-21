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

#include <climits>
#include <cstddef>
#include <cstdint>

#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {

FloatAttr ExtractSingleElementAsFloat(ElementsAttr attr) {
  if (attr.getShapedType().getNumElements() != 1 ||
      !mlir::isa<FloatType>(attr.getShapedType().getElementType())) {
    return {};
  }
  return attr.getSplatValue<FloatAttr>();
}

FloatAttr GetSingleElementAsFloatOrSelf(Attribute attr) {
  if (auto m = mlir::dyn_cast_or_null<ElementsAttr>(attr)) {
    return ExtractSingleElementAsFloat(m);
  } else {
    return mlir::dyn_cast_or_null<FloatAttr>(attr);
  }
}

IntegerAttr ExtractSingleElementAsInteger(ElementsAttr attr) {
  if (attr.getShapedType().getNumElements() != 1 ||
      !attr.getShapedType().getElementType().isSignlessInteger()) {
    return {};
  }
  return attr.getSplatValue<IntegerAttr>();
}

size_t GetDenseElementBitWidth(Type elt_type) {
  // Align the width for complex to 8 to make storage and interpretation easier.
  if (ComplexType comp = llvm::dyn_cast<ComplexType>(elt_type))
    return llvm::alignTo<8>(GetDenseElementBitWidth(comp.getElementType())) * 2;
  if (elt_type.isIndex()) return IndexType::kInternalStorageBitWidth;
  return elt_type.getIntOrFloatBitWidth();
}

bool IsValidIntOrFloat(Type type, int64_t data_element_size, bool is_int,
                       bool is_signed) {
  // Make sure that the data element size is the same as the type element width.
  auto dense_elt_bit_width = GetDenseElementBitWidth(type);
  auto data_size = static_cast<size_t>(data_element_size * CHAR_BIT);
  if (dense_elt_bit_width != data_size) {
    return false;
  }

  // Check that the element type is either float or integer or index.
  if (!is_int) {
    return llvm::isa<FloatType>(type);
  }
  if (type.isIndex()) return true;

  auto int_type = llvm::dyn_cast<IntegerType>(type);
  if (!int_type) {
    return false;
  }

  // Make sure signedness semantics is consistent.
  if (int_type.isSignless()) return true;

  return int_type.isSigned() == is_signed;
}

}  // namespace TFL
}  // namespace mlir
